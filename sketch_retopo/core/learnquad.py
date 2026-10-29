"""Patch database lookup: a port of LearningQuadrangulationsCore, the code behind "Data-Driven Interactive
Quadrangulation" (Marcias, Takayama, Pietroni, Panozzo, Sorkine-Hornung, Puppo, Cignoni; SIGGRAPH 2015), as used by
SketchRetopo2's query database mode and by its patch generation for loops with more than 6 corners.

The templates come from the demo's sqlite file (patches.db).  The original runs the lookup on worker threads and
notifies the UI; here it runs incrementally (`Lookup.step(time_budget)`) so that the host can drive it from a timer.

Meshes are kept as plain lists (face -> 4 vertices, face -> 4 adjacent faces and their edge indices) and walked with
`Pos`, which follows vcg::face::Pos exactly, so that the walks of the original code can be ported literally."""

import bisect
import math
import os
import sqlite3
import sys
import time
import urllib.request

import numpy as np

SEPARATOR = '-'
BORDER = '#'
VARLABEL = 'x'
VALENCELABEL = 'v'
VALENCESEP = ':'
NUM_HEX = 16                    # hex digits of num_type
LABEL_HEX = 4                   # hex digits of var_label_type / valence_type
BORDER_STR = BORDER * NUM_HEX
NC_MAX = 0xFFFF                 # numeric_limits<n_corners_type>::max()
LOOP = (NC_MAX, NC_MAX)         # FlowTopology of a closed flow
SCALAR_MAX = sys.float_info.max
INF = float('inf')
PENALTY = 10000.0               # vcg::ImplicitSmoother
SMOOTH_LAMBDA = 0.2
NUM_FLOW_SAMPLES = 200

# PatchCompare::CompareCriteria
PATCH_DEFAULT = 0
SINGULARITYTYPES = 1
SINGULARITYNUM = 2
FLOW_DISTANCE = 4
QUADS_QUALITY = 8
SORT_BY_FLOW = (FLOW_DISTANCE, SINGULARITYNUM, SINGULARITYTYPES, QUADS_QUALITY)
SORT_BY_QUALITY = (QUADS_QUALITY, SINGULARITYNUM, SINGULARITYTYPES, FLOW_DISTANCE)
SORT_BY_NUMSING = (SINGULARITYNUM, SINGULARITYTYPES, FLOW_DISTANCE, QUADS_QUALITY)


class MalformedTemplate(Exception):
    pass


# ---- quad meshes and vcg-style positions -----------------------------------------------------------------------------
class QuadMesh:
    """Quad mesh with vcg's face-face adjacency: ffp[f][z] is the face across edge z (f itself on the border) and
    ffi[f][z] the index of that edge there.  Edge z of a face goes from fv[f][z] to fv[f][(z + 1) % 4]."""
    __slots__ = ('fv', 'ffp', 'ffi', 'nv', 'P', 'N')

    def __init__(self, fv, ffp, ffi, nv):
        self.fv = fv
        self.ffp = ffp
        self.ffi = ffi
        self.nv = nv
        self.P = np.zeros((nv, 3))
        self.N = np.zeros((nv, 3))

    @property
    def nf(self):
        return len(self.fv)

    def border_vertices(self):
        b = np.zeros(self.nv, dtype=bool)
        for f, (fv, fp) in enumerate(zip(self.fv, self.ffp)):
            for z in range(4):
                if fp[z] == f:
                    b[fv[z]] = True
                    b[fv[(z + 1) & 3]] = True
        return b

    def walk_limit(self):
        return 8 * len(self.fv) + 16


class Pos:
    """vcg::face::Pos: a face, one of its edges and one of the edge's vertices."""
    __slots__ = ('m', 'f', 'z', 'v')

    def __init__(self, m, f, z, v):
        self.m = m
        self.f = f
        self.z = z
        self.v = v

    def copy(self):
        return Pos(self.m, self.f, self.z, self.v)

    def __eq__(self, o):
        return self.f == o.f and self.z == o.z and self.v == o.v

    def __ne__(self, o):
        return self.f != o.f or self.z != o.z or self.v != o.v

    __hash__ = None

    def flip_e(self):
        if self.m.fv[self.f][(self.z + 1) & 3] == self.v:
            self.z = (self.z + 1) & 3
        else:
            self.z = (self.z + 3) & 3

    def flip_v(self):
        fv = self.m.fv[self.f]
        if fv[(self.z + 1) & 3] == self.v:
            self.v = fv[self.z]
        else:
            self.v = fv[(self.z + 1) & 3]

    def flip_f(self):
        f, z = self.f, self.z
        self.f = self.m.ffp[f][z]
        self.z = self.m.ffi[f][z]

    def is_border(self):
        return self.m.ffp[self.f][self.z] == self.f

    def fflip(self):
        return self.m.ffp[self.f][self.z]

    def vflip(self):
        fv = self.m.fv[self.f]
        return fv[self.z] if fv[(self.z + 1) & 3] == self.v else fv[(self.z + 1) & 3]

    def advance_polychord(self):
        """FlipE FlipV FlipE FlipF: across the face to the opposite edge, then into the next face."""
        self.flip_e()
        self.flip_v()
        self.flip_e()
        self.flip_f()


class _Guard:
    def __init__(self, limit):
        self.n = limit

    def __call__(self):
        self.n -= 1
        if self.n < 0:
            raise MalformedTemplate('walk does not terminate')


# ---- template decoding (Patch::reconstructTopologyFromString / reconstructPatch) -----------------------------------
def decode_topology(topology):
    n = len(topology) // (3 * NUM_HEX)
    out = [[-1, -1, -1, -1] for _ in range(n)]
    last_seen = 0
    pos = 0
    for i in range(n):
        row = out[i]
        for j in (1, 2, 3):
            s = topology[pos:pos + NUM_HEX]
            pos += NUM_HEX
            if s != BORDER_STR:
                val = int(s, 16)
                if val >= n:
                    raise MalformedTemplate('face index out of range')
                row[j] = val
                if val > last_seen:
                    last_seen = val
                    out[last_seen][0] = i
    return out


def build_template_mesh(ff, n_vertices):
    """Face-face adjacency and face-vertex assignment of Patch::reconstructPatch."""
    nf = len(ff)
    ffp = [[0, 0, 0, 0] for _ in range(nf)]
    ffi = [[0, 0, 0, 0] for _ in range(nf)]
    for i in range(nf):
        for j in range(4):
            a = ff[i][j]
            if a >= 0:
                ffp[i][j] = a
                fa = ff[a]
                k = 0
                while k < 4 and fa[k] != i:
                    k += 1
                if k == 4:
                    raise MalformedTemplate('inconsistent adjacency')
                g = 0
                while fa[(k + 1) % 4] == i:     # last common edge
                    k = (k + 1) % 4
                    g += 1
                    if g > 4:
                        raise MalformedTemplate('inconsistent adjacency')
                kk = (j + 3) % 4
                g = 0
                while ff[i][kk] == a:           # go back by the common edges already processed
                    k = (k + 3) % 4
                    kk = (kk + 3) % 4
                    g += 1
                    if g > 4:
                        raise MalformedTemplate('inconsistent adjacency')
                ffi[i][j] = k
            else:
                ffp[i][j] = i
                ffi[i][j] = j
    fv = [[-1, -1, -1, -1] for _ in range(nf)]
    assigned = [False] * nf
    last = 0
    limit = 4 * nf + 8
    for i in range(nf):
        for j in range(4):
            cf = ffp[i][j]
            ce = (ffi[i][j] + 1) % 4
            g = 0
            while not assigned[cf] and cf != i and ffp[cf][ce] != cf:
                tmp = cf
                cf = ffp[cf][ce]
                ce = (ffi[tmp][ce] + 1) % 4
                g += 1
                if g > limit:
                    raise MalformedTemplate('vertex assignment does not terminate')
            if not assigned[cf]:
                cf = ffp[i][(j + 3) % 4]
                ce = ffi[i][(j + 3) % 4]
                g = 0
                while not assigned[cf] and cf != i and ffp[cf][(ce + 3) % 4] != cf:
                    tmp = cf
                    cf = ffp[cf][(ce + 3) % 4]
                    ce = ffi[tmp][(ce + 3) % 4]
                    g += 1
                    if g > limit:
                        raise MalformedTemplate('vertex assignment does not terminate')
            if assigned[cf]:
                fv[i][j] = fv[cf][ce]
            else:
                fv[i][j] = last
                last += 1
        assigned[i] = True
    if last != n_vertices:
        raise MalformedTemplate('vertex count mismatch')
    return QuadMesh(fv, ffp, ffi, last)


def update_start_corner(pos, n_corners):
    """The 'update the starting corner' step of Patch::reconstructPatch."""
    g = _Guard(pos.m.walk_limit())
    pos.flip_e()
    while not pos.is_border() and n_corners > 0:
        g()
        pos.flip_f()
        pos.flip_e()
        if not pos.is_border():     # concave corner
            pos.flip_e()
            while not pos.is_border():
                g()
                pos.flip_f()
                pos.flip_e()
            pos.flip_e()
            break
        pos.flip_v()
        pos.flip_e()
    pos.flip_e()


def rotate_start_corner(pos, n_rotations):
    g = _Guard(pos.m.walk_limit() * max(n_rotations, 1))
    for _ in range(n_rotations):
        pos.flip_v()
        pos.flip_e()
        while not pos.is_border():
            g()
            pos.flip_f()
            pos.flip_e()
            if not pos.is_border():     # concave corner
                while not pos.is_border():
                    g()
                    pos.flip_f()
                    pos.flip_e()
                break
            pos.flip_v()
            pos.flip_e()


def build_topology_str(m, start):
    """Patch::BuildTopologyStr: breadth-first face order from `start`, listing for each face the ids of the faces
    across its edges 1..3 (edge 0 leads back to where the face was reached from).  A position (f, z, v) walks the
    edges of its face forward when v is the first vertex of edge z and backward otherwise."""
    fv, ffp, ffi = m.fv, m.ffp, m.ffi
    ids = [-1] * len(fv)
    f0 = start.f
    order = [(f0, start.z, 1 if fv[f0][start.z] == start.v else -1)]
    ids[f0] = 0
    cur_id = 0
    qi = 0
    while qi < len(order):
        f, z, d = order[qi]
        qi += 1
        pf = ffp[f]
        for k in range(4):
            e = (z + d * k) & 3
            g = pf[e]
            if g != f and ids[g] < 0:
                cur_id += 1
                ids[g] = cur_id
                v = fv[f][e] if d == 1 else fv[f][(e + 1) & 3]
                ge = ffi[f][e]
                order.append((g, ge, 1 if fv[g][(ge + 1) & 3] == v else -1))
    parts = []
    for f, z, d in order:
        pf = ffp[f]
        for k in (1, 2, 3):
            g = pf[(z + d * k) & 3]
            parts.append(BORDER_STR if g == f else '%016x' % ids[g])
    return ''.join(parts)


# ---- polychords (PolychordSupport) --------------------------------------------------------------------------------
def find_polygon_corners(m, start, convex_only=False, collect_regular=False, end=None):
    """PolychordSupport::FindPolygonCorners.  Returns (number of corners or -1, corner positions)."""
    corners = []
    run = start.copy()
    g = _Guard(m.walk_limit())
    while not run.is_border():
        g()
        run.flip_v()
        run.flip_e()
        if not run.is_border() and run != start:
            run.flip_f()
            run.flip_e()
        if run == start:
            break
    if not run.is_border():
        run.flip_v()
        run.flip_e()
        tmp = run.copy()
        while not run.is_border():
            g()
            run.flip_v()
            run.flip_e()
            if not run.is_border() and run != tmp:
                run.flip_f()
                run.flip_e()
            if run == tmp:
                break
    if not run.is_border():
        run = None
        for f in range(m.nf):
            for z in range(4):
                if m.ffp[f][z] == f:
                    run = Pos(m, f, z, m.fv[f][z])
                    break
            if run is not None:
                break
        if run is None:
            return -1, []
    init = run.copy()
    if end is None or not end.is_border():
        end = run.copy()
    while True:
        g()
        if collect_regular:
            corners.append(run.copy())
        run.flip_v()
        run.flip_e()
        if run.is_border():
            if not collect_regular:
                corners.append(run.copy())
        else:
            run.flip_f()
            run.flip_e()
            if not run.is_border():
                while not run.is_border():
                    g()
                    run.flip_f()
                    run.flip_e()
                if not collect_regular:
                    corners.append(run.copy())
                if convex_only:
                    return -1, corners
        if run == end or run == init:
            break
    if not corners:
        corners.append(run.copy())
        return 0, corners
    return len(corners), corners


def count_concave_corners(corners):
    n = 0
    for c in corners:
        p = c.copy()
        p.flip_e()
        if not p.is_border():
            n += 1
    return n


def corners_from_start(m, start_corner):
    """The corners of a patch from its start corner, the first one being the start corner itself."""
    sp = start_corner.copy()
    sp.flip_e()
    g = _Guard(m.walk_limit())
    while not sp.is_border():
        g()
        sp.flip_f()
        sp.flip_e()
    sp.flip_v()
    return find_polygon_corners(m, sp)[1]


def find_loop_polychords(m):
    """PolychordCollapse::FindPolychords(mesh, polychords, loopsOnly=true): one position on each closed polychord."""
    visited = set()
    loops = []
    for f in range(m.nf):
        for d in (0, 1):
            if (f, d) in visited:
                continue
            closed = False
            p = Pos(m, f, d, m.fv[f][d])
            g = _Guard(2 * m.nf + 4)
            while True:
                g()
                visited.add((p.f, p.z & 1))
                p.flip_e()
                p.flip_v()
                p.flip_e()
                if p.is_border():
                    break
                p.flip_f()
                if p.f == f and (p.z & 1) == d:
                    closed = True
                    break
            if closed:
                loops.append(Pos(m, f, d, m.fv[f][d]))
                continue
            p = Pos(m, f, d, m.fv[f][d])
            while not p.is_border():
                g()
                p.flip_f()
                visited.add((p.f, p.z & 1))
                p.flip_e()
                p.flip_v()
                p.flip_e()
    return loops


def find_polychords_with_topology(m, corners, flow):
    """PolychordSupport::FindPolychordsWithTopology: the polychords starting on side flow[0] and ending on side
    flow[1] (all closed polychords for a loop flow)."""
    if flow[0] >= len(corners):
        return find_loop_polychords(m)
    polychords = []
    selected = set()
    start = corners[flow[0]].copy()
    stop = corners[(flow[0] + 1) % len(corners)]
    g = _Guard(m.walk_limit() * 4)
    while True:
        g()
        if not start.is_border():
            return []
        if start.f not in selected:
            run = start.copy()
            selected.add(run.f)
            while True:
                g()
                run.advance_polychord()
                if run.is_border():
                    break
            selected.add(run.f)
            if flow[1] < len(corners):
                run.flip_v()
                run.flip_e()
                while not run.is_border():
                    g()
                    run.flip_f()
                    run.flip_e()
                    if not run.is_border():     # concave corner
                        run.flip_e()
                        run.flip_f()
                        break
                    run.flip_v()
                    run.flip_e()
                run.flip_e()
                if run == corners[flow[1]]:
                    polychords.append(start.copy())
            else:
                polychords.append(start.copy())
        start.flip_v()
        start.flip_e()
        while not start.is_border():
            g()
            start.flip_f()
            start.flip_e()
        if start == stop:
            break
    return polychords


# ---- templates (Patch as loaded by PatchDBServer) -----------------------------------------------------------------
def parse_boundary_ids(s, n_corners):
    """Boundary ids per side from a boundaryIDs string ('x%04x' per edge, sides separated by '-')."""
    n_sides = n_corners if n_corners > 0 else 1
    sides = [[] for _ in range(n_sides)]
    side = 0
    i = 0
    n = len(s)
    step = 1 + LABEL_HEX
    while i < n:
        while i < n and s[i] == SEPARATOR:
            side += 1
            i += 1
        if i >= n:
            break
        if side >= n_sides:
            raise MalformedTemplate('too many sides')
        sides[side].append(int(s[i + 1:i + step], 16))
        i += step
    return sides


def parse_singularities(s):
    out = []
    if not s:
        return out
    for part in s.split(SEPARATOR):
        out.append((int(part[1:1 + LABEL_HEX], 16), int(part[2 + LABEL_HEX:2 + LABEL_HEX + NUM_HEX], 16)))
    return out


def singularity_str(vec):
    return SEPARATOR.join('%s%04x%s%016x' % (VALENCELABEL, val, VALENCESEP, num) for val, num in vec)


def boundary_ids_str(sides):
    return SEPARATOR.join(''.join('%s%04x' % (VARLABEL, b) for b in side) for side in sides)


class Template:
    """One database row.  The mesh is decoded on demand."""
    __slots__ = ('index', 'topology', 'boundary_ids', 'ends', 'side_len', 'n_vars', 'n_sing', 'sing_vec', 'n_corners',
                 'n_faces', 'n_vertices', 'n_occ', 'n_concave', 'perimeter', 'mesh', 'start', 'border_edges')

    def __init__(self, index, row):
        topology, boundary_ids, singularities, n_sing, n_corners, n_faces, n_vertices, n_occ, n_concave = row
        self.index = index
        self.topology = topology
        self.n_corners = int(n_corners)
        self.n_sing = int(n_sing)
        self.n_faces = int(n_faces)
        self.n_vertices = int(n_vertices)
        self.n_occ = int(n_occ)
        self.n_concave = int(n_concave)
        self.sing_vec = tuple(parse_singularities(singularities))
        self.boundary_ids = parse_boundary_ids(boundary_ids, self.n_corners)
        self.side_len = [len(s) for s in self.boundary_ids]
        self.perimeter = sum(self.side_len)
        self.n_vars = self.perimeter // 2
        ends = [[NC_MAX, NC_MAX] for _ in range(self.n_vars)]
        for side, ids in enumerate(self.boundary_ids):
            for b in ids:
                if b >= self.n_vars:
                    raise MalformedTemplate('boundary id out of range')
                if ends[b][0] == NC_MAX:
                    ends[b][0] = side
                else:
                    ends[b][1] = side
        self.ends = [tuple(e) for e in ends]
        self.mesh = None
        self.start = None
        self.border_edges = None

    def ensure_mesh(self):
        if self.mesh is not None:
            return
        m = build_template_mesh(decode_topology(self.topology), self.n_vertices)
        if m.nf != self.n_faces:
            raise MalformedTemplate('face count mismatch')
        # walk on the border as Patch::reconstructPatch does, recording the boundary id of each border edge
        start = Pos(m, 0, 0, m.fv[0][0])
        run = start.copy()
        side = 0
        edge = 0
        border_edges = []
        g = _Guard(m.walk_limit())
        while True:
            g()
            try:
                bid = self.boundary_ids[side][edge]
            except IndexError:
                raise MalformedTemplate('boundary walk out of range')
            border_edges.append((run.f, run.z, bid))
            run.flip_v()
            run.flip_e()
            if not run.is_border():
                run.flip_f()
                run.flip_e()
                edge += 1
                if not run.is_border():     # concave corner
                    while not run.is_border():
                        g()
                        run.flip_f()
                        run.flip_e()
                    side += 1
                    edge = 0
            else:
                side += 1
                edge = 0
            if run == start:
                break
        update_start_corner(start, self.n_corners)
        self.mesh = m
        self.start = start
        self.border_edges = border_edges

    def widths(self, var):
        """Per face, the number of parallel polychords replacing its polychords through edges 0/2 and 1/3."""
        self.ensure_mesh()
        t = self.mesh
        nf = t.nf
        width = [[1, 1] for _ in range(nf)]
        for f, z, bid in self.border_edges:
            w = var[bid]
            if w == 1:
                continue
            pf, pz = f, z
            g = _Guard(2 * nf + 4)
            while True:
                g()
                width[pf][pz & 1] = w
                oz = (pz + 2) & 3
                nf_ = t.ffp[pf][oz]
                if nf_ == pf:
                    break
                pf, pz = nf_, t.ffi[pf][oz]
        return width

    def rotated_start(self, n_rotations):
        """The start corner of Patch::reconstructPatch on the template (corners are the same once expanded)."""
        self.ensure_mesh()
        p = self.start.copy()
        rotate_start_corner(p, n_rotations)
        return p

    def expand_adjacency(self, var, n_rotations):
        """Face adjacency of the expanded patch and its start corner (cell, edge, vertex, direction)."""
        width = self.widths(var)
        ex = _Expansion(self.mesh, width)
        st = self.rotated_start(n_rotations)
        t = self.mesh
        f = st.f
        k = t.fv[f].index(st.v)
        a, b = ex.A[f], ex.B[f]
        cell = ex.base[f] + (0, a - 1, (b - 1) * a + a - 1, (b - 1) * a)[k]
        d = 1 if t.fv[f][st.z] == st.v else -1
        return ex, (cell, st.z, st.v, d)

    def expand(self, var, n_rotations):
        """Patch::reconstructPatch with the polychord widths `var`: every polychord starting at a border edge with
        id b is split into var[b] parallel polychords.  Returns the expanded mesh and its start corner."""
        ex, (cell, z, v, _) = self.expand_adjacency(var, n_rotations)
        m = ex.mesh()
        return m, Pos(m, cell, z, v)


def _topology_str_adj(ffp, ffi, f0, z0, d):
    """build_topology_str for a consistently oriented mesh given by its face adjacency only."""
    ids = [-1] * len(ffp)
    order = [(f0, z0)]
    ids[f0] = 0
    cur_id = 0
    qi = 0
    while qi < len(order):
        f, z = order[qi]
        qi += 1
        pf = ffp[f]
        for k in range(4):
            e = (z + d * k) & 3
            g = pf[e]
            if g != f and ids[g] < 0:
                cur_id += 1
                ids[g] = cur_id
                order.append((g, ffi[f][e]))
    parts = []
    for f, z in order:
        pf = ffp[f]
        for k in (1, 2, 3):
            g = pf[(z + d * k) & 3]
            parts.append(BORDER_STR if g == f else '%016x' % ids[g])
    return ''.join(parts)


class _Expansion:
    """Replaces every face f of `t` by an a x b grid of cells (a = width[f][0] cuts of edges 0 and 2, b = width[f][1]
    cuts of edges 1 and 3), keeping the local edge numbering of the parent face.  Cell (i, j) of face f is face
    base[f] + j * a + i of the expanded mesh.  The adjacency is computed first; the vertices on demand."""

    def __init__(self, t, width):
        self.t = t
        nf = t.nf
        A = self.A = [w[0] for w in width]
        B = self.B = [w[1] for w in width]
        base = self.base = []
        total = 0
        for f in range(nf):
            base.append(total)
            total += A[f] * B[f]
        tffp, tffi = t.ffp, t.ffi

        def n_of(f, e):
            return A[f] if e % 2 == 0 else B[f]

        def cross(f, e, k, own):
            g = tffp[f][e]
            if g == f:
                return own, e
            ge = tffi[f][e]
            n = n_of(f, e)
            if n_of(g, ge) != n:
                raise MalformedTemplate('inconsistent polychord widths')
            k = n - 1 - k
            a, b = A[g], B[g]
            if ge == 0:
                return base[g] + k, 0
            if ge == 1:
                return base[g] + k * a + a - 1, 1
            if ge == 2:
                return base[g] + (b - 1) * a + a - 1 - k, 2
            return base[g] + (b - 1 - k) * a, 3

        ffp = [None] * total
        ffi = [None] * total
        for f in range(nf):
            a, b = A[f], B[f]
            c = base[f]
            for j in range(b):
                for i in range(a):
                    p0, i0 = cross(f, 0, i, c) if j == 0 else (c - a, 2)
                    p1, i1 = cross(f, 1, j, c) if i == a - 1 else (c + 1, 3)
                    p2, i2 = cross(f, 2, a - 1 - i, c) if j == b - 1 else (c + a, 0)
                    p3, i3 = cross(f, 3, b - 1 - j, c) if i == 0 else (c - 1, 1)
                    ffp[c] = [p0, p1, p2, p3]
                    ffi[c] = [i0, i1, i2, i3]
                    c += 1
        self.ffp = ffp
        self.ffi = ffi

    def mesh(self):
        t = self.t
        A, B, base = self.A, self.B, self.base
        nv = t.nv
        edge_pts = {}
        tffp, tffi = t.ffp, t.ffi

        def edge_point(f, e, k):
            nonlocal nv
            g = tffp[f][e]
            n = A[f] if e % 2 == 0 else B[f]
            key = (f, e)
            kk = k
            if g != f:
                other = (g, tffi[f][e])
                if other < key:
                    key = other
                    kk = n - k
            lst = edge_pts.get(key)
            if lst is None:
                lst = list(range(nv, nv + n - 1))
                nv += n - 1
                edge_pts[key] = lst
            return lst[kk - 1]

        fv = [None] * len(self.ffp)
        for f in range(t.nf):
            a, b = A[f], B[f]
            V = t.fv[f]
            G = [[0] * (b + 1) for _ in range(a + 1)]
            for i in range(a + 1):
                Gi = G[i]
                for j in range(b + 1):
                    if i == 0 and j == 0:
                        vid = V[0]
                    elif i == a and j == 0:
                        vid = V[1]
                    elif i == a and j == b:
                        vid = V[2]
                    elif i == 0 and j == b:
                        vid = V[3]
                    elif j == 0:
                        vid = edge_point(f, 0, i)
                    elif i == a:
                        vid = edge_point(f, 1, j)
                    elif j == b:
                        vid = edge_point(f, 2, a - i)
                    elif i == 0:
                        vid = edge_point(f, 3, b - j)
                    else:
                        vid = nv
                        nv += 1
                    Gi[j] = vid
            c = base[f]
            for j in range(b):
                for i in range(a):
                    fv[c] = [G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]]
                    c += 1
        return QuadMesh(fv, self.ffp, self.ffi, nv)


# ---- template encoding (Patch::setPatch), used to build databases --------------------------------------------------
def mesh_from_faces(faces):
    """QuadMesh with face-face adjacency from a list of quads (consistently oriented, manifold disk)."""
    nv = max(max(f) for f in faces) + 1 if faces else 0
    he = {}
    for fi, f in enumerate(faces):
        for z in range(4):
            he[(f[z], f[(z + 1) & 3])] = (fi, z)
    ffp = []
    ffi = []
    for fi, f in enumerate(faces):
        p = []
        q = []
        for z in range(4):
            o = he.get((f[(z + 1) & 3], f[z]))
            if o is None:
                p.append(fi)
                q.append(z)
            else:
                p.append(o[0])
                q.append(o[1])
        ffp.append(p)
        ffi.append(q)
    return QuadMesh([list(f) for f in faces], ffp, ffi, nv)


def internal_singularities(m):
    border = m.border_vertices()
    val = np.zeros(m.nv, dtype=np.int64)
    for fv in m.fv:
        for v in fv:
            val[v] += 1
    out = {}
    for v in range(m.nv):
        if not border[v] and val[v] != 4:
            out[int(val[v])] = out.get(int(val[v]), 0) + 1
    return sorted(out.items())


def encode_patch(faces, n_occurrences=1):
    """Database row (topology, boundaryIDs, singularities, nSingularities, nCorners, nFaces, nVertices, nOccurrences,
    nConcaveCorners) of a quad patch, as Patch::setPatch and PatchDBInterface store it."""
    m = mesh_from_faces(faces)
    start = None
    for f in range(m.nf):
        for z in range(4):
            if m.ffp[f][z] == f:
                start = Pos(m, f, z, m.fv[f][z])
                break
        if start is not None:
            break
    n_corners, corners = find_polygon_corners(m, start)
    if n_corners < 0 or not corners:
        raise ValueError('not a disk')
    n_concave = count_concave_corners(corners) if n_corners > 0 else 0
    topology = build_topology_str(m, corners[0])
    b = 0
    lengths = []
    nc = len(corners)
    for i in range(nc):
        run = corners[i].copy()
        length = 0
        g = _Guard(m.walk_limit())
        while True:
            g()
            if n_corners == 0:
                s = build_topology_str(m, run)
                if s < topology:
                    topology = s
            run.flip_v()
            run.flip_e()
            length += 1
            while not run.is_border():
                g()
                run.flip_f()
                run.flip_e()
            if run == corners[(i + 1) % nc]:
                break
        lengths.append(length)
        if n_corners > 0:
            s = build_topology_str(m, corners[i])
            if s < topology:
                topology = s
                b = i
    lengths = [lengths[(i + b) % nc] for i in range(nc)]
    ids = _boundary_ids_from_mesh(m, corners[b], lengths)
    sing = internal_singularities(m)
    return (topology, boundary_ids_str(ids), singularity_str(sing), sum(n for _, n in sing), n_corners, m.nf, m.nv,
            n_occurrences, n_concave)


def _boundary_ids_from_mesh(m, start, lengths):
    """Patch::buildBoundaryIDsFromMesh."""
    ids = [[0] * n for n in lengths]
    boundary_map = {}
    border = start.copy()
    side = 0
    edge = 0
    new_id = 0
    g = _Guard(m.walk_limit() * 2)
    while True:
        g()
        key = (border.f, border.z)
        if key not in boundary_map:
            cur = new_id
            new_id += 1
            boundary_map[key] = cur
            run = border.copy()
            run.flip_v()
            while True:
                g()
                run.flip_e()
                run.flip_v()
                run.flip_e()
                if not run.is_border():
                    run.flip_f()
                if run.is_border():
                    break
            boundary_map.setdefault((run.f, run.z), cur)
            ids[side][edge] = cur
        else:
            ids[side][edge] = boundary_map[key]
        border.flip_v()
        border.flip_e()
        if border != start and not border.is_border():
            border.flip_f()
            border.flip_e()
            if not border.is_border():  # concave corner
                while not border.is_border():
                    g()
                    border.flip_f()
                    border.flip_e()
                side += 1
                edge = 0
            else:
                edge += 1
        else:
            side += 1
            edge = 0
        if border == start:
            break
    return ids


# ---- integer linear systems (stand-in for SCIP: all integer solutions of A x = b within bounds) ---------------------
def _ceil_div(p, q):
    return -((-p) // q) if q > 0 else -((p) // (-q))


def _floor_div(p, q):
    return p // q if q > 0 else (-p) // (-q)


def _propagate(rows, rhs, lo, hi, var_rows, start_rows):
    queue = list(start_rows)
    inq = set(queue)
    while queue:
        r = queue.pop()
        inq.discard(r)
        terms = rows[r]
        b = rhs[r]
        smin = 0
        smax = 0
        nmin = 0
        nmax = 0
        cmins = []
        cmaxs = []
        for k, a in terms:
            if a > 0:
                cmin = a * lo[k]
                cmax = a * hi[k] if hi[k] != INF else None
            else:
                cmin = a * hi[k] if hi[k] != INF else None
                cmax = a * lo[k]
            if cmin is None:
                nmin += 1
            else:
                smin += cmin
            if cmax is None:
                nmax += 1
            else:
                smax += cmax
            cmins.append(cmin)
            cmaxs.append(cmax)
        if nmin == 0 and smin > b:
            return False
        if nmax == 0 and smax < b:
            return False
        for (k, a), cmin, cmax in zip(terms, cmins, cmaxs):
            if cmin is None:
                rest_min = smin if nmin == 1 else None
            else:
                rest_min = smin - cmin if nmin == 0 else None
            if cmax is None:
                rest_max = smax if nmax == 1 else None
            else:
                rest_max = smax - cmax if nmax == 0 else None
            if a > 0:
                nlo = _ceil_div(b - rest_max, a) if rest_max is not None else None
                nhi = _floor_div(b - rest_min, a) if rest_min is not None else None
            else:
                nlo = _ceil_div(b - rest_min, a) if rest_min is not None else None
                nhi = _floor_div(b - rest_max, a) if rest_max is not None else None
            changed = False
            if nlo is not None and nlo > lo[k]:
                lo[k] = nlo
                changed = True
            if nhi is not None and nhi < hi[k]:
                hi[k] = nhi
                changed = True
            if lo[k] > hi[k]:
                return False
            if changed:
                for r2 in var_rows[k]:
                    if r2 not in inq:
                        inq.add(r2)
                        queue.append(r2)
    return True


def enumerate_ilp(cons, rhs, lo, hi):
    """Yields every integer x with cons . x == rhs and lo <= x <= hi (hi entries may be INF)."""
    n = len(lo)
    rows = [[(k, int(a)) for k, a in enumerate(row) if a != 0] for row in cons]
    rhs = [int(b) for b in rhs]
    for terms, b in zip(rows, rhs):
        if not terms and b != 0:
            return
    var_rows = [[] for _ in range(n)]
    for r, terms in enumerate(rows):
        for k, _ in terms:
            var_rows[k].append(r)
    lo = [int(x) for x in lo]
    hi = [x if x == INF else int(x) for x in hi]
    if not _propagate(rows, rhs, lo, hi, var_rows, range(len(rows))):
        return
    stack = [(lo, hi)]
    while stack:
        lo, hi = stack.pop()
        best = -1
        best_dom = INF
        for k in range(n):
            d = hi[k] - lo[k]
            if 0 < d < best_dom:
                best_dom = d
                best = k
        if best < 0:
            if any(lo[k] != hi[k] for k in range(n)):
                continue    # unbounded: infinitely many solutions
            if all(sum(a * lo[k] for k, a in terms) == b for terms, b in zip(rows, rhs)):
                yield list(lo)
            continue
        children = []
        for val in range(lo[best], hi[best] + 1):
            nlo = list(lo)
            nhi = list(hi)
            nlo[best] = nhi[best] = val
            if _propagate(rows, rhs, nlo, nhi, var_rows, var_rows[best]):
                children.append((nlo, nhi))
        stack.extend(reversed(children))


def constraint_ilp_by_flows(t, n_rotations, boundary_length, flows, lo, hi, cons, rhs, tolerance):
    """ILPSupport::ConstraintILPbyFlows: adds constraints forcing a polychord through every border-to-border flow.
    Returns the constrained system (lo, hi, cons, rhs) of the first feasible choice of polychords, or None."""
    n_sides = len(boundary_length)
    if n_rotations >= n_sides or not flows:
        return None
    topo_vars = {}
    serialized = []
    for key in sorted(flows):
        if key[0] >= n_sides or key[1] >= n_sides:
            continue
        ss = (key[0] + n_rotations) % n_sides
        es = (key[1] + n_rotations) % n_sides
        vars_ = [b for b in t.boundary_ids[ss] if t.ends[b] == (ss, es) or t.ends[b] == (es, ss)]
        if not vars_:
            return None
        topo_vars[key] = vars_
        for fp in flows[key]:
            if (fp.start_edge < 0 or fp.start_edge >= boundary_length[key[0]] or
                    fp.end_edge < 0 or fp.end_edge >= boundary_length[key[1]]):
                continue
            serialized.append((key, fp.start_edge, fp.end_edge))
    if not serialized:
        return lo, hi, cons, rhs
    stack = [(0, (lo, hi, cons, rhs))]
    while stack:
        idx, (lo0, hi0, cons0, rhs0) = stack.pop()
        key, start_edge, end_edge = serialized[idx]
        ss = (key[0] + n_rotations) % n_sides
        es = (key[1] + n_rotations) % n_sides
        ids_s = t.boundary_ids[ss]
        ids_e = t.boundary_ids[es]
        for v in topo_vars[key]:
            lo2 = list(lo0)
            hi2 = list(hi0)
            n = len(lo2)
            d1, d2 = n, n + 1
            n += 2
            lo2 += [0, 0]
            hi2 += [INF, INF]
            if tolerance > 0:
                m1, m2 = n, n + 1
                n += 2
                lo2 += [-tolerance, -tolerance]
                hi2 += [tolerance, tolerance]
            cons2 = [row + [0] * (n - len(row)) for row in cons0]
            rhs2 = list(rhs0)
            c = [0] * n
            c[v] = 1
            c[d1] = -1
            c[d2] = -1
            cons2.append(c)
            rhs2.append(1)
            c = [0] * n
            e = 0
            while ids_s[e] != v:
                c[ids_s[e]] += 1
                e += 1
            c[d1] = 1
            if tolerance > 0:
                c[m1] = -1
            cons2.append(c)
            rhs2.append(start_edge)
            c = [0] * n
            e = len(ids_s) - 1
            while ids_s[e] != v:
                c[ids_s[e]] += 1
                e -= 1
            if ss != es:
                c[d2] = 1
            else:
                c[d1] = 1
            if tolerance > 0:
                if ss != es:
                    c[m1] = 1
                else:
                    c[m2] = 1
            cons2.append(c)
            rhs2.append(boundary_length[key[0]] - start_edge - 1 if ss != es else
                        boundary_length[key[0]] - end_edge - 1)
            if ss != es:
                c = [0] * n
                e = 0
                while ids_e[e] != v:
                    c[ids_e[e]] += 1
                    e += 1
                c[d2] = 1
                if tolerance > 0:
                    c[m2] = -1
                cons2.append(c)
                rhs2.append(end_edge)
                c = [0] * n
                e = len(ids_e) - 1
                while ids_e[e] != v:
                    c[ids_e[e]] += 1
                    e -= 1
                c[d1] = 1
                if tolerance > 0:
                    c[m2] = 1
                cons2.append(c)
                rhs2.append(boundary_length[key[1]] - end_edge - 1)
            if idx + 1 < len(serialized):
                stack.append((idx + 1, (lo2, hi2, cons2, rhs2)))
            elif next(enumerate_ilp(cons2, rhs2, lo2, hi2), None) is not None:
                return lo2, hi2, cons2, rhs2
    return None


# ---- flows ---------------------------------------------------------------------------------------------------------
class FlowPoints:
    __slots__ = ('start_edge', 'end_edge', 'points', 'arc', 'samples')

    def __init__(self, points=None):
        self.start_edge = -1
        self.end_edge = -1
        self.points = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
        self.arc = np.zeros(0)
        self.samples = np.zeros((0, 3))

    def update_arc(self):
        if len(self.points) == 0:
            self.arc = np.zeros(0)
            return
        d = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.concatenate(([0.0], np.cumsum(d)))

    def compute_samples(self, n):
        self.samples = np.zeros((0, 3))
        if len(self.points) == 0 or len(self.points) != len(self.arc) or n < 2:
            return
        total = self.arc[-1]
        if total <= 0:
            return
        s = np.arange(n - 1) * (total / (n - 1))
        seg = np.clip(np.searchsorted(self.arc, s, side='right') - 1, 0, len(self.points) - 2)
        ln = self.arc[seg + 1] - self.arc[seg]
        with np.errstate(divide='ignore', invalid='ignore'):
            tt = np.where(ln > 0, (s - self.arc[seg]) / ln, 0.0)
        pts = self.points[seg] * (1 - tt)[:, None] + self.points[seg + 1] * tt[:, None]
        self.samples = np.vstack((pts, self.points[-1:]))

    def invert(self):
        self.start_edge, self.end_edge = self.end_edge, self.start_edge
        self.points = self.points[::-1].copy()
        self.samples = self.samples[::-1].copy()


class PolychordPoints(FlowPoints):
    __slots__ = ('start', 'distances')

    def __init__(self):
        FlowPoints.__init__(self)
        self.start = None
        self.distances = None

    def polychord_to_points(self, P):
        run = self.start.copy()
        pts = []
        g = _Guard(2 * run.m.nf + 4)
        while True:
            g()
            pts.append((P[run.v] + P[run.vflip()]) / 2.0)
            run.flip_e()
            run.flip_v()
            run.flip_e()
            pts.append((pts[-1] + (P[run.v] + P[run.vflip()]) / 2.0) / 2.0)
            run.flip_f()
            if run == self.start or run.is_border():
                break
        pts.append((P[run.v] + P[run.vflip()]) / 2.0)
        self.points = np.array(pts)
        self.update_arc()


def _segment_closest(p0, p1, p):
    e = p1 - p0
    en = float(e @ e)
    if en < sys.float_info.min:
        c = (p0 + p1) / 2
    else:
        t = float((p - p0) @ e) / en
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        c = p0 + e * t
    return c, float(np.linalg.norm(p - c))


def points_to_flow(points, is_loop, boundary, flow_map, n_samples=NUM_FLOW_SAMPLES):
    """PatchRetopologizer::PointsToFlow: snaps the ends of a stroke to the patch boundary and files it by the sides
    it connects (closed strokes are filed as loops)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    if not pts or n_samples < 2:
        return
    fp = FlowPoints()
    key = [NC_MAX, NC_MAX]
    if is_loop:
        if np.any(pts[0] != pts[-1]):
            pts.append(pts[0].copy())
    else:
        for end in (0, 1):
            q = pts[0] if end == 0 else pts[-1]
            best = INF
            closest = None
            for s in range(len(boundary)):
                bs = boundary[s]
                for p in range(len(bs) - 1):
                    c, d = _segment_closest(bs[p], bs[p + 1], q)
                    if d < best:
                        best, closest, key[end] = d, c, s
                        if end == 0:
                            fp.start_edge = p
                        else:
                            fp.end_edge = p
                nxt = boundary[(s + 1) % len(boundary)]
                if np.any(bs[-1] != nxt[0]):
                    c, d = _segment_closest(bs[-1], nxt[0], q)
                    if d < best:
                        best, closest, key[end] = d, c, s
                        if end == 0:
                            fp.start_edge = len(bs)
                        else:
                            fp.end_edge = len(bs)
            if closest is not None:
                if end == 0 and np.any(pts[0] != closest):
                    pts.insert(0, closest)
                elif end == 1 and np.any(pts[-1] != closest):
                    pts.append(closest)
        fp.points = np.array(pts)
        if key[1] < key[0]:
            key = [key[1], key[0]]
            fp.invert()
        elif key[1] == key[0] and fp.end_edge < fp.start_edge:
            fp.invert()
    fp.points = np.array(pts) if is_loop else fp.points
    fp.update_arc()
    fp.compute_samples(n_samples)
    flow_map.setdefault(tuple(key), []).append(fp)


def _points_distance(a, b, values=False):
    """PatchFlattener::ComputePointsToPointsDistance (Euclidean): root of the summed squared distances."""
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=1) if n else np.zeros(0)
    total = float(np.sqrt((d * d).sum()))
    return (total, d) if values else total


def _scaled_distance(dist, flow_len, poly_len):
    if flow_len >= poly_len:
        return SCALAR_MAX if poly_len == 0 else dist * flow_len / poly_len
    return SCALAR_MAX if flow_len == 0 else dist * poly_len / flow_len


def _greedy_match(distances, n_rows, n_cols):
    rows = list(range(n_rows))
    cols = list(range(n_cols))
    pairs = []
    total = 0.0
    while cols:
        best = SCALAR_MAX
        bi = bj = None
        for i in rows:
            for j in cols:
                if distances[i][j] < best:
                    best = distances[i][j]
                    bi, bj = i, j
        if bi is None:
            bi, bj = rows[0], cols[0]
        total += best
        pairs.append((bi, bj))
        rows.remove(bi)
        cols.remove(bj)
    return total, pairs


def flows_polychords_distance(n_sides, key, flows, polychord_vec, P):
    """PatchFlattener::ComputeFlowsPolychordsDistance: matches the flows with polychords of the same topology.
    Returns (total distance, polychord points per flow)."""
    if len(polychord_vec) < len(flows) or not flows:
        return SCALAR_MAX, None
    nr = len(polychord_vec)
    nc = len(flows)
    distances = [[SCALAR_MAX] * nc for _ in range(nr)]
    is_loop = key[0] >= n_sides or key[1] >= n_sides
    if is_loop or key[0] == key[1]:
        centers = []
        for fp in flows:
            if is_loop:
                centers.append(fp.points[:-1].mean(axis=0))
            else:
                centers.append((fp.points[0] + fp.points[-1]) / 2.0)
        cand = [[None] * nc for _ in range(nr)]
        for i in range(nr):
            for j in range(nc):
                fp = flows[j]
                pp = PolychordPoints()
                if is_loop:
                    mids = []
                    run = polychord_vec[i].copy()
                    g = _Guard(2 * run.m.nf + 4)
                    while True:
                        g()
                        mids.append(((P[run.v] + P[run.vflip()]) / 2.0, run.copy()))
                        run.advance_polychord()
                        if run == polychord_vec[i]:
                            break
                    center = np.mean([mm for mm, _ in mids], axis=0)
                    shift = center - centers[j]
                    best = INF
                    for mm, ps in mids:
                        d = float(np.linalg.norm(mm - shift - fp.points[0]))
                        if d < best:
                            best = d
                            pp.start = ps
                    pp.polychord_to_points(P)
                    pp.points = pp.points - shift
                    pp.compute_samples(len(fp.samples))
                    v1 = _points_distance(fp.samples, pp.samples)
                    v2 = _points_distance(fp.samples, pp.samples[::-1])
                    if v2 < v1:
                        pp.start.flip_f()
                        pp.invert()
                        pp.update_arc()
                    pp.points = pp.points + shift
                else:
                    run = polychord_vec[i].copy()
                    pp.start = run.copy()
                    c0 = (P[run.v] + P[run.vflip()]) / 2.0
                    g = _Guard(2 * run.m.nf + 4)
                    while True:
                        g()
                        run.advance_polychord()
                        if run.is_border():
                            break
                    center = (c0 + (P[run.v] + P[run.vflip()]) / 2.0) / 2.0
                    shift = center - centers[j]
                    p1 = (P[pp.start.v] + P[pp.start.vflip()]) / 2.0 - shift
                    p2 = (P[run.v] + P[run.vflip()]) / 2.0 - shift
                    if np.linalg.norm(p2 - fp.points[0]) < np.linalg.norm(p1 - fp.points[0]):
                        pp.start = run.copy()
                    pp.polychord_to_points(P)
                pp.compute_samples(len(fp.samples))
                d, pp.distances = _points_distance(fp.samples, pp.samples, values=True)
                distances[i][j] = _scaled_distance(d, fp.arc[-1] if len(fp.arc) else 0.0,
                                                   pp.arc[-1] if len(pp.arc) else 0.0)
                cand[i][j] = pp
        total, pairs = _greedy_match(distances, nr, nc)
        out = [None] * nc
        for i, j in pairs:
            out[j] = cand[i][j]
        return total, out
    # border to border
    polys = []
    for i in range(nr):
        pp = PolychordPoints()
        pp.start = polychord_vec[i].copy()
        pp.polychord_to_points(P)
        pp.compute_samples(len(flows[0].samples))
        polys.append(pp)
    for i, pp in enumerate(polys):
        for j, fp in enumerate(flows):
            d, pp.distances = _points_distance(fp.samples, pp.samples, values=True)
            distances[i][j] = _scaled_distance(d, fp.arc[-1] if len(fp.arc) else 0.0,
                                               pp.arc[-1] if len(pp.arc) else 0.0)
    total, pairs = _greedy_match(distances, nr, nc)
    out = [None] * nc
    for i, j in pairs:
        out[j] = polys[i]
    return total, out


# ---- geometry (PatchFlattener / PatchRetopologizer) ----------------------------------------------------------------
def _solve_spd(rows, cols, vals, n, rhs):
    from .engine import _solve_sparse_spd
    return _solve_sparse_spd(rows, cols, vals, n, rhs)


def _laplacian_triplets(m, weight):
    F = np.asarray(m.fv, dtype=np.int64)
    a = F.reshape(-1)
    b = np.roll(F, -1, axis=1).reshape(-1)
    w = np.full(len(a), float(weight))
    return [a, b, a, b], [a, b, b, a], [w, w, -w, -w]


def implicit_smooth(m, border, lap_weight, constraints=(), target=None):
    """vcg::ImplicitSmoother::Compute (no mass matrix, uniform weights, border fixed by a penalty, lambda 0.2) with
    optional face constraints (face, barycentric weights, target), solved in its reduced symmetric form."""
    n = m.nv
    rows, cols, vals = _laplacian_triplets(m, SMOOTH_LAMBDA * lap_weight)
    bidx = np.flatnonzero(border)
    rows.append(bidx)
    cols.append(bidx)
    vals.append(np.full(len(bidx), PENALTY))
    rhs = np.zeros((n, 3))
    rhs[bidx] = PENALTY * m.P[bidx]
    if constraints:
        pen = 1.0 - lap_weight
        c = pen * pen / 3.0
        for f, w, t in constraints:
            fv = m.fv[f]
            for j in range(4):
                for k in range(4):
                    rows.append(np.array([fv[j]]))
                    cols.append(np.array([fv[k]]))
                    vals.append(np.array([c * w[j] * w[k]]))
                rhs[fv[j]] += c * w[j] * np.asarray(t)
    x = _solve_spd(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), n, rhs)
    m.P = np.asarray(x, dtype=float).reshape(n, 3)


def face_normals(m):
    F = np.asarray(m.fv, dtype=np.int64)
    P = m.P
    nrm = np.zeros((len(F), 3))
    for i in range(4):
        nrm += np.cross(P[F[:, i]], P[F[:, (i + 1) % 4]])
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1
    return nrm / ln[:, None]


def _rotation_to_z(nrm):
    """vcg::RotationMatrix(nrm, (0, 0, 1))."""
    z = np.array((0.0, 0.0, 1.0))
    dot = float(nrm @ z)
    if dot > 1 - 0.00001:
        return np.eye(3)
    axis = np.cross(nrm, z)
    ln = np.linalg.norm(axis)
    if ln > 0:
        axis = axis / ln
    u, v, w = axis
    phi = math.acos(max(-1.0, min(1.0, dot)))
    c = math.cos(phi)
    s = math.sin(phi)
    return np.array([[c + u * u * (1 - c), -w * s + u * v * (1 - c), v * s + u * w * (1 - c)],
                     [w * s + v * u * (1 - c), c + v * v * (1 - c), -u * s + v * w * (1 - c)],
                     [-v * s + w * u * (1 - c), u * s + w * v * (1 - c), c + w * w * (1 - c)]])


def quad_barycentric(points, nrm, p):
    """PatchFlattener::ComputeQuadFaceBarycentricCoordinates (bilinear coordinates in the face's plane)."""
    m = points.mean(axis=0)
    R = _rotation_to_z(nrm)
    q = (points - m) @ R.T
    x = R @ np.asarray(p, dtype=float)      # (sic) the original centers the face but not the point
    v00, v10, v11, v01 = q[0, :2], q[1, :2], q[2, :2], q[3, :2]
    xx = x[:2]

    def cr(a, b):
        return a[0] * b[1] - a[1] * b[0]

    aa = v00 - xx
    bb = v10 - v00
    cc = v01 - v00
    dd = v00 - v01 - v10 + v11
    lam = l1 = l2 = -1.0
    a = cr(bb, dd)
    b = cr(bb, cc) + cr(aa, dd)
    c = cr(aa, cc)
    disc = b * b - 4 * a * c
    if a != 0 and disc >= 0:
        l1 = (-b + math.sqrt(disc)) / (2 * a)
        l2 = (-b - math.sqrt(disc)) / (2 * a)
    if 0 <= l1 <= 1:
        lam = l1
    elif 0 <= l2 <= 1:
        lam = l2
    u = u1 = u2 = -1.0
    a = cr(cc, dd)
    b = cr(cc, bb) + cr(aa, dd)
    c = cr(aa, bb)
    disc = b * b - 4 * a * c
    if a != 0 and disc >= 0:
        u1 = (-b + math.sqrt(disc)) / (2 * a)
        u2 = (-b - math.sqrt(disc)) / (2 * a)
    if 0 <= u1 <= 1:
        u = u1
    elif 0 <= u2 <= 1:
        u = u2
    if 0 <= lam <= 1 and 0 <= u <= 1:
        return [(1 - lam) * (1 - u), lam * (1 - u), lam * u, (1 - lam) * u], True
    return [0.25, 0.25, 0.25, 0.25], False


def quads_quality(m):
    """PatchFlattener::ComputeQuadsQuality: the largest deviation of a corner angle from 90 degrees, over pi/2."""
    F = np.asarray(m.fv, dtype=np.int64)
    if len(F) == 0:
        return SCALAR_MAX
    P = m.P
    worst = 0.0
    for i in range(4):
        d0 = P[F[:, i]] - P[F[:, (i + 1) % 4]]
        d1 = P[F[:, (i + 2) % 4]] - P[F[:, (i + 1) % 4]]
        w = np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cosv = np.clip((d0 * d1).sum(1) / w, -1, 1)
        ang = np.where(w > 0, np.arccos(cosv), -1.0)
        worst = max(worst, float(np.abs(ang - math.pi / 2).max()))
    return min(worst / (math.pi / 2), 1.0)


class UVLocator:
    """UVGrid: finds the base mesh triangle under a point of the parameter domain."""

    def __init__(self, xyz, nrm, uv, tris):
        self.xyz = np.asarray(xyz, dtype=float)
        self.nrm = np.asarray(nrm, dtype=float)
        self.uv = np.asarray(uv, dtype=float)[:, :2]
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        nf = len(self.tris)
        size = max(int(math.sqrt(nf)), 2)
        self.size = size
        lo = self.uv.min(axis=0) if len(self.uv) else np.zeros(2)
        hi = self.uv.max(axis=0) if len(self.uv) else np.zeros(2)
        half = (hi - lo) / size * 0.5
        self.lo = lo - half
        self.hi = hi + half
        self.edge = (self.hi - self.lo) / size
        self.edge[self.edge == 0] = 1
        T = self.uv[self.tris] if nf else np.zeros((0, 3, 2))
        self.T = T
        cells = [[[] for _ in range(size)] for _ in range(size)]
        if nf:
            cmin = np.floor((T.min(axis=1) - self.lo) / self.edge).astype(np.int64)
            cmax = np.floor((T.max(axis=1) - self.lo) / self.edge).astype(np.int64)
            cmin = np.clip(cmin, 0, size - 1)
            cmax = np.clip(cmax, 0, size - 1)
            for t in range(nf):
                for x in range(cmin[t, 0], cmax[t, 0] + 1):
                    row = cells[x]
                    for y in range(cmin[t, 1], cmax[t, 1] + 1):
                        row[y].append(t)
        self.cells = [[np.array(c, dtype=np.int64) for c in row] for row in cells]
        e1 = T[:, 1] - T[:, 0]
        e2 = T[:, 2] - T[:, 0]
        self.area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    def _bary(self, idx, p):
        T = self.T[idx]
        x1, y1 = T[:, 0, 0], T[:, 0, 1]
        x2, y2 = T[:, 1, 0], T[:, 1, 1]
        x3, y3 = T[:, 2, 0], T[:, 2, 1]
        x, y = p
        with np.errstate(divide='ignore', invalid='ignore'):
            l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
            l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y3 - y1) * (x2 - x3) + (x1 - x3) * (y2 - y3))
        l3 = 1 - l1 - l2
        bad = np.isnan(l1) | np.isnan(l2) | np.isnan(l3)
        l1 = np.where(bad, 1 / 3, l1)
        l2 = np.where(bad, 1 / 3, l2)
        l3 = np.where(bad, 1 / 3, l3)
        return np.stack((l1, l2, l3), axis=1)

    def locate(self, p):
        """(triangle, barycentric coordinates) of uv point p; the closest triangle when p is outside."""
        p = np.asarray(p, dtype=float)[:2]
        if len(self.tris) == 0:
            return None
        if np.all(p >= self.lo) and np.all(p <= self.hi):
            c = np.floor((p - self.lo) / self.edge).astype(np.int64)
            cand = self.cells[min(max(c[0], 0), self.size - 1)][min(max(c[1], 0), self.size - 1)]
            if len(cand):
                bc = self._bary(cand, p)
                eps = 0.0001
                inside = np.all((bc >= -eps) & (bc <= 1 + eps), axis=1) & (self.area[cand] > 0.00001)
                k = np.flatnonzero(inside)
                if len(k):
                    return int(cand[k[0]]), bc[k[0]]
        # closest triangle (by distance to its edges, as UVGrid::getClosest)
        T = self.T
        best_d = None
        best_c = None
        for i in range(3):
            a = T[:, i]
            b = T[:, (i + 1) % 3]
            e = b - a
            en = (e * e).sum(1)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.clip(((p - a) * e).sum(1) / en, 0, 1)
            t = np.where(en > 0, t, 0.5)
            c = a + e * t[:, None]
            d = np.linalg.norm(c - p, axis=1)
            if best_d is None:
                best_d, best_c = d, c
            else:
                better = d < best_d
                best_d = np.where(better, d, best_d)
                best_c = np.where(better[:, None], c, best_c)
        k = int(np.argmin(best_d))
        return k, self._bary(np.array([k]), best_c[k])[0]

    def map(self, uv):
        """Position and normal on the base mesh for each uv point."""
        n = len(uv)
        P = np.zeros((n, 3))
        N = np.zeros((n, 3))
        for i in range(n):
            res = self.locate(uv[i])
            if res is None:
                continue
            t, bc = res
            tri = self.tris[t]
            P[i] = bc @ self.xyz[tri]
            nn = bc @ self.nrm[tri]
            ln = np.linalg.norm(nn)
            N[i] = nn / ln if ln > 0 else nn
        return P, N


class Retopologizer:
    """PatchRetopologizer: places a candidate patch in the parameter domain of the base mesh region under the query
    patch (fixing its border, pulling its polychords along the user's flows), then maps it to 3D."""

    def __init__(self):
        self.lap_weight = 0.8
        self.quality_threshold = 1.0
        self.constrained_smooth = True
        self.locator = None
        self.boundary = []
        self.boundary_uv = []
        self.flow_map = {}
        self.flow_map_uv = {}
        self.n_corners = NC_MAX
        self.boundary_len = []

    def set_basemesh(self, xyz, nrm, uv, tris):
        self.locator = UVLocator(xyz, nrm, uv, tris) if len(tris) else None

    def set_boundary(self, num_subdiv, boundary_xyz, boundary_uv):
        self.n_corners = len(num_subdiv)
        self.boundary_len = list(num_subdiv)
        total = len(boundary_xyz)
        bxyz = np.asarray(boundary_xyz, dtype=float).reshape(-1, 3)
        buv = np.zeros((total, 3))
        if total:
            buv[:, :2] = np.asarray(boundary_uv, dtype=float).reshape(-1, 2)
        self.boundary = []
        self.boundary_uv = []
        offset = 0
        for s, ns in enumerate(num_subdiv):
            idx = [(offset + i) % total for i in range(ns + 1)]
            self.boundary.append(bxyz[idx])
            self.boundary_uv.append(buv[idx])
            offset += ns

    def set_strokes(self, strokes):
        self.flow_map = {}
        self.flow_map_uv = {}
        for pts_xyz, pts_uv, is_loop in strokes:
            uv3 = np.zeros((len(pts_uv), 3))
            if len(pts_uv):
                uv3[:, :2] = np.asarray(pts_uv, dtype=float).reshape(-1, 2)
            points_to_flow(pts_xyz, is_loop, self.boundary, self.flow_map)
            points_to_flow(uv3, is_loop, self.boundary_uv, self.flow_map_uv)

    def _boundary_to_boundary(self, m, start, boundary):
        run = start.copy()
        if not run.is_border() or not boundary:
            return False
        side = 0
        edge = 0
        g = _Guard(m.walk_limit())
        while True:
            g()
            bs = boundary[side]
            if edge >= len(bs):
                return False
            m.P[run.v] = bs[edge]
            run.flip_v()
            run.flip_e()
            if not run.is_border():
                run.flip_f()
                run.flip_e()
                if not run.is_border():     # concave corner
                    while not run.is_border():
                        g()
                        run.flip_f()
                        run.flip_e()
                    side = (side + 1) % len(boundary)
                    edge = 0
                else:
                    edge += 1
            else:
                side = (side + 1) % len(boundary)
                edge = 0
            if run == start:
                break
        return True

    def _to_3d(self, m):
        if self.locator is None:
            return
        P, N = self.locator.map(m.P[:, :2])
        m.P = P
        m.N = N

    def geometrize(self, r, flows):
        """PatchRetopologizer::geometrizePatch.  Fills r.mesh's positions, r.quality and r.flow_distance; returns
        False when the patch is rejected."""
        m = r.mesh
        m.P = np.zeros((m.nv, 3))
        if not self._boundary_to_boundary(m, r.start, self.boundary_uv):
            return False
        border = m.border_vertices()
        implicit_smooth(m, border, 1.0)
        if not flows:
            self._to_3d(m)
            r.quality = quads_quality(m)
            return r.quality <= self.quality_threshold
        poly_map = self._flows_polychords(r, flows)
        if poly_map is None:
            return False
        if self.constrained_smooth:
            self._smooth_with_flow_constraints(m, poly_map, flows, border)
            self._to_3d(m)
            self._update_flow_polychords(r, poly_map, self.flow_map)
        else:
            self._to_3d(m)
        r.quality = quads_quality(m)
        return r.quality <= self.quality_threshold

    def _flows_polychords(self, r, flows):
        """PatchFlattener::ComputeFlowsPolychords."""
        m = r.mesh
        n_sides = r.template.n_corners or 1
        corners = corners_from_start(m, r.start)
        poly_map = {}
        total = 0.0
        for key in sorted(flows):
            fc = flows[key]
            pv = find_polychords_with_topology(m, corners, key)
            if len(pv) < len(fc):
                r.flow_distance = SCALAR_MAX
                return None
            d, pc = flows_polychords_distance(n_sides, key, fc, pv, m.P)
            if pc is None:
                r.flow_distance = SCALAR_MAX
                return None
            poly_map[key] = pc
            total += d
        r.flow_distance = total
        return poly_map

    def _smooth_with_flow_constraints(self, m, poly_map, flows, border):
        normals = face_normals(m)
        constraints = []
        for key in sorted(poly_map):
            fc = flows.get(key, [])
            for pp, fp in zip(poly_map[key], fc):
                if pp is None or len(pp.samples) < 2 or len(pp.arc) == 0:
                    continue
                step = pp.arc[-1] / (len(pp.samples) - 1)
                if step <= 0:
                    continue
                run = pp.start.copy()
                b_ind = 1
                seen = set()
                g = _Guard(2 * m.nf + 4)
                while True:
                    g()
                    if run.f not in seen:
                        if b_ind >= len(pp.arc):
                            break
                        s_ind = int(math.floor(pp.arc[b_ind] / step + 0.5))
                        s_ind = min(s_ind, len(pp.samples) - 1, len(fp.samples) - 1)
                        fv = m.fv[run.f]
                        w, _ = quad_barycentric(m.P[fv], normals[run.f], pp.samples[s_ind])
                        constraints.append((run.f, w, fp.samples[s_ind]))
                        seen.add(run.f)
                    b_ind += 2
                    run.advance_polychord()
                    if run.is_border() or run == pp.start:
                        break
        implicit_smooth(m, border, 1.0 - self.lap_weight, constraints)

    def _update_flow_polychords(self, r, poly_map, flows):
        """PatchFlattener::UpdateFlowPolychords (with the flows in 3D)."""
        total = 0.0
        for fkey, pkey in zip(sorted(flows), sorted(poly_map)):
            for fp, pp in zip(flows[fkey], poly_map[pkey]):
                if pp is None:
                    continue
                pp.polychord_to_points(r.mesh.P)
                pp.compute_samples(len(fp.samples))
                d, pp.distances = _points_distance(fp.samples, pp.samples, values=True)
                total += _scaled_distance(d, fp.arc[-1] if len(fp.arc) else 0.0, pp.arc[-1] if len(pp.arc) else 0.0)
        r.flow_distance = total


# ---- the database --------------------------------------------------------------------------------------------------
class _CornerTable:
    """Templates with a given number of corners, in the order of PatchDBInterface::querySelect."""

    def __init__(self, conn, n_corners):
        cur = conn.execute(
            'SELECT rowid, boundaryIDs, singularities, nSingularities, nConcaveCorners FROM Patches '
            'WHERE nCorners = ? ORDER BY nSingularities ASC, singularities ASC, nOccurrences DESC, nFaces ASC',
            (n_corners,))
        n_sides = n_corners if n_corners > 0 else 1
        rowids = []
        n_sing = []
        minv = []
        maxv = []
        n_conc = []
        side_len = []
        for rowid, ids, sing, ns, nconc in cur:
            sides = [0] * n_sides
            side = 0
            ok = True
            i = 0
            ids = ids or ''
            while i < len(ids):
                if ids[i] == SEPARATOR:
                    side += 1
                    i += 1
                    continue
                if side >= n_sides:
                    ok = False
                    break
                sides[side] += 1
                i += 1 + LABEL_HEX
            if not ok:
                continue
            vals = [int(p[1:1 + LABEL_HEX], 16) for p in sing.split(SEPARATOR)] if sing else []
            rowids.append(rowid)
            n_sing.append(ns)
            minv.append(min(vals) if vals else 1 << 30)
            maxv.append(max(vals) if vals else -1)
            n_conc.append(nconc)
            side_len.append(sides)
        self.n_corners = n_corners
        self.rowids = np.array(rowids, dtype=np.int64)
        self.n_sing = np.array(n_sing, dtype=np.int64)
        self.minv = np.array(minv, dtype=np.int64)
        self.maxv = np.array(maxv, dtype=np.int64)
        self.n_concave = np.array(n_conc, dtype=np.int64)
        self.side_len = np.array(side_len, dtype=np.int64).reshape(-1, n_sides)
        self.perimeter = self.side_len.sum(axis=1)

    def __len__(self):
        return len(self.rowids)


class PatchDB:
    """Read-only access to a patches.db file (PatchDBServer)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        uri = 'file:%s?mode=ro' % urllib.request.pathname2url(self.path)
        self.conn = sqlite3.connect(uri, uri=True)
        try:
            self.corner_numbers = sorted(r[0] for r in self.conn.execute('SELECT DISTINCT nCorners FROM Patches'))
            self.num_templates = int(self.conn.execute('SELECT count(*) FROM Patches').fetchone()[0])
        except sqlite3.DatabaseError:
            self.conn.close()
            raise
        self._tables = {}
        self._templates = {}

    def close(self):
        self.conn.close()

    def table(self, n_corners):
        t = self._tables.get(n_corners)
        if t is None:
            t = _CornerTable(self.conn, n_corners)
            self._tables[n_corners] = t
        return t

    def template(self, n_corners, index):
        """The decoded template at `index` of the n_corners table (None if malformed)."""
        key = (n_corners, index)
        t = self._templates.get(key, False)
        if t is not False:
            return t
        if len(self._templates) > 50000:
            self._templates.clear()
        rowid = int(self.table(n_corners).rowids[index])
        row = self.conn.execute(
            'SELECT topology, boundaryIDs, singularities, nSingularities, nCorners, nFaces, nVertices, nOccurrences, '
            'nConcaveCorners FROM Patches WHERE rowid = ?', (rowid,)).fetchone()
        try:
            t = Template(index, row)
            t.ensure_mesh()
        except (MalformedTemplate, ValueError, IndexError, TypeError):
            t = None
        self._templates[key] = t
        return t


def create_db(path, rows):
    """Writes rows (as returned by encode_patch) into a new database with the demo's schema."""
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE IF NOT EXISTS Patches (topology TEXT PRIMARY KEY, boundaryIDs TEXT NOT NULL, '
                 'singularities TEXT NOT NULL, nSingularities INTEGER CHECK (nSingularities >= 0), '
                 'nCorners INTEGER CHECK (nCorners >= 0), nFaces INTEGER CHECK (nFaces > 0), '
                 'nVertices INTEGER CHECK (nVertices > 0), nOccurrences INTEGER CHECK (nOccurrences > 0), '
                 'nConcaveCorners INTEGER CHECK (nConcaveCorners >= 0))')
    conn.execute('CREATE INDEX IF NOT EXISTS Patches_index ON Patches (nCorners ASC, nSingularities ASC, '
                 'singularities ASC, nOccurrences DESC, nFaces ASC)')
    conn.executemany('INSERT OR IGNORE INTO Patches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
    conn.commit()
    conn.close()


# ---- the lookup (PatchPriorityLookup / MT_PatchPriorityLookup) ----------------------------------------------------
class Result:
    """A patch found by the lookup: a template with its rotation and polychord widths, expanded and placed."""
    __slots__ = ('template', 'rot', 'var', 'mesh', 'start', 'flow_distance', 'quality', 'boundary_len', 'perimeter')

    def default_key(self):
        t = self.template
        return (t.n_sing, -t.n_occ, t.n_faces, t.n_vertices, t.n_corners, self.perimeter, tuple(self.boundary_len),
                t.topology, tuple(tuple(s) for s in t.boundary_ids), self.rot, self.var)

    def sort_key(self, criteria):
        k = []
        for c in criteria:
            if c == FLOW_DISTANCE:
                k.append(self.flow_distance)
            elif c == SINGULARITYNUM:
                k.append(self.template.n_sing)
            elif c == SINGULARITYTYPES:
                k.append(self.template.sing_vec)
            elif c == QUADS_QUALITY:
                k.append(self.quality)
            elif c == PATCH_DEFAULT:
                k.append(self.default_key())
        k.append(self.default_key())
        return tuple(k)


class Lookup:
    """Finds the templates that realize a boundary (and, optionally, follow flows), expands and places them, and
    keeps them sorted.  Call start(), then step() until status is 'complete'."""

    def __init__(self, db=None):
        self.db = db
        self.max_singularities = 16
        self.max_concavities = 0
        self.min_valence = 3
        self.max_valence = 5
        self.min_size_threshold_ilp = 1
        self.flow_tolerance = 1
        self.sort_criteria = SORT_BY_FLOW
        self.status = 'idle'
        self.results = []
        self._keys = []
        self._unique = set()
        self._phase = None
        self.n_templates = 0

    def num_results(self):
        return len(self.results)

    def stop(self):
        """Interrupts the lookup, keeping what was found (the original then notifies completion)."""
        if self._phase is not None:
            self._phase = None
            self.status = 'complete'

    def clear(self):
        self._phase = None
        self.status = 'idle'
        self.results = []
        self._keys = []
        self._unique = set()

    def start(self, n_corners, boundary_length, flows=None, geometrizer=None):
        self.clear()
        self.n_corners = n_corners
        self.boundary_length = [int(x) for x in boundary_length]
        self.perimeter = sum(self.boundary_length)
        self.flows = flows or {}
        self.geometrizer = geometrizer
        # the settings in effect for this lookup (the original ignores changes while it runs)
        self._criteria = tuple(self.sort_criteria)
        self._max_sing = self.max_singularities
        self._max_conc = self.max_concavities
        self._min_val = self.min_valence
        self._max_val = self.max_valence
        self._ilp_th = self.min_size_threshold_ilp
        self._tol = self.flow_tolerance
        self.status = 'started'
        self._phase = 'structure'
        self._cand = []
        self._ci = 0
        self._sol_iter = None

    @property
    def running(self):
        return self._phase is not None

    def step(self, time_budget=0.03):
        """Advances the lookup for about `time_budget` seconds.  Returns True while not complete."""
        if self._phase is None:
            return False
        t0 = time.perf_counter()
        if self.status == 'started':
            self.status = 'progress'
        while self._phase is not None:
            if self._phase == 'structure':
                self._structure_filter()
                self._ci = 0
                self._phase = 'flows' if self.flows and self._cand else 'ilp'
                if self._phase == 'flows':
                    self._passed = []
            elif self._phase == 'flows':
                if self._ci >= len(self._cand):
                    self._cand = self._passed
                    self._ci = 0
                    self._phase = 'ilp'
                else:
                    idx, rot = self._cand[self._ci]
                    self._ci += 1
                    if self._flow_filter(idx, rot):
                        self._passed.append((idx, rot))
                    else:
                        self.n_templates -= 1
            elif self._phase == 'ilp':
                if self._sol_iter is None:
                    if self._ci >= len(self._cand):
                        self._phase = None
                        self.status = 'complete'
                        self._unique = set()
                        break
                    idx, rot = self._cand[self._ci]
                    self._ci += 1
                    self._sol_template = self.db.template(self.n_corners, idx) if self.db else None
                    self._sol_rot = rot
                    self._sol_iter = self._solutions(self._sol_template, rot) if self._sol_template else None
                    if self._sol_iter is None:
                        self.n_templates -= 1
                else:
                    var = next(self._sol_iter, None)
                    if var is None:
                        self._sol_iter = None
                    else:
                        self._process_solution(self._sol_template, self._sol_rot, var)
            if time.perf_counter() - t0 >= time_budget:
                break
        return self._phase is not None

    def run(self, time_limit=None):
        t0 = time.perf_counter()
        while self.step(0.1):
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                self.stop()
                break

    # -- filters --
    def _structure_filter(self):
        """structureFilterWorker: singularities, valences, concavities, perimeter, and side lengths per rotation."""
        self._cand = []
        self.n_templates = 0
        if self.db is None or self.perimeter < 4 or self.perimeter % 2 != 0:
            return
        if self.n_corners not in self.db.corner_numbers:
            return
        tab = self.db.table(self.n_corners)
        n = len(tab)
        if n == 0:
            return
        min_val, max_val = self._min_val, self._max_val
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        has_sing = tab.maxv >= 0
        ok = (tab.n_sing <= self._max_sing) & (tab.n_concave <= self._max_conc) & \
             (tab.perimeter <= self.perimeter) & (~has_sing | ((tab.minv >= min_val) & (tab.maxv <= max_val)))
        q = np.array(self.boundary_length, dtype=np.int64)
        ns = len(q)
        n_sides_t = tab.side_len.shape[1]
        rot_ok = np.zeros((n, ns), dtype=bool)
        for rot in range(ns):
            cols = [(side + rot) % n_sides_t for side in range(min(ns, n_sides_t))]
            rot_ok[:, rot] = np.all(q[None, :len(cols)] >= tab.side_len[:, cols], axis=1)
        rot_ok &= ok[:, None]
        idx, rot = np.nonzero(rot_ok)
        self._cand = list(zip(idx.tolist(), rot.tolist()))
        self.n_templates = len(self._cand)

    def _flow_filter(self, idx, rot):
        """flowFilterWorker."""
        t = self.db.template(self.n_corners, idx)
        if t is None:
            return False
        ns = len(self.boundary_length)
        try:
            corners = corners_from_start(t.mesh, t.start)
            for key in sorted(self.flows):
                if key[0] < ns and key[1] < ns:
                    ss = (key[0] + rot) % ns
                    es = (key[1] + rot) % ns
                    if not any(t.ends[b] == (ss, es) or t.ends[b] == (es, ss) for b in t.boundary_ids[ss]):
                        return False
                elif len(find_polychords_with_topology(t.mesh, corners, key)) < len(self.flows[key]):
                    return False
        except (MalformedTemplate, IndexError):
            return False
        return True

    def _solutions(self, t, rot):
        """solveILP: the polychord widths (one per boundary id) realizing the boundary lengths, as a generator."""
        ns = len(self.boundary_length)
        nv = t.n_vars
        cons = [[0] * nv for _ in range(ns)]
        try:
            for v in range(nv):
                cons[t.ends[v][0]][v] += 1
                cons[t.ends[v][1]][v] += 1
        except IndexError:
            return None
        rhs = [0] * ns
        for i in range(ns):
            rhs[(i + rot) % ns] = self.boundary_length[i]
        lo = [1] * nv
        hi = [INF] * nv
        if self.flows and min(self.boundary_length) > self._ilp_th:
            res = constraint_ilp_by_flows(t, rot, self.boundary_length, self.flows, lo, hi, cons, rhs, self._tol)
            if res is None:
                return None
            lo, hi, cons, rhs = res

        def gen():
            seen = set()
            for x in enumerate_ilp(cons, rhs, lo, hi):
                var = tuple(x[:nv])
                if var not in seen:
                    seen.add(var)
                    yield var
        return gen()

    def _process_solution(self, t, rot, var):
        """One solution of ilpFilterWorker: expand, check the flows' polychords and uniqueness, place, insert.
        (Uniqueness is looked up before the flow check: a known topology is dropped either way, and it is only
        registered once the flow check passed, as in the original.)"""
        try:
            ex, (cell, z, v, d) = t.expand_adjacency(var, rot)
            h = hash(_topology_str_adj(ex.ffp, ex.ffi, cell, z, d))
            if h in self._unique:
                return
            m = ex.mesh()
        except MalformedTemplate:
            return
        start = Pos(m, cell, z, v)
        r = Result()
        r.template = t
        r.rot = rot
        r.var = var
        r.mesh = m
        r.start = start
        r.flow_distance = SCALAR_MAX
        r.quality = SCALAR_MAX
        r.boundary_len = self.boundary_length
        r.perimeter = self.perimeter
        if self.flows:
            try:
                corners = corners_from_start(m, start)
                for key in sorted(self.flows):
                    if len(find_polychords_with_topology(m, corners, key)) < len(self.flows[key]):
                        return
            except MalformedTemplate:
                return
        self._unique.add(h)
        if self.geometrizer is not None:
            try:
                if not self.geometrizer.geometrize(r, self.flows):
                    return
            except MalformedTemplate:
                return
        if r.flow_distance != r.flow_distance:
            r.flow_distance = SCALAR_MAX
        if r.quality != r.quality:
            r.quality = SCALAR_MAX
        key = r.sort_key(self._criteria)
        i = bisect.bisect_left(self._keys, key)
        self._keys.insert(i, key)
        self.results.insert(i, r)


def build_patch(r, patch):
    """convert_patch: replaces the content of a curvenetwork patch with a lookup result."""
    m = r.mesh
    patch.clear()
    for v in range(m.nv):
        patch.add_vertex(np.concatenate((m.P[v], m.N[v])))
    # add the faces breadth-first so that each new face touches the ones already added
    order = []
    seen = [False] * m.nf
    queue = [0]
    seen[0] = True
    while queue:
        f = queue.pop(0)
        order.append(f)
        for g in m.ffp[f]:
            if not seen[g]:
                seen[g] = True
                queue.append(g)
    for f in range(m.nf):
        if not seen[f]:
            order.append(f)
    for f in order:
        if patch.add_face(list(m.fv[f])) < 0:
            patch.clear()
            return False
    h = patch.halfedge_handle(r.start.v)
    num_subdiv = patch.num_subdiv_all()
    for i, ns in enumerate(num_subdiv):
        patch.v_corner[patch.from_vertex_handle(h)] = i
        for _ in range(ns):
            h = patch.prev_halfedge_handle(h)
    patch.is_failure = False
    patch.param = None
    patch.set_halfedgeData()
    patch.set_boundary_pn()
    patch.invalidate_displist()
    return True


_open_dbs = {}


def open_db(path):
    """The PatchDB of a file, shared by all sessions (reopened when the file changed)."""
    path = os.path.abspath(path)
    stamp = os.path.getmtime(path)
    entry = _open_dbs.get(path)
    if entry is not None and entry[1] == stamp:
        return entry[0]
    db = PatchDB(path)
    if entry is not None:
        entry[0].close()
    _open_dbs[path] = (db, stamp)
    return db


class LearnQuad:
    """The patch database feature of SketchRetopo2: the database, the interactive lookup of the query database mode
    and its settings, and patch generation for loops with more than 6 corners."""

    def __init__(self):
        self.db = None
        self.lookup = Lookup()
        self.retopologizer = Retopologizer()
        self.querydata = None

    # -- database --
    def load(self, path):
        db = open_db(path)
        self.lookup.clear()
        self.db = db
        self.lookup.db = db
        return db.num_templates

    def close(self):
        self.lookup.clear()
        self.db = None
        self.lookup.db = None

    @property
    def path(self):
        return self.db.path if self.db is not None else ''

    def is_loaded(self):
        return self.db is not None

    def num_templates(self):
        return self.db.num_templates if self.db is not None else 0

    # -- interactive lookup (StateQueryDB) --
    @property
    def status(self):
        return self.lookup.status

    def num_results(self):
        return self.lookup.num_results()

    def stop(self):
        self.lookup.stop()

    def step(self, time_budget=0.03):
        return self.lookup.step(time_budget)

    def send_query(self, qd):
        """QueryData::send_basemesh / send_boundary / send_strokes."""
        rt = self.retopologizer
        rt.set_basemesh(qd.basemesh_vertices_xyz, qd.basemesh_vertices_normal, qd.basemesh_vertices_uv,
                        qd.basemesh_faces)
        rt.set_boundary(qd.num_subdiv, qd.boundary_xyz, qd.boundary_uv)
        rt.set_strokes(qd.strokes)
        self.querydata = qd

    def send_strokes(self, qd):
        self.retopologizer.set_strokes(qd.strokes)

    def start_lookup(self, qd):
        """StateQueryDB::mouse_up: the base mesh and boundary were sent when the patch was selected."""
        if self.querydata is not qd:
            self.send_query(qd)
        else:
            self.send_strokes(qd)
        self.restart(qd)

    def restart(self, qd):
        self.lookup.stop()
        if self.db is None:
            return
        self.lookup.start(len(qd.num_subdiv), qd.num_subdiv, self.retopologizer.flow_map_uv, self.retopologizer)

    def convert_result(self, index, patch):
        if index < 0 or index >= self.lookup.num_results():
            return False
        return build_patch(self.lookup.results[index], patch)

    # -- SketchRetopo::generate_patch for more than 6 corners --
    def generate_patch(self, engine, patch, num_subdiv, time_limit=10.0):
        """Takes the best template found within the first moments of a lookup (the original converts the first
        result notified, ~0.1 s after the start, and interrupts the lookup)."""
        if self.db is None:
            return False
        lk = Lookup(self.db)
        for name in ('max_singularities', 'max_concavities', 'min_valence', 'max_valence', 'min_size_threshold_ilp',
                     'flow_tolerance', 'sort_criteria'):
            setattr(lk, name, getattr(self.lookup, name))
        lk.start(len(num_subdiv), num_subdiv)
        t0 = time.perf_counter()
        while lk.step(0.02):
            el = time.perf_counter() - t0
            if (lk.num_results() > 0 and el >= 0.1) or el > time_limit:
                lk.stop()
                break
        if lk.num_results() == 0:
            return False
        return build_patch(lk.results[0], patch)
