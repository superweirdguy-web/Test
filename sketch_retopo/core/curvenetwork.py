# Port of SketchRetopo/curvenetwork/*: the halfedge-based curve network
# (Vertex, Halfedge, Halfchain, Edgechain, Patch, Core) and its editing
# operations (adding/erasing curves, snapping, intersections, corners).
#
# C++ ElemPtr semantics: an ElemPtr evaluates to true only if it is non-null
# AND not deleted.  alive() reproduces that; plain `is None` checks are used
# where the original compares raw pointers.

import math

import numpy as np

from .pn import (Polyline, pn_norm, pn_normalize, normalized, cross3, dot3,
                 orthonormalize, acos_clamped, distance_to_triangle)
from .polymesh import PolyMesh
from . import patchgen


def alive(x):
    return x is not None and not x.is_deleted


# ---- elements ---------------------------------------------------------------------------------

class Vertex:
    __slots__ = ('id', 'is_deleted', 'flag', 'halfedge', 'pn', 'is_hidden')

    def __init__(self, id_=-1):
        self.id = id_
        self.is_deleted = False
        self.flag = 0
        self.halfedge = None
        self.pn = np.zeros(6)
        self.is_hidden = False

    def is_openend(self):
        return self.halfedge is not None and self.halfedge.prev is None

    def is_corner(self):
        for h in vih_iter(self):
            if not h.is_deleted and h.is_corner:
                return True
        return False

    def is_boundary(self):
        for c in voc_iter(self):
            if c is not None and c.is_boundary():
                return True
        return False

    def valence(self):
        return sum(1 for h in voh_iter(self) if not h.is_deleted)

    def on_symmetry_plane(self):
        for c in voc_iter(self):
            if c is not None and c.edgechain is not None and c.edgechain.on_symmetry_plane():
                return True
        return False


class Halfedge:
    __slots__ = ('id', 'is_deleted', 'flag', 'vertex', 'next', 'prev', 'opposite', 'halfchain',
                 'is_corner', 'imaginary_patch')

    def __init__(self, id_=-1):
        self.id = id_
        self.is_deleted = False
        self.flag = 0
        self.vertex = None
        self.next = None
        self.prev = None
        self.opposite = None
        self.halfchain = None
        self.is_corner = False
        self.imaginary_patch = None

    def from_vertex(self):
        return self.opposite.vertex

    def patch(self):
        return self.halfchain.patch if self.halfchain is not None else None

    def to_vector(self):
        return self.vertex.pn[:3] - self.opposite.vertex.pn[:3]


class Halfchain:
    __slots__ = ('id', 'is_deleted', 'flag', 'halfedge_front', 'halfedge_back', 'patch', 'edgechain')

    def __init__(self, id_=-1):
        self.id = id_
        self.is_deleted = False
        self.flag = 0
        self.halfedge_front = None
        self.halfedge_back = None
        self.patch = None
        self.edgechain = None

    def next(self):
        n = self.halfedge_back.next
        if alive(n):
            return n.halfchain
        return None

    def prev(self):
        p = self.halfedge_front.prev
        if alive(p):
            return p.halfchain
        return None

    def opposite(self):
        return self.halfedge_front.opposite.halfchain

    def is_corner(self):
        return self.halfedge_back.is_corner

    def num_subdiv(self):
        return self.edgechain.num_subdiv

    def vertex_front(self):
        return self.halfedge_front.from_vertex()

    def vertex_back(self):
        return self.halfedge_back.vertex

    def is_loop(self):
        return self.halfedge_front is self.halfedge_back.next

    def to_polyline(self):
        pts = [self.halfedge_front.from_vertex().pn.copy()]
        for h in ch_iter(self):
            pts.append(h.vertex.pn.copy())
        result = Polyline()
        result.points = pts
        if self.is_loop():
            result.points.pop()
            result.is_loop = True
        return result

    def length(self):
        return self.to_polyline().length()

    def resample(self):
        polyline = self.to_polyline()
        polyline.resample()
        i = 1
        for h in ch_iter(self):
            if h is self.halfedge_back:
                break
            h.vertex.pn = polyline[i]
            i += 1

    def pn_at(self, t):
        return self.to_polyline().point_at(t)

    def patch_vertex_at(self, index):
        patch = self.patch
        if not Patch.is_ordinary(patch) or index < 0 or self.num_subdiv() < index:
            return -1
        h_start = patch.prev_halfedge_handle(patch.halfedge_handle(patch.patch_corner(0)))
        h = h_start
        while True:
            if patch.h_chain[h] is self:
                for _ in range(index):
                    h = patch.prev_halfedge_handle(h)
                return patch.to_vertex_handle(h)
            h = patch.prev_halfedge_handle(h)
            if h == h_start:
                break
        return -1

    def is_boundary(self):
        opp = self.opposite()
        return (not Patch.is_ordinary(self.patch)) and opp is not None and Patch.is_ordinary(opp.patch)


class Edgechain:
    __slots__ = ('id', 'is_deleted', 'flag', 'halfchain', 'num_subdiv')

    def __init__(self, id_=-1):
        self.id = id_
        self.is_deleted = False
        self.flag = 0
        self.halfchain = [None, None]
        self.num_subdiv = 0

    def is_boundary(self):
        return self.halfchain[0].is_boundary() or self.halfchain[1].is_boundary()

    def on_symmetry_plane(self):
        c0, c1 = self.halfchain
        return ((c0 is not None and c0.patch is IMAGINARY_SYMMETRY) or
                (c1 is not None and c1.patch is IMAGINARY_SYMMETRY))


# ---- circulators --------------------------------------------------------------------------------

def voh_iter(v):
    """Outgoing halfedges around a curve network vertex."""
    start = v.halfedge
    cur = start
    while cur is not None:
        yield cur
        cur = cur.opposite.next
        if cur is start:
            break


def vih_iter(v):
    for h in voh_iter(v):
        yield h.opposite


def voc_iter(v):
    for h in voh_iter(v):
        yield h.halfchain


def pc_iter(patch):
    start = patch.halfchain
    cur = start
    while cur is not None:
        yield cur
        cur = cur.next()
        if cur is start:
            break


def ch_iter(c):
    cur = c.halfedge_front
    while cur is not None:
        yield cur
        if cur is c.halfedge_back:
            break
        cur = cur.next


# ---- helpers (curvenetwork/helper.cc) ------------------------------------------------------------

def set_next_prev(h_pred, h_succ):
    if h_pred is not None:
        h_pred.next = h_succ
    if h_succ is not None:
        h_succ.prev = h_pred


def set_vertex_halfedge(v0, v1, h01, h10):
    v0.halfedge = h01
    v1.halfedge = h10
    h01.vertex = v1
    h10.vertex = v0
    h01.opposite = h10
    h10.opposite = h01


def trace_halfchains(in_c):
    """Returns (out_c_front, out_c_back)."""
    c_forward = in_c
    c_backward = in_c
    out_front = None
    out_back = None
    i = 0
    while True:
        if i % 2 == 0 and c_forward is not None:
            nxt = c_forward.next()
            if not alive(nxt):
                out_back = c_forward
            c_forward = nxt
        elif i % 2 == 1 and c_backward is not None:
            prv = c_backward.prev()
            if not alive(prv):
                out_front = c_backward
            c_backward = prv
        if c_forward is c_backward:
            break
        i += 1
        if i > 10000000:
            break
    if c_forward is not None:
        out_front = in_c
        out_back = in_c.prev()
    return out_front, out_back


def delete_patch(p):
    if p is None or p.is_deleted or p.is_imaginary():
        return
    p.is_deleted = True
    for c in list(pc_iter(p)):
        c.patch = None
        opp = c.opposite()
        if alive(opp) and not Patch.is_ordinary(opp.patch):
            c.edgechain.num_subdiv = 0


def delete_halfchain(c):
    if c is None or c.is_deleted:
        return
    c.is_deleted = True
    delete_patch(c.patch)
    for h in list(ch_iter(c)):
        h.halfchain = None
    e = c.edgechain
    if e is not None:
        c0, c1 = e.halfchain
        if (c0 is None or c0.is_deleted) and (c1 is None or c1.is_deleted):
            e.is_deleted = True


def delete_halfedge(h):
    if h is None or h.is_deleted:
        return
    h.is_deleted = True
    delete_halfchain(h.halfchain)
    v = h.from_vertex()
    is_isolated = True
    for voh in voh_iter(v):
        if voh.is_deleted:
            continue
        v.halfedge = voh
        is_isolated = False
        break
    if is_isolated:
        v.is_deleted = True


# ---- Patch ------------------------------------------------------------------------------------------

class Patch(PolyMesh):
    __slots__ = ('id', 'is_deleted', 'flag', 'halfchain', 'is_failure', 'is_hidden', 'param', 'version')

    quadSize = 1.0
    use_even_num_subdiv = True
    prefer_rect_proc3 = True
    failure_patch_num_subdiv = 4

    def __init__(self, id_=-1):
        PolyMesh.__init__(self)
        self.id = id_
        self.is_deleted = False
        self.flag = 0
        self.halfchain = None
        self.is_failure = False
        self.is_hidden = False
        self.param = patchgen.PatchParam()
        self.version = 0
        if id_ < -1:
            # imaginary patches hold a single vertex so that they are not seen as empty
            self.add_vertex()

    def __deepcopy__(self, memo):
        if self.id < -1:
            return self
        raise TypeError('Patch objects are copied through Core.snapshot()')

    @staticmethod
    def is_ordinary(patch):
        return patch is not None and not patch.is_failure and not patch.is_imaginary()

    def is_imaginary_symmetry(self):
        return self is IMAGINARY_SYMMETRY

    def is_imaginary_boundary(self):
        return self is IMAGINARY_BOUNDARY

    def is_imaginary(self):
        return self is IMAGINARY_SYMMETRY or self is IMAGINARY_BOUNDARY

    def invalidate_displist(self):
        self.version += 1

    def clear(self):
        self.clear_mesh()
        self.invalidate_displist()

    def is_boundary(self, v):
        return self.is_boundary_v(v)

    def num_corners(self):
        result = 0
        c = self.halfchain
        guard = 0
        while True:
            if not alive(c):
                return 0   # bug!
            if c.is_corner():
                result += 1
            c = c.next()
            if c is self.halfchain:
                break
            guard += 1
            if guard > 1000000:
                return 0
        return result

    def opposite_boundary_halfedge(self, h):
        c = self.h_chain[h]
        c_opposite = c.opposite()
        p_opposite = self.opposite_patch(h)
        if p_opposite is None:
            return -1
        target = c.num_subdiv() - 1 - self.h_index[h]
        for h2 in p_opposite.halfedges():
            if p_opposite.h_chain[h2] is c_opposite and p_opposite.h_index[h2] == target:
                return h2
        return -1

    def opposite_patch(self, h):
        c = self.h_chain[h]
        if c is None:
            return None
        c_opposite = c.opposite()
        if c_opposite is None or not Patch.is_ordinary(c_opposite.patch) or c.patch is c_opposite.patch:
            return None
        return c_opposite.patch

    def vertex_patch_to_curveNetwork(self, v):
        if not self.is_boundary_v(v):
            return None
        h = self.prev_halfedge_handle(self.halfedge_handle(v))
        if self.h_index[h] != 0:
            return None
        c = self.h_chain[h]
        if c is None:
            return None
        return c.halfedge_front.from_vertex()

    def vertex_curveNetwork_to_patch(self, cv):
        if not cv.is_corner():
            return -1
        for v in self.vertices():
            if not self.is_boundary_v(v):
                continue
            if self.vertex_patch_to_curveNetwork(v) is cv:
                return v
        return -1

    def patch_corner(self, corner_index):
        try:
            return self.v_corner.index(corner_index)
        except ValueError:
            return -1

    def num_subdiv(self, side_index):
        current_index = 0
        result = 0
        c = self.halfchain
        while True:
            result += c.num_subdiv()
            if c.is_corner():
                if side_index == current_index:
                    return result
                current_index += 1
                result = 0
            c = c.next()
            if c is self.halfchain or c is None:
                break
        return 0

    def num_free_halfchains(self, side_index):
        current_index = 0
        result = 0
        c = self.halfchain
        while True:
            if c.num_subdiv() == 0:
                result += 1
            if c.is_corner():
                if current_index == side_index:
                    return result
                current_index += 1
                result = 0
            c = c.next()
            if c is self.halfchain or c is None:
                break
        return 1   # the original returns `true` here

    def side_length(self, side_index):
        current_index = 0
        result = 0.0
        c = self.halfchain
        while True:
            result += c.length()
            if c.is_corner():
                if side_index == current_index:
                    return result
                current_index += 1
                result = 0.0
            c = c.next()
            if c is self.halfchain or c is None:
                break
        return 0.0

    def change_num_subdiv(self, side_index, target_num_subdiv):
        delta_target = target_num_subdiv - self.num_subdiv(side_index)
        if delta_target <= 0:
            return
        c_free = []
        current_index = 0
        c = self.halfchain
        while True:
            if current_index == side_index and c.num_subdiv() == 0:
                c_free.append(c)
            if c.is_corner():
                if current_index == side_index:
                    break
                current_index += 1
            c = c.next()
            if c is self.halfchain or c is None:
                break
        if not c_free:
            return
        lengths = [cf.length() for cf in c_free]
        length_sum = sum(lengths)
        for cf, length in zip(c_free, lengths):
            ns = int(math.floor(0.5 + delta_target * length / length_sum)) if length_sum > 0 else delta_target
            if Patch.use_even_num_subdiv and ns % 2:
                ns += 1
            cf.edgechain.num_subdiv = ns

    def get_side_halfedges(self, side_index):
        h_front = self.prev_halfedge_handle(self.halfedge_handle(self.patch_corner(side_index)))
        h = h_front
        while self.v_corner[self.from_vertex_handle(h)] == -1:
            h = self.prev_halfedge_handle(h)
        return h_front, h

    def get_side_polyline(self, side_index):
        h_front, h_back = self.get_side_halfedges(side_index)
        pts = [self._pn[self.to_vertex_handle(h_front)].copy()]
        h = h_front
        while True:
            pts.append(self._pn[self.from_vertex_handle(h)].copy())
            if h == h_back:
                break
            h = self.prev_halfedge_handle(h)
        return Polyline(pts, False)

    def distance(self, p):
        """Distance from p to the patch surface (quads split into two triangles)."""
        if self.n_faces() == 0:
            return float('inf')
        quads = np.array([self.fv(f)[:4] for f in self.faces()])
        P = self._pn[:self._nv, :3]
        return _min_distance_to_quads(P[quads], np.asarray(p, dtype=float))

    def generate_topology(self, init_param=True):
        num_sides = self.num_corners()
        if init_param:
            for i in range(num_sides):
                j = (i + 2) % num_sides
                is_fixed_i = self.num_free_halfchains(i) == 0
                is_fixed_j = self.num_free_halfchains(j) == 0
                if not is_fixed_i and not is_fixed_j:
                    target = max(int(self.side_length(i) / Patch.quadSize),
                                 int(self.side_length(j) / Patch.quadSize),
                                 self.num_subdiv(i) + self.num_free_halfchains(i),
                                 self.num_subdiv(j) + self.num_free_halfchains(j))
                    self.change_num_subdiv(i, target)
                    self.change_num_subdiv(j, target)
                elif not is_fixed_i:
                    self.change_num_subdiv(i, self.num_subdiv(j))
                elif not is_fixed_j:
                    self.change_num_subdiv(j, self.num_subdiv(i))
            l = [self.num_subdiv(i) for i in range(num_sides)]
            self.is_failure = sum(l) % 2 != 0
            if self.is_failure:
                l = [Patch.failure_patch_num_subdiv] * num_sides
            param = patchgen.get_default_parameter(l)
            if param is None:
                # no pattern can realize this boundary (the original asserts here); show it as a failure patch
                self.is_failure = True
                param = patchgen.get_default_parameter([Patch.failure_patch_num_subdiv] * num_sides)
            self.param = param
        if self.param is None or self.param.pattern_id < 0 or not (2 <= self.param.get_num_sides() <= 6):
            self.clear()
            return
        patchgen.generate_topology(self.param, self)
        self.set_halfedgeData()
        self.set_boundary_pn()
        self.invalidate_displist()

    def set_halfedgeData(self):
        if self.is_failure:
            return
        v_corner = self.patch_corner(0)
        if v_corner < 0:
            return
        h = self.prev_halfedge_handle(self.halfedge_handle(v_corner))
        c = self.halfchain
        if c is None:
            return
        guard = 0
        while True:
            index = 0
            for _ in range(c.num_subdiv()):
                self.h_chain[h] = c
                self.h_index[h] = index
                h = self.prev_halfedge_handle(h)
                index += 1
            c = c.next()
            if c is self.halfchain or c is None:
                break
            guard += 1
            if guard > 100000:
                break

    def set_boundary_pn(self):
        v_corner = self.patch_corner(0)
        if v_corner < 0:
            return
        h0 = self.prev_halfedge_handle(self.halfedge_handle(v_corner))
        if self.is_failure:
            h = h0
            c = self.halfchain
            for _ in range(self.num_corners()):
                polyline_all = Polyline()
                while True:
                    pc = c.to_polyline()
                    polyline_all.points.extend(pc.points)
                    if c.is_corner():
                        break
                    polyline_all.points.pop()
                    c = c.next()
                n = Patch.failure_patch_num_subdiv
                for j in range(n):
                    self._pn[self.to_vertex_handle(h)] = polyline_all.point_at(j / float(n))
                    h = self.prev_halfedge_handle(h)
                c = c.next()
        else:
            h = h0
            while True:
                c = self.h_chain[h]
                v = self.to_vertex_handle(h)
                h_opposite = self.opposite_boundary_halfedge(h)
                if h_opposite >= 0:
                    opposite_patch = c.opposite().patch
                    self._pn[v] = opposite_patch._pn[opposite_patch.from_vertex_handle(h_opposite)]
                elif c is not None:
                    t = self.h_index[h] / float(c.num_subdiv())
                    self._pn[v] = c.pn_at(t)
                h = self.prev_halfedge_handle(h)
                if h == h0:
                    break

    def irregular_vertices(self):
        """[(vertex, valence-in-closed-sense)] for vertices whose valence is not 4."""
        result = []
        for v in self.vertices():
            val = self.valence(v)
            if self.v_corner[v] != -1:
                val += 2
            elif self.is_boundary_v(v):
                val += 1
            if val != 4:
                result.append((v, val))
        return result


def _min_distance_to_quads(Q, p):
    """Q: (F, 4, 3) quad vertex positions.  Mirrors Patch::distance (two triangles per quad,
    kt84::distance_to_triangle with clamping)."""
    best = float('inf')
    for (a, b, c) in ((0, 1, 2), (0, 2, 3)):
        x0 = Q[:, a]
        x1 = Q[:, b]
        x2 = Q[:, c]
        d = _distance_to_triangles(x0, x1, x2, p)
        if d.size:
            best = min(best, float(np.nanmin(d)))
    return best


def _distance_to_triangles(x0, x1, x2, p):
    d01 = x1 - x0
    d02 = x2 - x0
    d0p = p[None, :] - x0
    a = (d01 * d01).sum(1)
    b = (d01 * d02).sum(1)
    c = (d02 * d02).sum(1)
    det = a * c - b * b
    b0 = (d01 * d0p).sum(1)
    b1 = (d02 * d0p).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (c * b0 - b * b1) / det
        v = (-b * b0 + a * b1) / det
    t0 = 1 - u - v
    inside = (u >= 0) & (v >= 0) & (det != 0)   # original: (clamp && t0 < 0) || t1 < 0 || t2 < 0 -> edges
    proj = t0[:, None] * x0 + u[:, None] * x1 + v[:, None] * x2
    d_inside = np.sqrt(((proj - p[None, :]) ** 2).sum(1))

    def seg(e0, e1):
        d = e1 - e0
        r = (d * d).sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = ((p[None, :] - e0) * d).sum(1) / r
        t0 = np.clip(1 - t1, 0.0, 1.0)
        t1 = 1 - t0
        q = t0[:, None] * e0 + t1[:, None] * e1
        out = np.sqrt(((q - p[None, :]) ** 2).sum(1))
        out[r == 0] = np.inf
        return out

    d_edges = np.minimum(np.minimum(seg(x0, x1), seg(x1, x2)), seg(x2, x0))
    result = np.where(inside & ~(t0 < 0), d_inside, d_edges)
    result[det == 0] = np.inf
    return result


IMAGINARY_SYMMETRY = Patch(-2)   # curves on the symmetry plane
IMAGINARY_BOUNDARY = Patch(-3)   # curves on base mesh boundaries


# ---- Core --------------------------------------------------------------------------------------

SNAPTYPE_OPENEND = 0
SNAPTYPE_CORNER = 1
SNAPTYPE_SIDE = 2


class Core:
    def __init__(self):
        self.vertices = []
        self.halfedges = []
        self.halfchains = []
        self.edgechains = []
        self.patches = []
        self.version = 0

    def invalidate_displist(self):
        self.version += 1

    # ---- element creation ----
    def new_vertex(self):
        v = Vertex(self.vertices[-1].id + 1 if self.vertices else 0)
        self.vertices.append(v)
        return v

    def new_halfedge(self):
        h = Halfedge(self.halfedges[-1].id + 1 if self.halfedges else 0)
        self.halfedges.append(h)
        return h

    def new_halfchain(self):
        c = Halfchain(self.halfchains[-1].id + 1 if self.halfchains else 0)
        self.halfchains.append(c)
        return c

    def new_edgechain(self):
        e = Edgechain(self.edgechains[-1].id + 1 if self.edgechains else 0)
        self.edgechains.append(e)
        return e

    def new_patch(self):
        p = Patch(self.patches[-1].id + 1 if self.patches else 0)
        self.patches.append(p)
        return p

    def garbage_collect(self):
        self.vertices = [x for x in self.vertices if not x.is_deleted]
        self.halfedges = [x for x in self.halfedges if not x.is_deleted]
        self.halfchains = [x for x in self.halfchains if not x.is_deleted]
        self.edgechains = [x for x in self.edgechains if not x.is_deleted]
        self.patches = [x for x in self.patches if not x.is_deleted]

    def clear(self):
        self.vertices = []
        self.halfedges = []
        self.halfchains = []
        self.edgechains = []
        self.patches = []
        self.invalidate_displist()

    def empty(self):
        return not self.vertices

    def set_flag(self, elements, op, arg):
        for e in elements:
            if e.is_deleted:
                continue
            if op == 'SUBST':
                e.flag = arg
            elif op == 'AND':
                e.flag &= arg
            elif op == 'OR':
                e.flag |= arg

    def set_flag_vertices(self, op, arg):
        self.set_flag(self.vertices, op, arg)

    def set_flag_halfedges(self, op, arg):
        self.set_flag(self.halfedges, op, arg)

    def set_flag_halfchains(self, op, arg):
        self.set_flag(self.halfchains, op, arg)

    def set_flag_edgechains(self, op, arg):
        self.set_flag(self.edgechains, op, arg)

    def set_flag_patches(self, op, arg):
        self.set_flag(self.patches, op, arg)

    # ---- intersection acceleration ----
    def _intersect_candidates(self):
        """Halfedges calc_intersect may test, in list order, one per edge (the first one encountered)."""
        seen = set()
        cands = []
        for h in self.halfedges:
            if h.is_deleted or (h.flag & 2):
                continue
            if id(h.opposite) in seen:
                continue
            seen.add(id(h))
            cands.append(h)
        if cands:
            W0 = np.array([h.opposite.vertex.pn[:3] for h in cands])
            W1 = np.array([h.vertex.pn[:3] for h in cands])
        else:
            W0 = W1 = np.zeros((0, 3))
        return [cands, W0, W1, np.ones(len(cands), dtype=bool)]

    @staticmethod
    def _segment_hits(v0, v1, W0, W1, mask, alive_check=None):
        """Vectorized version of the segment/segment test in calc_intersect / erase_curve.
        Returns (index of the first hit or -1, x)."""
        if W0.shape[0] == 0:
            return -1, None
        rv = np.abs(v1 - v0).max() * 0.5
        rw = np.abs(W1 - W0).max(axis=1) * 0.5
        rvw = np.abs((v0 + v1)[None, :] - (W0 + W1)).min(axis=1) * 0.5
        ok = mask & ~(rv + rw < rvw)
        p = v1 - v0
        q = W1 - W0
        b = W0 - v0[None, :]
        pp = float(np.dot(p, p))
        qq = (q * q).sum(1)
        pq = q @ p
        ok &= ~(pp * qq == pq * pq)
        det = pp * qq - pq * pq
        with np.errstate(divide='ignore', invalid='ignore'):
            pb = b @ p
            qb = (q * b).sum(1)
            # x = inv(A^T A) A^T b with A = [p, -q]
            x0 = (qq * pb - pq * qb) / det
            x1 = (pq * pb - pp * qb) / det
        ok &= ~((x0 < 0) | (x1 < 0) | (1 < x0) | (1 < x1))
        ok &= np.isfinite(x0) & np.isfinite(x1)
        with np.errstate(invalid='ignore'):
            res = x0[:, None] * p[None, :] - x1[:, None] * q - b
            dist = np.sqrt((res * res).sum(1))
            ok &= ~((math.sqrt(pp) + np.sqrt(qq)) < dist)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return -1, None
        if alive_check is not None:
            for i in idx.tolist():
                if alive_check(i):
                    return i, (float(x0[i]), float(x1[i]))
                mask[i] = False
            return -1, None
        i = int(idx[0])
        return i, (float(x0[i]), float(x1[i]))

    # ---- curve addition ----
    def add_curve_open(self, curve, dist_snap, corner_at_openend=True):
        self.set_flag_vertices('SUBST', 0)
        self.set_flag_halfedges('SUBST', 0)
        vertices_added = []
        for i in range(curve.size()):
            v = self.new_vertex()
            v.flag |= 2
            v.pn = np.array(curve[i], dtype=float)
            vertices_added.append(v)
        cache = self._intersect_candidates()
        halfedges_forward = []
        halfedges_backward = []
        n = curve.size()
        for i in range(n - 1):
            v0 = vertices_added[i]
            v1 = vertices_added[i + 1]
            snapped = None
            if i == 0:
                snapped = (self.calc_snap(SNAPTYPE_OPENEND, v0, v1, dist_snap, corner_at_openend) or
                           self.calc_snap(SNAPTYPE_CORNER, v0, v1, dist_snap, corner_at_openend) or
                           self.calc_snap(SNAPTYPE_SIDE, v0, v1, dist_snap, corner_at_openend))
                if snapped:
                    halfedges_forward.append(snapped.opposite)
                    halfedges_backward.append(snapped)
                    continue
            if i == n - 2:
                snapped = (self.calc_snap(SNAPTYPE_OPENEND, v1, v0, dist_snap, corner_at_openend) or
                           self.calc_snap(SNAPTYPE_CORNER, v1, v0, dist_snap, corner_at_openend) or
                           self.calc_snap(SNAPTYPE_SIDE, v1, v0, dist_snap, corner_at_openend))
                if snapped:
                    halfedges_forward.append(snapped)
                    halfedges_backward.append(snapped.opposite)
                    continue
            hits = self.calc_intersect(v0, v1, cache)
            if hits:
                h_v0_u, h_u_v1 = hits
                halfedges_forward.append(h_v0_u)
                halfedges_forward.append(h_u_v1)
                halfedges_backward.append(h_v0_u.opposite)
                halfedges_backward.append(h_u_v1.opposite)
            else:
                h01 = self.new_halfedge()
                h10 = self.new_halfedge()
                h01.flag = h10.flag = 2
                set_vertex_halfedge(v0, v1, h01, h10)
                halfedges_forward.append(h01)
                halfedges_backward.append(h10)
        m = len(halfedges_forward)
        for i in range(m):
            h0 = halfedges_forward[i]
            if not alive(h0.prev) and i != 0:
                h0.prev = halfedges_forward[i - 1]
            if not alive(h0.next) and i != m - 1:
                h0.next = halfedges_forward[i + 1]
            h1 = halfedges_backward[i]
            if not alive(h1.prev) and i != m - 1:
                h1.prev = halfedges_backward[i + 1]
            if not alive(h1.next) and i != 0:
                h1.next = halfedges_backward[i - 1]
        self.generate_halfchains()
        self.garbage_collect()

    def add_curve_loop(self, curve):
        self.set_flag_vertices('SUBST', 0)
        self.set_flag_halfedges('SUBST', 0)
        vertices_added = []
        for i in range(curve.size()):
            v = self.new_vertex()
            v.flag = 2
            v.pn = np.array(curve[i], dtype=float)
            vertices_added.append(v)
        cache = self._intersect_candidates()
        halfedges_forward = []
        halfedges_backward = []
        n = curve.size()
        for i in range(n):
            v0 = vertices_added[i]
            v1 = vertices_added[i + 1 if i < n - 1 else 0]
            hits = self.calc_intersect(v0, v1, cache)
            if hits:
                h_v0_u, h_u_v1 = hits
                halfedges_forward.append(h_v0_u)
                halfedges_forward.append(h_u_v1)
                halfedges_backward.append(h_v0_u.opposite)
                halfedges_backward.append(h_u_v1.opposite)
            else:
                h01 = self.new_halfedge()
                h10 = self.new_halfedge()
                h01.flag = h10.flag = 2
                set_vertex_halfedge(v0, v1, h01, h10)
                halfedges_forward.append(h01)
                halfedges_backward.append(h10)
        m = len(halfedges_forward)
        for i in range(m):
            i_next = i + 1 if i < m - 1 else 0
            i_prev = m - 1 if i == 0 else i - 1
            h0 = halfedges_forward[i]
            if not alive(h0.prev):
                h0.prev = halfedges_forward[i_prev]
            if not alive(h0.next):
                h0.next = halfedges_forward[i_next]
            h1 = halfedges_backward[i]
            if not alive(h1.prev):
                h1.prev = halfedges_backward[i_next]
            if not alive(h1.next):
                h1.next = halfedges_backward[i_prev]
        self.generate_halfchains()
        self.garbage_collect()

    # ---- erasing ----
    def erase_curve(self, scribble):
        cache = None
        for i in range(1, scribble.size()):
            v0 = scribble[i - 1][:3]
            v1 = scribble[i][:3]
            if cache is None:
                cands = [h for h in self.halfedges
                         if not h.is_deleted and not (h.imaginary_patch or (h.opposite is not None and h.opposite.imaginary_patch))]
                if cands:
                    W0 = np.array([h.opposite.vertex.pn[:3] for h in cands])
                    W1 = np.array([h.vertex.pn[:3] for h in cands])
                else:
                    W0 = W1 = np.zeros((0, 3))
                cache = (cands, W0, W1, np.ones(len(cands), dtype=bool))
            cands, W0, W1, mask = cache
            idx, _ = self._segment_hits(v0, v1, W0, W1, mask)
            if idx < 0:
                continue
            h = cands[idx]
            if h.halfchain is None:
                break   # this should not happen!
            self.erase_curve_sub(h)
            cache = None

    def erase_curve_sub(self, h):
        if h.is_deleted:
            return
        h_front = h.halfchain.halfedge_front
        h_back = h.halfchain.halfedge_back
        h_endpoint = (h_front.opposite, h_back)
        valence = [0, 0]
        prev_next_pair = [None, None]
        for i in range(2):
            valence[i] = h_endpoint[i].vertex.valence()
            prev_next_pair[i] = (h_endpoint[i].opposite.prev, h_endpoint[i].next)
        h1 = h_front
        while True:
            delete_halfedge(h1)
            delete_halfedge(h1.opposite)
            if h1 is h_back:
                break
            h1 = h1.next
            if h1 is None:
                break
        for i in range(2):
            h_prev, h_next = prev_next_pair[i]
            if valence[i] == 2:
                if h_prev is not None:
                    h_prev.next = None
                    h_prev.is_corner = False
                if h_next is not None:
                    h_next.prev = None
            elif valence[i] > 2:
                set_next_prev(h_prev, h_next)
                if h_prev is not None:
                    num_corners = sum(1 for vih in vih_iter(h_prev.vertex) if vih.is_corner)
                    h_prev.is_corner = valence[i] == 3 or num_corners < 2
        self.garbage_collect()

    # ---- corners ----
    def flip_corner_valance2(self, v):
        if v.valence() != 2:
            return
        is_corner = v.halfedge.opposite.is_corner
        v.halfedge.opposite.is_corner = not is_corner
        v.halfedge.prev.is_corner = not is_corner
        delete_halfchain(v.halfedge.halfchain)
        delete_halfchain(v.halfedge.opposite.halfchain)
        delete_halfchain(v.halfedge.opposite.next.halfchain)
        delete_halfchain(v.halfedge.prev.halfchain)
        self.generate_halfchains()
        self.garbage_collect()

    def flip_corner_valanceN(self, v, reference_point):
        if v.valence() < 3:
            return
        d_ref = normalized(reference_point - v.pn[:3])
        h_max = None
        dot_max = -float('inf')
        for vih in vih_iter(v):
            d0 = -normalized(vih.to_vector())
            d1 = normalized(vih.next.to_vector())
            d = dot3(d0, d_ref) + dot3(d1, d_ref)
            if dot_max < d:
                dot_max = d
                h_max = vih
        h_max.is_corner = not h_max.is_corner
        delete_halfchain(h_max.halfchain)
        delete_halfchain(h_max.next.halfchain)
        self.generate_halfchains()
        self.garbage_collect()

    def generate_halfchains(self):
        for h in list(self.halfedges):
            if h.is_deleted:
                continue
            if alive(h.halfchain):
                continue
            c = self.new_halfchain()
            c.halfedge_back = h
            while True:
                c.halfedge_back.halfchain = c
                v = c.halfedge_back.vertex
                if v.is_corner() or v.is_openend():
                    break
                c.halfedge_back = c.halfedge_back.next
                if c.halfedge_back is h:
                    break   # loop with no corner
            if c.halfedge_back is h and not h.vertex.is_corner() and not h.vertex.is_openend():
                c.halfedge_front = h
                c.halfedge_back = h.prev
            else:
                c.halfedge_front = h
                while True:
                    c.halfedge_front.halfchain = c
                    v = c.halfedge_front.opposite.vertex
                    if v.is_corner() or v.is_openend():
                        break
                    c.halfedge_front = c.halfedge_front.prev
            opp = c.opposite()
            if alive(opp):
                e = opp.edgechain
                e.halfchain[1 if e.halfchain[0] is opp else 0] = c
                c.edgechain = e
            else:
                e = self.new_edgechain()
                e.halfchain[0] = c
                c.edgechain = e

    def calc_snap(self, snap_type, v_snap, v_other, dist_snap, corner_at_openend):
        """Returns h_snap (halfedge pointing toward the snapped vertex) or None."""
        cand = [v for v in self.vertices if not v.is_deleted and not (v.flag & 2)]
        if not cand:
            return None
        P = np.array([v.pn[:3] for v in cand])
        d = np.sqrt(((P - v_snap.pn[:3][None, :]) ** 2).sum(1))
        order = np.argsort(d, kind='stable')
        v_min = None
        for k in order:
            if d[k] > dist_snap:
                break
            v = cand[k]
            is_openend = v.is_openend()
            is_corner = v.is_corner()
            if snap_type == SNAPTYPE_OPENEND:
                if not is_openend:
                    continue
            elif snap_type == SNAPTYPE_CORNER:
                if is_openend or not is_corner:
                    continue
            else:
                if is_openend or is_corner:
                    continue
            v_min = v
            break
        if v_min is None:
            return None

        if snap_type in (SNAPTYPE_CORNER, SNAPTYPE_SIDE):
            n = v_min.pn[3:]
            d0 = None
            star_angles = []
            for h in voh_iter(v_min):
                dv = h.vertex.pn[:3] - v_min.pn[:3]
                dv = dv - dot3(n, dv) * n
                dv = normalized(dv)
                if h is v_min.halfedge:
                    d0 = dv
                    continue
                ang = math.atan2(dot3(cross3(d0, dv), n), dot3(d0, dv))
                if ang < 0:
                    ang += 2 * math.pi
                star_angles.append(ang)
            d1 = v_other.pn[:3] - v_min.pn[:3]
            d1 = normalized(d1 - dot3(n, d1) * n)
            angle = math.atan2(dot3(cross3(d0, d1), n), dot3(d0, d1))
            if angle < 0:
                angle += 2 * math.pi
            h_snap_next = None
            hs = list(voh_iter(v_min))
            for i in range(len(star_angles)):
                if star_angles[i] < angle:
                    h_snap_next = hs[(i + 1) % len(hs)] if i + 1 < len(hs) else None
                    break
            if h_snap_next is None:
                h_snap_next = v_min.halfedge
            h_snap_prev = h_snap_next.prev
        else:
            h_snap_next = v_min.halfedge
            h_snap_prev = h_snap_next.opposite

        if snap_type == SNAPTYPE_CORNER:
            if alive(h_snap_next.halfchain):
                delete_patch(h_snap_next.halfchain.patch)
            if h_snap_prev is not None and alive(h_snap_prev.halfchain):
                delete_patch(h_snap_prev.halfchain.patch)
        elif snap_type == SNAPTYPE_SIDE:
            delete_halfchain(h_snap_next.halfchain)
            delete_halfchain(h_snap_next.opposite.halfchain)

        h_snap = self.new_halfedge()
        h_snap_opposite = self.new_halfedge()
        h_snap.flag |= 2
        h_snap_opposite.flag |= 2
        set_vertex_halfedge(v_other, v_min, h_snap, h_snap_opposite)
        set_next_prev(h_snap, h_snap_next)
        set_next_prev(h_snap_prev, h_snap_opposite)
        is_corner = snap_type != SNAPTYPE_OPENEND or corner_at_openend
        h_snap.is_corner = is_corner
        if h_snap_prev is not None:
            h_snap_prev.is_corner = is_corner
        if not is_corner:
            delete_halfchain(h_snap_next.halfchain)
            delete_halfchain(h_snap_next.opposite.halfchain)
        v_snap.is_deleted = True
        return h_snap

    def calc_intersect(self, v0, v1, cache=None):
        """Returns (h_v0_u, h_u_v1) if the segment v0-v1 crosses an existing curve, else None."""
        if cache is None:
            cache = self._intersect_candidates()
        cands, W0, W1, mask = cache
        # candidates deleted since the cache was built are skipped
        idx, x = self._segment_hits(v0.pn[:3], v1.pn[:3], W0, W1, mask, lambda k: not cands[k].is_deleted)
        if idx < 0:
            return None
        h = cands[idx]
        mask[idx] = False
        w0 = h.opposite.vertex
        w1 = h.vertex
        u = self.new_vertex()
        delete_halfchain(h.halfchain)
        delete_halfchain(h.opposite.halfchain)
        u.flag = 2
        u.pn = 0.5 * ((1 - x[0]) * v0.pn + x[0] * v1.pn + (1 - x[1]) * w0.pn + x[1] * w1.pn)
        pn_normalize(u.pn)
        n = u.pn[3:]
        p = v1.pn[:3] - v0.pn[:3]
        q = w1.pn[:3] - w0.pn[:3]
        if dot3(cross3(p, q), n) < 0:
            w0, w1 = w1, w0
            h = h.opposite
        h_v0_u = self.new_halfedge()
        h_u_v0 = self.new_halfedge()
        h_v1_u = self.new_halfedge()
        h_u_v1 = self.new_halfedge()
        h_w0_u = self.new_halfedge()
        h_u_w0 = self.new_halfedge()
        h_w1_u = self.new_halfedge()
        h_u_w1 = self.new_halfedge()
        for hh in (h_v0_u, h_u_v0, h_v1_u, h_u_v1, h_w0_u, h_u_w0, h_w1_u, h_u_w1):
            hh.flag = 2
        set_vertex_halfedge(u, v0, h_u_v0, h_v0_u)
        set_vertex_halfedge(u, v1, h_u_v1, h_v1_u)
        set_vertex_halfedge(u, w0, h_u_w0, h_w0_u)
        set_vertex_halfedge(u, w1, h_u_w1, h_w1_u)
        set_next_prev(h_v0_u, h_u_w1)
        set_next_prev(h_w1_u, h_u_v1)
        set_next_prev(h_v1_u, h_u_w0)
        set_next_prev(h_w0_u, h_u_v0)
        if alive(h.next):
            set_next_prev(h_u_w1, h.next)
            set_next_prev(h.opposite.prev, h_w1_u)
        if alive(h.prev):
            set_next_prev(h.prev, h_w0_u)
            set_next_prev(h_u_w0, h.opposite.next)
        h_v0_u.is_corner = True
        h_v1_u.is_corner = True
        h_w0_u.is_corner = True
        h_w1_u.is_corner = True
        delete_halfedge(h)
        delete_halfedge(h.opposite)
        return h_v0_u, h_u_v1

    # ---- snapshots (memento / file IO) ----
    def snapshot(self):
        def rid(x):
            return x.id if x is not None else -1

        verts = [(v.id, rid(v.halfedge), v.pn.copy(), v.is_hidden, v.flag) for v in self.vertices]
        hes = [(h.id, rid(h.vertex), rid(h.next), rid(h.prev), rid(h.opposite), rid(h.halfchain),
                h.is_corner, h.imaginary_patch.id if h.imaginary_patch is not None else -1, h.flag)
               for h in self.halfedges]
        chains = [(c.id, rid(c.halfedge_front), rid(c.halfedge_back), rid(c.patch), rid(c.edgechain), c.flag)
                  for c in self.halfchains]
        echains = [(e.id, rid(e.halfchain[0]), rid(e.halfchain[1]), e.num_subdiv, e.flag) for e in self.edgechains]
        patches = []
        for p in self.patches:
            patches.append((p.id, rid(p.halfchain), p.is_failure, p.is_hidden, p.flag, p.param.copy(),
                            (list(p.v_he), list(p.he_to), list(p.he_next), list(p.he_prev), list(p.he_face),
                             list(p.f_he), p.pn.copy(), list(p.v_corner), list(p.v_tag))))
        return (verts, hes, chains, echains, patches)

    def restore(self, snap):
        verts, hes, chains, echains, patches = snap
        self.vertices = []
        self.halfedges = []
        self.halfchains = []
        self.edgechains = []
        self.patches = []
        vmap, hmap, cmap, emap, pmap = {}, {}, {}, {}, {}
        for (id_, _, pn, hidden, flag) in verts:
            v = Vertex(id_)
            v.pn = pn.copy()
            v.is_hidden = hidden
            v.flag = flag
            self.vertices.append(v)
            vmap[id_] = v
        for t in hes:
            h = Halfedge(t[0])
            h.is_corner = t[6]
            h.flag = t[8]
            if t[7] == -2:
                h.imaginary_patch = IMAGINARY_SYMMETRY
            elif t[7] == -3:
                h.imaginary_patch = IMAGINARY_BOUNDARY
            self.halfedges.append(h)
            hmap[t[0]] = h
        for t in chains:
            c = Halfchain(t[0])
            c.flag = t[5]
            self.halfchains.append(c)
            cmap[t[0]] = c
        for t in echains:
            e = Edgechain(t[0])
            e.num_subdiv = t[3]
            e.flag = t[4]
            self.edgechains.append(e)
            emap[t[0]] = e
        for t in patches:
            p = Patch(t[0])
            p.is_failure = t[2]
            p.is_hidden = t[3]
            p.flag = t[4]
            p.param = t[5].copy()
            (p.v_he, p.he_to, p.he_next, p.he_prev, p.he_face, p.f_he, pn, p.v_corner, p.v_tag) = (
                list(t[6][0]), list(t[6][1]), list(t[6][2]), list(t[6][3]), list(t[6][4]), list(t[6][5]),
                t[6][6], list(t[6][7]), list(t[6][8]))
            p._pn = np.array(pn, dtype=float).reshape(-1, 6).copy()
            if len(p._pn) == 0:
                p._pn = np.zeros((16, 6))
            p._nv = len(p.v_he)
            p.h_chain = [None] * len(p.he_to)
            p.h_index = [-1] * len(p.he_to)
            self.patches.append(p)
            pmap[t[0]] = p

        def patch_ref(pid):
            if pid == -2:
                return IMAGINARY_SYMMETRY
            if pid == -3:
                return IMAGINARY_BOUNDARY
            return pmap.get(pid)

        for (id_, hid, _, _, _) in verts:
            vmap[id_].halfedge = hmap.get(hid)
        for t in hes:
            h = hmap[t[0]]
            h.vertex = vmap.get(t[1])
            h.next = hmap.get(t[2])
            h.prev = hmap.get(t[3])
            h.opposite = hmap.get(t[4])
            h.halfchain = cmap.get(t[5])
        for t in chains:
            c = cmap[t[0]]
            c.halfedge_front = hmap.get(t[1])
            c.halfedge_back = hmap.get(t[2])
            c.patch = patch_ref(t[3])
            c.edgechain = emap.get(t[4])
        for t in echains:
            e = emap[t[0]]
            e.halfchain = [cmap.get(t[1]), cmap.get(t[2])]
        for t in patches:
            p = pmap[t[0]]
            p.halfchain = cmap.get(t[1])
            p.set_halfedgeData()
            p.invalidate_displist()
        self.invalidate_displist()
