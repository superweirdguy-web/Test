# A small polygon mesh with the exact connectivity semantics of OpenMesh's
# ArrayKernel / PolyConnectivity (as bundled with SketchRetopo), plus the kt84
# helpers built on top of it (insert_edgeloop, find_edgeloop, append_quad_strip,
# flip_faces).  Handle orders matter to SketchRetopo (e.g. which outgoing
# halfedge a boundary vertex stores), so the code follows OpenMesh line by line.
#
# Handles are plain ints, -1 is the invalid handle.  Edge e owns halfedges 2e and 2e+1.

import numpy as np

INVALID = -1


class PolyMesh:
    __slots__ = ('v_he', 'he_to', 'he_next', 'he_prev', 'he_face', 'f_he',
                 '_pn', '_nv', 'v_corner', 'v_tag', 'h_chain', 'h_index')

    def __init__(self):
        self.clear_mesh()

    # ---- storage --------------------------------------------------------------------------
    def clear_mesh(self):
        self.v_he = []
        self.he_to = []
        self.he_next = []
        self.he_prev = []
        self.he_face = []
        self.f_he = []
        self.v_corner = []       # patchgen corner_index per vertex
        self.v_tag = []          # patchgen PatchVertexTag per vertex (int, -1 = None)
        self.h_chain = []        # curvenetwork Halfchain per halfedge
        self.h_index = []        # index_wrt_halfchain per halfedge
        self._pn = np.zeros((16, 6))
        self._nv = 0

    @property
    def pn(self):
        """(n_vertices, 6) view of the per-vertex PointNormal data."""
        return self._pn[:self._nv]

    def n_vertices(self):
        return self._nv

    def n_halfedges(self):
        return len(self.he_to)

    def n_edges(self):
        return len(self.he_to) >> 1

    def n_faces(self):
        return len(self.f_he)

    def vertices(self):
        return range(self._nv)

    def halfedges(self):
        return range(len(self.he_to))

    def edges(self):
        return range(len(self.he_to) >> 1)

    def faces(self):
        return range(len(self.f_he))

    def add_vertex(self, pn=None):
        if self._nv == len(self._pn):
            grown = np.zeros((max(16, 2 * len(self._pn)), 6))
            grown[:self._nv] = self._pn[:self._nv]
            self._pn = grown
        v = self._nv
        self._nv += 1
        if pn is None:
            self._pn[v] = 0.0
        else:
            self._pn[v] = pn
        self.v_he.append(INVALID)
        self.v_corner.append(-1)
        self.v_tag.append(-1)
        return v

    def new_edge(self, start_vh, end_vh):
        h0 = len(self.he_to)
        self.he_to.extend((end_vh, start_vh))
        self.he_next.extend((INVALID, INVALID))
        self.he_prev.extend((INVALID, INVALID))
        self.he_face.extend((INVALID, INVALID))
        self.h_chain.extend((None, None))
        self.h_index.extend((-1, -1))
        return h0

    def new_face(self):
        self.f_he.append(INVALID)
        return len(self.f_he) - 1

    # ---- elementary connectivity ----------------------------------------------------------
    @staticmethod
    def opposite_halfedge_handle(h):
        return h ^ 1

    @staticmethod
    def edge_handle(h):
        return h >> 1

    @staticmethod
    def halfedge_handle_e(e, i):
        return 2 * e + i

    def halfedge_handle(self, v):
        return self.v_he[v]

    def set_halfedge_handle(self, v, h):
        self.v_he[v] = h

    def to_vertex_handle(self, h):
        return self.he_to[h]

    def from_vertex_handle(self, h):
        return self.he_to[h ^ 1]

    def set_vertex_handle(self, h, v):
        self.he_to[h] = v

    def next_halfedge_handle(self, h):
        return self.he_next[h]

    def prev_halfedge_handle(self, h):
        return self.he_prev[h]

    def set_next_halfedge_handle(self, h, n):
        self.he_next[h] = n
        self.he_prev[n] = h

    def face_handle(self, h):
        return self.he_face[h]

    def set_face_handle(self, h, f):
        self.he_face[h] = f

    def is_boundary_h(self, h):
        return self.he_face[h] == INVALID

    def is_boundary_e(self, e):
        return self.he_face[2 * e] == INVALID or self.he_face[2 * e + 1] == INVALID

    def is_boundary_v(self, v):
        h = self.v_he[v]
        return h == INVALID or self.he_face[h] == INVALID

    def cw_rotated_halfedge_handle(self, h):
        return self.he_next[h ^ 1]

    def ccw_rotated_halfedge_handle(self, h):
        return self.he_prev[h] ^ 1

    # ---- circulators ----------------------------------------------------------------------
    def voh(self, v):
        start = self.v_he[v]
        if start == INVALID:
            return []
        result = []
        h = start
        while True:
            result.append(h)
            h = self.he_next[h ^ 1]
            if h == start or h == INVALID:
                break
        return result

    def vv(self, v):
        return [self.he_to[h] for h in self.voh(v)]

    def vf(self, v):
        return [self.he_face[h] for h in self.voh(v) if self.he_face[h] != INVALID]

    def fh(self, f):
        start = self.f_he[f]
        result = []
        h = start
        while True:
            result.append(h)
            h = self.he_next[h]
            if h == start:
                break
        return result

    def fv(self, f):
        return [self.he_to[h] for h in self.fh(f)]

    def valence(self, v):
        return len(self.voh(v))

    def face_valence(self, f):
        return len(self.fh(f))

    def find_halfedge(self, v0, v1):
        for h in self.voh(v0):
            if self.he_to[h] == v1:
                return h
        return INVALID

    def adjust_outgoing_halfedge(self, v):
        for h in self.voh(v):
            if self.he_face[h] == INVALID:
                self.v_he[v] = h
                break

    def util_halfedge_to_vertex_pair(self, h):
        return self.he_to[h ^ 1], self.he_to[h]

    def util_edge_to_vertex_pair(self, e):
        return self.util_halfedge_to_vertex_pair(2 * e)

    # ---- PolyConnectivity::add_face ---------------------------------------------------------
    def add_face(self, vhs):
        n = len(vhs)
        assert n > 2
        edge_h = [INVALID] * n
        is_new = [False] * n
        needs_adjust = [False] * n
        next_cache = []

        for i in range(n):
            ii = (i + 1) % n
            if not self.is_boundary_v(vhs[i]):
                return INVALID   # complex vertex
            edge_h[i] = self.find_halfedge(vhs[i], vhs[ii])
            is_new[i] = edge_h[i] == INVALID
            if not is_new[i] and not self.is_boundary_h(edge_h[i]):
                return INVALID   # complex edge

        # re-link patches if necessary
        for i in range(n):
            ii = (i + 1) % n
            if not is_new[i] and not is_new[ii]:
                inner_prev = edge_h[i]
                inner_next = edge_h[ii]
                if self.he_next[inner_prev] != inner_next:
                    outer_prev = inner_next ^ 1
                    boundary_prev = outer_prev
                    while True:
                        boundary_prev = self.he_next[boundary_prev] ^ 1
                        if self.is_boundary_h(boundary_prev) and boundary_prev != inner_prev:
                            break
                    boundary_next = self.he_next[boundary_prev]
                    if boundary_next == inner_next:
                        return INVALID   # patch re-linking failed
                    patch_start = self.he_next[inner_prev]
                    patch_end = self.he_prev[inner_next]
                    next_cache.append((boundary_prev, patch_start))
                    next_cache.append((patch_end, boundary_next))
                    next_cache.append((inner_prev, inner_next))

        # create missing edges
        for i in range(n):
            if is_new[i]:
                edge_h[i] = self.new_edge(vhs[i], vhs[(i + 1) % n])

        # create the face
        fh = self.new_face()
        self.f_he[fh] = edge_h[n - 1]

        # setup halfedges
        for i in range(n):
            ii = (i + 1) % n
            vh = vhs[ii]
            inner_prev = edge_h[i]
            inner_next = edge_h[ii]
            id_ = (1 if is_new[i] else 0) | (2 if is_new[ii] else 0)
            if id_:
                outer_prev = inner_next ^ 1
                outer_next = inner_prev ^ 1
                if id_ == 1:      # prev is new, next is old
                    boundary_prev = self.he_prev[inner_next]
                    next_cache.append((boundary_prev, outer_next))
                    self.v_he[vh] = outer_next
                elif id_ == 2:    # next is new, prev is old
                    boundary_next = self.he_next[inner_prev]
                    next_cache.append((outer_prev, boundary_next))
                    self.v_he[vh] = boundary_next
                else:             # both are new
                    if self.v_he[vh] == INVALID:
                        self.v_he[vh] = outer_next
                        next_cache.append((outer_prev, outer_next))
                    else:
                        boundary_next = self.v_he[vh]
                        boundary_prev = self.he_prev[boundary_next]
                        next_cache.append((boundary_prev, outer_next))
                        next_cache.append((outer_prev, boundary_next))
                next_cache.append((inner_prev, inner_next))
            else:
                needs_adjust[ii] = self.v_he[vh] == inner_next
            self.he_face[edge_h[i]] = fh

        for a, b in next_cache:
            self.set_next_halfedge_handle(a, b)

        for i in range(n):
            if needs_adjust[i]:
                self.adjust_outgoing_halfedge(vhs[i])
        return fh

    # ---- PolyConnectivity::split_edge -------------------------------------------------------
    def split_edge(self, e, vh):
        h0 = 2 * e
        h1 = 2 * e + 1
        vfrom = self.from_vertex_handle(h0)
        ph0 = self.he_prev[h0]
        nh1 = self.he_next[h1]
        boundary0 = self.is_boundary_h(h0)
        boundary1 = self.is_boundary_h(h1)

        new_e = self.new_edge(self.from_vertex_handle(h0), vh)
        self.he_to[h1] = vh

        self.set_next_halfedge_handle(new_e, h0)
        self.set_next_halfedge_handle(h1, new_e ^ 1)
        self.set_next_halfedge_handle(ph0, new_e)
        self.set_next_halfedge_handle(new_e ^ 1, nh1)

        self.he_face[new_e] = INVALID if boundary0 else self.he_face[h0]
        self.he_face[new_e ^ 1] = INVALID if boundary1 else self.he_face[h1]

        self.v_he[vh] = h0
        self.adjust_outgoing_halfedge(vh)
        if self.v_he[vfrom] == h0:
            self.v_he[vfrom] = new_e
            self.adjust_outgoing_halfedge(vfrom)

    # ---- PolyConnectivity::insert_edge ------------------------------------------------------
    def insert_edge(self, prev_heh, next_heh):
        vh0 = self.to_vertex_handle(prev_heh)
        vh1 = self.from_vertex_handle(next_heh)
        heh0 = self.new_edge(vh0, vh1)
        heh1 = heh0 ^ 1
        next_prev_heh = self.he_next[prev_heh]
        prev_next_heh = self.he_prev[next_heh]
        self.set_next_halfedge_handle(prev_heh, heh0)
        self.set_next_halfedge_handle(heh0, next_heh)
        self.set_next_halfedge_handle(prev_next_heh, heh1)
        self.set_next_halfedge_handle(heh1, next_prev_heh)

        new_fh = self.new_face()
        self.f_he[new_fh] = heh0
        for h in self.fh(new_fh):
            self.he_face[h] = new_fh
        old_fh = self.he_face[next_prev_heh]
        self.he_face[heh1] = old_fh
        if old_fh != INVALID and self.he_face[self.f_he[old_fh]] == new_fh:
            self.f_he[old_fh] = heh1
        self.adjust_outgoing_halfedge(vh0)
        self.adjust_outgoing_halfedge(vh1)
        return heh0


# ---- kt84/openmesh/edgeloop.hh ----------------------------------------------------------------

def insert_edgeloop(mesh, h_start, t=0.5, func=None):
    # check if this edge loop forms a loop by walking toward h1 of e
    h = h_start ^ 1
    while True:
        h_opposite = h ^ 1
        h_next_next = mesh.he_next[mesh.he_next[h]]
        if mesh.is_boundary_h(h):
            h_start = h_opposite
            break
        elif h_start == h_next_next:
            break
        h = h_next_next ^ 1

    # for each edge, insert a vertex at its middle
    h = h_start
    while True:
        h_next = mesh.he_next[mesh.he_next[h]] ^ 1
        v0 = mesh.from_vertex_handle(h)
        v1 = mesh.to_vertex_handle(h)
        v_new = mesh.add_vertex((1 - t) * mesh._pn[v0] + t * mesh._pn[v1])
        if func is not None:
            func(mesh, h, v_new)
        mesh.split_edge(h >> 1, v_new)
        if mesh.is_boundary_h(h):
            break
        if h_next == h_start:
            break
        h = h_next

    # connect inserted vertices
    h = h_start
    while True:
        h_prev = mesh.he_next[mesh.he_next[h]]
        h_next = h_prev ^ 1
        mesh.insert_edge(h_prev, h)
        h = h_next
        if mesh.is_boundary_h(h) or h == h_start:
            break


def find_edgeloop(mesh, h_start):
    h = h_start ^ 1
    while True:
        h_opposite = h ^ 1
        h_next_next = mesh.he_next[mesh.he_next[h]]
        if mesh.is_boundary_h(h):
            h_start = h_opposite
            break
        elif h_start == h_next_next:
            break
        h = h_next_next ^ 1

    result = []
    h = h_start
    while True:
        result.append(h)
        if mesh.is_boundary_h(h):
            break
        h_next = mesh.he_next[mesh.he_next[h]] ^ 1
        if h_next == h_start:
            result.append(h_start)   # indication of loop
            break
        h = h_next
    return result


# ---- kt84/openmesh/append_quad_strip.hh ---------------------------------------------------------

def append_quad_strip(mesh, boundary_h_front, boundary_h_back):
    """Returns the updated (boundary_h_front, boundary_h_back), or None on failure."""
    if not mesh.is_boundary_h(boundary_h_front) or not mesh.is_boundary_h(boundary_h_back):
        return None
    h = boundary_h_front
    v_prev = INVALID
    while True:
        h_next = mesh.he_next[h]
        v_from = mesh.from_vertex_handle(h)
        v_to = mesh.to_vertex_handle(h)
        if h == boundary_h_front:
            v_prev = mesh.add_vertex()
        v_next = mesh.add_vertex()
        mesh.add_face([v_from, v_to, v_next, v_prev])
        if h == boundary_h_back:
            break
        v_prev = v_next
        h = h_next
    boundary_h_front = mesh.he_next[mesh.he_next[boundary_h_front]] ^ 1
    boundary_h_back = mesh.he_next[mesh.he_next[boundary_h_back]] ^ 1
    return boundary_h_front, boundary_h_back


# ---- kt84/openmesh/flip_faces.hh ----------------------------------------------------------------

def flip_faces(mesh):
    nv = mesh.n_vertices()
    pn = mesh.pn.copy()
    corner = list(mesh.v_corner)
    tag = list(mesh.v_tag)
    faces = [mesh.fv(f) for f in mesh.faces()]
    mesh.clear_mesh()
    for v in range(nv):
        mesh.add_vertex(pn[v])
        mesh.v_corner[v] = corner[v]
        mesh.v_tag[v] = tag[v]
    for fvs in faces:
        mesh.add_face(list(reversed(fvs)))
