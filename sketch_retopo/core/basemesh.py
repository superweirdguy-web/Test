# Port of SketchRetopo/BaseMesh plus the ray-casting part of SketchRetopo_intersect.cc.
#
# The base mesh is a triangle mesh stored in numpy arrays with an OpenMesh-like
# halfedge structure.  Ray casting uses Blender's mathutils BVHTree in place of
# Intel Embree.  All coordinates are in the base object's local space.

import heapq
import math

import numpy as np

try:
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError:      # pragma: no cover - only outside Blender
    Vector = None
    BVHTree = None


class Hit:
    __slots__ = ('face', 'u', 'v', 't')

    def __init__(self, face, u, v, t):
        self.face = face
        self.u = u
        self.v = v
        self.t = t

    def bc(self, index):
        return (1 - self.u - self.v, self.u, self.v)[index]


SNAKE_MEAN_CURVATURE = 0
SNAKE_NORMAL_VARIATION = 1
SNAKE_NORMAL_CURVATURE = 2


class BaseMesh:
    def __init__(self, verts, tris, normals=None, texcoords=None):
        self.verts = np.ascontiguousarray(verts, dtype=float).reshape(-1, 3)
        self.tris = np.ascontiguousarray(tris, dtype=np.int64).reshape(-1, 3)
        self.texcoords = None if texcoords is None else np.asarray(texcoords, dtype=float).reshape(-1, 3, 2)
        self.has_texture = self.texcoords is not None
        nv = len(self.verts)
        nf = len(self.tris)
        self.face_hidden = np.zeros(nf, dtype=bool)

        # face normals / vertex normals
        p0 = self.verts[self.tris[:, 0]]
        p1 = self.verts[self.tris[:, 1]]
        p2 = self.verts[self.tris[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        self.face_area = 0.5 * np.linalg.norm(fn, axis=1)
        fl = np.linalg.norm(fn, axis=1)
        fl[fl == 0] = 1
        self.face_normals = fn / fl[:, None]
        if normals is None:
            vn = np.zeros((nv, 3))
            for k in range(3):
                np.add.at(vn, self.tris[:, k], fn)
        else:
            vn = np.asarray(normals, dtype=float).reshape(-1, 3).copy()
        ln = np.linalg.norm(vn, axis=1)
        ln[ln == 0] = 1
        self.normals = vn / ln[:, None]

        bb_min = self.verts.min(axis=0) if nv else np.zeros(3)
        bb_max = self.verts.max(axis=0) if nv else np.zeros(3)
        self.bbox_min = bb_min
        self.bbox_max = bb_max
        self._bbox_diag = float(np.linalg.norm(bb_max - bb_min))
        self.center_of_mass = self.verts.mean(axis=0) if nv else np.zeros(3)

        self._build_halfedges()
        self._build_adjacency()
        self._build_bvh()

        # exponential map state
        self.expmap_uv = np.zeros((nv, 2))
        self.expmap_paramed = np.zeros(nv, dtype=bool)
        self.expmap_paramed_faces = np.zeros(0, dtype=np.int64)
        self._expmap_pre = None

        # snakes
        self.normal_smooth_iter = 0
        self.feature_smooth_iter = 0
        self.use_feature_as_energy = True
        self.negate_energy = True
        self.snake_feature_type = SNAKE_MEAN_CURVATURE
        self.snake_energy = np.zeros(nv)
        self.snake_energy_gradient = np.zeros((nv, 3))
        self.snake_energy_min = 0.0
        self.snake_energy_max = 0.0
        self.compute_snake_energy()

        self._cotan = None
        self.harmonic_uv = np.zeros((nv, 2))

    # ---- connectivity ---------------------------------------------------------------------
    def _build_halfedges(self):
        nf = len(self.tris)
        tris = self.tris
        h_from = tris.reshape(-1)
        h_to = np.roll(tris, -1, axis=1).reshape(-1)
        nfh = 3 * nf
        opp = np.full(nfh, -1, dtype=np.int64)
        if nfh:
            key_dir = h_from * (len(self.verts) + 1) + h_to
            key_rev = h_to * (len(self.verts) + 1) + h_from
            order = np.argsort(key_dir, kind='stable')
            sorted_keys = key_dir[order]
            pos = np.searchsorted(sorted_keys, key_rev)
            pos = np.minimum(pos, nfh - 1)
            found = sorted_keys[pos] == key_rev
            cand = np.where(found, order[pos], -1)
            # keep only mutual pairings (non-manifold edges keep the first pairing only)
            idx = np.arange(nfh)
            ok = cand >= 0
            mutual = np.zeros(nfh, dtype=bool)
            mutual[ok] = cand[cand[ok]] == idx[ok] if ok.any() else False
            # cand[cand] may not point back for duplicates; fall back to a greedy pairing
            opp[mutual] = cand[mutual]
            rest = np.flatnonzero(ok & ~mutual)
            if rest.size:
                taken = set(np.flatnonzero(opp >= 0).tolist())
                lookup = {}
                for h in np.flatnonzero(opp < 0).tolist():
                    lookup.setdefault((int(h_from[h]), int(h_to[h])), []).append(h)
                for h in rest.tolist():
                    if h in taken:
                        continue
                    lst = lookup.get((int(h_to[h]), int(h_from[h])), [])
                    while lst and (lst[0] in taken or lst[0] == h):
                        lst.pop(0)
                    if lst:
                        g = lst.pop(0)
                        opp[h] = g
                        opp[g] = h
                        taken.add(h)
                        taken.add(g)
        he_to = list(h_to.tolist())
        he_next = [3 * (h // 3) + (h % 3 + 1) % 3 for h in range(nfh)]
        he_face = [h // 3 for h in range(nfh)]
        opp_l = opp.tolist()
        # boundary halfedges
        unpaired = [h for h in range(nfh) if opp_l[h] < 0]
        h_from_l = h_from.tolist()
        for h in unpaired:
            g = len(he_to)
            he_to.append(h_from_l[h])
            he_next.append(-1)
            he_face.append(-1)
            opp_l.append(h)
            opp_l[h] = g
        self.he_to = np.array(he_to, dtype=np.int64)
        self.he_opp = np.array(opp_l, dtype=np.int64)
        he_prev = [-1] * len(he_to)
        for h in range(nfh):
            he_prev[he_next[h]] = h
        # next pointers of boundary halfedges: g = b->a ; find the boundary halfedge leaving a
        for g in range(nfh, len(he_to)):
            h = opp_l[g]               # face halfedge a->b
            x = h
            steps = 0
            while True:
                y = he_prev[x]         # c->a
                oy = opp_l[y]
                if he_face[oy] < 0:
                    he_next[g] = oy
                    he_prev[oy] = g
                    break
                x = oy
                steps += 1
                if steps > 10000:
                    break
        self.he_next = np.array(he_next, dtype=np.int64)
        self.he_prev = np.array(he_prev, dtype=np.int64)
        self.he_face = np.array(he_face, dtype=np.int64)
        nh = len(he_to)
        self.he_from = self.he_to[self.he_opp]
        # edge index of each halfedge
        self.he_edge = np.minimum(np.arange(nh), self.he_opp)
        uniq, inv = np.unique(self.he_edge, return_inverse=True)
        self.he_edge = inv
        self.n_edges = len(uniq)
        # outgoing halfedge per vertex, preferring boundary ones (OpenMesh adjust_outgoing_halfedge)
        nv = len(self.verts)
        v_he = np.full(nv, -1, dtype=np.int64)
        v_he[self.he_from[::-1]] = np.arange(nh)[::-1]
        bnd = np.flatnonzero(self.he_face < 0)
        if bnd.size:
            v_he[self.he_from[bnd[::-1]]] = bnd[::-1]
        self.v_he = v_he
        self.n_halfedges = nh

    def _build_adjacency(self):
        nv = len(self.verts)
        e0 = self.he_from
        e1 = self.he_to
        order = np.argsort(e0, kind='stable')
        self.adj_start = np.searchsorted(e0[order], np.arange(nv + 1))
        self.adj_nbr = e1[order]
        self.adj_he = order

    def _build_bvh(self):
        self.bvh = None
        self.bvh_faces = np.flatnonzero(~self.face_hidden)
        self._bvh_identity = len(self.bvh_faces) == len(self.tris)
        if BVHTree is None or len(self.bvh_faces) == 0:
            return
        verts = self.verts.tolist()
        polys = self.tris[self.bvh_faces].tolist()
        self.bvh = BVHTree.FromPolygons(verts, polys, all_triangles=True)

    def rebuild_bvh(self):
        self._build_bvh()

    def is_boundary_h(self, h):
        return self.he_face[h] < 0

    def voh(self, v):
        start = int(self.v_he[v])
        if start < 0:
            return []
        result = []
        h = start
        for _ in range(10000):
            result.append(h)
            h = int(self.he_next[self.he_opp[h]])
            if h == start or h < 0:
                break
        return result

    def fh(self, f):
        return [3 * f, 3 * f + 1, 3 * f + 2]

    def face_neighbors(self, f):
        result = []
        for h in self.fh(f):
            g = self.he_opp[h]
            ff = self.he_face[g]
            if ff >= 0:
                result.append(int(ff))
        return result

    def get_pn(self, v):
        pn = np.empty(6)
        pn[:3] = self.verts[v]
        pn[3:] = self.normals[v]
        return pn

    def bbox_diagonal_norm(self):
        return self._bbox_diag

    def bbox_center(self):
        return 0.5 * (self.bbox_min + self.bbox_max)

    # ---- ray casting ------------------------------------------------------------------------
    # These run thousands of times per interaction, so they use Python floats / mathutils
    # instead of small numpy arrays.
    def _barycentric(self, f, loc):
        T = self.tris
        V = self.verts
        a = T.item(f, 0)
        b = T.item(f, 1)
        c = T.item(f, 2)
        ax = V.item(a, 0)
        ay = V.item(a, 1)
        az = V.item(a, 2)
        d1x = V.item(b, 0) - ax
        d1y = V.item(b, 1) - ay
        d1z = V.item(b, 2) - az
        d2x = V.item(c, 0) - ax
        d2y = V.item(c, 1) - ay
        d2z = V.item(c, 2) - az
        dpx = loc[0] - ax
        dpy = loc[1] - ay
        dpz = loc[2] - az
        m00 = d1x * d1x + d1y * d1y + d1z * d1z
        m01 = d1x * d2x + d1y * d2y + d1z * d2z
        m11 = d2x * d2x + d2y * d2y + d2z * d2z
        det = m00 * m11 - m01 * m01
        if det == 0:
            return 0.0, 0.0
        b0 = d1x * dpx + d1y * dpy + d1z * dpz
        b1 = d2x * dpx + d2y * dpy + d2z * dpz
        return (m11 * b0 - m01 * b1) / det, (-m01 * b0 + m00 * b1) / det

    def _face_of(self, idx):
        return idx if self._bvh_identity else self.bvh_faces.item(idx)

    def ray_cast(self, origin, direction, require_front_facing=True):
        """First hit along the ray (used for the mouse cursor)."""
        if self.bvh is None:
            return None
        dvec = Vector((float(direction[0]), float(direction[1]), float(direction[2])))
        loc, nrm, idx, dist = self.bvh.ray_cast(Vector((float(origin[0]), float(origin[1]), float(origin[2]))), dvec)
        if loc is None:
            return None
        if require_front_facing and nrm.dot(dvec) > 0:
            return None
        f = self._face_of(idx)
        u, v = self._barycentric(f, loc)
        return Hit(f, u, v, dist)

    def intersect(self, pn):
        """SketchRetopo::intersect(PointNormal): 4 rays along +-n from p +- delta, keeping the
        closest front-facing hit."""
        bvh = self.bvh
        if bvh is None:
            return None
        px, py, pz, nx, ny, nz = pn.tolist()
        n = Vector((nx, ny, nz))
        p = Vector((px, py, pz))
        dn = n * self._bbox_diag * 0.0001
        neg_n = -n
        best_f = -1
        best_loc = None
        best_t = float('inf')
        for org, dvec in ((p - dn, n), (p + dn, n), (p - dn, neg_n), (p + dn, neg_n)):
            loc, nrm, idx, dist = bvh.ray_cast(org, dvec)
            if loc is None:
                continue
            if nrm.dot(n) < 0:
                continue
            if dist < best_t:
                best_t = dist
                best_f = idx
                best_loc = loc
        if best_loc is None:
            return None
        f = self._face_of(best_f)
        u, v = self._barycentric(f, best_loc)
        return Hit(f, u, v, best_t)

    def intersect_convert(self, hit):
        if hit is None:
            return None
        T = self.tris
        V = self.verts
        N = self.normals
        f = hit.face
        a = T.item(f, 0)
        b = T.item(f, 1)
        c = T.item(f, 2)
        u = hit.u
        v = hit.v
        w0 = 1 - u - v
        nx = w0 * N.item(a, 0) + u * N.item(b, 0) + v * N.item(c, 0)
        ny = w0 * N.item(a, 1) + u * N.item(b, 1) + v * N.item(c, 1)
        nz = w0 * N.item(a, 2) + u * N.item(b, 2) + v * N.item(c, 2)
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln > 0:
            nx /= ln
            ny /= ln
            nz /= ln
        return np.array((w0 * V.item(a, 0) + u * V.item(b, 0) + v * V.item(c, 0),
                         w0 * V.item(a, 1) + u * V.item(b, 1) + v * V.item(c, 1),
                         w0 * V.item(a, 2) + u * V.item(b, 2) + v * V.item(c, 2),
                         nx, ny, nz))

    def nearest(self, p):
        if self.bvh is None:
            return None
        loc, nrm, idx, dist = self.bvh.find_nearest(Vector((float(p[0]), float(p[1]), float(p[2]))))
        if loc is None:
            return None
        f = self._face_of(idx)
        u, v = self._barycentric(f, loc)
        return Hit(f, u, v, dist)

    # ---- exponential map (kt84/openmesh/base/ExpMap.hh) -----------------------------------
    @staticmethod
    def _intr_basis(normals):
        nx = np.abs(normals[:, 0])
        ny = np.abs(normals[:, 1])
        nz = np.abs(normals[:, 2])
        basis = np.zeros_like(normals)
        cx = (nx < ny) & (ny < nz)
        cy = ~cx & (ny < nz)
        cz = ~cx & ~cy
        basis[cx, 0] = 1
        basis[cy, 1] = 1
        basis[cz, 2] = 1
        basis -= (basis * normals).sum(1)[:, None] * normals
        ln = np.linalg.norm(basis, axis=1)
        ln[ln == 0] = 1
        return basis / ln[:, None]

    def _expmap_precompute(self):
        N = self.normals
        bx = self._intr_basis(N)
        by = np.cross(N, bx)
        p_idx = self.he_from[self.adj_he]
        q_idx = self.adj_nbr
        pq = self.verts[q_idx] - self.verts[p_idx]
        dist = np.linalg.norm(pq, axis=1)
        lu = (pq * bx[p_idx]).sum(1)
        lv = (pq * by[p_idx]).sum(1)
        ll = np.sqrt(lu * lu + lv * lv)
        scale = np.where(ll > 0, dist / np.where(ll > 0, ll, 1), 0)
        lu *= scale
        lv *= scale
        # rotation aligning q's intrinsic basis with p's (compute_rotation)
        pn_ = N[p_idx]
        qn = N[q_idx]
        q_basis = bx[q_idx].copy()
        axis = np.cross(qn, pn_)
        al = np.linalg.norm(axis, axis=1)
        nz = al > 0
        if nz.any():
            k = axis[nz] / al[nz, None]
            ang = np.arccos(np.clip((qn[nz] * pn_[nz]).sum(1), -1, 1))
            v = q_basis[nz]
            c = np.cos(ang)[:, None]
            s = np.sin(ang)[:, None]
            q_basis[nz] = v * c + np.cross(k, v) * s + k * (k * v).sum(1)[:, None] * (1 - c)
        p_basis = bx[p_idx]
        rot = np.arccos(np.clip((q_basis * p_basis).sum(1), -1, 1))
        neg = (np.cross(q_basis, p_basis) * pn_).sum(1) < 0
        rot[neg] *= -1
        with np.errstate(divide='ignore'):
            w = np.where(dist > 0, 1.0 / np.power(np.where(dist > 0, dist, 1), 0.25), 0.0)
        self._expmap_pre = (bx, by, lu.tolist(), lv.tolist(), np.cos(rot).tolist(), np.sin(rot).tolist(),
                            w.tolist(), dist.tolist(), self.adj_start.tolist(), q_idx.tolist())

    def expmap_compute(self, seeds, seeds_basis_u, seeds_uv, dist_max, uv_min=None, uv_max=None):
        if self._expmap_pre is None:
            self._expmap_precompute()
        bx, by, LU, LV, RC, RS, W, D, start, nbr = self._expmap_pre
        nv = len(self.verts)
        inf = float('inf')
        umin = -inf if uv_min is None else float(uv_min[0])
        vmin = -inf if uv_min is None else float(uv_min[1])
        umax = inf if uv_max is None else float(uv_max[0])
        vmax = inf if uv_max is None else float(uv_max[1])

        paramed = {}
        dist = {}
        weight = {}
        uv_u = {}
        uv_v = {}
        bu_x = {}
        bu_y = {}
        heap = []
        for s, basis_u, suv in zip(seeds, seeds_basis_u, seeds_uv):
            s = int(s)
            weight[s] = weight.get(s, 0.0) + 1
            dist[s] = 0.0
            uv_u[s] = uv_u.get(s, 0.0) + float(suv[0])
            uv_v[s] = uv_v.get(s, 0.0) + float(suv[1])
            bu_x[s] = bu_x.get(s, 0.0) + float(np.dot(bx[s], basis_u))
            bu_y[s] = bu_y.get(s, 0.0) + float(np.dot(by[s], basis_u))
            heapq.heappush(heap, (0.0, s))

        while heap:
            _, p = heapq.heappop(heap)
            if p in paramed:
                continue
            wsum = weight[p]
            pu = uv_u[p] / wsum
            pv = uv_v[p] / wsum
            bux = bu_x[p] / wsum
            buy = bu_y[p] / wsum
            uv_u[p] = pu
            uv_v[p] = pv
            bu_x[p] = bux
            bu_y[p] = buy
            paramed[p] = True
            pd = dist[p]
            if dist_max < pd:
                continue
            if not (umin <= pu <= umax and vmin <= pv <= vmax):
                continue
            bsq = bux * bux + buy * buy
            if bsq == 0:
                continue
            for k in range(start[p], start[p + 1]):
                q = nbr[k]
                if q in paramed:
                    continue
                dpq = D[k]
                nd = pd + dpq
                if nd < dist.get(q, inf):
                    dist[q] = nd
                w = W[k]
                weight[q] = weight.get(q, 0.0) + w
                lu = LU[k]
                lv = LV[k]
                # encode local uv with the propagated (non-unit) basis, account for scale
                luu = (bux * lu + buy * lv) / bsq
                lvv = (-buy * lu + bux * lv) / bsq
                uv_u[q] = uv_u.get(q, 0.0) + w * (pu + luu)
                uv_v[q] = uv_v.get(q, 0.0) + w * (pv + lvv)
                c = RC[k]
                s = RS[k]
                bu_x[q] = bu_x.get(q, 0.0) + w * (c * bux - s * buy)
                bu_y[q] = bu_y.get(q, 0.0) + w * (s * bux + c * buy)
                heapq.heappush(heap, (dist[q], q))

        self.expmap_paramed[:] = False
        self.expmap_uv[:] = 0.0
        if paramed:
            idx = np.fromiter(paramed.keys(), dtype=np.int64, count=len(paramed))
            self.expmap_paramed[idx] = True
            self.expmap_uv[idx, 0] = [uv_u[i] for i in idx.tolist()]
            self.expmap_uv[idx, 1] = [uv_v[i] for i in idx.tolist()]
        faces_paramed = self.expmap_paramed[self.tris].all(axis=1)
        self.expmap_paramed_faces = np.flatnonzero(faces_paramed)
        self._face_paramed = faces_paramed
        if self.expmap_paramed_faces.size:
            fuv = self.expmap_uv[self.tris[self.expmap_paramed_faces]]
            self._fuv = fuv
            self._fuv_min = fuv.min(axis=1)
            self._fuv_max = fuv.max(axis=1)
        else:
            self._fuv = np.zeros((0, 3, 2))
            self._fuv_min = self._fuv_max = np.zeros((0, 2))

    def fbc_to_uv(self, f, u, v):
        if not self._face_paramed[f]:
            return None
        a, b, c = self.tris[f]
        return (1 - u - v) * self.expmap_uv[a] + u * self.expmap_uv[b] + v * self.expmap_uv[c]

    def uv_to_fbc(self, uv):
        """Returns (face, u, v) or None."""
        if self._fuv.shape[0] == 0:
            return None
        uv = np.asarray(uv, dtype=float)
        inbox = np.all((self._fuv_min <= uv) & (uv <= self._fuv_max), axis=1)
        idx = np.flatnonzero(inbox)
        if idx.size == 0:
            return None
        F = self._fuv[idx]
        e1 = F[:, 1] - F[:, 0]
        e2 = F[:, 2] - F[:, 0]
        d = uv[None, :] - F[:, 0]
        det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            hu = (d[:, 0] * e2[:, 1] - e2[:, 0] * d[:, 1]) / det
            hv = (e1[:, 0] * d[:, 1] - d[:, 0] * e1[:, 1]) / det
        ok = (hu >= 0) & (hv >= 0) & (1 - hu - hv >= 0) & np.isfinite(hu) & np.isfinite(hv)
        k = np.flatnonzero(ok)
        if k.size == 0:
            return None
        j = int(k[0])
        return int(self.expmap_paramed_faces[idx[j]]), float(hu[j]), float(hv[j])

    # ---- snakes (BaseMesh::compute_snake_energy) ---------------------------------------
    def _laplace_iterative_scalar(self, values, num_iter, damping=0.25):
        start = self.adj_start
        nbr = self.adj_nbr
        counts = np.diff(start)
        counts_safe = np.where(counts > 0, counts, 1)
        owner = np.repeat(np.arange(len(values)), counts)
        x = values.astype(float).copy()
        for _ in range(num_iter):
            s = np.bincount(owner, weights=x[nbr], minlength=len(x))
            new = s / counts_safe
            x = np.where(counts > 0, (1 - damping) * new + damping * x, x)
        return x

    def _face_gradient(self, y):
        x0 = self.verts[self.tris[:, 0]]
        x1 = self.verts[self.tris[:, 1]]
        x2 = self.verts[self.tris[:, 2]]
        A = np.stack([x1 - x0, x2 - x0, np.cross(x1 - x0, x2 - x0)], axis=1)
        rhs = np.stack([y[self.tris[:, 1]] - y[self.tris[:, 0]], y[self.tris[:, 2]] - y[self.tris[:, 0]],
                        np.zeros(len(self.tris))], axis=1)
        det = np.linalg.det(A)
        good = np.abs(det) > 0
        g = np.zeros((len(self.tris), 3))
        if good.any():
            g[good] = np.linalg.solve(A[good], rhs[good][:, :, None])[:, :, 0]
        return g

    def compute_snake_energy(self):
        nv = len(self.verts)
        if self.snake_feature_type == SNAKE_MEAN_CURVATURE:
            feature = np.zeros(nv)      # principal curvatures are not computed in the original (TODO there)
        elif self.snake_feature_type == SNAKE_NORMAL_VARIATION:
            feature = np.full(nv, np.finfo(float).max)
            for k in range(3):
                d = (self.normals[self.tris[:, k]] * self.face_normals).sum(1)
                np.minimum.at(feature, self.tris[:, k], d)
        else:
            feature = np.zeros(nv)
            p = self.he_from
            q = self.he_to
            dp = self.verts[q] - self.verts[p]
            dn = self.normals[q] - self.normals[p]
            dd = (dp * dp).sum(1)
            with np.errstate(divide='ignore', invalid='ignore'):
                k1 = np.where(dd > 0, (dp * dn).sum(1) / np.where(dd > 0, dd, 1), 0)
            order = np.argsort(-np.abs(k1), kind='stable')
            seen = np.zeros(nv, dtype=bool)
            for h in order.tolist():
                v = p[h]
                if not seen[v]:
                    seen[v] = True
                    feature[v] = k1[h]
        if self.feature_smooth_iter > 0:
            feature = self._laplace_iterative_scalar(feature, self.feature_smooth_iter)
        if self.use_feature_as_energy:
            energy = feature.copy()
            if self.negate_energy:
                energy *= -1
        else:
            g = self._face_gradient(feature)
            gn = np.linalg.norm(g, axis=1)
            energy = np.zeros(nv)
            cnt = np.zeros(nv)
            for k in range(3):
                np.add.at(energy, self.tris[:, k], gn)
                np.add.at(cnt, self.tris[:, k], 1)
            energy /= np.where(cnt > 0, cnt, 1)
            if self.negate_energy:
                energy *= -1
        self.snake_energy = energy
        self.snake_energy_min = float(energy.min()) if nv else 0.0
        self.snake_energy_max = float(energy.max()) if nv else 0.0
        g = self._face_gradient(energy)
        grad = np.zeros((nv, 3))
        cnt = np.zeros(nv)
        for k in range(3):
            np.add.at(grad, self.tris[:, k], g)
            np.add.at(cnt, self.tris[:, k], 1)
        self.snake_energy_gradient = grad / np.where(cnt > 0, cnt, 1)[:, None]

    def smooth_normal(self, normals_original):
        """BaseMesh::smooth_normal (normalSmoothIter iterations of Laplacian smoothing)."""
        n = normals_original.copy()
        start = self.adj_start
        nbr = self.adj_nbr
        counts = np.diff(start)
        owner = np.repeat(np.arange(len(n)), counts)
        for _ in range(self.normal_smooth_iter):
            s = np.zeros_like(n)
            for k in range(3):
                s[:, k] = np.bincount(owner, weights=n[nbr, k], minlength=len(n))
            new = s / np.where(counts > 0, counts, 1)[:, None]
            n = np.where(counts[:, None] > 0, 0.75 * new + 0.25 * n, n)
            ln = np.linalg.norm(n, axis=1)
            n /= np.where(ln > 0, ln, 1)[:, None]
        self.normals = n
        self._expmap_pre = None

    # ---- cotangent weights (for the harmonic parameterization) ---------------------------
    def cotangents(self):
        """Per face-halfedge cotangent of the opposite corner (CotanWeight::cotanWeight_compute)."""
        if self._cotan is None:
            nh = self.n_halfedges
            cot = np.zeros(nh)
            nf = len(self.tris)
            if nf:
                L = np.linalg.norm(self.verts[self.he_to[:3 * nf]] - self.verts[self.he_from[:3 * nf]], axis=1).reshape(nf, 3)
                A = np.where(self.face_area > 0, self.face_area, 1)
                for i in range(3):
                    i1 = (i + 1) % 3
                    i2 = (i + 2) % 3
                    cot[np.arange(nf) * 3 + i] = (L[:, i1] ** 2 + L[:, i2] ** 2 - L[:, i] ** 2) / (4 * A)
            self._cotan = cot
        return self._cotan

    # ---- geodesics -------------------------------------------------------------------------
    def geodesic_path(self, hit0, hit1, max_nodes=2000000):
        """Approximate geodesic between two surface points: Dijkstra over mesh edges between the
        hit faces' corners, followed by iterative straightening on the surface.  Returns a list of
        positions (numpy (3,)) and a list of base mesh vertices crossed by the vertex path."""
        src = self.tris[hit0.face]
        dst = set(self.tris[hit1.face].tolist())
        p_src = self.intersect_convert(hit0)[:3]
        p_dst = self.intersect_convert(hit1)[:3]
        start = self.adj_start
        nbr = self.adj_nbr
        V = self.verts
        dist = {}
        prev = {}
        heap = []
        for v in src.tolist():
            d0 = float(np.linalg.norm(V[v] - p_src))
            dist[v] = d0
            prev[v] = -1
            heapq.heappush(heap, (d0, v))
        done = set()
        end = -1
        best_total = float('inf')
        while heap:
            d, v = heapq.heappop(heap)
            if v in done:
                continue
            done.add(v)
            if d > best_total:
                break
            if v in dst:
                total = d + float(np.linalg.norm(V[v] - p_dst))
                if total < best_total:
                    best_total = total
                    end = v
            if len(done) > max_nodes:
                break
            for k in range(start[v], start[v + 1]):
                w = int(nbr[k])
                nd = d + float(np.linalg.norm(V[w] - V[v]))
                if nd < dist.get(w, float('inf')):
                    dist[w] = nd
                    prev[w] = v
                    heapq.heappush(heap, (nd, w))
        if end < 0:
            return None, None
        path_v = []
        v = end
        while v != -1:
            path_v.append(v)
            v = prev[v]
        path_v.reverse()
        return [p_src] + [V[v].copy() for v in path_v] + [p_dst], path_v
