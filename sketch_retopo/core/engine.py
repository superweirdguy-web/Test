# Port of the SketchRetopo struct: SketchRetopo_core.cc, SketchRetopo_event.cc,
# SketchRetopo_misc.cc and SketchRetopo_intersect.cc.
#
# The engine is independent of Blender's UI.  The host (the Blender modal
# operator) provides a `view` object describing the camera:
#     view.width, view.height                  region size in pixels
#     view.camera_distance()                   |center - eye| in base-mesh local units
#     view.eye()                               eye position (local space)
#     view.eye_to_center()                     view direction (local space, unnormalized)
#     view.is_ortho()
#     view.ray(x, y) -> (origin, direction)    local-space ray through a pixel
#     view.project(points) -> (N, 2)           local positions -> pixel coordinates
# Mouse coordinates have their origin at the bottom-left of the region.

import math
import time

import numpy as np

from .pn import (Polyline, Polyline2d, pn_norm, pn_normalize, pn_normalized, normalized, norm3, dot3, cross3,
                 orthonormalize, rotate90, angle2d, acos_clamped, rbf_wendland, distance_to_line, project_to_line,
                 project_to_triangle)
from .curvenetwork import (Core, Patch, alive, voh_iter, vih_iter, voc_iter, pc_iter, ch_iter,
                           set_next_prev, set_vertex_halfedge, trace_halfchains, delete_patch,
                           delete_halfchain, delete_halfedge, IMAGINARY_SYMMETRY, IMAGINARY_BOUNDARY)
from .polymesh import find_edgeloop, insert_edgeloop
from .config import ConfigTemp, ConfigSaved, ConfigRender
from . import patchgen


class Memento:
    """kt84::Memento storing Core snapshots."""

    def __init__(self, buffer_size=150):
        self.init(buffer_size)

    def init(self, buffer_size=None):
        if buffer_size is None:
            buffer_size = len(self.undo_buffer)
        self.undo_buffer = [None] * buffer_size
        self.redo_buffer = []
        self.undo_begin = 0
        self.undo_size = 0

    def buffer_size(self):
        return len(self.undo_buffer)

    def store(self, data):
        n = self.buffer_size()
        if self.undo_size == n:
            self.undo_begin = (self.undo_begin + 1) % n
        else:
            self.undo_size += 1
        self.undo_buffer[(self.undo_begin + self.undo_size - 1) % n] = data
        self.redo_buffer = []

    def undo(self, current):
        if not self.undo_size:
            return None
        self.redo_buffer.append(current)
        n = self.buffer_size()
        data = self.undo_buffer[(self.undo_begin + self.undo_size - 1) % n]
        self.undo_size -= 1
        return data

    def redo(self, current):
        if not self.redo_buffer:
            return None
        self.undo_size += 1
        n = self.buffer_size()
        self.undo_buffer[(self.undo_begin + self.undo_size - 1) % n] = current
        return self.redo_buffer.pop()


class AutocmplInfo:
    def __init__(self):
        self.uv_lines = []
        self.corners = []

    def add_uv_line(self, uv0, uv1, dir0=None, dir1=None):
        self.uv_lines.append((np.array(uv0, dtype=float), np.array(uv1, dtype=float),
                              None if dir0 is None else np.array(dir0, dtype=float),
                              None if dir1 is None else np.array(dir1, dtype=float)))

    def set_corners(self, *pns):
        self.corners = [np.array(p, dtype=float) for p in pns]


# state ids (EnumState)
SKETCH, SPINE, AUTOCMPL, LASER, CYLINDER, DEFORMCURVE, EDITCORNER, EDITTOPOLOGY, EDGELOOP, MOVEVERTEX = range(10)

LEFT, RIGHT, MIDDLE = 'LEFT', 'RIGHT', 'MIDDLE'


def _unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        if id(x) not in seen:
            seen.add(id(x))
            out.append(x)
    return out


class Engine:
    def __init__(self):
        from . import states
        self.basemesh = None
        self.curvenetwork = Core()
        self.configTemp = ConfigTemp()
        self.configSaved = ConfigSaved()
        self.configRender = ConfigRender()
        self.memento = Memento(150)
        self.view = None

        self.common_dragMode = None          # None / 'Smooth' / 'Snakes'
        self.common_affected_edgechains = []
        self.common_pn_mouse = None
        self.common_mouse_pos = (0, 0)
        self.common_mouse_pos_prev = (0, 0)
        self.common_sketch_curve = Polyline()
        self.common_key_pressed = set()
        self.common_button_left_pressed = False
        self.common_button_right_pressed = False
        self.common_button_middle_pressed = False
        self.common_shift_pressed = False
        self.common_ctrl_pressed = False
        self.common_alt_pressed = False
        self.hide_stroke = Polyline2d()

        self.material_texture_png_base64 = ''
        self.messages = []                    # (level, text) for the host to report
        self.requests = []                    # host actions requested by keys (save, load, export, ...)

        self.stateSketch = states.StateSketch(self)
        self.stateSpine = states.StateSpine(self)
        self.stateAutocmpl = states.StateAutocmpl(self)
        self.stateLaser = states.StateLaser(self)
        self.stateCylinder = states.StateCylinder(self)
        self.stateDeformCurve = states.StateDeformCurve(self)
        self.stateEditCorner = states.StateEditCorner(self)
        self.stateEditTopology = states.StateEditTopology(self)
        self.stateEdgeLoop = states.StateEdgeLoop(self)
        self.stateMoveVertex = states.StateMoveVertex(self)
        self._states = [self.stateSketch, self.stateSpine, self.stateAutocmpl, self.stateLaser, self.stateCylinder,
                        self.stateDeformCurve, self.stateEditCorner, self.stateEditTopology, self.stateEdgeLoop,
                        self.stateMoveVertex]
        self.state = self.stateSketch
        self.state.init()

    # ---- messages -----------------------------------------------------------------------------
    def report(self, text, level='INFO'):
        self.messages.append((level, text))

    # ---- base mesh -----------------------------------------------------------------------------
    def set_basemesh(self, basemesh, reset_curves=True):
        """SketchRetopo::import_basemesh (after reading the file)."""
        self.basemesh = basemesh
        bb = basemesh.bbox_diagonal_norm()
        if reset_curves:
            self.curvenetwork.clear()
            self.memento.init()
            self.configRender = ConfigRender()
            self.configSaved = ConfigSaved()
            self.configTemp = ConfigTemp()
            self.configSaved.projOffset = 0.0005 * bb
        self.configTemp.quadSize.value = bb * ConfigTemp.default_quadSize_ratio
        self.configTemp.segmentSize.value = bb * ConfigTemp.default_segmentSize_ratio
        Patch.quadSize = self.configTemp.quadSize.value
        self.state.init()
        if reset_curves:
            self.trace_basemesh_boundary()

    # ---- intersection / projection (SketchRetopo_intersect.cc) --------------------------------
    def intersect(self, pn):
        if self.basemesh is None:
            return None
        return self.basemesh.intersect(pn)

    def intersect_convert(self, hit):
        if hit is None:
            return None
        return self.basemesh.intersect_convert(hit)

    def intersect_mouse(self, mouse_x, mouse_y):
        if self.basemesh is None or self.view is None:
            return None
        ray = self.view.ray(mouse_x, mouse_y)
        if ray is None:
            return None
        origin, direction = ray
        return self.basemesh.ray_cast(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float))

    def project(self, pn):
        """In-place projection onto the base mesh, offset by projOffset along the normal."""
        res = self.intersect_convert(self.intersect(pn))
        if res is None:
            return pn
        pn[:] = res
        pn[:3] += self.configSaved.projOffset * res[3:]
        return pn

    def project_polyline(self, polyline):
        for pn in polyline.points:
            self.project(pn)

    def reproject_all(self):
        for v in self.curvenetwork.vertices:
            self.project(v.pn)
        for p in self.curvenetwork.patches:
            for v in p.vertices():
                self.project(p.pn[v])
            p.invalidate_displist()
        self.curvenetwork.invalidate_displist()

    def uv_to_pn(self, uv):
        fbc = self.basemesh.uv_to_fbc(uv)
        if fbc is None:
            return None
        from .basemesh import Hit
        return self.intersect_convert(Hit(fbc[0], fbc[1], fbc[2], 0.0))

    def pn_to_uv(self, pn):
        hit = self.intersect(pn)
        if hit is None:
            return None
        return self.basemesh.fbc_to_uv(hit.face, hit.u, hit.v)

    # ---- patch generation ------------------------------------------------------------------------
    def is_loop_ccw(self, loop):
        n = loop.size()
        if n == 0:
            return False
        P = np.asarray(loop.points)
        avg = P.mean(axis=0)
        if np.linalg.norm(avg[3:]) < self.configTemp.loop_threshold_normal:
            return False
        d0 = P[:, :3] - avg[:3]
        d1 = np.roll(P[:, :3], -1, axis=0) - avg[:3]
        area_sum = float((0.5 * (np.cross(d0, d1) * P[:, 3:]).sum(1)).sum())
        loop_length = loop.length()
        area_max = loop_length * loop_length / (4 * math.pi)
        return area_max * self.configTemp.loop_threshold_area < area_sum

    def generate_patches(self):
        cn = self.curvenetwork
        cn.invalidate_displist()
        cn.set_flag_halfchains('SUBST', 0)
        for c in list(cn.halfchains):
            if c.is_deleted or c.flag or (c.patch is not None and not c.patch.is_deleted):
                continue
            if c.halfedge_front.imaginary_patch is not None:
                c.patch = c.halfedge_front.imaginary_patch
                continue
            c_front, c_back = trace_halfchains(c)
            if c_front is None or c_back is None:
                continue
            num_corners = 0
            c1 = c_front
            guard = 0
            while c1 is not None:
                c1.flag = 1
                if c1.is_corner():
                    num_corners += 1
                if c1 is c_back:
                    break
                c1 = c1.next()
                guard += 1
                if guard > 100000:
                    break
            if c_front.prev() is not c_back or num_corners < 2 or 6 < num_corners:
                continue
            polyline = Polyline()
            c1 = c_front
            while True:
                sub = c1.to_polyline()
                sub.points.pop()
                polyline.points.extend(sub.points)
                if c1 is c_back:
                    break
                c1 = c1.next()
            polyline.is_loop = True
            if polyline.length() > self.basemesh.bbox_diagonal_norm() * 2:
                continue
            if not self.is_loop_ccw(polyline):
                continue
            patch = cn.new_patch()
            c1 = c_front
            while True:
                c1.patch = patch
                c1 = c1.next()
                if c1 is c_front or c1 is None:
                    break
            patch.halfchain = c_front
            while not patch.halfchain.is_corner():
                patch.halfchain = patch.halfchain.next()
            patch.halfchain = patch.halfchain.next()
            patch.generate_topology()
            self.compute_patch_interior_pn(patch)

    def generate_patch_debug(self):
        cn = self.curvenetwork
        if len(cn.patches) != 1:
            return
        import random
        patch = cn.patches[0]
        total = 0
        for c in pc_iter(patch):
            ns = random.randint(1, 20)
            c.edgechain.num_subdiv = ns
            total += ns
        if total % 2:
            patch.halfchain.edgechain.num_subdiv += 1
        patch.generate_topology()
        self.compute_patch_interior_pn(patch)

    def _find_snap(self, pn):
        snap = self.configTemp.snapSize()
        best = {}
        for v in self.curvenetwork.vertices:
            if v.is_deleted:
                continue
            d = pn_norm(pn - v.pn)
            if snap < d:
                continue
            if v.is_openend():
                kind = 0
            elif v.is_corner():
                kind = 1
            else:
                kind = 2
            if kind not in best or d < best[kind][0]:
                best[kind] = (d, v)
        for kind in (0, 1, 2):
            if kind in best:
                return best[kind][1].pn.copy()
        return None

    def add_curve_open_delegate(self, curve, corner_at_openend=True):
        is_snapped = False
        pn_snap = self._find_snap(curve.front())
        if pn_snap is not None:
            curve.offset_front(pn_snap)
            is_snapped = True
        pn_snap = self._find_snap(curve.back())
        if pn_snap is not None:
            curve.offset_back(pn_snap)
            is_snapped = True
        if is_snapped:
            self.project_polyline(curve)
        self.curvenetwork.add_curve_open(curve, self.configTemp.snapSize(), corner_at_openend)

    # ---- patch interior (Laplacian smoothing + projection) ------------------------------------------
    @staticmethod
    def _patch_edges(patch):
        he_to = np.asarray(patch.he_to, dtype=np.int64)
        nh = len(he_to)
        he_from = he_to[np.arange(nh) ^ 1]
        return he_from, he_to

    @staticmethod
    def _laplace_direct(patch, fixed, values):
        """Uniform-weight harmonic interpolation of `values` (n, k) with `fixed` vertices held."""
        n = patch.n_vertices()
        he_from, he_to = Engine._patch_edges(patch)
        free = np.flatnonzero(~fixed)
        x = values.copy()
        if free.size == 0:
            return x
        deg = np.bincount(he_from, minlength=n).astype(float)
        index = -np.ones(n, dtype=np.int64)
        index[free] = np.arange(free.size)
        m = free.size
        k = values.shape[1]
        sel = ~fixed[he_from]
        rows = index[he_from[sel]]
        cols_v = he_to[sel]
        to_free = ~fixed[cols_v]
        rhs = np.zeros((m, k))
        bsel = ~to_free
        for j in range(k):
            rhs[:, j] = np.bincount(rows[bsel], weights=values[cols_v[bsel], j], minlength=m)
        diag = deg[free]
        diag[diag == 0] = 1
        r_ff = rows[to_free]
        c_ff = index[cols_v[to_free]]
        if m <= 900:
            A = np.zeros((m, m))
            A[np.arange(m), np.arange(m)] = diag
            np.add.at(A, (r_ff, c_ff), -1.0)
            try:
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        else:
            def matvec(z):
                out = diag[:, None] * z
                for j in range(k):
                    out[:, j] -= np.bincount(r_ff, weights=z[c_ff, j], minlength=m)
                return out
            sol = rhs / diag[:, None]
            r = rhs - matvec(sol)
            zv = r / diag[:, None]
            pvec = zv.copy()
            rz = (r * zv).sum(0)
            bnorm = np.linalg.norm(rhs, axis=0) + 1e-30
            for _ in range(10 * m):
                Ap = matvec(pvec)
                alpha = rz / np.where((pvec * Ap).sum(0) != 0, (pvec * Ap).sum(0), 1)
                sol += alpha * pvec
                r -= alpha * Ap
                if np.all(np.linalg.norm(r, axis=0) < 1e-10 * bnorm):
                    break
                zv = r / diag[:, None]
                rz_new = (r * zv).sum(0)
                beta = rz_new / np.where(rz != 0, rz, 1)
                pvec = zv + beta * pvec
                rz = rz_new
        x[free] = sol
        return x

    def compute_patch_interior_pn(self, patch):
        n = patch.n_vertices()
        if n == 0:
            return
        fixed = np.array([patch.is_boundary_v(v) for v in range(n)], dtype=bool)
        n_interior = int((~fixed).sum())
        values = self._laplace_direct(patch, fixed, patch.pn.copy())
        he_from, he_to = self._patch_edges(patch)
        deg = np.bincount(he_from, minlength=n).astype(float)
        deg[deg == 0] = 1
        interior = np.flatnonzero(~fixed)
        n_iterations = max(int(n_interior * 0.01), 5)
        for _ in range(n_iterations):
            s = np.zeros_like(values)
            for j in range(6):
                s[:, j] = np.bincount(he_from, weights=values[he_to, j], minlength=n)
            new = s / deg[:, None]
            values[interior] = 0.75 * new[interior] + 0.25 * values[interior]
            for v in interior.tolist():
                pn = values[v]
                if pn[3] == 0 and pn[4] == 0 and pn[5] == 0:
                    continue
                pn_normalize(pn)
                self.project(pn)
        patch.pn[interior] = values[interior]
        patch.invalidate_displist()

    def compute_patch_interior_pn_harmonic(self, patch, submesh_seed):
        """Hidden feature (ctrl+click in move-vertex mode): harmonic parameterization of the base mesh
        region enclosed by the patch boundary.  Geodesic paths are approximated by edge paths."""
        bm = self.basemesh
        p_corner0 = patch.patch_corner(0)
        if p_corner0 < 0 or patch.is_failure:
            return
        num_sides = patch.param.get_num_sides()
        l = patch.param.l
        n = patch.n_vertices()
        # step I: harmonic parameterization of the patch
        fixed = np.array([patch.is_boundary_v(v) for v in range(n)], dtype=bool)
        vals = np.zeros((n, 2))
        p_h = patch.halfedge_handle(p_corner0)
        for i in range(num_sides):
            for j in range(l[i]):
                t = i + j / float(l[i])
                vals[patch.from_vertex_handle(p_h)] = patchgen.get_boundary_geometry(num_sides, t)
                p_h = patch.prev_halfedge_handle(p_h)
        harmonic_uv = self._laplace_direct(patch, fixed, vals)

        # step II: harmonic parameterization of the base mesh region
        constraints = []    # (vertices, coords, uv)
        floodfill = np.zeros(len(bm.tris), dtype=bool)
        submesh_faces = []
        p_h = patch.halfedge_handle(p_corner0)
        for i in range(num_sides):
            for j in range(l[i]):
                pv0, pv1 = patch.util_halfedge_to_vertex_pair(p_h ^ 1)
                hit = self.intersect(patch.pn[pv0])
                hit_next = self.intersect(patch.pn[pv1])
                p_h = patch.prev_halfedge_handle(p_h)
                if hit is None or hit_next is None:
                    continue
                fv = bm.tris[hit.face]
                constraints.append((list(fv), [1 - hit.u - hit.v, hit.u, hit.v], harmonic_uv[pv0]))
                pts, path_v = bm.geodesic_path(hit, hit_next)
                if path_v is None:
                    continue
                total = sum(norm3(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)) or 1.0
                acc = 0.0
                for k in range(1, len(pts) - 1):
                    acc += norm3(pts[k] - pts[k - 1])
                    t = acc / total
                    uv = (1 - t) * harmonic_uv[pv0] + t * harmonic_uv[pv1]
                    vtx = path_v[k - 1]
                    constraints.append(([vtx], [1.0], uv))
                    for h in bm.voh(vtx):
                        f = bm.he_face[h]
                        if f >= 0:
                            floodfill[f] = True
                            submesh_faces.append(int(f))
        limit = max(len(submesh_faces), 1) * 10000
        candidates = [int(submesh_seed)]
        while candidates:
            if limit < len(submesh_faces):
                self.report('Sub-mesh extraction process does not seem to terminate!', 'WARNING')
                return
            f = candidates.pop()
            floodfill[f] = True
            submesh_faces.append(f)
            for ff in bm.face_neighbors(f):
                if not floodfill[ff]:
                    candidates.append(ff)
        submesh_faces = sorted(set(submesh_faces))
        verts = sorted(set(bm.tris[submesh_faces].reshape(-1).tolist()))
        vtx2var = {v: i for i, v in enumerate(verts)}
        m = len(verts)
        cot = bm.cotangents()
        L = np.zeros((m, m))
        for v in verts:
            i = vtx2var[v]
            wsum = 0.0
            for h in bm.voh(v):
                w_v = int(bm.he_to[h])
                j = vtx2var.get(w_v)
                if j is None:
                    continue
                weight = 0.0
                for hh in (h, int(bm.he_opp[h])):
                    f = bm.he_face[hh]
                    if f < 0:
                        continue
                    if floodfill[f]:
                        weight += cot[hh]
                L[i, j] += weight
                wsum += weight
            L[i, i] -= wsum
        cw = 0.1 * np.abs(L).sum()
        C = np.zeros((m, m))
        rhs = np.zeros((m, 2))
        for (vs, coords, uv) in constraints:
            for k0, v0 in enumerate(vs):
                i = vtx2var.get(int(v0))
                if i is None:
                    continue
                for k1, v1 in enumerate(vs):
                    j = vtx2var.get(int(v1))
                    if j is None:
                        continue
                    C[i, j] += cw * coords[k0] * coords[k1]
                rhs[i] += cw * coords[k0] * np.asarray(uv)
        A = -L + C
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, rhs, rcond=None)[0]
        b_uv = {v: x[i] for v, i in vtx2var.items()}
        # step III: locate each interior patch vertex by its harmonic uv
        for pv in range(n):
            if fixed[pv]:
                continue
            p_uv = harmonic_uv[pv]
            for f in submesh_faces:
                a, b, c = bm.tris[f]
                t = project_to_triangle(b_uv[a], b_uv[b], b_uv[c], p_uv)
                if t is None or min(t) < 0:
                    continue
                pn = t[0] * bm.get_pn(a) + t[1] * bm.get_pn(b) + t[2] * bm.get_pn(c)
                pn_normalize(pn)
                self.project(pn)
                patch.pn[pv] = pn
                break
        patch.invalidate_displist()

    # ---- suggestive tools ----------------------------------------------------------------------------
    def add_uv_line(self, uv0, uv1, unnormalized_dir0=None, unnormalized_dir1=None):
        curve = Polyline()
        seg = self.configTemp.segmentSize()
        num_segments = int(np.linalg.norm(uv1 - uv0) / seg) if seg > 0 else 0
        a = b = None
        if unnormalized_dir0 is not None and unnormalized_dir1 is not None:
            r = float(np.linalg.norm(uv1 - uv0))
            derivative0 = r * normalized(unnormalized_dir0)
            derivative1 = r * normalized(unnormalized_dir1)
            a = uv0 - uv1 + derivative0
            b = -uv0 + uv1 - derivative1
        for i in range(num_segments + 1):
            t = i / float(num_segments) if num_segments > 0 else 0.0
            if a is not None:
                uv = (1 - t) * uv0 + t * uv1 + t * (1 - t) * ((1 - t) * a + t * b)
            else:
                uv = (1 - t) * uv0 + t * uv1
            pn = self.uv_to_pn(uv)
            if pn is None:
                continue
            self.project(pn)
            curve.push_back(pn)
        if curve.size() < 3:
            return
        self.add_curve_open_delegate(curve)

    def _seed_vertex(self, hit):
        f = hit.face
        v0, v1, v2 = self.basemesh.tris[f]
        b0, b1, b2 = hit.bc(0), hit.bc(1), hit.bc(2)
        if b0 < b1 and b1 < b2:
            return int(v2)
        if b0 < b1:
            return int(v1)
        return int(v0)

    def add_spine_curve(self, spine_curve):
        cn = self.curvenetwork
        bm = self.basemesh
        brush_size = self.configTemp.brushSize_spine()
        curve_length = spine_curve.length()
        angle_threshold = math.cos(3.1415 / 6)

        seeds = []
        seeds_basis_u = []
        seeds_uv = []
        seed_v = 0.0
        n = spine_curve.size()
        for i in range(n):
            hit = self.intersect(spine_curve[i])
            if hit is None:
                return
            seed = self._seed_vertex(hit)
            seed_normal = bm.normals[seed]
            basis_v = np.zeros(3)
            if 0 < i:
                basis_v += spine_curve[i][:3] - spine_curve[i - 1][:3]
            if i < n - 1:
                basis_v += spine_curve[i + 1][:3] - spine_curve[i][:3]
            basis_v -= dot3(seed_normal, basis_v) * seed_normal
            basis_v = normalized(basis_v)
            basis_u = np.cross(basis_v, seed_normal)
            seed_uv = np.array((0.0, seed_v))
            d = bm.verts[seed] - spine_curve[i][:3]
            seed_uv[0] += dot3(d, basis_u)
            seed_uv[1] += dot3(d, basis_v)
            if i > 0:
                seed_v += pn_norm(spine_curve[i] - spine_curve[i - 1])
            seeds.append(seed)
            seeds_basis_u.append(basis_u)
            seeds_uv.append(seed_uv)

        uv_min = np.array((-1.5 * brush_size, -0.5 * brush_size))
        uv_max = np.array((1.5 * brush_size, curve_length + 0.5 * brush_size))
        bm.expmap_compute(seeds, seeds_basis_u, seeds_uv, brush_size * 1.8, uv_min, uv_max)
        self.expmap_isoline_interval = brush_size

        UNDETERMINED, INVALID, SNAPPED_V, OUTSIDE, INSIDE, CONNECTED_TO_CORNER = range(6)
        for h in cn.halfedges:
            h.flag = UNDETERMINED
            if h.is_deleted or h.patch() is not None or h.opposite.patch() is None:
                h.flag = INVALID

        class CornerInfo:
            __slots__ = ('is_snapped_u', 'is_snapped_v', 'snapped_h', 'uv')

            def __init__(self):
                self.is_snapped_u = False
                self.is_snapped_v = False
                self.snapped_h = None
                self.uv = None

        corners = {}
        for is_front in (0, 1):
            for is_left in (0, 1):
                c = CornerInfo()
                c.uv = np.array((-brush_size if is_left else brush_size, 0.0 if is_front else curve_length))
                corners[(is_front, is_left)] = c

        # check if corners snap in v-direction
        for is_front in (0, 1):
            pn_endpoint = spine_curve.front() if is_front else spine_curve.back()
            corner_left = corners[(is_front, 1)]
            corner_right = corners[(is_front, 0)]
            dist_min = float('inf')
            h_min = None
            for h in cn.halfedges:
                if h.flag != UNDETERMINED:
                    continue
                if h.vertex.is_corner():
                    continue
                d = pn_norm(h.vertex.pn - pn_endpoint)
                if d < dist_min:
                    dist_min = d
                    h_min = h
            if self.configTemp.snapSize() < dist_min:
                continue
            h_forward = h_min
            while True:
                if h_forward.vertex.is_corner():
                    break
                uv = self.pn_to_uv(h_forward.vertex.pn)
                if uv is None or abs(uv[0]) > brush_size:
                    h_forward = h_forward.prev
                    break
                h_forward = h_forward.next
            h_backward = h_min
            while True:
                if h_backward.vertex.is_corner():
                    break
                uv = self.pn_to_uv(h_backward.vertex.pn)
                if uv is None or abs(uv[0]) > brush_size:
                    h_backward = h_backward.next
                    break
                h_backward = h_backward.prev
            corner_forward = corner_right if is_front else corner_left
            corner_backward = corner_left if is_front else corner_right
            uvf = self.pn_to_uv(h_forward.vertex.pn)
            uvb = self.pn_to_uv(h_backward.vertex.pn)
            if uvf is None or uvb is None:
                continue
            corner_forward.is_snapped_v = True
            corner_backward.is_snapped_v = True
            corner_forward.snapped_h = h_forward
            corner_backward.snapped_h = h_backward
            corner_forward.uv = uvf
            corner_backward.uv = uvb
            h = h_forward
            guard = 0
            while h is not h_backward and h is not None:
                h.flag = SNAPPED_V
                h = h.prev
                guard += 1
                if guard > 1000000:
                    break

        # check if corners snap in u-directions
        for is_front in (0, 1):
            for is_left in (0, 1):
                corner = corners[(is_front, is_left)]
                if corner.is_snapped_v:
                    if not corner.snapped_h.vertex.is_corner():
                        continue
                    for h in voh_iter(corner.snapped_h.vertex):
                        uv = self.pn_to_uv(h.vertex.pn)
                        if uv is None:
                            continue
                        uv = normalized(uv - corner.uv)
                        if (is_front and uv[1] > angle_threshold) or (not is_front and uv[1] < -angle_threshold):
                            corner.is_snapped_u = True
                            break
                else:
                    corner_dist_min = float('inf')
                    corner_h_min = None
                    nocorner_dist_min = float('inf')
                    nocorner_h_min = None
                    for h in cn.halfedges:
                        if h.flag != UNDETERMINED:
                            continue
                        uv = self.pn_to_uv(h.vertex.pn)
                        if uv is None:
                            continue
                        dist_v = abs(uv[1] if is_front else uv[1] - curve_length)
                        if self.configTemp.snapSize() < dist_v:
                            continue
                        dist_u = -uv[0] if is_left else uv[0]
                        if brush_size < dist_u or dist_u <= 0:
                            continue
                        is_corner = h.vertex.is_corner()
                        if is_corner and dist_u < corner_dist_min:
                            corner_dist_min = dist_u
                            corner_h_min = h
                        elif not is_corner and dist_u < nocorner_dist_min:
                            nocorner_dist_min = dist_u
                            nocorner_h_min = h
                    if corner_h_min is not None or nocorner_h_min is not None:
                        corner.snapped_h = corner_h_min if corner_h_min is not None else nocorner_h_min
                        corner.uv = self.pn_to_uv(corner.snapped_h.vertex.pn)
                        corner.is_snapped_u = True

        # trace halfedges from snapped corner in v-direction
        for is_front in (0, 1):
            for is_left in (0, 1):
                corner = corners[(is_front, is_left)]
                corner_opposite = corners[(1 - is_front, is_left)]
                if not corner.is_snapped_u:
                    continue
                is_forward = (is_front and not is_left) or (not is_front and is_left)
                h = None
                angle_max = -float('inf')
                for voh in voh_iter(corner.snapped_h.vertex):
                    uv = self.pn_to_uv(voh.vertex.pn)
                    if uv is None:
                        continue
                    uv = normalized(uv - corner.uv)
                    ang = uv[1] if is_front else -uv[1]
                    if angle_max < ang:
                        angle_max = ang
                        h = voh if is_forward else voh.opposite
                if h is None:
                    continue
                for voh in voh_iter(corner.snapped_h.vertex):
                    if voh is not h and voh.flag == UNDETERMINED:
                        voh.flag = OUTSIDE
                    if voh.opposite is not h and voh.opposite.flag == UNDETERMINED:
                        voh.opposite.flag = OUTSIDE
                guard = 0
                while h is not None:
                    if h.flag != UNDETERMINED or h is corner_opposite.snapped_h:
                        break
                    h.flag = CONNECTED_TO_CORNER
                    if h is not corner.snapped_h:
                        h.is_corner = False
                    h_new = h.next if is_forward else h.prev
                    if h_new is None:
                        break
                    uv = self.pn_to_uv((h.vertex if is_forward else h.from_vertex()).pn)
                    uv_new = self.pn_to_uv((h_new.vertex if is_forward else h_new.from_vertex()).pn)
                    if uv is None or uv_new is None:
                        break
                    d = normalized(uv_new - uv)
                    ang = d[1] if is_front else -d[1]
                    if (is_left and uv_new[0] < -brush_size) or (not is_left and brush_size < uv_new[0]) or ang < angle_threshold:
                        break
                    h = h_new
                    guard += 1
                    if guard > 1000000:
                        break

        def make_huv(h):
            return [h, self.pn_to_uv(h.from_vertex().pn), self.pn_to_uv(h.vertex.pn)]

        def huv_ok(huv):
            return huv[0] is not None and (huv[1] is not None or huv[2] is not None)

        def huv_uv(huv):
            return huv[1] if huv[1] is not None else huv[2]

        def is_inside(huv):
            if huv[1] is None or huv[2] is None:
                return False
            for uv in (huv[2], huv[1]):
                if uv[0] < -brush_size or brush_size < uv[0] or uv[1] < 0 or curve_length < uv[1]:
                    return False
            d = normalized(huv[2] - huv[1])
            return angle_threshold < abs(d[1])

        for h in cn.halfedges:
            if h.flag != UNDETERMINED:
                continue
            huv = make_huv(h)
            if not huv_ok(huv) or not is_inside(huv):
                h.flag = OUTSIDE
                continue
            is_left = huv_uv(huv)[0] < 0
            v_min = corners[(1, 1 if is_left else 0)].uv[1]
            v_max = corners[(0, 1 if is_left else 0)].uv[1]
            hv = huv_uv(huv)[1]
            h.flag = INSIDE if (v_min < hv < v_max) else OUTSIDE

        snapped_vertices = [c.snapped_h.vertex for c in corners.values() if c.snapped_h is not None]

        def is_corner_vertex(v):
            return any(v is sv for sv in snapped_vertices)

        seq_left = []
        seq_right = []
        for h in cn.halfedges:
            if h.flag < INSIDE:
                continue
            if h.next.flag >= INSIDE and h.prev.flag >= INSIDE:
                if not is_corner_vertex(h.vertex):
                    h.is_corner = False
                if not is_corner_vertex(h.next.vertex):
                    h.next.is_corner = False
                if not is_corner_vertex(h.prev.vertex):
                    h.prev.is_corner = False
                continue
            huv = make_huv(h)
            is_left = huv_uv(huv)[0] < 0
            if h.next.flag < INSIDE and not is_corner_vertex(h.vertex):
                tmp = [huv[0], None, huv[2]]
                (seq_left if is_left else seq_right).append(tmp)
            if h.prev.flag < INSIDE and not is_corner_vertex(h.from_vertex()):
                tmp = [huv[0], huv[1], None]
                (seq_left if is_left else seq_right).append(tmp)

        def sort_key(huv):
            return huv_uv(huv)[1] if huv_ok(huv) else 0.0

        seq_left.sort(key=sort_key)
        seq_right.sort(key=sort_key)

        for is_left in (0, 1):
            corner_front = corners[(1, is_left)]
            corner_back = corners[(0, is_left)]
            seq = seq_left if is_left else seq_right
            if not corner_front.is_snapped_u and not corner_back.is_snapped_u and not seq:
                self.add_uv_line(corner_front.uv, corner_back.uv)
                continue
            for is_front in (0, 1):
                corner = corner_front if is_front else corner_back
                if not corner.is_snapped_u and seq:
                    huv = seq[0] if is_front else seq[-1]
                    self.add_uv_line(corner.uv, huv_uv(huv))
                    target = huv[0] if ((is_left and is_front) or (not is_left and not is_front)) else huv[0].prev
                    if target is not None:
                        target.is_corner = False
                    if is_front:
                        seq.pop(0)
                    else:
                        seq.pop()
            for k in range(0, len(seq) - 1, 2):
                uv0 = huv_uv(seq[k])
                h0 = seq[k][0]
                uv1 = huv_uv(seq[k + 1])
                h1 = seq[k + 1][0]
                self.add_uv_line(uv0, uv1)
                (h0.prev if is_left else h0).is_corner = False
                (h1 if is_left else h1.prev).is_corner = False

        for is_front in (0, 1):
            corner_left = corners[(is_front, 1)]
            corner_right = corners[(is_front, 0)]
            if not corner_left.is_snapped_v:
                self.add_uv_line(corner_left.uv, corner_right.uv)

    # ---- auto-completion -------------------------------------------------------------------------------
    def compute_autocompl(self, info):
        for (uv0, uv1, d0, d1) in info.uv_lines:
            self.add_uv_line(uv0, uv1, d0, d1)

    def suggest_autocmpl(self, h, favor_triangle):
        cn = self.curvenetwork
        bm = self.basemesh
        brush_size = self.configTemp.brushSize_autocmpl()
        seed_basis = h.to_vector()
        seed_uv = np.zeros(2)
        hit = self.intersect(h.vertex.pn)
        if hit is None:
            return []
        seed = self._seed_vertex(hit)
        seed_normal = bm.normals[seed]
        seed_basis = orthonormalize(seed_normal, seed_basis)
        d = bm.verts[seed] - h.vertex.pn[:3]
        seed_uv[0] += dot3(d, seed_basis)
        seed_uv[1] += dot3(d, np.cross(seed_normal, seed_basis))
        uv_min = brush_size * np.array((-1.2, -0.3))
        uv_max = brush_size * np.array((1.2, 2.2))
        bm.expmap_compute([seed], [seed_basis], [seed_uv], brush_size * 3.8, uv_min, uv_max)
        self.expmap_isoline_interval = brush_size

        relevant = []
        uv_cache = {}
        for hh in cn.halfedges:
            if hh.is_deleted or hh.patch() is not None or hh.opposite.patch() is None:
                continue
            uv_to = self._uv_of_vertex(hh.vertex, uv_cache)
            uv_from = self._uv_of_vertex(hh.from_vertex(), uv_cache)
            if uv_to is None or uv_from is None:
                continue
            relevant.append(hh)

        result = []
        num_steps = 10
        OUTSIDE, INSIDE, INSIDE_GROUPED = 0, 1, 2

        def is_sharp(a):
            return 0 < a < math.pi * 0.66

        def is_very_sharp(a):
            return 0 < a < math.pi * 0.33

        for step in range(1, num_steps + 1):
            bsc = step * brush_size / num_steps
            info = AutocmplInfo()

            def inside(uv):
                x = uv / bsc
                return -1.0 <= x[0] <= 1.0 and -0.25 <= x[1] <= 2.0

            for hh in relevant:
                hh.flag = OUTSIDE
                for v in (hh.vertex, hh.from_vertex()):
                    if inside(self._uv_of_vertex(v, uv_cache)):
                        hh.flag = INSIDE
                        break

            sequences = []
            for hh in relevant:
                if hh.flag != INSIDE:
                    continue
                s = _HalfedgeSequence()
                s.h_back = hh
                while True:
                    s.h_back.flag = INSIDE_GROUPED
                    if s.h_back.next is None or s.h_back.next.flag != INSIDE:
                        break
                    s.h_back = s.h_back.next
                s.h_front = hh
                while True:
                    s.h_front.flag = INSIDE_GROUPED
                    if s.h_front.prev is None or s.h_front.prev.flag != INSIDE:
                        break
                    s.h_front = s.h_front.prev
                sequences.append(s)

            if not sequences or len(sequences) > 2:
                continue

            p2u = self.pn_to_uv
            u2p = self.uv_to_pn
            if len(sequences) == 1:
                s = sequences[0]
                nc = s.num_corners()
                uv_front = p2u(s.pn_front())
                uv_back = p2u(s.pn_back())
                if uv_front is None or uv_back is None:
                    continue
                uv0 = p2u(s.get_corner(0).vertex.pn) if 0 < nc else None
                uv1 = p2u(s.get_corner(1).vertex.pn) if 1 < nc else None
                uv2 = p2u(s.get_corner(2).vertex.pn) if 2 < nc else None
                uv3 = p2u(s.get_corner(3).vertex.pn) if 3 < nc else None
                sub_uv_front = p2u(s.sub_pn_front()) if nc != 0 else None
                sub_uv_back = p2u(s.sub_pn_back()) if nc != 0 else None
                if nc == 0:
                    dd = normalized(rotate90(uv_back - uv_front))
                    uv_c0 = uv_front + (2 * bsc) * dd
                    uv_c1 = uv_back + (2 * bsc) * dd
                    pn_c0 = u2p(uv_c0)
                    pn_c1 = u2p(uv_c1)
                    if pn_c0 is None or pn_c1 is None:
                        continue
                    info.add_uv_line(uv_front, uv_c0)
                    info.add_uv_line(uv_back, uv_c1)
                    info.add_uv_line(uv_c0, uv_c1)
                    info.set_corners(s.pn_front(), s.pn_back(), pn_c1, pn_c0)
                elif nc == 1:
                    if uv0 is None:
                        continue
                    d0 = uv_front - uv0
                    d1 = uv_back - uv0
                    ang = angle2d(d1, d0)
                    if is_very_sharp(ang) or favor_triangle:
                        if sub_uv_back is None or sub_uv_front is None:
                            continue
                        if np.linalg.norm(d0) < np.linalg.norm(d1):
                            info.add_uv_line(uv_front, sub_uv_back, rotate90(-d0), rotate90(-d1))
                            info.set_corners(s.pn_front(), s.pn_corner(0), s.sub_pn_back())
                        else:
                            info.add_uv_line(sub_uv_front, uv_back, rotate90(-d0), rotate90(-d1))
                            info.set_corners(s.sub_pn_front(), s.pn_corner(0), s.pn_back())
                    elif is_sharp(ang):
                        uv_diag = uv_front + uv_back - uv0
                        pn_diag = u2p(uv_diag)
                        if pn_diag is None:
                            continue
                        info.add_uv_line(uv_front, uv_diag)
                        info.add_uv_line(uv_back, uv_diag)
                        info.set_corners(s.pn_front(), s.pn_corner(0), s.pn_back(), pn_diag)
                elif nc == 2:
                    if uv0 is None or uv1 is None:
                        continue
                    d0 = normalized(uv0 - uv_front)
                    d1 = normalized(uv1 - uv0)
                    d2 = normalized(uv_back - uv1)
                    angle0 = angle2d(d1, -d0)
                    angle1 = angle2d(d2, -d1)
                    sharp0 = is_sharp(angle0)
                    sharp1 = is_sharp(angle1)
                    r = float(np.linalg.norm(uv1 - uv0))
                    uv_c0 = uv0 + r * rotate90(d1)
                    uv_c1 = uv1 + r * rotate90(d1)
                    if sharp1:
                        uv_c0 = uv0 + r * d2
                    if sharp0:
                        uv_c1 = uv1 - r * d0
                    pn_c0 = u2p(uv_c0)
                    pn_c1 = u2p(uv_c1)
                    if pn_c0 is None or pn_c1 is None:
                        continue
                    if sharp0 and sharp1:
                        info.add_uv_line(uv_front, uv_back)
                        info.set_corners(s.pn_front(), s.pn_corner(0), s.pn_corner(1), s.pn_back())
                    elif not sharp0 and sharp1:
                        if sub_uv_back is None:
                            continue
                        if is_very_sharp(angle1) or favor_triangle:
                            info.add_uv_line(uv0, sub_uv_back, rotate90(d1), rotate90(-d2))
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), s.sub_pn_back())
                        else:
                            info.add_uv_line(uv0, uv_c0)
                            info.add_uv_line(sub_uv_back, uv_c0)
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), s.sub_pn_back(), pn_c0)
                    elif sharp0 and not sharp1:
                        if sub_uv_front is None:
                            continue
                        if is_very_sharp(angle0) or favor_triangle:
                            info.add_uv_line(uv1, sub_uv_front, rotate90(d1), rotate90(-d0))
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), s.sub_pn_front())
                        else:
                            info.add_uv_line(uv1, uv_c1)
                            info.add_uv_line(sub_uv_front, uv_c1)
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), pn_c1, s.sub_pn_front())
                    else:
                        info.add_uv_line(uv0, uv_c0)
                        info.add_uv_line(uv1, uv_c1)
                        info.add_uv_line(uv_c0, uv_c1)
                        info.set_corners(s.pn_corner(0), s.pn_corner(1), pn_c1, pn_c0)
                elif nc == 3:
                    if uv0 is None or uv1 is None or uv2 is None:
                        continue
                    d0 = uv0 - uv_front
                    d1 = uv1 - uv0
                    d2 = uv2 - uv1
                    d3 = uv_back - uv2
                    angle0 = angle2d(d1, -d0)
                    angle1 = angle2d(d2, -d1)
                    angle2 = angle2d(d3, -d2)
                    sharp0 = is_sharp(angle0)
                    sharp1 = is_sharp(angle1)
                    sharp2 = is_sharp(angle2)
                    if not sharp1 or (sharp0 and sharp2):
                        continue
                    if is_very_sharp(angle1) or favor_triangle:
                        info.add_uv_line(uv0, uv2, rotate90(d1), rotate90(-d2))
                        info.set_corners(s.pn_corner(0), s.pn_corner(1), s.pn_corner(2))
                    elif sharp0:
                        if sub_uv_front is None:
                            continue
                        info.add_uv_line(sub_uv_front, uv2)
                        info.set_corners(s.sub_pn_front(), s.pn_corner(0), s.pn_corner(1), s.pn_corner(2))
                    elif sharp2:
                        if sub_uv_back is None:
                            continue
                        info.add_uv_line(sub_uv_back, uv0)
                        info.set_corners(s.sub_pn_back(), s.pn_corner(0), s.pn_corner(1), s.pn_corner(2))
                    else:
                        uv_c = uv0 + uv2 - uv1
                        pn_c = u2p(uv_c)
                        if pn_c is None:
                            continue
                        info.add_uv_line(uv0, uv_c)
                        info.add_uv_line(uv2, uv_c)
                        info.set_corners(s.pn_corner(0), s.pn_corner(1), s.pn_corner(2), pn_c)
                elif nc == 4:
                    if uv0 is None or uv1 is None or uv2 is None or uv3 is None:
                        continue
                    d0 = uv1 - uv0
                    d1 = uv2 - uv1
                    d2 = uv3 - uv2
                    angle0 = angle2d(d1, -d0)
                    angle1 = angle2d(d2, -d1)
                    sharp0 = is_sharp(angle0)
                    sharp1 = is_sharp(angle1)
                    if sharp0 and sharp1:
                        info.add_uv_line(uv0, uv3, rotate90(d0), rotate90(-d2))
                        info.set_corners(s.pn_corner(0), s.pn_corner(1), s.pn_corner(2), s.pn_corner(3))
                    elif sharp0:
                        if is_very_sharp(angle0) or favor_triangle:
                            info.add_uv_line(uv0, uv2, rotate90(d0), rotate90(-d1))
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), s.pn_corner(2))
                        else:
                            uv_c = uv0 + uv2 - uv1
                            pn_c = u2p(uv_c)
                            if pn_c is None:
                                continue
                            info.add_uv_line(uv0, uv_c)
                            info.add_uv_line(uv2, uv_c)
                            info.set_corners(s.pn_corner(0), s.pn_corner(1), s.pn_corner(2), pn_c)
                    elif sharp1:
                        if is_very_sharp(angle1) or favor_triangle:
                            info.add_uv_line(uv1, uv3, rotate90(d1), rotate90(-d2))
                            info.set_corners(s.pn_corner(1), s.pn_corner(2), s.pn_corner(3))
                        else:
                            uv_c = uv1 + uv3 - uv2
                            pn_c = u2p(uv_c)
                            if pn_c is None:
                                continue
                            info.add_uv_line(uv1, uv_c)
                            info.add_uv_line(uv3, uv_c)
                            info.set_corners(s.pn_corner(1), s.pn_corner(2), s.pn_corner(3), pn_c)
            else:
                s0, s1 = sequences
                if s0.num_corners() < s1.num_corners():
                    s0, s1 = s1, s0
                nc0 = s0.num_corners()
                nc1 = s1.num_corners()
                s0_front = p2u(s0.pn_front())
                s0_back = p2u(s0.pn_back())
                s1_front = p2u(s1.pn_front())
                s1_back = p2u(s1.pn_back())
                if s0_front is None or s0_back is None or s1_front is None or s1_back is None:
                    continue
                s0_uv0 = p2u(s0.get_corner(0).vertex.pn) if nc0 == 2 else None
                s0_uv1 = p2u(s0.get_corner(1).vertex.pn) if nc0 == 2 else None
                s1_uv0 = p2u(s1.get_corner(0).vertex.pn) if nc1 == 2 else None
                s1_uv1 = p2u(s1.get_corner(1).vertex.pn) if nc1 == 2 else None
                if nc0 == 0 and nc1 == 0:
                    d0 = s0_back - s0_front
                    d1 = s1_back - s1_front
                    info.add_uv_line(s0_front, s1_back, rotate90(d0), rotate90(-d1))
                    info.add_uv_line(s0_back, s1_front, rotate90(d0), rotate90(-d1))
                    info.set_corners(s0.pn_front(), s0.pn_back(), s1.pn_front(), s1.pn_back())
                elif nc0 == 2 and nc1 == 0:
                    if s0_uv0 is None or s0_uv1 is None:
                        continue
                    d0 = normalized(s0_uv0 - s0_front)
                    d1 = normalized(s0_uv1 - s0_uv0)
                    d2 = normalized(s0_back - s0_uv1)
                    if is_sharp(angle2d(d1, -d0)) or is_sharp(angle2d(d2, -d1)):
                        continue
                    e0 = s0_uv1 - s0_uv0
                    e1 = s1_back - s1_front
                    info.add_uv_line(s0_uv0, s1_back, rotate90(e0), rotate90(-e1))
                    info.add_uv_line(s0_uv1, s1_front, rotate90(e0), rotate90(-e1))
                    info.set_corners(s0.pn_corner(0), s0.pn_corner(1), s1.pn_front(), s1.pn_back())
                elif nc0 == 2 and nc1 == 2:
                    if s0_uv0 is None or s0_uv1 is None or s1_uv0 is None or s1_uv1 is None:
                        continue
                    s0_d0 = normalized(s0_uv0 - s0_front)
                    s0_d1 = normalized(s0_uv1 - s0_uv0)
                    s0_d2 = normalized(s0_back - s0_uv1)
                    s1_d0 = normalized(s1_uv0 - s1_front)
                    s1_d1 = normalized(s1_uv1 - s1_uv0)
                    s1_d2 = normalized(s1_back - s1_uv1)
                    if (is_sharp(angle2d(s0_d1, -s0_d0)) or is_sharp(angle2d(s0_d2, -s0_d1)) or
                            is_sharp(angle2d(s1_d1, -s1_d0)) or is_sharp(angle2d(s1_d2, -s1_d1))):
                        continue
                    e0 = s0_uv1 - s0_uv0
                    e1 = s1_uv1 - s1_uv0
                    info.add_uv_line(s0_uv0, s1_uv1, rotate90(e0), rotate90(-e1))
                    info.add_uv_line(s0_uv1, s1_uv0, rotate90(e0), rotate90(-e1))
                    info.set_corners(s0.pn_corner(0), s0.pn_corner(1), s1.pn_corner(0), s1.pn_corner(1))
            if info.uv_lines:
                result.append(info)
        return result

    def _uv_of_vertex(self, v, cache):
        key = id(v)
        if key not in cache:
            cache[key] = self.pn_to_uv(v.pn)
        return cache[key]

    # ---- cylinder sketching -----------------------------------------------------------------------------
    def add_cylinder_curve(self, cylinder_curve):
        cn = self.curvenetwork
        cn.set_flag_halfedges('SUBST', 0)
        for c in cn.halfchains:
            if c.patch is not None:
                continue
            if c.halfedge_front.flag:
                continue
            c_front, c_back = trace_halfchains(c)
            if c_front is not None and c_front.prev() is c_back:
                num_corner = 0
                h = c.halfedge_front
                while True:
                    h.flag = 1
                    if h.vertex.is_corner():
                        num_corner += 1
                    h = h.next
                    if h is c.halfedge_front or h is None:
                        break
                if num_corner == 0:
                    h = c.halfedge_front
                    while True:
                        h.flag = 2
                        h = h.next
                        if h is c.halfedge_front or h is None:
                            break

        h_snapped = [None, None]
        for i in range(2):
            pn_end = cylinder_curve.front() if i == 0 else cylinder_curve.back()
            if i == 0:
                d = cylinder_curve[1][:3] - cylinder_curve[0][:3]
            else:
                d = cylinder_curve[-2][:3] - cylinder_curve[-1][:3]
            d = normalized(d)
            dist_min = float('inf')
            for h in cn.halfedges:
                if h.is_deleted or not h.flag:
                    continue
                if h.flag == 1 and not h.vertex.is_corner():
                    continue
                dist = pn_norm(h.vertex.pn - pn_end)
                dp = dot3(cross3(h.to_vector(), d), pn_end[3:])
                if dist < dist_min and dist < self.configTemp.snapSize() and dp > 0:
                    dist_min = dist
                    h_snapped[i] = h
        if h_snapped[0] is None or h_snapped[1] is None:
            return

        num_corners = [0, 0]
        for i in range(2):
            h = h_snapped[i]
            while True:
                if h is h_snapped[(i + 1) % 2]:
                    return   # two halfedges are on the same edge flow!
                if h.vertex.is_corner():
                    num_corners[i] += 1
                h = h.next
                if h is h_snapped[i] or h is None:
                    break
        if num_corners[0] and num_corners[1] and num_corners[0] != num_corners[1]:
            return

        loop_curve = [Polyline(), Polyline()]
        for i in range(2):
            h = h_snapped[i]
            while True:
                loop_curve[i].push_back(h.vertex.pn)
                h = h.next if i == 0 else h.prev
                if h is h_snapped[i] or h is None:
                    break
            loop_curve[i].is_loop = True

        if num_corners[0] > num_corners[1]:
            cylinder_curve.points.reverse()
            h_snapped.reverse()
            num_corners.reverse()
            loop_curve.reverse()
            loop_curve[0].reverse()
            loop_curve[1].reverse()

        geodesic_endpoint = [[], []]
        if num_corners[0] > 0:
            for i in range(2):
                h = h_snapped[i].next if i == 0 else h_snapped[i].prev
                while True:
                    if h.vertex.is_corner():
                        geodesic_endpoint[i].append(h.vertex.pn.copy())
                    if h is h_snapped[i]:
                        break
                    h = h.next if i == 0 else h.prev
        elif num_corners[1] > 0:
            arc = [0.0] * num_corners[1]
            c = h_snapped[1].halfchain
            for i in range(num_corners[1]):
                arc[i] = (0 if i == 0 else arc[i - 1]) + c.length()
                c = c.prev()
            for i in range(num_corners[1]):
                arc[i] /= arc[-1]
            for i in range(num_corners[1]):
                geodesic_endpoint[0].append(loop_curve[0].point_at(arc[i]))
            h = h_snapped[1].prev
            while True:
                if h.vertex.is_corner():
                    geodesic_endpoint[1].append(h.vertex.pn.copy())
                if h is h_snapped[1]:
                    break
                h = h.prev
        else:
            nd = self.configTemp.cylinder_num_div
            for i in range(2):
                for j in range(1, nd + 1):
                    geodesic_endpoint[i].append(loop_curve[i].point_at(j / float(nd)))
        if len(geodesic_endpoint[0]) != len(geodesic_endpoint[1]) or not geodesic_endpoint[0]:
            return
        geodesic_endpoint[0].pop()
        geodesic_endpoint[1].pop()

        self.memento_store()
        for pn_from, pn_to in zip(geodesic_endpoint[0], geodesic_endpoint[1]):
            geodesic_curve = self.geodesic_compute(pn_from, pn_to)
            if geodesic_curve.empty():
                continue
            geodesic_curve.resample_by_length(self.configTemp.segmentSize())
            self.project_polyline(geodesic_curve)
            self.add_curve_open_delegate(geodesic_curve)
        self.add_curve_open_delegate(cylinder_curve)

        for i in range(2):
            if i == 0:
                loop_curve[i].reverse()
            if not self.is_loop_ccw(loop_curve[i]):
                continue
            h_start = h_snapped[i].opposite
            h = h_start
            while h is not None:
                if h.vertex.is_corner():
                    h.is_corner = True
                h = h.next
                if h is h_start:
                    break

    # ---- laser cutting -----------------------------------------------------------------------------------
    def trace_on_plane(self, pn_start, plane_normal, dist_max):
        bm = self.basemesh
        hit = self.intersect(pn_start)
        if hit is None:
            return Polyline()
        p_start = pn_start[:3].copy()
        nrm = np.asarray(plane_normal, dtype=float)
        V = bm.verts

        def plane_func(p):
            return float(nrm @ (p - p_start))

        vals = (V - p_start) @ nrm

        def is_crossing(h):
            return vals[bm.he_from[h]] * vals[bm.he_to[h]] <= 0

        h_start = -1
        for h in bm.fh(hit.face):
            if is_crossing(h):
                h_start = h
                break
        if h_start < 0:
            # robustness addition: the plane may pass exactly through the hit face's corner(s)
            for v in bm.tris[hit.face].tolist():
                for hv in bm.voh(v):
                    f = bm.he_face[hv]
                    if f < 0:
                        continue
                    for h in bm.fh(int(f)):
                        if is_crossing(h):
                            h_start = h
                            break
                    if h_start >= 0:
                        break
                if h_start >= 0:
                    break
        if h_start < 0:
            return Polyline()
        h = h_start
        for _ in range(4 * bm.n_halfedges + 4):
            h1 = int(bm.he_next[h])
            h2 = int(bm.he_next[h1])
            h_next = h1 if is_crossing(h1) else h2
            h_next = int(bm.he_opp[h_next])
            if bm.is_boundary_h(h_next):
                h_start = int(bm.he_opp[h_next])
                break
            elif h_next == h_start:
                break
            h = h_next
        else:
            return Polyline()

        result = Polyline()
        dist_sum = 0.0
        h = h_start
        for _ in range(4 * bm.n_halfedges + 4):
            v0 = int(bm.he_from[h])
            v1 = int(bm.he_to[h])
            pn0 = bm.get_pn(v0)
            pn1 = bm.get_pn(v1)
            t0 = vals[v0]
            t1 = vals[v1]
            u = -t0 / (t1 - t0) if t1 != t0 else 0.5
            pn = (1 - u) * pn0 + u * pn1
            pn_normalize(pn)
            self.project(pn)
            if not result.empty():
                dist_sum += pn_norm(result.back() - pn)
                if dist_max < dist_sum:
                    return Polyline()
            result.push_back(pn)
            h1 = int(bm.he_next[h])
            h2 = int(bm.he_next[h1])
            h_next = h1 if is_crossing(h1) else h2
            h_next = int(bm.he_opp[h_next])
            if bm.is_boundary_h(h_next) or h_next == h_start:
                if h_next == h_start:
                    result.is_loop = True
                result.insert_points_per_segment(9)
                self.project_polyline(result)
                return result
            h = h_next
        return Polyline()

    # ---- edge loops --------------------------------------------------------------------------------------
    def _edgeloop_find_start(self, patch_start, h_start, t):
        patch = patch_start
        h = h_start
        for _ in range(100000):
            edge_loop = find_edgeloop(patch, h)
            h_opposite = edge_loop[0] ^ 1
            patch_next = patch.opposite_patch(h_opposite)
            h_next = patch.opposite_boundary_halfedge(h_opposite)
            if patch_next is None or patch_next.is_imaginary():
                return patch, h_opposite, 1 - t, False
            elif patch_next is patch_start and h_next == h_start:
                return patch_start, h_start, t, True
            patch = patch_next
            h = h_next
        return patch_start, h_start, t, True

    def edgeLoop_insert(self, patch_start, h_start, t):
        if patch_start is None or h_start < 0 or not patch_start.is_boundary_h(h_start) or t <= 0 or 1 <= t:
            return
        self.memento_store()
        patch_start, h_start, t, _ = self._edgeloop_find_start(patch_start, h_start, t)
        e_affected = []
        patch_affected = []
        patch = patch_start
        h = h_start

        def func(mesh, h_old, v_new):
            v0, v1 = mesh.util_halfedge_to_vertex_pair(h_old)
            pn = (1 - t) * mesh.pn[v0] + t * mesh.pn[v1]
            pn_normalize(pn)
            self.project(pn)
            mesh.pn[v_new] = pn

        for _ in range(100000):
            edge_loop = find_edgeloop(patch, h)
            h_opposite = edge_loop[0] ^ 1
            if patch.h_chain[h] is not None:
                e_affected.append(patch.h_chain[h].edgechain)
            if patch.h_chain[h_opposite] is not None:
                e_affected.append(patch.h_chain[h_opposite].edgechain)
            patch_affected.append(patch)
            patch_next = patch.opposite_patch(h_opposite)
            h_next = patch.opposite_boundary_halfedge(h_opposite)
            insert_edgeloop(patch, h, t, func)
            if patch_next is None or patch_next.is_imaginary() or (patch_next is patch_start and h_next == h_start):
                break
            patch = patch_next
            h = h_next
        for e in _unique(e_affected):
            e.num_subdiv += 1
        for p in _unique(patch_affected):
            p.set_halfedgeData()
            p.invalidate_displist()
        self.curvenetwork.invalidate_displist()

    def edgeLoop_preview(self, patch_start, h_start, t):
        result = Polyline()
        if patch_start is None or h_start < 0 or not patch_start.is_boundary_h(h_start) or t <= 0 or 1 <= t:
            return result
        patch_start, h_start, t, _ = self._edgeloop_find_start(patch_start, h_start, t)
        patch = patch_start
        h = h_start
        for _ in range(100000):
            edge_loop = find_edgeloop(patch, h)
            for hh in reversed(edge_loop):
                v0, v1 = patch.util_halfedge_to_vertex_pair(hh)
                pn = (1 - t) * patch.pn[v0] + t * patch.pn[v1]
                pn_normalize(pn)
                result.push_back(pn)
            h_opposite = edge_loop[0] ^ 1
            patch_next = patch.opposite_patch(h_opposite)
            h_next = patch.opposite_boundary_halfedge(h_opposite)
            if patch_next is None or patch_next.is_imaginary() or (patch_next is patch_start and h_next == h_start):
                break
            patch = patch_next
            h = h_next
        return result

    def edgeLoop_walk(self, patch_start, h_start):
        result = []
        if patch_start is None or h_start < 0 or not patch_start.is_boundary_h(h_start):
            return result
        patch_start, h_start, _, _ = self._edgeloop_find_start(patch_start, h_start, 0.5)
        patch = patch_start
        h = h_start
        for _ in range(100000):
            result.append((patch, h))
            edge_loop = find_edgeloop(patch, h)
            h_opposite = edge_loop[0] ^ 1
            patch_next = patch.opposite_patch(h_opposite)
            h_next = patch.opposite_boundary_halfedge(h_opposite)
            if patch_next is None or patch_next.is_imaginary():
                result.append((patch, h_opposite))
                break
            if patch_next is patch_start and h_next == h_start:
                break
            patch = patch_next
            h = h_next
        return result

    # ---- vertex moving / smoothing ---------------------------------------------------------------------
    def _on_symmetry_boundary(self, patch, v):
        h = patch.halfedge_handle(v)
        for c in (patch.h_chain[h], patch.h_chain[patch.prev_halfedge_handle(h)]):
            if c is None:
                continue
            opp = c.opposite()
            if opp is not None and opp.patch is not None and opp.patch.is_imaginary_symmetry():
                return True
        return False

    def vertices_move(self, pn_center, center_offset, radius):
        if radius <= 0:
            return
        for patch in self.curvenetwork.patches:
            P = patch.pn
            if len(P) == 0:
                continue
            dist = np.sqrt(((P[:, :3] - pn_center[:3]) ** 2).sum(1))
            dot = P[:, 3:] @ pn_center[3:]
            sel = np.flatnonzero((dist <= radius) & (dot >= 0))
            if sel.size == 0:
                continue
            for v in sel.tolist():
                pn = patch.pn[v]
                pn[:3] += rbf_wendland(dist[v] / radius) * center_offset
                self.project(pn)
                if self.configSaved.symmetric and patch.is_boundary_v(v) and self._on_symmetry_boundary(patch, v):
                    pn[0] = 0.0
                    pn[3] = 0.0
            patch.invalidate_displist()
        affected = []
        for v in self.curvenetwork.vertices:
            dist = pn_norm(pn_center - v.pn)
            dot = dot3(pn_center[3:], v.pn[3:])
            if dist > radius or dot < 0:
                continue
            for c in voc_iter(v):
                if c is not None:
                    affected.append(c.edgechain)
        for e in _unique(affected):
            for i in range(2):
                c = e.halfchain[i]
                if c is not None and Patch.is_ordinary(c.patch):
                    self.set_halfchain_pn_from_patch(c)
                    break

    def vertices_smooth(self, pn_center, radius):
        if radius <= 0:
            return
        affected = []
        temps = []
        for patch in self.curvenetwork.patches:
            P = patch.pn
            pn_temp = P.copy()
            temps.append(pn_temp)
            if len(P) == 0:
                continue
            dist = np.sqrt(((P[:, :3] - pn_center[:3]) ** 2).sum(1))
            dot = P[:, 3:] @ pn_center[3:]
            sel = np.flatnonzero((dist <= radius) & (dot >= 0))
            for v in sel.tolist():
                average = self.vertices_smooth_sub(patch, v, affected)
                if average is None:
                    continue
                falloff = rbf_wendland(dist[v] / radius)
                t = (1 - falloff) * P[v] + falloff * average
                pn_normalize(t)
                self.project(t)
                pn_temp[v] = t
            if sel.size:
                patch.invalidate_displist()
        for patch, pn_temp in zip(self.curvenetwork.patches, temps):
            patch.pn[:] = pn_temp
        for e in _unique(affected):
            for i in range(2):
                c = e.halfchain[i]
                if c is not None and Patch.is_ordinary(c.patch):
                    self.set_halfchain_pn_from_patch(c)
                    break

    def vertices_smooth_sub(self, patch, patch_v, affected):
        sum_pn = np.zeros(6)
        sum_weight = 0.0
        if patch.is_boundary_v(patch_v):
            if patch.is_failure:
                return None
            v = patch.vertex_patch_to_curveNetwork(patch_v)
            if v is not None:
                if v.is_boundary():
                    return None
                for c in voc_iter(v):
                    if c is None:
                        continue
                    c_patch = c.patch
                    if c_patch is None or c_patch.is_imaginary() or c_patch.is_failure:
                        continue
                    pv2 = c_patch.vertex_curveNetwork_to_patch(v)
                    if pv2 < 0:
                        continue
                    for w in c_patch.vv(pv2):
                        weight = 0.5 if c_patch.is_boundary_v(w) else 1.0
                        sum_pn += weight * c_patch.pn[w]
                        sum_weight += weight
                    affected.append(c.edgechain)
            else:
                patch_h = patch.prev_halfedge_handle(patch.halfedge_handle(patch_v))
                c = patch.h_chain[patch_h]
                if c is None:
                    return None
                affected.append(c.edgechain)
                pn_prev = patch.pn[patch.from_vertex_handle(patch_h)]
                pn_next = patch.pn[patch.to_vertex_handle(patch.next_halfedge_handle(patch_h))]
                sum_pn += pn_prev + pn_next
                sum_weight += 2
                patch_opposite = patch.opposite_patch(patch_h)
                if Patch.is_ordinary(patch_opposite):
                    for w in patch.vv(patch_v):
                        if patch.is_boundary_v(w):
                            continue
                        sum_pn += patch.pn[w]
                        sum_weight += 1
                    poh = patch.opposite_boundary_halfedge(patch_h)
                    if poh >= 0:
                        pov = patch_opposite.from_vertex_handle(poh)
                        for w in patch_opposite.vv(pov):
                            if patch_opposite.is_boundary_v(w):
                                continue
                            sum_pn += patch_opposite.pn[w]
                            sum_weight += 1
        else:
            for w in patch.vv(patch_v):
                sum_pn += patch.pn[w]
                sum_weight += 1
        if sum_weight == 0:
            return None
        return sum_pn / sum_weight

    def vertices_smooth_global(self):
        temps = []
        for patch in self.curvenetwork.patches:
            P = patch.pn
            pn_temp = P.copy()
            for v in patch.vertices():
                average = self.vertices_smooth_sub(patch, v, [])
                if average is None:
                    continue
                pn_normalize(average)
                self.project(average)
                pn_temp[v] = average
            temps.append(pn_temp)
            patch.invalidate_displist()
        for patch, pn_temp in zip(self.curvenetwork.patches, temps):
            patch.pn[:] = pn_temp
        for e in self.curvenetwork.edgechains:
            for i in range(2):
                c = e.halfchain[i]
                if c is not None and Patch.is_ordinary(c.patch):
                    self.set_halfchain_pn_from_patch(c)
                    break

    def set_halfchain_pn_from_patch(self, c):
        patch = c.patch
        if not Patch.is_ordinary(patch):
            return
        corner0 = patch.patch_corner(0)
        if corner0 < 0:
            return
        h_start = patch.prev_halfedge_handle(patch.halfedge_handle(corner0))
        ph = h_start
        while True:
            if patch.h_chain[ph] is c:
                break
            ph = patch.prev_halfedge_handle(ph)
            if ph == h_start:
                return
        curve = Polyline()
        curve.push_back(patch.pn[patch.to_vertex_handle(ph)])
        for _ in range(c.num_subdiv()):
            curve.push_back(patch.pn[patch.from_vertex_handle(ph)])
            ph = patch.prev_halfedge_handle(ph)
        n = c.to_polyline().size() - 1
        h = c.halfedge_front
        h.from_vertex().pn = curve.front().copy()
        for i in range(1, n + 1):
            t = i / float(n)
            h.vertex.pn = curve.point_at(t)
            self.project(h.vertex.pn)
            h = h.next
            if h is None:
                break
        self.curvenetwork.invalidate_displist()

    # ---- geometric snakes --------------------------------------------------------------------------------
    def _snake_gradient(self, pn):
        hit = self.intersect(pn)
        if hit is None:
            return np.zeros(3)
        g = np.zeros(3)
        tri = self.basemesh.tris[hit.face]
        for i in range(3):
            g += hit.bc(i) * self.basemesh.snake_energy_gradient[tri[i]]
        return g

    def snakes_move_edgechain(self, e):
        c = e.halfchain[0]
        ct = self.configTemp
        wsum = ct.snakes_internal1 + ct.snakes_internal2 + ct.snakes_external
        result = []
        h = c.halfedge_front.next
        hs = []
        while h is not None and h.next is not None and h.next is not c.halfedge_back:
            pn_m2 = h.prev.from_vertex().pn
            pn_m1 = h.prev.vertex.pn
            pn_0 = h.vertex.pn.copy()
            pn_p1 = h.next.vertex.pn
            pn_p2 = h.next.next.vertex.pn
            g1 = ((pn_m1 + pn_p1) / 2.0 - pn_0)[:3]
            g2 = ((-pn_m2 + 4 * pn_m1 + 4 * pn_p1 - pn_p2) / 6.0 - pn_0)[:3]
            ge = self._snake_gradient(pn_0)
            gradient = (ct.snakes_internal1 * g1 + ct.snakes_internal2 * g2 + ct.snakes_external * ge) / wsum
            pn_0[:3] += ct.snakes_damping * gradient
            self.project(pn_0)
            result.append(pn_0)
            hs.append(h)
            h = h.next
            if h is c.halfedge_front:
                break
        for hh, pn in zip(hs, result):
            hh.vertex.pn = pn

    def snakes_move_vertex(self, v):
        if self.intersect(v.pn) is None:
            return
        ct = self.configTemp
        ge = self._snake_gradient(v.pn)
        offset = ct.snakes_damping * ct.snakes_external / (ct.snakes_internal1 + ct.snakes_internal2 + ct.snakes_external) * ge

        def offset_func(pn, w):
            pn[:3] += w * offset
            self.project(pn)

        offset_func(v.pn, 1)
        for c in voc_iter(v):
            if c is None:
                continue
            n = c.to_polyline().size() - 1
            h = c.halfedge_front
            for i in range(1, n):
                offset_func(h.vertex.pn, 1.0 - i / float(n))
                h = h.next

    # ---- symmetry / boundaries ----------------------------------------------------------------------------
    def toggle_symmetric(self):
        """Flips the symmetry mode.  Discards all curves (the caller asks for confirmation)."""
        cn = self.curvenetwork
        bm = self.basemesh
        cn.clear()
        self.memento.init()
        self.configTemp.autoSave.unsaved = True
        self.configSaved.symmetric = not self.configSaved.symmetric
        if self.configSaved.symmetric:
            X = bm.verts[:, 0]
            hf = bm.he_from
            ht = bm.he_to

            def is_crossing(h):
                return X[hf[h]] * X[ht[h]] < 0

            def is_touching(h):
                a = X[hf[h]]
                b = X[ht[h]]
                return a * b == 0 and (a != 0 or b != 0)

            def is_coincide(h):
                return X[hf[h]] == 0 and X[ht[h]] == 0

            nxt = bm.he_next
            opp = bm.he_opp
            edge_visited = np.zeros(bm.n_edges, dtype=bool)
            candidates = np.flatnonzero((X[hf] * X[ht] < 0) | ((X[hf] == 0) & (X[ht] == 0)))
            limit = 4 * bm.n_halfedges + 4
            for h_iter in candidates.tolist():
                e_idx = bm.he_edge[h_iter]
                if edge_visited[e_idx]:
                    continue
                edge_visited[e_idx] = True
                h_start = h_iter
                is_loop = False
                h_current = h_start
                for _ in range(limit):
                    h_next = -1
                    if is_coincide(h_current):
                        v = int(ht[h_current])
                        for voh in bm.voh(v):
                            if opp[voh] == h_current:
                                continue
                            if is_coincide(voh):
                                h_next = voh
                                break
                            h2 = int(nxt[voh])
                            if h2 >= 0 and is_crossing(h2):
                                h_next = h2
                                break
                    else:
                        h_opposite = int(opp[h_current])
                        if bm.is_boundary_h(h_opposite):
                            h_start = h_opposite
                            break
                        h0 = int(nxt[h_opposite])
                        h1 = int(nxt[h0])
                        if is_touching(h0):
                            v = int(ht[h0])
                            for voh in bm.voh(v):
                                if is_coincide(voh):
                                    h_next = voh
                                    break
                                h2 = int(nxt[voh])
                                if h2 < 0 or opp[h2] == h_current:
                                    continue
                                if is_crossing(h2):
                                    h_next = h2
                                    break
                        else:
                            h_next = h0 if is_crossing(h0) else h1
                    if h_next == h_start:
                        is_loop = True
                        break
                    if h_next < 0:
                        h_start = int(opp[h_current])
                        break
                    h_current = h_next

                def vertex_by_v(bv):
                    pn = bm.get_pn(bv)
                    pn[0] = pn[3] = 0.0
                    pn_normalize(pn)
                    pn[:3] += self.configSaved.projOffset * pn[3:]
                    cv = cn.new_vertex()
                    cv.pn = pn
                    return cv

                def vertex_by_h(bh, t):
                    pn = (1 - t) * bm.get_pn(int(hf[bh])) + t * bm.get_pn(int(ht[bh]))
                    pn[0] = pn[3] = 0.0
                    pn_normalize(pn)
                    pn[:3] += self.configSaved.projOffset * pn[3:]
                    cv = cn.new_vertex()
                    cv.pn = pn
                    return cv

                verts = []
                if not is_loop:
                    if is_coincide(h_start):
                        verts.append(vertex_by_v(int(hf[h_start])))
                    elif not bm.is_boundary_h(h_start):
                        hh = int(nxt[h_start])
                        verts.append(vertex_by_v(int(ht[hh])))
                h_current = h_start
                for _ in range(limit):
                    edge_visited[bm.he_edge[h_current]] = True
                    h_next = -1
                    if is_coincide(h_current):
                        v = int(ht[h_current])
                        verts.append(vertex_by_v(v))
                        for voh in bm.voh(v):
                            if opp[voh] == h_current:
                                continue
                            if is_coincide(voh):
                                h_next = voh
                                break
                            h2 = int(nxt[voh])
                            if h2 >= 0 and is_crossing(h2):
                                h_next = h2
                                break
                    else:
                        a = X[hf[h_current]]
                        b = X[ht[h_current]]
                        t = -a / (b - a)
                        verts.append(vertex_by_h(h_current, t))
                        h_opposite = int(opp[h_current])
                        if bm.is_boundary_h(h_opposite):
                            break
                        h0 = int(nxt[h_opposite])
                        h1 = int(nxt[h0])
                        if is_touching(h0):
                            v = int(ht[h0])
                            for voh in bm.voh(v):
                                if is_coincide(voh):
                                    h_next = voh
                                    break
                                h2 = int(nxt[voh])
                                if h2 < 0 or h2 == h_opposite:
                                    continue
                                if is_crossing(h2):
                                    h_next = h2
                                    break
                        else:
                            h_next = h0 if is_crossing(h0) else h1
                    if h_next < 0:
                        if not is_coincide(h_current) and not bm.is_boundary_h(h_current):
                            hh = int(nxt[h_current])
                            verts.append(vertex_by_v(int(ht[hh])))
                        break
                    if h_next == h_start:
                        break
                    h_current = h_next
                if len(verts) < 2:
                    for cv in verts:
                        cv.is_deleted = True
                    continue

                # insert more between each pair of vertices
                pairs = len(verts) if is_loop else len(verts) - 1
                dense = []
                for k in range(len(verts)):
                    dense.append(verts[k])
                    if k < pairs:
                        a = verts[k]
                        b = verts[(k + 1) % len(verts)]
                        d = (b.pn - a.pn) / 10.0
                        pn = a.pn + d
                        for _ in range(9):
                            nv_ = cn.new_vertex()
                            nv_.pn = pn_normalized(pn)
                            dense.append(nv_)
                            pn = pn + d
                verts = dense
                d = verts[1].pn[:3] - verts[0].pn[:3]
                if cross3(d, verts[0].pn[3:])[0] < 0:
                    verts.reverse()
                fwd = []
                bwd = []
                pairs = len(verts) if is_loop else len(verts) - 1
                for k in range(pairs):
                    v0 = verts[k]
                    v1 = verts[(k + 1) % len(verts)]
                    h01 = cn.new_halfedge()
                    h10 = cn.new_halfedge()
                    set_vertex_halfedge(v0, v1, h01, h10)
                    fwd.append(h01)
                    bwd.append(h10)
                    h01.imaginary_patch = IMAGINARY_SYMMETRY
                nh = len(fwd)
                for k in range(nh):
                    fwd[k].next = None if (not is_loop and k == nh - 1) else fwd[(k + 1) % nh]
                    fwd[k].prev = None if (not is_loop and k == 0) else fwd[(k + nh - 1) % nh]
                    bwd[k].prev = None if (not is_loop and k == nh - 1) else bwd[(k + 1) % nh]
                    bwd[k].next = None if (not is_loop and k == 0) else bwd[(k + nh - 1) % nh]
            cn.garbage_collect()
        self.trace_basemesh_boundary()
        cn.invalidate_displist()

    def trace_basemesh_boundary(self):
        cn = self.curvenetwork
        bm = self.basemesh
        visited = np.zeros(bm.n_halfedges, dtype=bool)
        boundary = np.flatnonzero(bm.he_face < 0)
        for h in boundary.tolist():
            if visited[h]:
                continue
            curve = []
            i = h
            for _ in range(bm.n_halfedges + 1):
                visited[i] = True
                curve.append(bm.get_pn(int(bm.he_from[i])))
                i = int(bm.he_next[i])
                if i == h or i < 0:
                    break
            is_symmetric_open = False
            if self.configSaved.symmetric:
                if all(pn[0] < 0 for pn in curve):
                    continue
                n = len(curve)
                for k in range(n):
                    pn0 = curve[k]
                    pn1 = curve[(k + 1) % n]
                    if pn0[0] * pn1[0] >= 0:
                        continue
                    t = -pn0[0] / (pn1[0] - pn0[0])
                    pn = (1 - t) * pn0 + t * pn1
                    pn[0] = pn[3] = 0.0
                    pn_normalize(pn)
                    curve[k] = pn
                rotate_pos = -1
                for k in range(n):
                    kp = n - 1 if k == 0 else k - 1
                    if curve[kp][0] < 0 and curve[k][0] >= 0:
                        rotate_pos = k
                        break
                if rotate_pos != -1:
                    curve = curve[rotate_pos:] + curve[:rotate_pos]
                    curve = [pn for pn in curve if pn[0] >= 0]
                    is_symmetric_open = True
            for pn in curve:
                pn[:3] += self.configSaved.projOffset * pn[3:]
            n = len(curve)
            if n < 2:
                continue
            verts = []
            for k in range(n):
                pn = curve[k]
                if is_symmetric_open and (k == 0 or k == n - 1):
                    best = None
                    best_d = float('inf')
                    for v in cn.vertices:
                        d = float(np.linalg.norm(pn - v.pn))
                        if d < best_d:
                            best_d = d
                            best = v
                    verts.append(best)
                else:
                    v = cn.new_vertex()
                    v.pn = pn.copy()
                    verts.append(v)
            if is_symmetric_open and (verts[0] is None or verts[-1] is None or verts[0].halfedge is None or verts[-1].halfedge is None):
                continue
            fwd = []
            bwd = []
            for k in range(n):
                if is_symmetric_open and k == n - 1:
                    continue
                v0 = verts[k]
                v1 = verts[k + 1 if k < n - 1 else 0]
                h01 = cn.new_halfedge()
                h10 = cn.new_halfedge()
                if is_symmetric_open and k == 0:
                    set_next_prev(v0.halfedge.opposite, h01)
                    set_next_prev(h10, v0.halfedge)
                    h01.prev.is_corner = True
                    h10.is_corner = True
                elif is_symmetric_open and k == n - 2:
                    set_next_prev(h01, v1.halfedge)
                    set_next_prev(v1.halfedge.opposite, h10)
                    h01.is_corner = True
                    h10.prev.is_corner = True
                set_vertex_halfedge(v0, v1, h01, h10)
                fwd.append(h01)
                bwd.append(h10)
                h01.imaginary_patch = IMAGINARY_BOUNDARY
            nh = len(fwd)
            for k in range(nh):
                if not (is_symmetric_open and k == nh - 1):
                    fwd[k].next = fwd[(k + 1) % nh]
                    bwd[k].prev = bwd[(k + 1) % nh]
                if not (is_symmetric_open and k == 0):
                    fwd[k].prev = fwd[(k + nh - 1) % nh]
                    bwd[k].next = bwd[(k + nh - 1) % nh]
        cn.generate_halfchains()
        self.generate_patches()

    # ---- geodesic (approximation of the exact geodesic used by the original) ----------------------------
    def geodesic_compute(self, pn0, pn1):
        bm = self.basemesh
        hit0 = self.intersect(pn0)
        hit1 = self.intersect(pn1)
        if hit0 is None or hit1 is None:
            return Polyline()
        pts, _ = bm.geodesic_path(hit0, hit1)
        if pts is None:
            return Polyline()
        pts = [np.array(p, dtype=float) for p in pts]
        # straighten the edge path on the surface (approaches the geodesic)
        if len(pts) > 2:
            for _ in range(60):
                new = [pts[0]]
                for k in range(1, len(pts) - 1):
                    q = 0.5 * pts[k] + 0.25 * (pts[k - 1] + pts[k + 1])
                    nh = bm.nearest(q)
                    new.append(bm.intersect_convert(nh)[:3] if nh is not None else q)
                new.append(pts[-1])
                pts = new
        result = Polyline()
        for p in pts:
            nh = bm.nearest(p)
            pn = bm.intersect_convert(nh) if nh is not None else None
            if pn is None:
                continue
            pn[:3] = p
            result.push_back(pn)
        return result

    # ---- common event handling (SketchRetopo_event.cc) ---------------------------------------------------
    # Camera navigation is handled by the host before these are called.
    def update_mouse_pn(self, mouse_x, mouse_y):
        self.common_mouse_pos_prev = self.common_mouse_pos
        self.common_mouse_pos = (mouse_x, mouse_y)
        pn = self.intersect_convert(self.intersect_mouse(mouse_x, mouse_y))
        if pn is not None:
            self.project(pn)
        self.common_pn_mouse = pn
        return pn

    def common_mouse_down(self, mouse_x, mouse_y, button, shift, ctrl, alt):
        if button == LEFT:
            self.common_button_left_pressed = True
        if button == RIGHT:
            self.common_button_right_pressed = True
        if button == MIDDLE:
            self.common_button_middle_pressed = True
        self.common_shift_pressed = shift
        self.common_ctrl_pressed = ctrl
        self.common_alt_pressed = alt
        ct = self.configTemp
        if button == RIGHT and shift and ctrl:
            ct.segmentSize.adjust_init(mouse_x)
            return True
        if button == RIGHT and shift:
            ct.snapSize.adjust_init(mouse_x)
            return True
        if button == RIGHT and ctrl:
            ct.quadSize.adjust_init(mouse_x)
            return True
        if button == LEFT and 'h' in self.common_key_pressed:
            self.hide_stroke.clear()
            self.hide_stroke.push_back((mouse_x, mouse_y))
            return True
        curve_level = self.state.edit_level == 'CURVE_NETWORK'
        if button == LEFT and shift and ctrl and curve_level:
            self.common_dragMode = 'Snakes'
            self.memento_store()
            return True
        if button == LEFT and shift and curve_level:
            self.common_dragMode = 'Smooth'
            self.memento_store()
            return True
        if button == LEFT and self.common_pn_mouse is not None and self.state in (
                self.stateSketch, self.stateSpine, self.stateAutocmpl, self.stateCylinder, self.stateDeformCurve):
            self.common_sketch_curve.push_back(self.common_pn_mouse)
        return False

    def common_mouse_up(self, mouse_x, mouse_y, button, shift, ctrl, alt):
        if button == LEFT:
            self.common_button_left_pressed = False
        if button == RIGHT:
            self.common_button_right_pressed = False
        if button == MIDDLE:
            self.common_button_middle_pressed = False
        self.common_shift_pressed = shift
        self.common_ctrl_pressed = ctrl
        self.common_alt_pressed = alt
        ct = self.configTemp
        if ct.snapSize.is_adjusting():
            ct.snapSize.adjust_done()
            return True
        if ct.quadSize.is_adjusting():
            ct.quadSize.adjust_done()
            Patch.quadSize = ct.quadSize()
            return True
        if ct.segmentSize.is_adjusting():
            ct.segmentSize.adjust_done()
            return True
        if not self.hide_stroke.empty():
            self.hide_mesh_by_stroke()
            self.hide_stroke.clear()
            return True
        if self.common_dragMode in ('Smooth', 'Snakes'):
            affected_patches = []
            for e in _unique(self.common_affected_edgechains):
                for i in range(2):
                    c = e.halfchain[i]
                    if c is None:
                        continue
                    patch = c.patch
                    if patch is None or patch.is_imaginary():
                        continue
                    affected_patches.append(patch)
            affected_patches = _unique(affected_patches)
            for patch in affected_patches:
                if not patch.is_deleted:
                    patch.clear()
            for patch in affected_patches:
                if patch.is_deleted:
                    continue
                patch.generate_topology(False)
                self.compute_patch_interior_pn(patch)
            self.curvenetwork.invalidate_displist()
            self.common_affected_edgechains = []
            self.common_dragMode = None
            return True
        if not self.common_sketch_curve.empty():
            c = self.common_sketch_curve
            c.is_loop = pn_norm(c.back() - c.front()) < ct.snapSize()
            c.resample_by_length(ct.segmentSize())
            c.smooth(2, 0.5, 0.5)
            self.project_polyline(c)
        return False

    def common_mouse_move(self, mouse_x, mouse_y):
        ct = self.configTemp
        if ct.snapSize.is_adjusting():
            ct.snapSize.adjust_move(mouse_x)
            if self.state is self.stateSpine:
                ct.snapSize.value = min(ct.snapSize.value, ct.brushSize_spine.value * 0.9)
            if self.state is self.stateAutocmpl:
                ct.snapSize.value = min(ct.snapSize.value, ct.brushSize_autocmpl.value * 0.9)
            if self.state is self.stateMoveVertex:
                ct.snapSize.value = min(ct.snapSize.value, ct.brushSize_moveVertex.value * 0.9)
            return True
        for brush in (ct.brushSize_spine, ct.brushSize_autocmpl, ct.brushSize_moveVertex):
            if brush.is_adjusting():
                brush.adjust_move(mouse_x)
                brush.value = max(brush.value, ct.snapSize.value * 0.9)
                return True
        if ct.quadSize.is_adjusting():
            ct.quadSize.adjust_move(mouse_x)
            return True
        if ct.segmentSize.is_adjusting():
            ct.segmentSize.adjust_move(mouse_x)
            return True
        if not self.hide_stroke.empty():
            self.hide_stroke.push_back((mouse_x, mouse_y))
            return True

        pn_mouse = self.update_mouse_pn(mouse_x, mouse_y)
        if pn_mouse is None:
            return True
        cn = self.curvenetwork

        if self.common_dragMode == 'Smooth':
            best = None
            best_d = float('inf')
            for v in cn.vertices:
                if v.is_corner() or v.is_openend():
                    continue
                d = pn_norm(v.pn - pn_mouse)
                if d < best_d:
                    best_d = d
                    best = v
            if best is not None and best_d < ct.snapSize():
                c = best.halfedge.halfchain
                if c is None:
                    return True
                polyline = c.to_polyline()
                n = polyline.size()
                if not polyline.is_loop and n < 5:
                    return True
                polyline.smooth(25, ct.snakes_internal1, ct.snakes_internal2, ct.snakes_damping)
                h = c.halfedge_front
                for i in range(n):
                    pn = h.from_vertex().pn
                    pn[:] = polyline[i]
                    self.project(pn)
                    h = h.next
                    if h is None:
                        break
                self.common_affected_edgechains.append(c.edgechain)
                cn.invalidate_displist()
            return True

        if self.common_dragMode == 'Snakes':
            v_corner = None
            v_corner_d = float('inf')
            v_noncorner = None
            v_noncorner_d = float('inf')
            for v in cn.vertices:
                if self.configSaved.symmetric and v.on_symmetry_plane():
                    continue
                d = pn_norm(v.pn - pn_mouse)
                if ct.snapSize() < d:
                    continue
                if v.is_corner() or v.is_openend():
                    if d < v_corner_d:
                        v_corner_d = d
                        v_corner = v
                else:
                    if d < v_noncorner_d:
                        v_noncorner_d = d
                        v_noncorner = v
            if v_corner is not None:
                for _ in range(5):
                    self.snakes_move_vertex(v_corner)
                for c in voc_iter(v_corner):
                    if c is not None:
                        self.common_affected_edgechains.append(c.edgechain)
                cn.invalidate_displist()
            elif v_noncorner is not None and v_noncorner.halfedge.halfchain is not None:
                e = v_noncorner.halfedge.halfchain.edgechain
                for _ in range(5):
                    self.snakes_move_edgechain(e)
                self.common_affected_edgechains.append(e)
                cn.invalidate_displist()
            return True

        if not self.common_sketch_curve.empty():
            self.common_sketch_curve.push_back(pn_mouse)
        return False

    MODE_KEYS = {'q': SKETCH, 'w': SPINE, 'e': AUTOCMPL, 'r': LASER, 't': CYLINDER, 'y': DEFORMCURVE,
                 'f': EDITCORNER, 'a': EDITTOPOLOGY, 's': EDGELOOP, 'd': MOVEVERTEX}

    def common_keyboard(self, key):
        """`key` is a character as GLUT delivers it ('B' for shift+b) or a control name such as 'ctrl+z'."""
        if key in self.MODE_KEYS:
            self.state_set(self.MODE_KEYS[key])
            return True
        if key in ('ctrl+y', 'ctrl+shift+z'):
            self.memento_redo()
            return True
        if key == 'ctrl+z':
            self.memento_undo()
            return True
        if key == ' ':
            cr = self.configRender
            cr.mode = ConfigRender.DEFAULT if cr.mode == ConfigRender.QUADMESH_ONLY else ConfigRender.QUADMESH_ONLY
            return True
        if key == 'ctrl+b':
            cr = self.configRender
            cr.mode = ConfigRender.DEFAULT if cr.mode == ConfigRender.BASEMESH_ONLY else ConfigRender.BASEMESH_ONLY
            return True
        if key in ('ctrl+o', 'ctrl+s', 'ctrl+i', 'ctrl+e', 'ctrl+f', '0', '8', '6'):
            self.requests.append(key)
            return True
        if key == '9':
            self.generate_patch_debug()
            return True
        if key == '7':
            self.report(self.stats_report())
            return True
        if len(key) == 1:
            self.common_key_pressed.add(key)
        return False

    def common_keyboardup(self, key):
        self.common_key_pressed.discard(key)
        return False

    def stats_report(self):
        n_quad = 0
        n_tri = n_rect = n_pent = 0
        for p in self.curvenetwork.patches:
            n_quad += p.n_faces()
            nc = p.num_corners()
            if nc == 3:
                n_tri += 1
            elif nc == 4:
                n_rect += 1
            else:
                n_pent += 1
        return '# tri: %d  # quad: %d  # patch: %d/%d/%d' % (len(self.basemesh.tris) if self.basemesh else 0,
                                                            n_quad, n_tri, n_rect, n_pent)

    # ---- event entry points used by the host ------------------------------------------------------------
    def mouse_down(self, x, y, button, shift=False, ctrl=False, alt=False):
        self.update_mouse_pn(x, y)
        if not self.common_mouse_down(x, y, button, shift, ctrl, alt):
            self.state.mouse_down(x, y, button, shift, ctrl, alt)

    def mouse_up(self, x, y, button, shift=False, ctrl=False, alt=False):
        if not self.common_mouse_up(x, y, button, shift, ctrl, alt):
            self.state.mouse_up(x, y, button, shift, ctrl, alt)

    def mouse_move(self, x, y):
        if not self.common_mouse_move(x, y):
            self.state.mouse_move(x, y)

    def key_down(self, key):
        if not self.common_keyboard(key):
            self.state.keyboard(key)

    def key_up(self, key):
        if not self.common_keyboardup(key):
            self.state.keyboardup(key)

    def update_camera_distance(self):
        if self.view is None:
            return
        d = self.view.camera_distance()
        ct = self.configTemp
        for p in (ct.snapSize, ct.brushSize_spine, ct.brushSize_autocmpl, ct.brushSize_moveVertex):
            p.set_camera_distance(d)
        ct.quadSize.adjust_rate = d * 0.0002
        ct.segmentSize.adjust_rate = d * 0.0002

    # ---- hiding by stroke (SketchRetopo_misc.cc) ---------------------------------------------------------
    def hide_mesh_by_stroke(self):
        bm = self.basemesh
        cn = self.curvenetwork
        if self.hide_stroke.size() < 3:
            bm.face_hidden[:] = False
            for v in cn.vertices:
                v.is_hidden = False
            for p in cn.patches:
                p.is_hidden = False
        else:
            s = (self.view.width + self.view.height) * 0.001
            self.hide_stroke.resample_by_length(s)
            pts = self.hide_stroke.points
            centers = []
            values = []
            for k in range(len(pts) - 1):
                p = (pts[k] + pts[k + 1]) * 0.5
                d = normalized(rotate90(pts[k + 1] - pts[k])) * s
                centers.append(p + d)
                values.append(s)
                centers.append(p - d)
                values.append(-s)
            rbf = _ThinPlateRBF(np.array(centers), np.array(values))
            face_centers = bm.verts[bm.tris].mean(axis=1)
            screen = self.view.project(face_centers)
            bm.face_hidden[:] = rbf(screen) < 0
            if cn.vertices:
                vs = np.array([v.pn[:3] for v in cn.vertices])
                vval = rbf(self.view.project(vs))
                for v, val in zip(cn.vertices, vval):
                    v.is_hidden = bool(val < -1.5 * s)
            for p in cn.patches:
                p.is_hidden = True
                for c in pc_iter(p):
                    if any(not h.vertex.is_hidden for h in ch_iter(c)):
                        p.is_hidden = False
                        break
        bm.rebuild_bvh()
        cn.invalidate_displist()
        for p in cn.patches:
            p.invalidate_displist()

    # ---- state ------------------------------------------------------------------------------------------
    def state_set(self, enum_state):
        new = self._states[enum_state]
        if new is self.state:
            return
        self.state = new
        self.state.init()

    def state_get(self):
        return self._states.index(self.state)

    # ---- undo / redo ------------------------------------------------------------------------------------
    def memento_store(self):
        self.memento.store(self.curvenetwork.snapshot())
        self.configTemp.autoSave.unsaved = True

    def memento_undo(self):
        data = self.memento.undo(self.curvenetwork.snapshot())
        if data is not None:
            self.curvenetwork.restore(data)
            self.state.init()
            self.configTemp.autoSave.unsaved = True

    def memento_redo(self):
        data = self.memento.redo(self.curvenetwork.snapshot())
        if data is not None:
            self.curvenetwork.restore(data)
            self.state.init()
            self.configTemp.autoSave.unsaved = True

    # ---- export -----------------------------------------------------------------------------------------
    def retopo_mesh(self):
        """SketchRetopo::export_retopomesh: (vertices (N,3), faces list).  Patches are concatenated,
        mirrored when symmetric, and nearby vertices merged."""
        verts = []
        faces = []
        for patch in self.curvenetwork.patches:
            base = len(verts)
            verts.extend(patch.pn[:, :3].tolist())
            for f in patch.faces():
                faces.append([base + v for v in patch.fv(f)])
        if self.configSaved.symmetric:
            for patch in self.curvenetwork.patches:
                base = len(verts)
                for p in patch.pn[:, :3].tolist():
                    verts.append([-p[0], p[1], p[2]])
                for f in patch.faces():
                    faces.append([base + v for v in reversed(patch.fv(f))])
        V = np.array(verts, dtype=float).reshape(-1, 3)
        threshold = (self.basemesh.bbox_diagonal_norm() if self.basemesh else 1.0) * 0.0000001
        return merge_nearby_vertices(V, faces, threshold)


class _HalfedgeSequence:
    """Helper of SketchRetopo::suggest_autocmpl."""

    def __init__(self):
        self.h_front = None
        self.h_back = None

    def _walk(self):
        h = self.h_front
        guard = 0
        while h is not None and h is not self.h_back:
            yield h
            h = h.next
            guard += 1
            if guard > 1000000:
                break

    def num_corners(self):
        return sum(1 for h in self._walk() if h.vertex.is_corner())

    def get_corner(self, i):
        k = 0
        for h in self._walk():
            if h.vertex.is_corner():
                if k == i:
                    return h
                k += 1
        return None

    def pn_front(self):
        return self.h_front.from_vertex().pn

    def pn_back(self):
        return self.h_back.vertex.pn

    def pn_corner(self, i):
        return self.get_corner(i).vertex.pn

    def sub_pn_front(self):
        nc = self.num_corners()
        if nc < 2:
            dist_target = pn_norm(self.pn_back() - self.pn_corner(0))
        else:
            dist_target = pn_norm(self.pn_corner(nc - 1) - self.pn_corner(nc - 2))
        h_start = self.get_corner(0)
        pn_start = h_start.vertex.pn
        h = h_start
        guard = 0
        while h is not None and h is not self.h_front:
            pn = h.from_vertex().pn
            if dist_target < pn_norm(pn - pn_start):
                return pn
            h = h.prev
            guard += 1
            if guard > 1000000:
                break
        return self.pn_front()

    def sub_pn_back(self):
        nc = self.num_corners()
        if nc < 2:
            dist_target = pn_norm(self.pn_front() - self.pn_corner(0))
        else:
            dist_target = pn_norm(self.pn_corner(1) - self.pn_corner(0))
        h_start = self.get_corner(nc - 1)
        pn_start = h_start.vertex.pn
        h = h_start.next
        guard = 0
        while h is not None and h is not self.h_back:
            pn = h.vertex.pn
            if dist_target < pn_norm(pn - pn_start):
                return pn
            h = h.next
            guard += 1
            if guard > 1000000:
                break
        return self.pn_back()


class _ThinPlateRBF:
    """kt84::RBF<2, 1, RBFKernel_SquaredLog, 1>: r^2 log r kernel with a linear polynomial."""

    def __init__(self, centers, values):
        n = len(centers)
        self.centers = centers
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(2))
        K = self._kernel(d)
        P = np.hstack([np.ones((n, 1)), centers])
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        b = np.zeros(n + 3)
        b[:n] = values
        try:
            self.w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(A, b, rcond=None)[0]

    @staticmethod
    def _kernel(r):
        with np.errstate(divide='ignore', invalid='ignore'):
            k = r * r * np.log(r)
        k[r == 0] = 0.0
        return k

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        n = len(self.centers)
        out = np.empty(len(pts))
        for s in range(0, len(pts), 4096):
            q = pts[s:s + 4096]
            d = np.sqrt(((q[:, None, :] - self.centers[None, :, :]) ** 2).sum(2))
            out[s:s + 4096] = self._kernel(d) @ self.w[:n] + self.w[n] + q @ self.w[n + 1:]
        return out


def merge_nearby_vertices(V, faces, threshold):
    """kt84::merge_nearby_vertices: vertices closer than `threshold` are merged."""
    if len(V) == 0:
        return V, faces
    cell = max(threshold, 1e-12)
    keys = np.floor(V / cell).astype(np.int64)
    buckets = {}
    remap = np.arange(len(V))
    for i, k in enumerate(map(tuple, keys.tolist())):
        found = -1
        kx, ky, kz = k
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                        if np.linalg.norm(V[i] - V[j]) <= threshold:
                            found = j
                            break
                    if found >= 0:
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found >= 0:
            remap[i] = remap[found]
        else:
            buckets.setdefault(k, []).append(i)
    used = sorted(set(remap.tolist()))
    new_index = {old: new for new, old in enumerate(used)}
    V2 = V[used]
    faces2 = []
    for f in faces:
        g = [new_index[remap[v]] for v in f]
        # drop collapsed duplicates within a face
        h = []
        for v in g:
            if not h or h[-1] != v:
                h.append(v)
        if len(h) > 1 and h[0] == h[-1]:
            h.pop()
        if len(h) >= 3:
            faces2.append(h)
    return V2, faces2
