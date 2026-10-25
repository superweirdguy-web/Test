# Port of SketchRetopo/state/*: the ten editing modes.
#
# display(drawer) receives an object with these primitives (positions in base-mesh local space):
#     drawer.line_strip(points, color, width=1, depth=True)
#     drawer.lines(points, color, width=1, depth=True)
#     drawer.triangles(points, color, depth=True)     # flat, possibly translucent (rgba)
#     drawer.points(points, color, size, depth=True)
#     drawer.patch_fill(patch, rgba)                  # phong-tessellated patch fill
#     drawer.phong_line_strip(pns, color, width)

import math
import time

import numpy as np

from .pn import (Polyline, pn_norm, pn_normalize, normalized, norm3, dot3, cross3, distance_to_line,
                 project_to_line, acos_clamped, rbf_wendland)
from .curvenetwork import (Patch, alive, voh_iter, vih_iter, voc_iter, ch_iter, pc_iter, set_next_prev,
                           set_vertex_halfedge, delete_halfedge, delete_halfchain, IMAGINARY_SYMMETRY)
from . import patchgen

LEFT, RIGHT, MIDDLE = 'LEFT', 'RIGHT', 'MIDDLE'


class State:
    state_name = ''
    state_color = (1, 1, 1)
    edit_level = 'CURVE_NETWORK'

    def __init__(self, core):
        self.core = core

    def show_brushSize_snap(self):
        return True

    def init(self):
        pass

    def display(self, drawer):
        pass

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        pass

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        pass

    def mouse_move(self, x, y):
        pass

    def keyboard(self, key):
        pass

    def keyboardup(self, key):
        pass


def _curve_points(curve):
    return [p[:3] for p in curve.points]


# ---- basic sketching ----------------------------------------------------------------------------------------

class StateSketch(State):
    state_name = 'basic sketching'
    state_color = (1, 0.5, 0.1)

    def init(self):
        self.core.common_sketch_curve.clear()

    def display(self, drawer):
        c = self.core.common_sketch_curve
        if c.size() >= 2:
            drawer.line_strip(_curve_points(c), self.state_color, 1)

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if core.common_pn_mouse is None:
            return
        if 'z' not in core.common_key_pressed:
            return
        # fix weird problem around clicked area
        cn = core.curvenetwork
        pn_mouse = core.common_pn_mouse
        crossing = []
        thr = core.configTemp.snapSize()
        for h in list(cn.halfedges):
            if h.is_deleted:
                continue
            d0 = pn_norm(pn_mouse - h.from_vertex().pn)
            d1 = pn_norm(pn_mouse - h.vertex.pn)
            if d0 < thr < d1:
                crossing.append([0.0, h])
            if d0 < thr and d1 < thr:
                delete_halfedge(h)
        crossing = [p for p in crossing if not p[1].is_deleted]
        if not crossing:
            return
        core.memento_store()
        v_center = cn.new_vertex()
        v_center.pn = pn_mouse.copy()
        d0 = crossing[0][1].from_vertex().pn[:3] - v_center.pn[:3]
        n = v_center.pn[3:]
        d1 = np.cross(n, d0)
        for p in crossing:
            d = p[1].from_vertex().pn[:3] - v_center.pn[:3]
            ang = math.atan2(dot3(d1, d), dot3(d0, d))
            if ang < 0:
                ang += 2 * math.pi
            p[0] = ang
        crossing.sort(key=lambda p: p[0])
        new_halfedges = []
        for _, hh in crossing:
            h0 = cn.new_halfedge()
            h1 = cn.new_halfedge()
            new_halfedges.append(h0)
            set_vertex_halfedge(v_center, hh.from_vertex(), h0, h1)
            set_next_prev(h0, hh)
            set_next_prev(hh.opposite, h1)
            h1.is_corner = True
        for i in range(len(crossing)):
            h0 = new_halfedges[i]
            h1 = new_halfedges[(i + 1) % len(new_halfedges)]
            set_next_prev(h1.opposite, h0)
        cn.garbage_collect()
        cn.generate_halfchains()
        core.generate_patches()
        cn.invalidate_displist()

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        curve = core.common_sketch_curve
        if curve.size() < 3:
            self.init()
            return
        core.memento_store()
        if 'c' in core.common_key_pressed:
            core.curvenetwork.erase_curve(curve)
        elif 'x' in core.common_key_pressed:
            pass   # TODO in the original
        else:
            has_negative = any(pn[0] < 0 for pn in curve.points)
            if core.configSaved.symmetric and has_negative:
                pts = curve.points
                if curve.is_loop:
                    pos = len(pts)
                    for k in range(len(pts) - 1):
                        if pts[k][0] < 0 and not pts[k + 1][0] < 0:
                            pos = k
                            break
                    pts = pts[pos:] + pts[:pos]
                split_curves = []
                is_new = True
                for pn in pts:
                    if pn[0] < 0:
                        is_new = True
                        continue
                    if is_new:
                        split_curves.append(Polyline())
                    is_new = False
                    split_curves[-1].push_back(pn)
                for sc in split_curves:
                    if sc.size() >= 2:
                        core.add_curve_open_delegate(sc, not ctrl)
            elif curve.is_loop:
                core.curvenetwork.add_curve_loop(curve)
            else:
                core.add_curve_open_delegate(curve, not ctrl)
        core.generate_patches()
        self.init()

    def keyboard(self, key):
        core = self.core
        if key != 'g':
            return
        erased = []
        for v in core.curvenetwork.vertices:
            if v.is_openend() and not v.on_symmetry_plane() and v.halfedge.halfchain is not None:
                erased.append(v.halfedge.halfchain.edgechain)
        seen = set()
        erased_u = []
        for e in erased:
            if id(e) not in seen:
                seen.add(id(e))
                erased_u.append(e)
        if not erased_u:
            return
        core.memento_store()
        hs = [e.halfchain[0].halfedge_front for e in erased_u]
        for h in hs:
            core.curvenetwork.erase_curve_sub(h)
        core.generate_patches()


# ---- spine sketching ----------------------------------------------------------------------------------------

class StateSpine(State):
    state_name = 'spine sketching'
    state_color = (0.4, 0.5, 1)

    def init(self):
        self.core.common_sketch_curve.clear()

    def display(self, drawer):
        c = self.core.common_sketch_curve
        if c.size() >= 2:
            drawer.line_strip(_curve_points(c), self.state_color, 1)

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        if button == RIGHT:
            self.core.configTemp.brushSize_spine.adjust_init(x)

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if button == RIGHT:
            core.configTemp.brushSize_spine.adjust_done()
            return
        curve = core.common_sketch_curve
        is_symmetric_negative = core.configSaved.symmetric and any(pn[0] < 0 for pn in curve.points)
        if curve.size() < 3 or curve.is_loop or is_symmetric_negative:
            self.init()
            return
        core.memento_store()
        core.add_spine_curve(curve)
        core.generate_patches()
        self.init()


# ---- auto-completion ----------------------------------------------------------------------------------------

class StateAutocmpl(State):
    state_name = 'auto-completion'
    state_color = (0.5, 1, 0.5)

    def init(self):
        self.suggested_autocmpl = []
        self.selected_autocmpl_index = -1
        self.h_selected = None
        self.last_time = 0.0
        self.favor_triangle = False

    def update_preview(self):
        """Called by the host on each redraw (the original does this in display())."""
        core = self.core
        if (core.common_button_left_pressed and core.common_pn_mouse is not None and
                core.common_sketch_curve.size() > 3 and
                (time.time() - self.last_time) * 1000 > core.configTemp.autocmpl_preview_interval):
            self.init()
            self.last_time = time.time()
            best = None
            best_d = float('inf')
            for h in core.curvenetwork.halfedges:
                if h.is_deleted or h.patch() is not None or h.opposite.patch() is None:
                    continue
                d = pn_norm(core.common_pn_mouse - h.vertex.pn)
                if d > core.configTemp.snapSize():
                    continue
                if d < best_d:
                    best_d = d
                    best = h
            if best is None:
                return False
            self.h_selected = best
            self.suggested_autocmpl = core.suggest_autocmpl(best, False)
            self.selected_autocmpl_index = len(self.suggested_autocmpl) - 1
            return True
        return False

    def display(self, drawer):
        if self.selected_autocmpl_index == -1:
            return
        info = self.suggested_autocmpl[self.selected_autocmpl_index]
        drawer.phong_fan(info.corners, self.state_color + (0.2,))

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        if button == RIGHT:
            self.core.configTemp.brushSize_autocmpl.adjust_init(x)

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if button == RIGHT:
            core.configTemp.brushSize_autocmpl.adjust_done()
            return
        if self.selected_autocmpl_index != -1 and core.common_pn_mouse is not None and core.common_sketch_curve.size() == 1:
            core.memento_store()
            core.compute_autocompl(self.suggested_autocmpl[self.selected_autocmpl_index])
            core.generate_patches()
            self.init()
        core.common_sketch_curve.clear()

    def keyboard(self, key):
        if self.selected_autocmpl_index == -1 or self.core.common_button_left_pressed:
            return
        if key in ('z', 'x'):
            i = self.selected_autocmpl_index + (-1 if key == 'z' else 1)
            self.selected_autocmpl_index = min(max(i, 0), len(self.suggested_autocmpl) - 1)
        elif key == 'c':
            self.favor_triangle = not self.favor_triangle
            self.suggested_autocmpl = self.core.suggest_autocmpl(self.h_selected, self.favor_triangle)
            self.selected_autocmpl_index = len(self.suggested_autocmpl) - 1


# ---- laser cutting ------------------------------------------------------------------------------------------

class StateLaser(State):
    state_name = 'laser sketching'
    state_color = (0.5, 0, 1)

    def show_brushSize_snap(self):
        return False

    def init(self):
        self.pn_prev = None
        self.pn_current = None

    def display(self, drawer):
        if self.pn_prev is None or self.pn_current is None:
            return
        drawer.lines([self.pn_prev[:3], self.pn_current[:3]], self.state_color, 5, depth=False)

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        if self.core.common_pn_mouse is None:
            return
        self.pn_prev = self.core.common_pn_mouse.copy()

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if self.pn_prev is None or self.pn_current is None:
            self.init()
            return
        view = core.view
        if view.is_ortho():
            d1 = normalized(view.eye_to_center())
            d2 = (self.pn_current - self.pn_prev)[:3]
            d2 = normalized(d2 - dot3(d1, d2) * d1)
            plane_normal = np.cross(d1, d2)
        else:
            e0 = self.pn_current[:3] - view.eye()
            e1 = self.pn_prev[:3] - view.eye()
            plane_normal = normalized(np.cross(e0, e1))
        traced = core.trace_on_plane(self.pn_prev, plane_normal, pn_norm(self.pn_current - self.pn_prev) * 10)
        if traced.empty():
            self.init()
            return
        core.memento_store()
        if core.configSaved.symmetric:
            if traced.is_loop:
                n = traced.size()
                rotate_pos = -1
                for i in range(n):
                    ip = n - 1 if i == 0 else i - 1
                    if traced[ip][0] < 0 and traced[i][0] >= 0:
                        rotate_pos = i
                        break
                if rotate_pos != -1:
                    traced.points = traced.points[rotate_pos:] + traced.points[:rotate_pos]
                    traced.is_loop = False
            traced.points = [pn for pn in traced.points if not pn[0] < 0]
        if traced.size() >= 2:
            if traced.is_loop:
                core.curvenetwork.add_curve_loop(traced)
            else:
                core.add_curve_open_delegate(traced)
        core.generate_patches()
        self.init()

    def mouse_move(self, x, y):
        if self.pn_prev is not None and self.core.common_pn_mouse is not None:
            self.pn_current = self.core.common_pn_mouse.copy()


# ---- cylinder sketching -------------------------------------------------------------------------------------

class StateCylinder(State):
    state_name = 'cylinder sketching'
    state_color = (0.5, 0.7, 0.3)

    def init(self):
        self.core.common_sketch_curve.clear()

    def display(self, drawer):
        c = self.core.common_sketch_curve
        if c.size() >= 2:
            drawer.line_strip(_curve_points(c), self.state_color, 1)

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if core.common_sketch_curve.size() < 3:
            self.init()
            return
        core.add_cylinder_curve(core.common_sketch_curve.copy())
        core.generate_patches()
        self.init()


# ---- deform curve -------------------------------------------------------------------------------------------

class StateDeformCurve(State):
    state_name = 'deform curve'
    state_color = (0.3, 0.8, 0.2)

    def init(self):
        self.mode = None            # None / 'DragCorner' / 'DragSide' / 'OverSketch'
        self.selected_vertex = None
        self.prev_pos = None
        self.core.common_sketch_curve.clear()
        self.core.common_affected_edgechains = []

    def display(self, drawer):
        if self.mode == 'OverSketch':
            c = self.core.common_sketch_curve
            if c.size() >= 2:
                drawer.line_strip(_curve_points(c), self.state_color, 1)

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if button != LEFT or core.common_pn_mouse is None:
            return
        v_corner = None
        v_corner_d = float('inf')
        v_side = None
        v_side_d = float('inf')
        for v in core.curvenetwork.vertices:
            if core.configSaved.symmetric and v.on_symmetry_plane():
                continue
            d = pn_norm(v.pn - core.common_pn_mouse)
            if core.configTemp.snapSize() < d:
                continue
            if v.is_corner() or v.is_openend():
                if d < v_corner_d:
                    v_corner_d = d
                    v_corner = v
            elif d < v_side_d:
                v_side_d = d
                v_side = v
        if v_corner is not None:
            self.selected_vertex = v_corner
        elif v_side is not None:
            self.selected_vertex = v_side
        else:
            self.init()
            return
        sv = self.selected_vertex
        if ctrl:
            self.mode = 'OverSketch'
        elif sv.is_corner() or sv.is_openend():
            self.mode = 'DragCorner'
        else:
            self.mode = 'DragSide'
        if self.mode in ('DragCorner', 'DragSide'):
            for c in voc_iter(sv):
                if c is not None:
                    core.common_affected_edgechains.append(c.edgechain)
            self.prev_pos = sv.pn[:3].copy()
        core.memento_store()

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        cn = core.curvenetwork
        if self.mode is None:
            return
        if self.mode == 'OverSketch':
            if core.common_pn_mouse is None:
                self.init()
                return
            other = None
            other_d = float('inf')
            for v in cn.vertices:
                if core.configSaved.symmetric and v.on_symmetry_plane():
                    continue
                if v is self.selected_vertex:
                    continue
                d = pn_norm(v.pn - core.common_pn_mouse)
                if d < other_d:
                    other_d = d
                    other = v
            if other is None or other_d > core.configTemp.snapSize():
                self.init()
                return
            h_found = None
            sequence_length = 0
            for h in voh_iter(self.selected_vertex):
                sequence_length = 1
                h2 = h
                while h2 is not None:
                    v = h2.vertex
                    if v is other:
                        h_found = h
                        break
                    if v.is_corner() or v.is_openend():
                        break
                    h2 = h2.next
                    sequence_length += 1
                    if sequence_length > 1000000:
                        break
                if h_found is not None:
                    break
            if h_found is None:
                self.init()
                return
            h = h_found
            for i in range(1, sequence_length):
                t = i / float(sequence_length)
                h.vertex.pn = core.common_sketch_curve.point_at(t)
                h = h.next
            core.common_affected_edgechains.append(h.halfchain.edgechain)

        affected_patches = []
        seen = set()
        for e in core.common_affected_edgechains:
            if e is None or id(e) in seen:
                continue
            seen.add(id(e))
            for i in range(2):
                c = e.halfchain[i]
                if c is None:
                    continue
                patch = c.patch
                if patch is None or patch.is_imaginary():
                    continue
                if all(patch is not p for p in affected_patches):
                    affected_patches.append(patch)
        for patch in affected_patches:
            if not patch.is_deleted:
                patch.clear()
        try:
            self._snap_selected_vertex()
        finally:
            for patch in affected_patches:
                if patch.is_deleted:
                    continue
                patch.generate_topology(False)
                core.compute_patch_interior_pn(patch)
                patch.invalidate_displist()
            cn.invalidate_displist()
            core.common_affected_edgechains = []
            cn.garbage_collect()
            self.init()

    def _snap_selected_vertex(self):
        core = self.core
        cn = core.curvenetwork
        sv = self.selected_vertex
        if not sv.is_corner() and not sv.is_openend():
            return
        snap = core.configTemp.snapSize()
        best = {'openend': (float('inf'), None), 'corner_adj': (float('inf'), None),
                'corner': (float('inf'), None), 'side': (float('inf'), None)}
        e_adjacent = [c.edgechain for c in voc_iter(sv) if c is not None]
        for v in cn.vertices:
            if v is sv:
                continue
            d = pn_norm(sv.pn - v.pn)
            if d > snap:
                continue
            if v.is_openend():
                if v.halfedge.halfchain is not None and v.halfedge.halfchain.vertex_back() is sv:
                    continue
                kind = 'openend'
            elif v.is_corner():
                if sv.is_openend() and sv.halfedge.halfchain is not None and sv.halfedge.halfchain.vertex_back() is v:
                    continue
                is_adjacent = any(c is not None and c.halfedge_back.vertex is v for c in voc_iter(sv))
                kind = 'corner_adj' if is_adjacent else 'corner'
            else:
                if v.halfedge.halfchain is None or any(v.halfedge.halfchain.edgechain is e for e in e_adjacent):
                    continue
                kind = 'side'
            if d < best[kind][0]:
                best[kind] = (d, v)
        v_snapped = None
        snapped_kind = None
        for kind in ('openend', 'corner_adj', 'corner', 'side'):
            if best[kind][1] is not None:
                v_snapped = best[kind][1]
                snapped_kind = kind
                break
        if v_snapped is None:
            return

        offset_value = (v_snapped.pn - sv.pn)[:3]

        def offset_func(pn, w):
            pn[:3] += w * offset_value
            core.project(pn)

        for c in voc_iter(sv):
            if c is None:
                continue
            n = c.to_polyline().size() - 1
            h = c.halfedge_front
            for i in range(1, n):
                offset_func(h.vertex.pn, 1.0 - i / float(n))
                h = h.next

        if snapped_kind == 'corner_adj':
            c_deleted = None
            for c in voc_iter(sv):
                if c is not None and c.vertex_back() is v_snapped:
                    c_deleted = c
                    break
            if c_deleted is None:
                return
            for h in list(vih_iter(sv)):
                h.vertex = v_snapped
            sv.is_deleted = True
            h0 = (c_deleted.halfedge_front.prev, c_deleted.halfedge_back.next)
            co = c_deleted.opposite()
            h1 = (co.halfedge_front.prev, co.halfedge_back.next)
            h_front = c_deleted.halfedge_front
            h_back = c_deleted.halfedge_back
            h = h_front
            while True:
                delete_halfedge(h.opposite)
                delete_halfedge(h)
                if h is h_back:
                    break
                h = h.next
            set_next_prev(h0[0], h0[1])
            set_next_prev(h1[0], h1[1])
        else:
            def get_boundary_halfedges(v):
                h_prev = h_next = None
                for h in vih_iter(v):
                    if h.patch() is not None:
                        continue
                    if h_prev is not None:
                        return None
                    h_prev = h
                for h in voh_iter(v):
                    if h.patch() is not None:
                        continue
                    if h_next is not None:
                        return None
                    h_next = h
                return h_prev, h_next

            src = get_boundary_halfedges(sv)
            tgt = get_boundary_halfedges(v_snapped)
            if src is None or tgt is None or None in src or None in tgt:
                return
            hsrc_prev, hsrc_next = src
            htgt_prev, htgt_next = tgt
            for h in list(vih_iter(sv)):
                h.vertex = v_snapped
            sv.is_deleted = True
            if snapped_kind == 'side':
                delete_halfchain(htgt_prev.halfchain)
                delete_halfchain(htgt_prev.opposite.halfchain)
            set_next_prev(hsrc_prev, htgt_next)
            set_next_prev(htgt_prev, hsrc_next)
            hsrc_prev.is_corner = True
            htgt_prev.is_corner = True
            if snapped_kind == 'side':
                cn.generate_halfchains()
            csrc_prev = hsrc_prev.halfchain
            csrc_next = hsrc_next.halfchain
            ctgt_prev = htgt_prev.halfchain
            ctgt_next = htgt_next.halfchain
            if (csrc_next is not None and ctgt_prev is not None and not csrc_next.is_deleted and
                    csrc_next.vertex_back() is ctgt_prev.vertex_front()):
                set_next_prev(htgt_prev, hsrc_next.opposite.next)
                set_next_prev(csrc_next.halfedge_back.opposite.prev, csrc_next.halfedge_back.next)
                h_front = csrc_next.halfedge_front
                h_back = csrc_next.halfedge_back
                h = h_front
                while True:
                    delete_halfedge(h.opposite)
                    delete_halfedge(h)
                    if h is h_back:
                        break
                    h = h.next
            if (csrc_prev is not None and ctgt_next is not None and not csrc_prev.is_deleted and
                    csrc_prev.vertex_front() is ctgt_next.vertex_back()):
                set_next_prev(hsrc_prev.opposite.prev, htgt_next)
                set_next_prev(csrc_prev.halfedge_front.prev, csrc_prev.halfedge_front.opposite.next)
                h_front = csrc_prev.halfedge_front
                h_back = csrc_prev.halfedge_back
                h = h_front
                while True:
                    delete_halfedge(h.opposite)
                    delete_halfedge(h)
                    if h is h_back:
                        break
                    h = h.next
        core.generate_patches()

    def mouse_move(self, x, y):
        core = self.core
        if core.common_pn_mouse is None or self.mode is None:
            return
        if self.mode in ('DragCorner', 'DragSide'):
            offset_value = core.common_pn_mouse[:3] - self.prev_pos
            sv = self.selected_vertex

            def offset_func(pn, w):
                pn[:3] += w * offset_value
                core.project(pn)

            if self.mode == 'DragCorner':
                for c in voc_iter(sv):
                    if c is None:
                        continue
                    n = c.to_polyline().size() - 1
                    h = c.halfedge_front
                    for i in range(1, n):
                        offset_func(h.vertex.pn, 1.0 - i / float(n))
                        h = h.next
            else:
                hs = list(voh_iter(sv))[:2]
                for h_start in hs:
                    num_halfedges = 1
                    h = h_start
                    looped = False
                    while True:
                        if h.vertex.is_corner() or h.vertex.is_openend():
                            break
                        num_halfedges += 1
                        h = h.next
                        if h is h_start or h is None:
                            looped = True
                            break
                    if looped:
                        return
                    h = h_start
                    for j in range(1, num_halfedges):
                        w = rbf_wendland(j / float(num_halfedges))
                        offset_func(h.vertex.pn, w)
                        h = h.next
            offset_func(sv.pn, 1)
            self.prev_pos = core.common_pn_mouse[:3].copy()
        core.curvenetwork.invalidate_displist()


# ---- edit corner --------------------------------------------------------------------------------------------

class StateEditCorner(State):
    state_name = 'edit corner'
    state_color = (0.8, 0.5, 0.6)

    def init(self):
        self.pn_prev = None

    def display(self, drawer):
        if self.pn_prev is None or self.core.common_pn_mouse is None:
            return
        drawer.lines([self.pn_prev[:3], self.core.common_pn_mouse[:3]], self.state_color, 5, depth=False)

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if core.common_pn_mouse is None:
            return
        if ctrl:
            v_corner = None
            v_corner_d = float('inf')
            v_side = None
            v_side_d = float('inf')
            for v in core.curvenetwork.vertices:
                if v.is_deleted or v.valence() != 2:
                    continue
                d = pn_norm(core.common_pn_mouse - v.pn)
                if core.configTemp.snapSize() < d:
                    continue
                if v.is_corner():
                    if d < v_corner_d:
                        v_corner_d = d
                        v_corner = v
                elif d < v_side_d:
                    v_side_d = d
                    v_side = v
            if v_corner is None and v_side is None:
                return
            core.memento_store()
            core.curvenetwork.flip_corner_valance2(v_corner if v_corner is not None else v_side)
            core.generate_patches()
            return
        self.pn_prev = core.common_pn_mouse.copy()

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if self.pn_prev is None or core.common_pn_mouse is None:
            self.init()
            return
        v_min = None
        d_min = float('inf')
        for v in core.curvenetwork.vertices:
            if v.is_deleted or not v.is_corner():
                continue
            d = pn_norm(core.common_pn_mouse - v.pn)
            if d < d_min:
                d_min = d
                v_min = v
        if v_min is None or d_min > core.configTemp.snapSize():
            self.init()
            return
        num_corners = sum(1 for h in vih_iter(v_min) if h.is_corner)
        n = v_min.pn[3:]
        d = normalized((self.pn_prev - core.common_pn_mouse)[:3])
        h_selected = None
        for h in vih_iter(v_min):
            if h.next is None:
                continue
            d0 = -normalized(h.to_vector())
            d1 = normalized(h.next.to_vector())
            angle = acos_clamped(dot3(d1, d))
            angle0 = acos_clamped(dot3(d1, d0))
            if dot3(np.cross(d1, d), n) < 0:
                angle = 2 * math.pi - angle
            if dot3(np.cross(d1, d0), n) < 0:
                angle0 = 2 * math.pi - angle0
            if angle < angle0:
                h_selected = h
                break
        if h_selected is None:
            self.init()
            return
        if num_corners == 1 and h_selected.is_corner:
            self.init()
            return
        core.memento_store()
        h_selected.is_corner = not h_selected.is_corner
        patch = h_selected.patch()
        if patch is not None and not patch.is_imaginary():
            patch.generate_topology()
            core.compute_patch_interior_pn(patch)
        core.curvenetwork.invalidate_displist()
        self.init()


# ---- edit topology ------------------------------------------------------------------------------------------

class StateEditTopology(State):
    state_name = 'edit topology'
    state_color = (1, 0, 0)
    edit_level = 'QUAD_MESH'

    def init(self):
        self.current_patch = None
        self.selected_variable = 0
        self.p_prev = None
        self.average_edge_length = 0.0
        self.core.curvenetwork.set_flag_edgechains('SUBST', 0)

    def _side_arrow(self, patch, j):
        polyline = patch.get_side_polyline(j)
        pn = polyline.point_at(0.5)
        n = pn[3:]
        d = (polyline.point_at(0.51) - pn)[:3]
        d = normalized(np.cross(n, d)) * self.average_edge_length * 0.5
        return pn[:3] - d, pn[:3] + d, n

    def display(self, drawer):
        core = self.core
        cp = self.current_patch
        if cp is not None and not cp.is_deleted:
            drawer.patch_fill(cp, self.state_color + (0.3,))
            param = cp.param
            num_sides = param.get_num_sides()
            num_variables = patchgen.get_num_variables(num_sides, param.pattern_id)
            if num_sides != 4 and num_sides < num_variables:
                for i in range(num_sides):
                    j = param.permutation[i]
                    p0, p1, n = self._side_arrow(cp, j)
                    color = (1, 0, 0) if i == self.selected_variable else (0.5, 0, 0)
                    drawer.double_arrow(p0, p1, n, color)
            color_table = [(0.6, 0.4, 0), (0, 0.5, 0), (0, 0.5, 0.6), (0.5, 0, 0.5)]
            indicators = patchgen.get_variable_indicators(num_sides, param.pattern_id)
            for i in range(num_sides, num_variables):
                k = i - num_sides
                if k >= len(indicators):
                    continue
                for tag0, tag1 in indicators[k]:
                    v0 = patchgen.find_tagged_vertex(cp, tag0)
                    v1 = patchgen.find_tagged_vertex(cp, tag1)
                    if v0 < 0 or v1 < 0:
                        continue
                    base = color_table[k % 4]
                    scale = 2 if i == self.selected_variable else 1
                    color = tuple(min(1.0, c * scale) for c in base)
                    pn0 = cp.pn[v0]
                    pn1 = cp.pn[v1]
                    drawer.double_arrow(pn0[:3], pn1[:3], normalized((pn0 + pn1)[3:]), color)
        # selected curves
        pts = []
        for e in core.curvenetwork.edgechains:
            if not e.flag:
                continue
            c = e.halfchain[0]
            strip = [c.halfedge_front.from_vertex().pn[:3]]
            for h in ch_iter(c):
                pn = h.vertex.pn
                strip.append(pn[:3] + pn[3:] * core.configSaved.projOffset * 0.1)
            for a, b in zip(strip[:-1], strip[1:]):
                pts.append(a)
                pts.append(b)
        if pts:
            drawer.lines(pts, self.state_color + (0.5,), 5)
        drawer.num_subdiv_points((0.1, 0.2, 1.0))

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if core.common_pn_mouse is None:
            self.init()
            return
        mouse_pos = core.common_pn_mouse[:3]
        cp = self.current_patch
        if cp is not None and not cp.is_deleted:
            param = cp.param
            num_sides = param.get_num_sides()
            if num_sides == 4:
                if cp.distance(mouse_pos) < core.configTemp.snapSize():
                    self.p_prev = mouse_pos.copy()
                    core.memento_store()
                else:
                    self.init()
            else:
                best = None
                best_d = float('inf')
                for i in range(num_sides):
                    j = param.permutation[i]
                    p0, p1, _ = self._side_arrow(cp, j)
                    d = distance_to_line(p0, p1, mouse_pos, True)
                    if d is not None and d < best_d:
                        best_d = d
                        best = i
                num_variables = patchgen.get_num_variables(num_sides, param.pattern_id)
                indicators = patchgen.get_variable_indicators(num_sides, param.pattern_id)
                for i in range(num_sides, num_variables):
                    k = i - num_sides
                    if k >= len(indicators):
                        continue
                    for tag0, tag1 in indicators[k]:
                        v0 = patchgen.find_tagged_vertex(cp, tag0)
                        v1 = patchgen.find_tagged_vertex(cp, tag1)
                        if v0 < 0 or v1 < 0:
                            continue
                        d = distance_to_line(cp.pn[v0][:3], cp.pn[v1][:3], mouse_pos, True)
                        if d is not None and d < best_d:
                            best_d = d
                            best = i
                if best is not None and best_d < core.configTemp.snapSize():
                    self.selected_variable = best
                else:
                    self.init()
            return

        # select closest edgechain
        best_e = None
        best_d = float('inf')
        for v in core.curvenetwork.vertices:
            if v.is_corner():
                continue
            d = pn_norm(v.pn - core.common_pn_mouse)
            if d > core.configTemp.snapSize():
                continue
            if v.halfedge is None or v.halfedge.halfchain is None:
                continue
            if d < best_d:
                best_d = d
                best_e = v.halfedge.halfchain.edgechain
        if best_e is not None:
            if shift:
                best_e.flag = 0 if best_e.flag else 1
            elif ctrl:
                c0 = best_e.halfchain[0]
                patch = best_e.halfchain[0 if Patch.is_ordinary(c0.patch) else 1].patch
                if not Patch.is_ordinary(patch):
                    return
                h_best = -1
                hd = float('inf')
                for h in patch.halfedges():
                    if not patch.is_boundary_h(h):
                        continue
                    v0, v1 = patch.util_halfedge_to_vertex_pair(h)
                    d = distance_to_line(patch.pn[v0][:3], patch.pn[v1][:3], mouse_pos, True)
                    if d is not None and d < hd:
                        hd = d
                        h_best = h
                for p, h in core.edgeLoop_walk(patch, h_best):
                    if not p.is_boundary_h(h):
                        continue
                    if p.h_chain[h] is not None:
                        p.h_chain[h].edgechain.flag = 1
            else:
                self.init()
                best_e.flag = 1
            return

        self.init()
        best_p = None
        best_pd = float('inf')
        snap = core.configTemp.snapSize()
        for patch in core.curvenetwork.patches:
            num_sides = patch.param.get_num_sides()
            if num_sides == 4 and patch.param.pattern_id == 0:
                continue
            P = patch.pn[:, :3]
            if len(P) == 0:
                continue
            lo = P.min(axis=0) - snap
            hi = P.max(axis=0) + snap
            if not (np.all(lo <= mouse_pos) and np.all(mouse_pos <= hi)):
                continue
            d = patch.distance(mouse_pos)
            if d < best_pd:
                best_pd = d
                best_p = patch
        if best_p is not None and best_pd < snap:
            self.current_patch = best_p
            self.selected_variable = best_p.param.get_num_sides()
            total = 0.0
            ne = best_p.n_edges()
            for e in range(ne):
                v0, v1 = best_p.util_edge_to_vertex_pair(e)
                total += norm3(best_p.pn[v1][:3] - best_p.pn[v0][:3])
            self.average_edge_length = total / ne if ne else 0.0

    def mouse_move(self, x, y):
        core = self.core
        cp = self.current_patch
        if cp is None or core.common_pn_mouse is None or not core.common_button_left_pressed or self.p_prev is None:
            return
        param = cp.param
        if param.get_num_sides() != 4:
            return
        d = core.common_pn_mouse[:3] - self.p_prev
        if norm3(d) < self.average_edge_length:
            return
        d = normalized(d)
        p = [cp.pn[cp.patch_corner(i)][:3] for i in range(4)]
        e = [None] * 4
        e[0] = normalized(p[3] - p[0] + p[2] - p[1])
        e[1] = normalized(p[0] - p[1] + p[3] - p[2])
        e[2] = -e[0]
        e[3] = -e[1]
        scores = [-dot3(d, e[i]) for i in range(4)]
        sel = int(np.argmin(scores))
        if patchgen.adjust_parameter(param, param.permutation[sel], param.permutation.is_flipped()):
            cp.generate_topology(False)
            core.compute_patch_interior_pn(cp)
        self.p_prev = core.common_pn_mouse[:3].copy()

    def keyboard(self, key):
        core = self.core
        cp = self.current_patch
        if cp is not None and not cp.is_deleted:
            if key not in ('z', 'x', 'c', 'v', 'b', 'B'):
                return
            core.memento_store()
            do_update = False
            if key in ('z', 'x', 'c', 'v'):
                do_update = patchgen.adjust_parameter(cp.param, self.selected_variable, key in ('c', 'v'))
            if key in ('b', 'B'):
                do_update = patchgen.switch_pattern(cp.param, key == 'b')
            if do_update:
                cp.generate_topology(False)
                core.compute_patch_interior_pn(cp)
            return
        if key not in ('z', 'x', 'c', 'v'):
            return
        core.memento_store()
        updated = []
        for e in core.curvenetwork.edgechains:
            if not e.flag:
                continue
            old = e.num_subdiv
            e.num_subdiv += {'z': -1, 'v': 1, 'x': -2, 'c': 2}[key]
            if e.num_subdiv < 1:
                e.num_subdiv = old
                continue
            for i in range(2):
                p = e.halfchain[i].patch if e.halfchain[i] is not None else None
                if p is None or p.is_imaginary():
                    continue
                if all(p is not q for q in updated):
                    updated.append(p)
                p.clear()
        for p in updated:
            p.generate_topology()
            core.compute_patch_interior_pn(p)
        core.curvenetwork.invalidate_displist()


# ---- edge loop insertion ------------------------------------------------------------------------------------

class StateEdgeLoop(State):
    state_name = 'edge loop'
    state_color = (0, 0.5, 0)
    edit_level = 'QUAD_MESH'

    def init(self):
        self.edgeLoop_preview = Polyline()

    def display(self, drawer):
        if self.edgeLoop_preview.size() >= 2:
            drawer.phong_line_strip(self.edgeLoop_preview.points, self.state_color, 3)

    def _closest_boundary_halfedge(self, p_mouse):
        core = self.core
        h_min = -1
        patch_min = None
        dist_min = float('inf')
        snap = core.configTemp.snapSize()
        for patch in core.curvenetwork.patches:
            if patch.is_failure:
                continue
            P = patch.pn[:, :3]
            if len(P) == 0:
                continue
            if np.any(P.min(axis=0) - snap > p_mouse) or np.any(P.max(axis=0) + snap < p_mouse):
                continue
            for h in patch.halfedges():
                if not patch.is_boundary_h(h):
                    continue
                v0, v1 = patch.util_halfedge_to_vertex_pair(h)
                d = distance_to_line(P[v0], P[v1], p_mouse, True)
                if d is not None and d < dist_min and d < snap:
                    h_min = h
                    patch_min = patch
                    dist_min = d
        return patch_min, h_min

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if core.common_pn_mouse is None:
            return
        p_mouse = core.common_pn_mouse[:3]
        self.init()
        patch, h = self._closest_boundary_halfedge(p_mouse)
        if h < 0:
            return
        v0, v1 = patch.util_halfedge_to_vertex_pair(h)
        t = project_to_line(patch.pn[v0][:3], patch.pn[v1][:3], p_mouse)
        if t is None:
            return
        core.edgeLoop_insert(patch, h, t[1])

    def mouse_move(self, x, y):
        core = self.core
        if core.common_pn_mouse is None:
            return
        p_mouse = core.common_pn_mouse[:3]
        self.init()
        patch, h = self._closest_boundary_halfedge(p_mouse)
        if h < 0:
            return
        v0, v1 = patch.util_halfedge_to_vertex_pair(h)
        t = project_to_line(patch.pn[v0][:3], patch.pn[v1][:3], p_mouse)
        if t is None:
            return
        self.edgeLoop_preview = core.edgeLoop_preview(patch, h, t[1])


# ---- move vertex --------------------------------------------------------------------------------------------

class StateMoveVertex(State):
    state_name = 'move vertex'
    state_color = (0.8, 1, 0.2)
    edit_level = 'QUAD_MESH'

    def show_brushSize_snap(self):
        return False

    def init(self):
        self.pn_prev = None
        self.is_smoothing = False

    def mouse_down(self, x, y, button, shift, ctrl, alt):
        core = self.core
        if button == RIGHT:
            core.configTemp.brushSize_moveVertex.adjust_init(x)
            return
        if core.common_pn_mouse is None:
            return
        mouse_pos = core.common_pn_mouse[:3]
        if ctrl:
            best = None
            best_d = float('inf')
            snap = core.configTemp.snapSize()
            for patch in core.curvenetwork.patches:
                P = patch.pn[:, :3]
                if len(P) == 0:
                    continue
                if np.any(P.min(axis=0) - snap > mouse_pos) or np.any(P.max(axis=0) + snap < mouse_pos):
                    continue
                d = patch.distance(mouse_pos)
                if d < best_d:
                    best_d = d
                    best = patch
            if best is not None and best_d < snap:
                hit = core.intersect(core.common_pn_mouse)
                if hit is not None:
                    core.memento_store()
                    core.compute_patch_interior_pn_harmonic(best, hit.face)
        else:
            core.memento_store()
            self.pn_prev = core.common_pn_mouse.copy()
            self.is_smoothing = shift

    def mouse_up(self, x, y, button, shift, ctrl, alt):
        if button == RIGHT:
            self.core.configTemp.brushSize_moveVertex.adjust_done()
            return
        self.init()

    def mouse_move(self, x, y):
        core = self.core
        if self.pn_prev is None or core.common_pn_mouse is None:
            return
        if self.is_smoothing:
            core.vertices_smooth(core.common_pn_mouse, core.configTemp.brushSize_moveVertex())
        else:
            core.vertices_move(core.common_pn_mouse, (core.common_pn_mouse - self.pn_prev)[:3],
                               core.configTemp.brushSize_moveVertex())
        core.curvenetwork.invalidate_displist()
        self.pn_prev = core.common_pn_mouse.copy()

    def keyboard(self, key):
        if key == 'g':
            core = self.core
            core.memento_store()
            core.vertices_smooth_global()
            core.curvenetwork.invalidate_displist()
