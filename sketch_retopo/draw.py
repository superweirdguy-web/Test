# Viewport drawing for a SketchRetopo session (port of SketchRetopo_render.cc).
#
# Everything is drawn in the base object's local space (the object matrix is
# pushed onto the GPU matrix stack).  Geometry batches are cached per patch and
# per curve-network version.

import math

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from .core import phong
from .core.config import ConfigRender
from .core.curvenetwork import Patch, alive, vih_iter, ch_iter
from .core.pn import normalized, acos_clamped

_shaders = {}


def _builtin(name):
    sh = _shaders.get(name)
    if sh is None:
        try:
            sh = gpu.shader.from_builtin(name)
        except ValueError:
            # POINT_UNIFORM_COLOR (round points) appeared in Blender 4.2; square points before that
            sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        _shaders[name] = sh
    return sh


def _lit_shader():
    sh = _shaders.get('lit')
    if sh is not None:
        return sh
    iface = gpu.types.GPUStageInterfaceInfo('sketch_retopo_lit_iface')
    iface.smooth('VEC3', 'v_pos')
    iface.smooth('VEC3', 'v_nrm')
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', 'mvp')
    info.push_constant('VEC4', 'color')
    info.push_constant('VEC3', 'light_pos')          # in object space
    info.push_constant('INT', 'use_shading')
    # pushes the surface away from the viewer (eye.w=1: eye point, eye.w=0: view direction) so that
    # quad edges drawn on top of it don't z-fight (the original relied on legacy GL line rasterization)
    info.push_constant('VEC4', 'eye')
    info.push_constant('FLOAT', 'depth_offset')
    info.vertex_in(0, 'VEC3', 'pos')
    info.vertex_in(1, 'VEC3', 'nrm')
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', 'frag_color')
    info.vertex_source(
        'void main() {\n'
        '  v_pos = pos;\n'
        '  v_nrm = nrm;\n'
        '  vec3 d = eye.w > 0.5 ? normalize(pos - eye.xyz) : eye.xyz;\n'
        '  gl_Position = mvp * vec4(pos + depth_offset * d, 1.0);\n'
        '}\n')
    info.fragment_source(
        'void main() {\n'
        '  frag_color = color;\n'
        '  if (use_shading != 0) {\n'
        '    vec3 n = normalize(v_nrm);\n'
        '    vec3 l = normalize(light_pos - v_pos);\n'
        '    frag_color.rgb *= max(dot(l, n), 0.0);\n'
        '  }\n'
        '}\n')
    sh = gpu.shader.create_from_info(info)
    _shaders['lit'] = sh
    return sh


def _corner_shader():
    sh = _shaders.get('corner')
    if sh is not None:
        return sh
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', 'mvp')
    info.push_constant('FLOAT', 'fan_radius')
    info.push_constant('VEC4', 'color')
    info.vertex_in(0, 'VEC3', 'pos')
    info.vertex_in(1, 'VEC3', 'dir')
    info.vertex_in(2, 'VEC3', 'nrm')
    info.fragment_out(0, 'VEC4', 'frag_color')
    info.vertex_source(
        'void main() {\n'
        '  vec3 v = pos + fan_radius * dir + (fan_radius * 0.1) * nrm;\n'
        '  gl_Position = mvp * vec4(v, 1.0);\n'
        '}\n')
    info.fragment_source('void main() { frag_color = color; }\n')
    sh = gpu.shader.create_from_info(info)
    _shaders['corner'] = sh
    return sh


def _negx_shader():
    sh = _shaders.get('negx')
    if sh is not None:
        return sh
    iface = gpu.types.GPUStageInterfaceInfo('sketch_retopo_negx_iface')
    iface.smooth('FLOAT', 'v_x')
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', 'mvp')
    info.push_constant('VEC4', 'color')
    info.vertex_in(0, 'VEC3', 'pos')
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', 'frag_color')
    info.vertex_source(
        'void main() {\n'
        '  v_x = pos.x;\n'
        '  gl_Position = mvp * vec4(pos, 1.0);\n'
        '}\n')
    info.fragment_source(
        'void main() {\n'
        '  if (v_x >= 0.0) discard;\n'
        '  frag_color = color;\n'
        '}\n')
    sh = gpu.shader.create_from_info(info)
    _shaders['negx'] = sh
    return sh


def _mvp():
    return gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()


def _sphere(slices=36, stacks=18):
    key = ('sphere', slices, stacks)
    if key in _shaders:
        return _shaders[key]
    th = np.linspace(0, math.pi, stacks + 1)
    ph = np.linspace(0, 2 * math.pi, slices + 1)
    P = np.stack([np.outer(np.sin(th), np.cos(ph)), np.outer(np.sin(th), np.sin(ph)),
                  np.outer(np.cos(th), np.ones_like(ph))], axis=-1)
    tris = []
    for i in range(stacks):
        for j in range(slices):
            a = P[i, j]
            b = P[i + 1, j]
            c = P[i + 1, j + 1]
            d = P[i, j + 1]
            tris.extend([a, b, c, a, c, d])
    arr = np.array(tris)
    _shaders[key] = arr
    return arr


class Drawer:
    """Primitive API used by the states' display() (see core/states.py)."""

    def __init__(self, renderer):
        self.r = renderer

    def _polyline(self, pts, color, width, kind, depth):
        if len(pts) < 2:
            return
        sh = _builtin('POLYLINE_UNIFORM_COLOR')
        batch = batch_for_shader(sh, kind, {'pos': [tuple(map(float, p[:3])) for p in pts]})
        gpu.state.depth_test_set('LESS_EQUAL' if depth else 'NONE')
        sh.bind()
        sh.uniform_float('viewportSize', (self.r.region.width, self.r.region.height))
        sh.uniform_float('lineWidth', float(width))
        sh.uniform_float('color', _rgba(color))
        batch.draw(sh)
        gpu.state.depth_test_set('LESS_EQUAL')

    def line_strip(self, points, color, width=1, depth=True):
        self._polyline(points, color, width, 'LINE_STRIP', depth)

    def lines(self, points, color, width=1, depth=True):
        self._polyline(points, color, width, 'LINES', depth)

    def points(self, points, color, size, depth=True):
        if not len(points):
            return
        sh = _builtin('POINT_UNIFORM_COLOR')
        batch = batch_for_shader(sh, 'POINTS', {'pos': [tuple(map(float, p[:3])) for p in points]})
        gpu.state.point_size_set(float(size))
        gpu.state.depth_test_set('LESS_EQUAL' if depth else 'NONE')
        sh.bind()
        sh.uniform_float('color', _rgba(color))
        batch.draw(sh)
        gpu.state.depth_test_set('LESS_EQUAL')

    def triangles(self, positions, color, depth=True):
        if not len(positions):
            return
        sh = _builtin('UNIFORM_COLOR')
        batch = batch_for_shader(sh, 'TRIS', {'pos': np.asarray(positions, dtype=np.float32).reshape(-1, 3)})
        gpu.state.depth_test_set('LESS_EQUAL' if depth else 'NONE')
        sh.bind()
        sh.uniform_float('color', _rgba(color))
        batch.draw(sh)
        gpu.state.depth_test_set('LESS_EQUAL')

    def patch_fill(self, patch, rgba):
        self.r.draw_patch_fill(patch, rgba, shading=False)

    def phong_fan(self, pns, rgba):
        cr = self.r.engine.configRender
        pos, _ = phong.fan(pns, True, 5, 0.7)
        self.triangles(pos, rgba, depth=False)

    def phong_line_strip(self, pns, color, width):
        cr = self.r.engine.configRender
        segs = phong.line_strip(pns, cr.phong_tessellation, cr.phong_subdiv, cr.phong_weight)
        self.lines(segs.reshape(-1, 3), color, width)

    def double_arrow(self, p0, p1, n, color):
        eye = self.r.view.eye()
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        arrow_size = float(np.linalg.norm(0.5 * p0 + 0.5 * p1 - eye)) * 0.01
        d = p1 - p0
        p0 = p0 + 0.1 * d
        p1 = p1 - 0.1 * d
        r = float(np.linalg.norm(d))
        if r == 0:
            return
        arrow_size = min(arrow_size, r * 0.3)
        d = d * (arrow_size / r)
        self.lines([p0 + d, p1 - d], color, 2, depth=False)
        e = 0.4 * np.cross(n, d)
        self.triangles([p0, p0 + d - e, p0 + d + e, p1, p1 - d + e, p1 - d - e], color, depth=False)

    def num_subdiv_points(self, color):
        """Core::render_edgechains_num_subdiv: the edges of curves without patches, shown as points."""
        pts = []
        for e in self.r.engine.curvenetwork.edgechains:
            c0, c1 = e.halfchain
            if e.is_deleted or (c0 is not None and alive(c0.patch)) or (c1 is not None and alive(c1.patch)):
                continue
            if c0 is None or any(h.vertex.is_hidden for h in ch_iter(c0)):
                continue
            for i in range(1, e.num_subdiv):
                pts.append(c0.pn_at(i / float(e.num_subdiv))[:3])
        self.points(pts, color, 10)

    def text(self, pos, string, color=(1, 1, 1)):
        """Text centred at a 3D position (drawn in the pixel pass)."""
        self.r.texts.append((np.asarray(pos, dtype=float)[:3].copy(), str(string), color))

    def screen_text(self, string, color=(1, 1, 1)):
        """Text centred at the top of the viewport (drawn in the pixel pass)."""
        self.r.screen_texts.append((str(string), color))


def _rgba(c):
    if len(c) == 4:
        return tuple(float(x) for x in c)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


class Renderer:
    def __init__(self, session):
        self.session = session
        self.engine = session.engine
        self.patch_cache = {}
        self.cn_cache = None
        self.base_cache = None
        self.view = None
        self.region = None
        self.texts = []           # (local position, text, color) collected by Drawer.text
        self.screen_texts = []

    # ---- caches ---------------------------------------------------------------------------
    def _patch_data(self, patch):
        cr = self.engine.configRender
        key = (patch.version, cr.phong_tessellation, cr.phong_subdiv, cr.phong_weight, patch.n_vertices(), patch.n_faces())
        entry = self.patch_cache.get(id(patch))
        if entry is not None and entry[0] is patch and entry[1] == key:
            return entry[2]
        P = patch.pn.copy()
        faces = [patch.fv(f) for f in patch.faces()]
        tris = []
        for fv in faces:
            for k in range(1, len(fv) - 1):
                tris.append((fv[0], fv[k], fv[k + 1]))
        tris = np.array(tris, dtype=np.int64).reshape(-1, 3)
        pos, nrm = phong.triangles(P[tris] if len(tris) else np.zeros((0, 3, 6)),
                                   cr.phong_tessellation, cr.phong_subdiv, cr.phong_weight)
        interior = []
        boundary = []
        for e in range(patch.n_edges()):
            v0, v1 = patch.util_edge_to_vertex_pair(e)
            (boundary if patch.is_boundary_e(e) else interior).append((v0, v1))
        il = phong.lines(P[np.array(interior, dtype=np.int64).reshape(-1, 2)], cr.phong_tessellation, cr.phong_subdiv, cr.phong_weight)
        bl = phong.lines(P[np.array(boundary, dtype=np.int64).reshape(-1, 2)], cr.phong_tessellation, cr.phong_subdiv, cr.phong_weight)
        irr3 = []
        irr5 = []
        if not patch.is_failure:
            for v, val in patch.irregular_vertices():
                (irr3 if val == 3 else irr5).append(P[v, :3])
        data = {
            'fill_pos': pos.reshape(-1, 3).astype(np.float32),
            'fill_nrm': nrm.reshape(-1, 3).astype(np.float32),
            'lines_i': il.reshape(-1, 3).astype(np.float32),
            'lines_b': bl.reshape(-1, 3).astype(np.float32),
            'irr3': np.array(irr3, dtype=np.float32).reshape(-1, 3),
            'irr5': np.array(irr5, dtype=np.float32).reshape(-1, 3),
            'batches': {},
        }
        self.patch_cache[id(patch)] = (patch, key, data)
        return data

    def _batch(self, data, name, shader, kind, mirror=False):
        key = (name, mirror)
        b = data['batches'].get(key)
        if b is not None:
            return b
        if name == 'fill':
            pos = data['fill_pos']
            nrm = data['fill_nrm']
            if mirror:
                pos = pos * np.array([-1, 1, 1], dtype=np.float32)
                nrm = nrm * np.array([-1, 1, 1], dtype=np.float32)
            b = batch_for_shader(shader, kind, {'pos': pos, 'nrm': nrm}) if len(pos) else None
        else:
            pos = data[name]
            if mirror:
                pos = pos * np.array([-1, 1, 1], dtype=np.float32)
            b = batch_for_shader(shader, kind, {'pos': pos}) if len(pos) else None
        data['batches'][key] = b
        return b

    def _prune(self):
        alive = {id(p) for p in self.engine.curvenetwork.patches}
        for k in list(self.patch_cache.keys()):
            if k not in alive:
                del self.patch_cache[k]

    def _cn_data(self):
        cn = self.engine.curvenetwork
        if self.cn_cache is not None and self.cn_cache[0] == cn.version and self.cn_cache[1] is cn:
            return self.cn_cache[2]
        free_lines = []
        bound_lines = []
        for e in cn.edgechains:
            if e.is_deleted or e.halfchain[0] is None:
                continue
            target = free_lines if e.num_subdiv == 0 else bound_lines
            for h in ch_iter(e.halfchain[0]):
                v0 = h.vertex
                v1 = h.from_vertex()
                if v0.is_hidden or v1.is_hidden:
                    continue
                target.append(v0.pn[:3])
                target.append(v1.pn[:3])
        endpoints = [v.pn[:3] for v in cn.vertices if not v.is_deleted and not v.is_hidden and v.is_openend()]
        corner_c = ([], [], [])
        corner_t = ([], [], [])
        step = math.pi / 16
        for v in cn.vertices:
            if v.is_deleted or v.is_hidden or not v.is_corner():
                continue
            n = v.pn[3:]
            for h in vih_iter(v):
                if h.is_deleted or h.next is None or h.next.is_deleted:
                    continue
                target = corner_c if (h.is_corner or h.imaginary_patch is not None) else corner_t
                d0 = -h.to_vector()
                d1 = h.next.to_vector()
                d0 = normalized(d0 - np.dot(n, d0) * n)
                d1 = normalized(d1 - np.dot(n, d1) * n)
                ang = acos_clamped(float(np.dot(d0, d1)))
                if np.dot(np.cross(d1, d0), n) < 0:
                    ang = 2 * math.pi - ang
                num_fans = int(ang / step) + 1
                a = ang / num_fans
                # rotate d1 around n by multiples of a (Rodrigues)
                ks = np.arange(num_fans + 1) * a
                c = np.cos(ks)[:, None]
                s = np.sin(ks)[:, None]
                dirs = d1[None, :] * c + np.cross(n, d1)[None, :] * s + n[None, :] * np.dot(n, d1) * (1 - c)
                for k in range(num_fans):
                    for dd in (np.zeros(3), dirs[k], dirs[k + 1]):
                        target[0].append(v.pn[:3])
                        target[1].append(dd)
                        target[2].append(n)
        data = {
            'free': np.array(free_lines, dtype=np.float32).reshape(-1, 3),
            'bound': np.array(bound_lines, dtype=np.float32).reshape(-1, 3),
            'endpoints': np.array(endpoints, dtype=np.float32).reshape(-1, 3),
            'corner_c': [np.array(x, dtype=np.float32).reshape(-1, 3) for x in corner_c],
            'corner_t': [np.array(x, dtype=np.float32).reshape(-1, 3) for x in corner_t],
            'batches': {},
        }
        self.cn_cache = (cn.version, cn, data)
        return data

    def _cn_batch(self, data, name, mirror=False):
        key = (name, mirror)
        if key in data['batches']:
            return data['batches'][key]
        flip = np.array([-1, 1, 1], dtype=np.float32) if mirror else np.ones(3, dtype=np.float32)
        if name in ('corner_c', 'corner_t'):
            pos, dirs, nrm = data[name]
            b = batch_for_shader(_corner_shader(), 'TRIS', {'pos': pos * flip, 'dir': dirs * flip, 'nrm': nrm * flip}) if len(pos) else None
        else:
            pos = data[name]
            kind = 'POINTS' if name == 'endpoints' else 'LINES'
            sh = _builtin('POINT_UNIFORM_COLOR' if kind == 'POINTS' else 'POLYLINE_UNIFORM_COLOR')
            b = batch_for_shader(sh, kind, {'pos': pos * flip}) if len(pos) else None
        data['batches'][key] = b
        return b

    def _base_batches(self):
        bm = self.engine.basemesh
        key = (id(bm), int(bm.face_hidden.sum()), self.engine.configSaved.symmetric)
        if self.base_cache is not None and self.base_cache[0] == key:
            return self.base_cache[1]
        data = {}
        vis = np.flatnonzero(~bm.face_hidden)
        tris = bm.tris[vis]
        data['visible'] = batch_for_shader(_lit_shader(), 'TRIS', {
            'pos': bm.verts[tris].reshape(-1, 3).astype(np.float32),
            'nrm': bm.normals[tris].reshape(-1, 3).astype(np.float32)}) if len(tris) else None
        neg = tris[(bm.verts[tris][:, :, 0] < 0).any(axis=1)]
        off = bm.bbox_diagonal_norm() * 0.0002
        data['negx'] = batch_for_shader(_negx_shader(), 'TRIS', {
            'pos': (bm.verts[neg] + off * bm.normals[neg]).reshape(-1, 3).astype(np.float32)}) if len(neg) else None
        self.base_cache = (key, data)
        return data

    # ---- drawing ------------------------------------------------------------------------------
    def _light_local(self):
        rv3d = self.view.rv3d
        world = rv3d.view_matrix.inverted() @ Vector(self.engine.configRender.light_pos)
        return self.obj.matrix_world.inverted_safe() @ world

    def _eye_local(self):
        if self.view.is_ortho():
            d = normalized(self.view.eye_to_center())
            return (float(d[0]), float(d[1]), float(d[2]), 0.0)
        e = self.view.eye()
        return (float(e[0]), float(e[1]), float(e[2]), 1.0)

    def _lit(self, batch, color, shading, depth_offset=0.0):
        if batch is None:
            return
        sh = _lit_shader()
        sh.bind()
        sh.uniform_float('mvp', _mvp())
        sh.uniform_float('light_pos', self._light_local())
        sh.uniform_float('color', _rgba(color))
        sh.uniform_int('use_shading', 1 if shading else 0)
        sh.uniform_float('eye', self._eye_local())
        sh.uniform_float('depth_offset', float(depth_offset))
        batch.draw(sh)

    def _polyline_batch(self, batch, color, width):
        if batch is None:
            return
        sh = _builtin('POLYLINE_UNIFORM_COLOR')
        sh.bind()
        sh.uniform_float('viewportSize', (self.region.width, self.region.height))
        sh.uniform_float('lineWidth', float(width))
        sh.uniform_float('color', _rgba(color))
        batch.draw(sh)

    def _points_batch(self, batch, color, size):
        if batch is None:
            return
        sh = _builtin('POINT_UNIFORM_COLOR')
        gpu.state.point_size_set(float(size))
        sh.bind()
        sh.uniform_float('color', _rgba(color))
        batch.draw(sh)

    def draw_patch_fill(self, patch, rgba, shading=False, mirror=False, depth_offset=0.0):
        data = self._patch_data(patch)
        self._lit(self._batch(data, 'fill', _lit_shader(), 'TRIS', mirror), rgba, shading, depth_offset)

    def _points(self, arr, color, size, mirror=False):
        if len(arr) == 0:
            return
        if mirror:
            arr = arr * np.array([-1, 1, 1], dtype=np.float32)
        sh = _builtin('POINT_UNIFORM_COLOR')
        self._points_batch(batch_for_shader(sh, 'POINTS', {'pos': arr}), color, size)

    def draw_corners(self, mirror=False):
        cam_d = self.view.camera_distance()
        data = self._cn_data()
        sh = _corner_shader()
        for name, color in (('corner_c', (1, 0, 0.5, 0.5)), ('corner_t', (0, 0.5, 1, 0.5))):
            b = self._cn_batch(data, name, mirror)
            if b is None:
                continue
            sh.bind()
            sh.uniform_float('mvp', _mvp())
            sh.uniform_float('fan_radius', float(cam_d * self.engine.configRender.corner_radius))
            sh.uniform_float('color', color)
            b.draw(sh)

    def draw_view(self, context, view, obj):
        """POST_VIEW callback body."""
        self.view = view
        self.region = context.region
        self.obj = obj
        eng = self.engine
        cr = eng.configRender
        cs = eng.configSaved
        self._prune()
        self.texts = []
        self.screen_texts = []
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.matrix.push()
        gpu.matrix.multiply_matrix(obj.matrix_world)
        try:
            if eng.basemesh.face_hidden.any() and cr.mode != ConfigRender.QUADMESH_ONLY:
                gpu.state.depth_mask_set(True)
                self._lit(self._base_batches()['visible'], (0.8, 0.8, 0.8, 1.0), True)
            if cr.mode == ConfigRender.DEFAULT:
                self._render_default()
            elif cr.mode == ConfigRender.QUADMESH_ONLY:
                self._render_quadmesh_only()
            elif cr.mode == ConfigRender.BASEMESH_ONLY:
                if cs.symmetric and not cr.quadmesh_only_show_left_side:
                    self._draw_negx()
            gpu.state.depth_mask_set(False)
            eng.state.display(Drawer(self))
            self._display_post()
        finally:
            gpu.matrix.pop()
            gpu.state.depth_mask_set(False)
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')

    def _draw_negx(self):
        b = self._base_batches()['negx']
        if b is None:
            return
        sh = _negx_shader()
        sh.bind()
        sh.uniform_float('mvp', _mvp())
        sh.uniform_float('color', (0.0, 0.0, 0.0, 0.3))
        gpu.state.depth_mask_set(False)
        b.draw(sh)

    def _render_default(self):
        eng = self.engine
        cr = eng.configRender
        cs = eng.configSaved
        cn = eng.curvenetwork
        if cs.symmetric:
            self._draw_negx()
        gpu.state.depth_mask_set(False)
        for p in cn.patches:
            if p.is_hidden:
                continue
            self.draw_patch_fill(p, tuple(cr.color_patch) + (cr.color_patch_alpha,))
        if cs.symmetric:
            for p in cn.patches:
                if p.is_hidden:
                    continue
                self.draw_patch_fill(p, (0.5, 0.5, 0.5, 0.5), mirror=True)
        show_quads = cr.always_show_quads or eng.state.edit_level == 'QUAD_MESH'
        polyline = _builtin('POLYLINE_UNIFORM_COLOR')
        if show_quads:
            for p in cn.patches:
                if p.is_failure or p.is_hidden:
                    continue
                d = self._patch_data(p)
                self._polyline_batch(self._batch(d, 'lines_i', polyline, 'LINES'), (0, 0, 0, 1), 1)
            if cs.symmetric:
                for p in cn.patches:
                    if p.is_failure or p.is_hidden:
                        continue
                    d = self._patch_data(p)
                    self._polyline_batch(self._batch(d, 'lines_i', polyline, 'LINES', True), (0, 0, 0, 0.5), 1)
                    self._polyline_batch(self._batch(d, 'lines_b', polyline, 'LINES', True), (0, 0, 0, 0.5), 1)
        data = self._cn_data()
        self._polyline_batch(self._cn_batch(data, 'free'), (1, 0.5, 0, 1), 2)
        self._polyline_batch(self._cn_batch(data, 'bound'), (1, 0.8, 0, 1), 2)
        self.draw_corners()
        self._points_batch(self._cn_batch(data, 'endpoints'), (0, 0.5, 0, 1), 7)
        if show_quads:
            for p in cn.patches:
                if p.is_failure or p.is_hidden:
                    continue
                d = self._patch_data(p)
                self._points(d['irr3'], (0.2, 0.5, 1, 1), 7)
                self._points(d['irr5'], (1, 0.5, 0.2, 1), 7)
        # number of edges of curves without patches
        Drawer(self).num_subdiv_points((0.1, 0.2, 1.0))
        self._axis_bbox()

    def _render_quadmesh_only(self):
        eng = self.engine
        cr = eng.configRender
        cs = eng.configSaved
        cn = eng.curvenetwork
        offset = self.view.camera_distance() * 0.001
        gpu.state.depth_mask_set(True)
        rgba = tuple(cr.color_patch) + (1.0,)
        for p in cn.patches:
            if p.is_hidden:
                continue
            self.draw_patch_fill(p, rgba, shading=True, depth_offset=offset)
        if cs.symmetric:
            for p in cn.patches:
                if p.is_hidden:
                    continue
                self.draw_patch_fill(p, rgba, shading=True, mirror=True, depth_offset=offset)
        polyline = _builtin('POLYLINE_UNIFORM_COLOR')
        for p in cn.patches:
            if p.is_failure or p.is_hidden:
                continue
            d = self._patch_data(p)
            self._polyline_batch(self._batch(d, 'lines_i', polyline, 'LINES'), (0, 0, 0, 1), cr.edge_width_interior)
            if not cr.quadmesh_only_show_auxiliary:
                self._polyline_batch(self._batch(d, 'lines_b', polyline, 'LINES'), (0, 0, 0, 1), cr.edge_width_interior)
            if cs.symmetric:
                self._polyline_batch(self._batch(d, 'lines_i', polyline, 'LINES', True), (0, 0, 0, 1), cr.edge_width_interior)
                if not cr.quadmesh_only_show_auxiliary or not cr.quadmesh_only_show_left_side:
                    self._polyline_batch(self._batch(d, 'lines_b', polyline, 'LINES', True), (0, 0, 0, 1), cr.edge_width_interior)
        if cr.quadmesh_only_show_auxiliary:
            for p in cn.patches:
                if p.is_hidden:
                    continue
                d = self._patch_data(p)
                self._polyline_batch(self._batch(d, 'lines_b', polyline, 'LINES'), (1, 0.8, 0, 1), cr.edge_width_boundary)
                self._points(d['irr3'], (0.2, 0.5, 1, 1), cr.singularity_radius)
                self._points(d['irr5'], (1, 0.5, 0.2, 1), cr.singularity_radius)
                if cs.symmetric and cr.quadmesh_only_show_left_side:
                    self._polyline_batch(self._batch(d, 'lines_b', polyline, 'LINES', True), (1, 0.8, 0, 1), cr.edge_width_boundary)
                    self._points(d['irr3'], (0.2, 0.5, 1, 1), cr.singularity_radius, True)
                    self._points(d['irr5'], (1, 0.5, 0.2, 1), cr.singularity_radius, True)
            self.draw_corners()
            if cs.symmetric and cr.quadmesh_only_show_left_side:
                self.draw_corners(mirror=True)
        gpu.state.depth_mask_set(False)
        self._axis_bbox()

    def _axis_bbox(self):
        cr = self.engine.configRender
        dr = Drawer(self)
        if cr.show_axis:
            o = np.zeros(3)
            dr.lines([o, (1, 0, 0)], (1, 0, 0), 2)
            dr.lines([o, (0, 1, 0)], (0, 1, 0), 2)
            dr.lines([o, (0, 0, 1)], (0, 0, 1), 2)
        if cr.show_bbox:
            lo = self.engine.basemesh.bbox_min
            hi = self.engine.basemesh.bbox_max
            c = [np.array((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
            pts = []
            for a, b in edges:
                pts.extend([c[a], c[b]])
            dr.lines(pts, (1, 1, 1), 2)

    def _display_post(self):
        eng = self.engine
        pn = eng.common_pn_mouse
        if pn is None:
            return
        ct = eng.configTemp
        alpha = 0.1
        state = eng.state
        brush = state.state_color + (alpha,)
        spheres = []
        if ct.quadSize.is_adjusting():
            spheres.append((ct.quadSize() * 0.7, (0, 0.2, 0.5, alpha)))
        elif ct.segmentSize.is_adjusting():
            spheres.append((ct.segmentSize() * 3, (0.5, 0.2, 0, alpha)))
        else:
            if state.show_brushSize_snap():
                k = 1.0 if ct.snapSize.is_adjusting() else 0.8
                spheres.append((ct.snapSize(), (k, k, k, alpha * k)))
            if state is eng.stateSpine:
                k = 1.0 if ct.brushSize_spine.is_adjusting() else 0.8
                spheres.append((ct.brushSize_spine(), tuple(c * k for c in brush)))
            if state is eng.stateAutocmpl and ct.brushSize_autocmpl.is_adjusting():
                spheres.append((ct.brushSize_autocmpl(), brush))
            if state is eng.stateMoveVertex:
                k = 1.0 if ct.brushSize_moveVertex.is_adjusting() else 0.8
                spheres.append((ct.brushSize_moveVertex(), (k, k, 0, alpha * k)))
        unit = _sphere()
        gpu.state.depth_mask_set(False)
        for radius, color in spheres:
            if radius <= 0:
                continue
            pos = (unit * radius + pn[:3]).astype(np.float32)
            b = batch_for_shader(_lit_shader(), 'TRIS', {'pos': pos, 'nrm': unit.astype(np.float32)})
            self._lit(b, color, True)
        Drawer(self).points([pn[:3]], (1, 0, 0), 3, depth=False)

    def draw_pixel(self, context):
        """POST_PIXEL callback body: hiding stroke and current mode name."""
        eng = self.engine
        gpu.state.blend_set('ALPHA')
        hs = eng.hide_stroke
        if not hs.empty() and hs.size() >= 2:
            sh = _builtin('POLYLINE_UNIFORM_COLOR')
            batch = batch_for_shader(sh, 'LINE_STRIP', {'pos': [(float(p[0]), float(p[1]), 0.0) for p in hs.points]})
            sh.bind()
            sh.uniform_float('viewportSize', (context.region.width, context.region.height))
            sh.uniform_float('lineWidth', 5.0)
            sh.uniform_float('color', (1, 0, 0, 1))
            batch.draw(sh)
        font_id = 0
        region = context.region
        if self.texts and self.view is not None:
            blf.size(font_id, 18)
            blf.enable(font_id, blf.SHADOW)
            blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 1.0)
            blf.shadow_offset(font_id, 1, -1)
            xy = self.view.project(np.array([t[0] for t in self.texts]))
            for (pos, text, color), (x, y) in zip(self.texts, xy):
                w, h = blf.dimensions(font_id, text)
                blf.color(font_id, color[0], color[1], color[2], 1.0)
                blf.position(font_id, float(x) - w / 2, float(y) - h / 2, 0)
                blf.draw(font_id, text)
            blf.disable(font_id, blf.SHADOW)
        # status texts at the top center, below the header bars when they overlap the viewport
        top = region.height
        for other in context.area.regions:
            if other.type in ('HEADER', 'TOOL_HEADER') and other.height > 1 and region.y < other.y < region.y + region.height:
                top = min(top, other.y - region.y)
        y = top - 30
        for text, color in self.screen_texts:
            blf.size(font_id, 18)
            for line in text.split('\n'):
                w, h = blf.dimensions(font_id, line)
                blf.color(font_id, color[0], color[1], color[2], 1.0)
                blf.position(font_id, (region.width - w) / 2, y, 0)
                blf.draw(font_id, line)
                y -= 24
        # current mode name at the bottom right (as SketchRetopo2)
        state = eng.state
        blf.size(font_id, 18)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        w, h = blf.dimensions(font_id, state.state_name)
        right = region.width
        for other in context.area.regions:
            # keep clear of the sidebar when it overlaps the viewport
            if other.type == 'UI' and other.width > 1 and region.x < other.x < region.x + region.width:
                right = min(right, other.x - region.x)
        blf.position(font_id, right - 20 - w, 20, 0)
        blf.draw(font_id, state.state_name)
        gpu.state.blend_set('NONE')
