import math
import os
import tempfile
import time

import numpy as np

from helpers import grid_mesh, uv_sphere, make_engine, stroke, click, hover, TopView
from sketch_retopo.core.basemesh import BaseMesh
from sketch_retopo.core.engine import Engine
from sketch_retopo.core import fileio


def quad_count(eng):
    return sum(p.n_faces() for p in eng.curvenetwork.patches)


def sides(eng):
    return sorted(p.num_corners() for p in eng.curvenetwork.patches)


def cylinder_mesh(nu=64, nx=80, r=0.5, length=3.0):
    verts = []
    for i in range(nx + 1):
        x = -length / 2 + length * i / nx
        for j in range(nu):
            a = 2 * math.pi * j / nu
            verts.append((x, r * math.cos(a), r * math.sin(a)))
    tris = []
    for i in range(nx):
        for j in range(nu):
            a = i * nu + j
            b = i * nu + (j + 1) % nu
            c = (i + 1) * nu + (j + 1) % nu
            d = (i + 1) * nu + j
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.array(verts), np.array(tris)


def test_nsided():
    V, T = grid_mesh(60, 2.0)
    eng = make_engine(V, T)
    # triangle
    stroke(eng, [(-0.8, -0.8), (-0.2, -0.8)])
    stroke(eng, [(-0.2, -0.8), (-0.5, -0.3)])
    stroke(eng, [(-0.5, -0.3), (-0.8, -0.8)])
    # pentagon
    pts = [(0.5 + 0.35 * math.cos(2 * math.pi * k / 5 + 0.3), 0.4 + 0.35 * math.sin(2 * math.pi * k / 5 + 0.3)) for k in range(5)]
    for k in range(5):
        stroke(eng, [pts[k], pts[(k + 1) % 5]])
    # hexagon
    pts = [(-0.4 + 0.33 * math.cos(2 * math.pi * k / 6), 0.45 + 0.33 * math.sin(2 * math.pi * k / 6)) for k in range(6)]
    for k in range(6):
        stroke(eng, [pts[k], pts[(k + 1) % 6]])
    # 2-sided: two arcs
    arc1 = [(0.2 + 0.5 * t, -0.7 + 0.25 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 12)]
    arc2 = [(0.7 - 0.5 * t, -0.7 - 0.25 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 12)]
    stroke(eng, arc1)
    stroke(eng, arc2)
    print('n-sided patches:', sides(eng), 'failures', [p.is_failure for p in eng.curvenetwork.patches])
    assert sides(eng) == [2, 3, 5, 6], sides(eng)
    return eng


def test_tjunction():
    V, T = grid_mesh(60, 2.0)
    eng = make_engine(V, T)
    for a, b in [((-0.6, -0.4), (0.6, -0.4)), ((0.6, -0.4), (0.6, 0.4)), ((0.6, 0.4), (-0.6, 0.4)), ((-0.6, 0.4), (-0.6, -0.4))]:
        stroke(eng, [a, b])
    # curve from the middle of the bottom side up to the middle of the top side -> two patches with T-junction ends? (snaps to sides)
    stroke(eng, [(0.0, -0.4), (0.0, 0.4)])
    print('T-junction split:', sides(eng), [p.param.l for p in eng.curvenetwork.patches])
    # half curve from the middle: dangling T-junction removes nothing but no new patch
    stroke(eng, [(0.3, -0.4), (0.3, 0.0)])
    print('dangling curve: patches', len(eng.curvenetwork.patches))
    # edit corner: ctrl+click a side vertex (valence 2) of the right patch top side to create a corner
    eng.key_down('t')
    before = sides(eng)
    click(eng, -0.3, 0.4, ctrl=True)
    print('edit corner (ctrl+click):', before, '->', sides(eng))
    return eng


def test_laser_cylinder():
    V, T = cylinder_mesh()
    view = TopView(scale=0.004, distance=4.0)
    eng = make_engine(V, T, view)
    eng.key_down('e')
    stroke(eng, [(-0.61, -0.45), (-0.61, 0.45)], steps=40)
    stroke(eng, [(0.61, -0.45), (0.61, 0.45)], steps=40)
    loops = [e for e in eng.curvenetwork.edgechains]
    print('laser: edgechains', len(loops), 'vertices', len(eng.curvenetwork.vertices))
    # SketchRetopo2 does not trace the open ends of the cylinder automatically (B / N + click does)
    assert len(loops) == 2
    eng.key_down('r')
    stroke(eng, [(-0.6, 0.0), (0.6, 0.0)])
    print('cylinder: patches', len(eng.curvenetwork.patches), sides(eng), 'quads', quad_count(eng))
    assert len(eng.curvenetwork.patches) == 4, len(eng.curvenetwork.patches)
    return eng


def test_spine_autocmpl():
    V, T = grid_mesh(80, 2.0)
    eng = make_engine(V, T)
    eng.key_down('y')
    t0 = time.time()
    stroke(eng, [(-0.6, 0.0), (0.0, 0.05), (0.5, 0.0)])
    print('spine: patches', len(eng.curvenetwork.patches), sides(eng), 'quads', quad_count(eng), '%.2fs' % (time.time() - t0))
    assert len(eng.curvenetwork.patches) == 1
    # auto-completion along the top boundary of the spine patch
    eng.key_down('u')
    st = eng.stateAutocmpl
    x0, y0 = eng.view.to_pixel(-0.2, 0.05 + eng.configTemp.brushSize_spine())
    p = eng.curvenetwork.patches[0]
    # find a point on the upper boundary
    P = p.pn[:, :3]
    top = P[np.argmax(P[:, 1])]
    hover(eng, top[0], top[1])
    x, y = eng.view.to_pixel(top[0], top[1])
    eng.mouse_down(x, y, 'LEFT', False, False, False)
    for k in range(1, 8):
        xx, yy = eng.view.to_pixel(top[0] + 0.02 * k, top[1])
        eng.mouse_move(xx, yy)
    st.last_time = 0
    t0 = time.time()
    st.update_preview()
    print('autocmpl suggestions', len(st.suggested_autocmpl), '%.2fs' % (time.time() - t0))
    eng.mouse_up(xx, yy, 'LEFT', False, False, False)
    if st.selected_autocmpl_index >= 0:
        click(eng, top[0], top[1] + 0.1)
    print('after accepting: patches', len(eng.curvenetwork.patches), sides(eng))
    return eng


def test_deform_move_smooth():
    V, T = grid_mesh(60, 2.0, z_func=lambda x, y: 0.1 * math.sin(2 * x) * math.cos(2 * y))
    eng = make_engine(V, T)
    for a, b in [((-0.5, -0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5)), ((0.5, 0.5), (-0.5, 0.5)), ((-0.5, 0.5), (-0.5, -0.5))]:
        stroke(eng, [a, b])
    eng.key_down('w')
    stroke(eng, [(0.5, 0.5), (0.6, 0.65)])   # drag corner
    p = eng.curvenetwork.patches[0]
    print('deform corner: patches', len(eng.curvenetwork.patches), 'max x', round(float(p.pn[:, 0].max()), 3))
    stroke(eng, [(0.0, -0.5), (0.0, -0.6)])  # drag side
    eng.key_down('d')
    before = eng.curvenetwork.patches[0].pn.copy()
    stroke(eng, [(0.0, 0.0), (0.05, 0.05)])
    after = eng.curvenetwork.patches[0].pn
    print('move vertex: moved', int((np.linalg.norm(after[:, :3] - before[:, :3], axis=1) > 1e-6).sum()), 'vertices')
    stroke(eng, [(0.0, 0.0), (0.05, 0.05)], shift=True)
    eng.key_down('g')
    # smoothing a curve with shift+drag in sketch mode
    eng.key_down('q')
    stroke(eng, [(0.2, -0.55), (0.3, -0.55)], shift=True)
    stroke(eng, [(0.2, -0.55), (0.3, -0.55)], shift=True, ctrl=True)
    print('smoothing/snakes ok, patches', len(eng.curvenetwork.patches))
    # topology: 4-sided patch with unequal sides, then drag the singularity
    eng.key_down('a')
    click(eng, 0.0, -0.55)
    eng.key_down('c')
    eng.key_down('c')
    p = eng.curvenetwork.patches[0]
    print('edit topo l', p.param.l, 'pattern', p.param.pattern_id)
    click(eng, 0.0, 0.0)
    st = eng.stateEditTopology
    print('selected patch', st.current_patch is not None)
    if st.current_patch is not None:
        x, y = eng.view.to_pixel(0.0, 0.0)
        eng.mouse_down(x, y, 'LEFT', False, False, False)
        for k in range(1, 30):
            eng.mouse_move(*eng.view.to_pixel(0.012 * k, 0.0))
        eng.mouse_up(*eng.view.to_pixel(0.35, 0.0), 'LEFT', False, False, False)
        print('after dragging singularity: pattern', p.param.pattern_id, 'p', p.param.p)
        eng.key_down('b')
        print('switch pattern ->', p.param.pattern_id, p.param.permutation.id)
    return eng


def test_symmetry_xml_hide():
    V, T = uv_sphere()
    view = TopView(scale=0.003, distance=4.0)
    eng = make_engine(V, T, view)
    # SketchRetopo2: symmetry is a mirrored preview; curves are drawn freely and snapped to X=0 on demand
    eng.set_symmetric(True)
    x0 = 0.004
    for a, b in [((x0, 0.3), (0.5, 0.3)), ((0.5, 0.3), (0.5, -0.3)), ((0.5, -0.3), (x0, -0.3)), ((x0, -0.3), (x0, 0.3))]:
        stroke(eng, [a, b])
    print('symmetric patches', sides(eng), 'quads', quad_count(eng))
    assert len(eng.curvenetwork.patches) == 1
    Vr, Fr = eng.retopo_mesh()
    n_before = len(Vr)
    eng.snap_symmetry()
    p = eng.curvenetwork.patches[0]
    on_plane = int((p.pn[:, 0] == 0).sum())
    Vr, Fr = eng.retopo_mesh()
    print('snap symmetry: %d patch vertices on X=0; retopo mesh %d -> %d verts, %d faces' % (on_plane, n_before, len(Vr), len(Fr)))
    assert on_plane > 0 and len(Vr) == n_before - on_plane and len(Fr) == 2 * quad_count(eng)
    fd, path = tempfile.mkstemp(suffix='.xml')
    os.close(fd)
    fileio.save_xml(eng, path, eng.basemesh.verts, eng.basemesh.tris)
    text = open(path).read()
    # SketchRetopo2 format: no version attribute, 7 columns per halfedge, no pattern parameters
    assert '<sketchretopo>' in text and '<param' not in text
    data = fileio.read_xml(path)
    eng2 = Engine()
    eng2.view = view
    eng2.set_basemesh(BaseMesh(data['verts'], data['tris']), reset_curves=False)
    data['restore'](eng2)
    eng2.update_camera_distance()
    print('xml roundtrip: patches', sides(eng2), 'quads', quad_count(eng2), 'symmetric', eng2.configSaved.symmetric)
    assert sides(eng2) == sides(eng) and quad_count(eng2) == quad_count(eng) and eng2.configSaved.symmetric
    # edit after load
    eng2.key_down('a')
    click(eng2, 0.5, 0.0)
    eng2.key_down('c')
    print('edit after load: quads', quad_count(eng2))
    os.remove(path)
    # hide stroke
    eng2.key_down('h')
    stroke(eng2, [(-1.2, 0.0), (1.2, 0.05)])
    eng2.key_up('h')
    print('hidden faces', int(eng2.basemesh.face_hidden.sum()), 'of', len(eng2.basemesh.tris))
    eng2.key_down('h')
    click(eng2, 0.0, 0.0)
    eng2.key_up('h')
    print('after reset', int(eng2.basemesh.face_hidden.sum()))
    return eng2


def test_sketchretopo2_tools():
    V, T = grid_mesh(60, 2.0)
    eng = make_engine(V, T)
    square = [((-0.5, -0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5)), ((0.5, 0.5), (-0.5, 0.5)), ((-0.5, 0.5), (-0.5, -0.5))]
    for a, b in square:
        stroke(eng, [a, b])
    assert len(eng.curvenetwork.patches) == 1
    # V + click deletes the patch, its curves keep their number of edges
    eng.key_down('v')
    click(eng, 0.0, 0.0)
    eng.key_up('v')
    spans = sorted(e.num_subdiv for e in eng.curvenetwork.edgechains)
    print('V+click: patches', len(eng.curvenetwork.patches), 'curve spans', spans)
    assert len(eng.curvenetwork.patches) == 0 and min(spans) > 0
    # X + drag from a curve into the empty region generates the patch again
    eng.key_down('x')
    stroke(eng, [(0.0, -0.5), (0.0, -0.2)])
    eng.key_up('x')
    print('X+drag: patches', len(eng.curvenetwork.patches), sides(eng))
    assert len(eng.curvenetwork.patches) == 1
    # topology editing: +1 on one curve leaves the region empty (odd total), +1 on another fixes it
    eng.key_down('a')
    click(eng, 0.0, -0.5)
    eng.key_down('v')
    eng.key_up('v')
    print('V (+1 edge): patches', len(eng.curvenetwork.patches))
    assert len(eng.curvenetwork.patches) == 0
    click(eng, 0.0, 0.5)
    eng.key_down('v')
    eng.key_up('v')
    # as in SketchRetopo2, the empty region is filled again by the next sketch action (or X+drag)
    print('V on the opposite curve: patches', len(eng.curvenetwork.patches))
    eng.key_down('q')
    stroke(eng, [(0.8, -0.9), (0.9, -0.8)])
    print('next stroke: patches', len(eng.curvenetwork.patches), [p.num_subdiv_all() for p in eng.curvenetwork.patches])
    assert len(eng.curvenetwork.patches) == 1 and sum(eng.curvenetwork.patches[0].num_subdiv_all()) == 50
    # B + click near the grid boundary: curve along the base mesh boundary
    eng.key_down('q')
    n_edgechains = len(eng.curvenetwork.edgechains)
    eng.key_down('b')
    click(eng, 0.99, 0.3)
    eng.key_up('b')
    print('B+click: edgechains', n_edgechains, '->', len(eng.curvenetwork.edgechains))
    assert len(eng.curvenetwork.edgechains) == n_edgechains + 1
    eng.key_down('ctrl+z')
    eng.key_down('n')
    click(eng, 0.99, 0.3)
    eng.key_up('n')
    loop = [e for e in eng.curvenetwork.edgechains if e.halfchain[0].is_loop()]
    print('N+click: boundary loop inset', [round(float(min(abs(h.vertex.pn[0]) for h in [e.halfchain[0].halfedge_front])), 3) for e in loop])
    assert len(loop) == 1
    return eng


if __name__ == '__main__':
    for fn in (test_nsided, test_tjunction, test_laser_cylinder, test_spine_autocmpl, test_deform_move_smooth,
               test_symmetry_xml_hide, test_sketchretopo2_tools):
        t0 = time.time()
        print('==', fn.__name__)
        fn()
        print('   (%.2fs)' % (time.time() - t0))
