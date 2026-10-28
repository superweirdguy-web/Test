"""End-to-end GUI test: runs inside a real Blender window and drives the modal operator with
simulated input events.  Usage (Linux, headless):

    xvfb-run -a -s "-screen 0 1600x1000x24" blender --factory-startup --enable-event-simulate \
        --python tests/gui_test.py -- <output_dir>
"""
import math
import os
import sys
import traceback

import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)

import sketch_retopo  # noqa: E402
from sketch_retopo import session  # noqa: E402

OUT = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else '/tmp/sketch_retopo_gui'
os.makedirs(OUT, exist_ok=True)
LOG = []


def log(*args):
    msg = ' '.join(str(a) for a in args)
    LOG.append(msg)
    print('[gui_test]', msg, flush=True)


def view3d():
    window = bpy.context.window_manager.windows[0]
    area = max((a for a in window.screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return window, area, region, area.spaces.active.region_3d


def to_window(world):
    window, area, region, rv3d = view3d()
    co = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(world))
    return int(region.x + co.x), int(region.y + co.y)


def sim(type_, value, xy, **mods):
    window = view3d()[0]
    window.event_simulate(type_, value, x=xy[0], y=xy[1], **mods)


def drag(points_world, button='LEFTMOUSE', steps=12, **mods):
    pts = [to_window(p) for p in points_world]
    sim('MOUSEMOVE', 'NOTHING', pts[0], **mods)
    sim(button, 'PRESS', pts[0], **mods)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        for k in range(1, steps + 1):
            t = k / steps
            sim('MOUSEMOVE', 'NOTHING', (int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))), **mods)
    sim(button, 'RELEASE', pts[-1], **mods)


def click(p, **mods):
    xy = to_window(p)
    sim('MOUSEMOVE', 'NOTHING', xy, **mods)
    sim('LEFTMOUSE', 'PRESS', xy, **mods)
    sim('LEFTMOUSE', 'RELEASE', xy, **mods)


def key(k, **mods):
    window, area, region, rv3d = view3d()
    xy = (region.x + region.width // 2, region.y + region.height // 2)
    sim(k, 'PRESS', xy, **mods)
    sim(k, 'RELEASE', xy, **mods)


def hover(p):
    sim('MOUSEMOVE', 'NOTHING', to_window(p))


def screenshot(name):
    """Grabs the 3D view from the X display (Blender's own screenshot reads back black under Xvfb)."""
    import subprocess
    window, area, region, rv3d = view3d()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    path = os.path.join(OUT, name + '.png')
    top = window.height - (area.y + area.height)
    geom = '%dx%d+%d+%d' % (area.width, area.height, area.x, top)
    subprocess.run(['import', '-display', os.environ.get('DISPLAY', ':99'), '-window', 'root', '-crop', geom, path],
                   check=False)
    log('screenshot', path)


def engine():
    s = session.running_session()
    return s.engine if s else None


def on_sphere(x, z, r=1.0):
    """Point on the front (-Y) side of the unit sphere seen from the front view."""
    y = -math.sqrt(max(r * r - x * x - z * z, 0.0))
    return (x, y, z)


def steps():
    sketch_retopo.register()
    window, area, region, rv3d = view3d()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    sphere = bpy.context.active_object
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected()
    area.spaces.active.overlay.show_floor = False
    area.spaces.active.overlay.show_axis_x = False
    area.spaces.active.overlay.show_axis_y = False
    yield 0.3
    with bpy.context.temp_override(window=window, area=area, region=region):
        res = bpy.ops.sketch_retopo.sketch('INVOKE_DEFAULT')
    log('start operator', res, 'running', session.running_session() is not None)
    yield 0.3
    eng = engine()
    assert eng is not None

    # --- basic sketching: a 4-sided patch and a crossing stroke
    s = 0.45
    corners = [on_sphere(-s, -s), on_sphere(s, -s), on_sphere(s, s), on_sphere(-s, s)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        drag([a, b])
        yield 0.05
    log('square: patches', len(eng.curvenetwork.patches), [p.num_corners() for p in eng.curvenetwork.patches])
    drag([on_sphere(0.0, -0.6), on_sphere(0.0, 0.6)])
    yield 0.1
    log('split: patches', len(eng.curvenetwork.patches))
    screenshot('01_sketch')

    # undo / redo with Ctrl+Z / Ctrl+Y
    key('Z', ctrl=True)
    yield 0.05
    n_undo = len(eng.curvenetwork.patches)
    key('Y', ctrl=True)
    yield 0.05
    log('undo/redo: patches', n_undo, '->', len(eng.curvenetwork.patches))

    # erase open ends with G
    key('G')
    yield 0.05

    # --- topology editing: select the middle curve and add two subdivisions
    key('A')
    yield 0.05
    click(on_sphere(0.0, 0.1))
    yield 0.05
    key('C')
    yield 0.1
    log('edit topology:', [p.num_subdiv_all() for p in eng.curvenetwork.patches])
    screenshot('02_edit_topology')

    # --- edge loop
    key('F')
    hover(on_sphere(-0.27, -0.45))
    yield 0.1
    screenshot('03_edge_loop_preview')
    q0 = sum(p.n_faces() for p in eng.curvenetwork.patches)
    click(on_sphere(-0.27, -0.45))
    yield 0.1
    log('edge loop: quads', q0, '->', sum(p.n_faces() for p in eng.curvenetwork.patches))

    # --- spine sketching (beside the square)
    key('Y')
    drag([on_sphere(0.65, -0.5), on_sphere(0.7, 0.0), on_sphere(0.65, 0.5)])
    yield 0.2
    log('spine: patches', len(eng.curvenetwork.patches))

    # --- triangle with a corner edit
    key('Q')
    tri = [on_sphere(-0.85, -0.2), on_sphere(-0.55, -0.15), on_sphere(-0.7, 0.25)]
    for a, b in zip(tri, tri[1:] + tri[:1]):
        drag([a, b])
        yield 0.05
    log('triangle: sides', sorted(p.num_corners() for p in eng.curvenetwork.patches))

    # --- move vertex + right-drag brush
    key('D')
    drag([on_sphere(-0.2, 0.2), on_sphere(-0.15, 0.25)])
    drag([on_sphere(0.0, 0.0), on_sphere(0.1, 0.0)], button='RIGHTMOUSE')
    yield 0.1
    hover(on_sphere(-0.2, 0.0))
    yield 0.05
    screenshot('04_move_vertex_brush')

    # --- auto-completion preview along the top of the square
    key('U')
    top = [on_sphere(-0.35 + 0.05 * k, s + 0.003) for k in range(10)]
    pts = [to_window(p) for p in top]
    sim('MOUSEMOVE', 'NOTHING', pts[0])
    sim('LEFTMOUSE', 'PRESS', pts[0])
    for p in pts[1:]:
        sim('MOUSEMOVE', 'NOTHING', p)
    yield 0.6
    st = eng.stateAutocmpl
    sim('LEFTMOUSE', 'RELEASE', pts[-1])
    yield 0.1
    log('autocmpl suggestions', len(st.suggested_autocmpl), 'selected', st.selected_autocmpl_index)
    screenshot('05_autocomplete_preview')
    if st.selected_autocmpl_index >= 0:
        click(on_sphere(0.0, 0.8))
        yield 0.2
    log('after auto-completion: patches', len(eng.curvenetwork.patches))

    # --- render modes
    key('SPACE')
    yield 0.1
    screenshot('06_quad_only')
    key('SPACE')
    yield 0.05

    # --- Alt+drag camera rotate
    rot0 = tuple(rv3d.view_rotation)
    drag([on_sphere(0.0, 0.0), on_sphere(0.3, 0.0)], alt=True)
    yield 0.1
    log('alt-rotate changed view:', tuple(rv3d.view_rotation) != rot0)
    screenshot('07_rotated')

    # --- side views to show that the quads follow the surface
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_axis(type='RIGHT')
    yield 0.1
    screenshot('08_side')
    key('SPACE')
    yield 0.1
    screenshot('09_side_quad_only')
    key('SPACE')
    yield 0.05

    # --- export and finish
    key('E', ctrl=True)
    yield 0.1
    out = bpy.context.scene.sketch_retopo.output_object
    log('export object', out.name if out else None, len(out.data.polygons) if out else 0)
    # every exported vertex should lie on the unit sphere (plus the small projection offset)
    radii = [(out.matrix_world @ v.co).length for v in out.data.vertices]
    log('export radius min %.4f max %.4f' % (min(radii), max(radii)))
    screenshot('10_side_after_export')
    log('export object hidden while running', out.hide_get())
    key('ESC')
    yield 0.2
    log('running after Esc', session.running_session() is not None, 'export object hidden', out.hide_get())


def main():
    gen = steps()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            finish(0)
        except Exception:
            traceback.print_exc()
            finish(1)
        return None

    def finish(code):
        with open(os.path.join(OUT, 'log.txt'), 'w') as f:
            f.write('\n'.join(LOG) + '\n')
        log('done', code)
        os._exit(code)

    bpy.app.timers.register(tick, first_interval=1.0)


main()
