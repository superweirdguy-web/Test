"""GUI test of the query database mode with a real patch database: runs inside a Blender window and drives the
modal operator with simulated input events.  Usage (Linux, headless):

    xvfb-run -a -s "-screen 0 1600x1000x24" blender --enable-event-simulate \\
        --python tests/gui_test_querydb.py -- <output_dir> <path/to/patches.db>

Screenshots go to <output_dir>; frames for an animation to <output_dir>/frames.
"""
import math
import os
import sys
import time
import traceback

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gui_test import (view3d, to_window, sim, drag, click, key, screenshot, engine, on_sphere, log, LOG,  # noqa: E402
                      OUT)
import sketch_retopo  # noqa: E402
from sketch_retopo import session  # noqa: E402
from sketch_retopo.core.curvenetwork import delete_patch  # noqa: E402

DB = sys.argv[sys.argv.index('--') + 2] if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 2 else ''
FRAMES = os.path.join(OUT, 'frames')
os.makedirs(FRAMES, exist_ok=True)
_frame = [0]
_delays = []


def frame(delay=50):
    """Saves the 3D view as the next animation frame, shown for `delay` hundredths of a second."""
    import subprocess
    window, area, region, rv3d = view3d()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    path = os.path.join(FRAMES, 'f%04d.png' % _frame[0])
    _frame[0] += 1
    top = window.height - (area.y + area.height)
    geom = '%dx%d+%d+%d' % (area.width, area.height, area.x, top)
    subprocess.run(['import', '-display', os.environ.get('DISPLAY', ':99'), '-window', 'root', '-crop', geom, path],
                   check=False)
    _delays.append(delay)
    with open(os.path.join(FRAMES, 'delays.txt'), 'w') as f:
        f.write('\n'.join(str(d) for d in _delays) + '\n')


def drag_slow(points_world, steps=10, frames_every=3, **mods):
    """Like drag(), but lets Blender run between moves and records frames while drawing."""
    pts = [to_window(p) for p in points_world]
    sim('MOUSEMOVE', 'NOTHING', pts[0], **mods)
    sim('LEFTMOUSE', 'PRESS', pts[0], **mods)
    k = 0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        for i in range(1, steps + 1):
            t = i / steps
            sim('MOUSEMOVE', 'NOTHING', (int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))), **mods)
            k += 1
            if k % frames_every == 0:
                yield 0.02
                frame(10)
    sim('LEFTMOUSE', 'RELEASE', pts[-1], **mods)
    yield 0.05


def hold(seconds, every=1.0):
    """Lets the lookup run, recording frames."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        yield every
        frame()


def set_spans(eng, spans):
    """Gives the only patch's curves the given numbers of edges and regenerates the patch."""
    cn = eng.curvenetwork
    for p in list(cn.patches):
        delete_patch(p)
    for e, n in zip(cn.edgechains, spans):
        e.num_subdiv = n
    eng.generate_patches()
    cn.garbage_collect()
    cn.invalidate_displist()


def show_query_panel():
    """Opens the sidebar on the add-on's panels with the Query Database panel expanded (test only: Blender 4.0 has
    no API to pick a sidebar tab, so the panels are registered again in the default one)."""
    from sketch_retopo import ui
    for c in reversed(ui.classes):
        bpy.utils.unregister_class(c)
    for c in ui.classes:
        c.bl_category = 'Item'
        if c.__name__ == 'SKETCHRETOPO_PT_query_db':
            c.bl_options = set()
        elif c.__name__ != 'SKETCHRETOPO_PT_main':
            c.bl_options = {'DEFAULT_CLOSED'}
    for c in ui.classes:
        bpy.utils.register_class(c)
    view3d()[1].spaces.active.show_region_ui = True


def steps():
    sketch_retopo.register()
    window, area, region, rv3d = view3d()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1.0)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.shade_smooth()
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected()
    sp = area.spaces.active
    sp.overlay.show_floor = False
    sp.overlay.show_axis_x = False
    sp.overlay.show_axis_y = False
    sp.overlay.show_ortho_grid = False
    sp.overlay.show_object_origins = False
    sp.overlay.show_outline_selected = False
    sp.overlay.show_text = False
    rv3d.view_distance *= 0.6
    yield 0.3
    with bpy.context.temp_override(window=window, area=area, region=region):
        res = bpy.ops.sketch_retopo.sketch('INVOKE_DEFAULT')
    log('start operator', res)
    yield 0.3
    eng = engine()
    assert eng is not None
    res = bpy.ops.sketch_retopo.load_patch_db(filepath=DB)
    log('load database', res, eng.learnquad.num_templates(), 'templates')
    assert eng.learnquad.is_loaded()

    # --- a 4-sided patch
    s = 0.42
    corners = [on_sphere(-s, -s), on_sphere(s, -s), on_sphere(s, s), on_sphere(-s, s)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        drag([a, b])
        yield 0.05
    set_spans(eng, [10, 8, 10, 8])
    yield 0.1
    patch = eng.curvenetwork.patches[0]
    log('patch spans', patch.num_subdiv_all(), 'quads', patch.n_faces())
    frame(150)
    screenshot('11_patch')

    # --- query database: select the patch, draw a flow
    key('S')
    yield 0.1
    click(on_sphere(0.05, 0.05))
    yield 0.1
    st = eng.stateQueryDB
    log('selected', st.selected_patch is patch, 'boundary', st.querydata.num_subdiv)
    frame(120)
    screenshot('12_selected')
    yield from drag_slow([on_sphere(-0.40, -0.12), on_sphere(-0.15, -0.06), on_sphere(0.15, 0.06),
                          on_sphere(0.40, 0.12)])
    log('flows', {k: [(f.start_edge, f.end_edge) for f in v]
                  for k, v in eng.learnquad.retopologizer.flow_map_uv.items()}, 'status', st.status)
    t0 = time.time()
    lk = eng.learnquad.lookup
    rt = eng.learnquad.retopologizer
    log('retopologizer: base tris', 0 if rt.locator is None else len(rt.locator.tris), 'boundary sides',
        [len(b) for b in rt.boundary_uv], 'lookup', lk.status, lk._phase)
    yield from hold(4.0)
    log('lookup phase %s, candidate %d/%d, templates %d' % (lk._phase, lk._ci, len(lk._cand), lk.n_templates))
    log('after %.1fs: %d results, status %r' % (time.time() - t0, eng.learnquad.num_results(), st.status_msg))
    screenshot('13_lookup_progress')
    yield from hold(6.0)
    log('after %.1fs: %d results, status %r' % (time.time() - t0, eng.learnquad.num_results(), st.status_msg))
    bpy.ops.sketch_retopo.query_stop()
    yield 0.1
    log('stopped: status %r' % st.status_msg)
    assert eng.learnquad.num_results() > 0
    best = patch.n_faces()
    screenshot('14_best_result')
    for i in range(3):
        key('X')
        yield 0.3
        frame(110)
    log('browsed to index', st.lookup_index, 'quads', patch.n_faces(), '(best had %d)' % best)
    screenshot('15_next_result')
    show_query_panel()
    yield 0.3
    window, area, region, rv3d = view3d()
    ui_region = next(r for r in area.regions if r.type == 'UI')
    for _ in range(12):
        with bpy.context.temp_override(window=window, area=area, region=ui_region):
            bpy.ops.view2d.scroll_down(deltax=0, deltay=400, page=True)
        yield 0.05
    screenshot('15b_sidebar')
    view3d()[1].spaces.active.show_region_ui = False
    yield 0.1
    key('Z')
    key('Z')
    key('Z')
    yield 0.2

    # --- a second flow across the first one (Shift)
    yield from drag_slow([on_sphere(0.25, -0.40), on_sphere(0.1, 0.0), on_sphere(-0.2, 0.40)], shift=True)
    t0 = time.time()
    yield from hold(6.0)
    bpy.ops.sketch_retopo.query_stop()
    yield 0.1
    log('two flows: %d results after %.1fs, status %r' % (eng.learnquad.num_results(), time.time() - t0, st.status_msg))
    screenshot('16_two_flows')
    frame(200)

    # --- quad-only look at the chosen result
    key('Q')
    yield 0.1
    key('SPACE')
    yield 0.1
    screenshot('17_quad_only')
    frame(250)
    key('SPACE')
    yield 0.05

    # --- a 7-corner loop: filled from the database with X + drag (no pattern exists for it)
    r = 0.33
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_axis(type='TOP')
    yield 0.2
    # points on the top cap of the sphere (z up), seen from above
    hept = [(r * math.cos(2 * math.pi * k / 7 + 0.2), r * math.sin(2 * math.pi * k / 7 + 0.2),
             math.sqrt(max(1 - r * r, 0.0))) for k in range(7)]
    n_before = len(eng.curvenetwork.patches)
    for a, b in zip(hept, hept[1:] + hept[:1]):
        drag([a, b])
        yield 0.05
    loop_chains = eng.curvenetwork.edgechains[-7:]
    for e in loop_chains:
        e.num_subdiv = 4
    mid = [(hept[0][i] + hept[1][i]) / 2 for i in range(3)]
    inner = [mid[0] * 0.5, mid[1] * 0.5, math.sqrt(max(1 - (mid[0] * 0.5) ** 2 - (mid[1] * 0.5) ** 2, 0.0))]
    window.event_simulate('X', 'PRESS', x=to_window(mid)[0], y=to_window(mid)[1])
    drag([mid, inner])
    window.event_simulate('X', 'RELEASE', x=to_window(inner)[0], y=to_window(inner)[1])
    yield 0.3
    new = [p for p in eng.curvenetwork.patches if p.num_corners() == 7]
    log('7-corner loop: patches', n_before, '->', len(eng.curvenetwork.patches),
        'with 7 corners:', [(p.num_subdiv_all(), p.n_faces()) for p in new])
    screenshot('18_seven_corners')
    frame(300)

    key('ESC')
    yield 0.2


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
        with open(os.path.join(OUT, 'log_querydb.txt'), 'w') as f:
            f.write('\n'.join(LOG) + '\n')
        log('done', code)
        os._exit(code)

    bpy.app.timers.register(tick, first_interval=1.0)


if __name__ == '__main__':
    main()
