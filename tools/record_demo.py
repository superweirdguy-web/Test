"""Records the demo video: sketch a square and a triangle on a sphere, then change curve span
counts in topology editing mode.  Runs inside a real Blender window and drives the modal operator
with simulated input, like tests/gui_test.py, while ffmpeg captures the 3D view.  Usage (Linux,
needs xvfb-run and ffmpeg):

    xvfb-run -a -s "-screen 0 1600x1000x24" blender --factory-startup --enable-event-simulate \
        --python tools/record_demo.py -- <output_dir>

Writes demo.mp4 and demo.gif to <output_dir>.  Simulated input does not move a real mouse
pointer, so the script draws one, together with a caption for each step.  The add-on itself
shows neither.
"""
import math
import os
import subprocess
import sys
import traceback

import blf
import bpy
import gpu
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)

import sketch_retopo  # noqa: E402
from sketch_retopo import session  # noqa: E402

OUT = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else '/tmp/sketch_retopo_demo'
os.makedirs(OUT, exist_ok=True)
MP4 = os.path.join(OUT, 'demo.mp4')
GIF = os.path.join(OUT, 'demo.gif')

pointer = None      # window coordinates of the last simulated mouse event
caption = ''


def log(*args):
    print('[record_demo]', *args, flush=True)


def view3d():
    window = bpy.context.window_manager.windows[0]
    area = max((a for a in window.screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return window, area, region, area.spaces.active.region_3d


def to_window(world):
    window, area, region, rv3d = view3d()
    co = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(world))
    return int(region.x + co.x), int(region.y + co.y)


def on_sphere(x, z, r=1.0):
    """Point on the front (-Y) side of the unit sphere seen from the front view."""
    y = -math.sqrt(max(r * r - x * x - z * z, 0.0))
    return (x, y, z)


# ---- overlay: pointer and caption ----------------------------------------------------------
def draw_overlay():
    region = bpy.context.region
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    if caption:
        font = 0
        blf.size(font, 26)
        w, h = blf.dimensions(font, caption)
        x0, y0 = (region.width - w) / 2, 40
        box = [(x0 - 16, y0 - 14), (x0 + w + 16, y0 - 14), (x0 + w + 16, y0 + h + 14), (x0 - 16, y0 + h + 14)]
        gpu.state.blend_set('ALPHA')
        shader.uniform_float('color', (0.0, 0.0, 0.0, 0.6))
        batch_for_shader(shader, 'TRI_FAN', {'pos': box}).draw(shader)
        blf.color(font, 1.0, 1.0, 1.0, 1.0)
        blf.position(font, x0, y0, 0)
        blf.draw(font, caption)
    if pointer is not None:
        x, y = pointer[0] - region.x, pointer[1] - region.y
        arrow = [(0, 0), (0, -26), (7, -20), (12, -31), (17, -29), (12, -18), (20, -18)]
        gpu.state.blend_set('ALPHA')
        for color, scale in (((0.0, 0.0, 0.0, 1.0), 1.15), ((1.0, 1.0, 1.0, 1.0), 1.0)):
            pts = [(x + scale * px - (scale - 1.0) * 4, y + scale * py + (scale - 1.0) * 6) for px, py in arrow]
            shader.uniform_float('color', color)
            batch_for_shader(shader, 'TRI_FAN', {'pos': pts}).draw(shader)
    gpu.state.blend_set('NONE')


def redraw():
    for a in bpy.context.window_manager.windows[0].screen.areas:
        a.tag_redraw()


def say(text):
    global caption
    caption = text
    log('caption:', text)
    redraw()


# ---- simulated input (slow enough to watch) ------------------------------------------------
def sim(type_, value, xy, **mods):
    global pointer
    pointer = xy
    view3d()[0].event_simulate(type_, value, x=xy[0], y=xy[1], **mods)
    redraw()


def move_to(xy, steps=8, dt=0.02):
    x0, y0 = pointer if pointer is not None else xy
    for k in range(1, steps + 1):
        t = k / steps
        sim('MOUSEMOVE', 'NOTHING', (int(x0 + t * (xy[0] - x0)), int(y0 + t * (xy[1] - y0))))
        yield dt


def drag(a, b, steps=16, dt=0.02):
    pa, pb = to_window(a), to_window(b)
    yield from move_to(pa)
    sim('LEFTMOUSE', 'PRESS', pa)
    yield 0.1
    for k in range(1, steps + 1):
        t = k / steps
        sim('MOUSEMOVE', 'NOTHING', (int(pa[0] + t * (pb[0] - pa[0])), int(pa[1] + t * (pb[1] - pa[1]))))
        yield dt
    sim('LEFTMOUSE', 'RELEASE', pb)
    yield 0.3


def click(p, **mods):
    xy = to_window(p)
    yield from move_to(xy)
    yield 0.2
    sim('LEFTMOUSE', 'PRESS', xy, **mods)
    sim('LEFTMOUSE', 'RELEASE', xy, **mods)
    yield 0.5


def key(k, **mods):
    xy = pointer if pointer is not None else to_window((0, -1, 0))
    view3d()[0].event_simulate(k, 'PRESS', x=xy[0], y=xy[1], **mods)
    view3d()[0].event_simulate(k, 'RELEASE', x=xy[0], y=xy[1], **mods)
    redraw()


def spans(eng):
    return [list(p.param.l) for p in eng.curvenetwork.patches]


# ---- recording -----------------------------------------------------------------------------
def start_ffmpeg():
    window, area, region, rv3d = view3d()
    top = window.height - (area.y + area.height)
    w, h = area.width // 2 * 2, area.height // 2 * 2
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-framerate', '30',
           '-video_size', '%dx%d' % (w, h), '-i', '%s+%d,%d' % (os.environ.get('DISPLAY', ':99'), area.x, top),
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', MP4]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def make_gif():
    vf = 'fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', MP4, '-vf', vf, GIF], check=True)


def steps():
    sketch_retopo.register()
    window, area, region, rv3d = view3d()
    # close the splash screen, which a fresh Blender shows over the 3D view
    window.event_simulate('ESC', 'PRESS', x=10, y=10)
    window.event_simulate('ESC', 'RELEASE', x=10, y=10)
    yield 0.3
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_axis(type='FRONT')
        bpy.ops.view3d.view_selected()
    bpy.context.active_object.select_set(False)     # no orange outline; it stays the active object
    space = area.spaces.active
    space.show_gizmo = space.show_region_toolbar = space.show_region_header = False
    overlay = space.overlay
    overlay.show_floor = overlay.show_axis_x = overlay.show_axis_y = False
    overlay.show_cursor = overlay.show_text = False
    bpy.types.SpaceView3D.draw_handler_add(draw_overlay, (), 'WINDOW', 'POST_PIXEL')
    yield 0.3
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.sketch_retopo.sketch('INVOKE_DEFAULT')
    yield 0.5
    eng = session.running_session().engine

    ffmpeg = start_ffmpeg()
    yield 0.5
    try:
        # --- basic sketching: a square, then a triangle
        say('Q: sketch four strokes to make a square')
        yield 1.0
        sq = [on_sphere(-0.75, -0.35), on_sphere(-0.05, -0.35), on_sphere(-0.05, 0.35), on_sphere(-0.75, 0.35)]
        for a, b in zip(sq, sq[1:] + sq[:1]):
            yield from drag(a, b)
        say('A patch appears as soon as the square closes')
        yield 1.5
        log('square spans', spans(eng))

        say('Three more strokes make a triangle')
        tri = [on_sphere(0.2, -0.3), on_sphere(0.75, -0.3), on_sphere(0.47, 0.3)]
        for a, b in zip(tri, tri[1:] + tri[:1]):
            yield from drag(a, b)
        yield 1.0
        log('triangle spans', spans(eng))

        # --- topology editing
        key('A')
        say('A: topology editing. Ctrl+click selects the whole strip')
        yield 0.8
        yield from click(on_sphere(-0.4, 0.35), ctrl=True)
        yield 1.0
        for _ in range(2):
            say('C adds two spans to the selected curves')
            key('C')
            yield 1.0
        log('after C C', spans(eng))
        yield 0.8

        say('Click one side only, then C: irregular vertices fill the gap')
        yield from click(on_sphere(-0.05, 0.0))
        key('C')
        yield 1.8
        log('after one side C', spans(eng))
        say('X takes two spans away again')
        key('X')
        yield 1.5

        say('Triangle sides work the same way')
        yield from click(on_sphere(0.47, -0.3))
        key('C')
        yield 1.2
        log('triangle after C', spans(eng))
        key('X')
        yield 1.2

        say('Space: show only the quad mesh')
        key('SPACE')
        yield 2.5
        key('SPACE')
        say('')
        yield 0.5
    finally:
        ffmpeg.communicate(b'q')
    key('ESC')
    yield 0.3
    make_gif()
    log('wrote', MP4, GIF)


def main():
    gen = steps()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            os._exit(0)
        except Exception:
            traceback.print_exc()
            os._exit(1)

    bpy.app.timers.register(tick, first_interval=1.0)


main()
