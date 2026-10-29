# Blender operators.  SKETCHRETOPO_OT_sketch is the interactive session: a modal
# operator that forwards mouse/keyboard events of the 3D view to the engine with
# the original program's controls.

import base64
import os
import time
import traceback

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector

from . import session
from .core import fileio
from .props import apply_query_settings, default_patch_db_path, addon_prefs
from .core.config import ConfigRender
from .core.engine import LEFT, RIGHT, MIDDLE, SKETCH
from .draw import Renderer
from .view import BlenderView, MayaCamera

# (engine state index, name, description) in SketchRetopo2's key order; spine sketching and auto-completion
# come from the earlier SketchRetopo release and sit on spare keys
MODE_ITEMS = [
    ('0', 'Sketch', 'Basic sketching (Q)'),
    ('5', 'Deform Curve', 'Deform curves (W)'),
    ('3', 'Laser', 'Laser sketching (E)'),
    ('4', 'Cylinder', 'Cylinder sketching (R)'),
    ('6', 'Edit Corner', 'Corner editing (T)'),
    ('7', 'Edit Topology', 'Topology editing (A)'),
    ('10', 'Query Database', 'Query the patch database (S)'),
    ('9', 'Move Vertex', 'Move vertices (D)'),
    ('8', 'Edge Loop', 'Edge loop insertion (F)'),
    ('1', 'Spine', 'Spine sketching (Y, from the earlier SketchRetopo)'),
    ('2', 'Auto-complete', 'Auto-completion (U, from the earlier SketchRetopo)'),
]
_MODE_ORDER_KEYS = {0: 'Q', 5: 'W', 3: 'E', 4: 'R', 6: 'T', 7: 'A', 10: 'S', 9: 'D', 8: 'F', 1: 'Y', 2: 'U'}
MODE_NAMES = {int(v): name for v, name, _ in MODE_ITEMS}

STATUS_TEXT = ('SketchRetopo  |  Q sketch  W deform  E laser  R cylinder  T corner  A topology  S query DB  '
               'D move  F edge loop  (Y spine  U auto-complete)  |  Alt+drag camera  |  Ctrl+Z/Y undo/redo  |  Esc exit')

PASS_THROUGH_TYPES = {
    'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
    'MOUSESMARTZOOM', 'NDOF_MOTION', 'NDOF_BUTTON_MENU', 'NDOF_BUTTON_FIT', 'HOME', 'N',
    'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8',
    'NUMPAD_9', 'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS', 'NUMPAD_SLASH', 'NUMPAD_ASTERIX',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'WINDOW_DEACTIVATE',
}
DIGITS = {'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5', 'SIX': '6', 'SEVEN': '7',
          'EIGHT': '8', 'NINE': '9'}
OVERLAP_REGIONS = {'UI', 'TOOLS', 'HEADER', 'TOOL_HEADER', 'NAVIGATION_BAR', 'ASSET_SHELF', 'ASSET_SHELF_HEADER',
                   'HUD', 'FOOTER', 'TOOL_PROPS'}

MATCAP_FILE = os.path.join(os.path.dirname(__file__), 'resources', 'sketchretopo_matcap.png')


def event_key(event):
    """Converts a Blender key event into the character GLUT would have delivered (or 'ctrl+x')."""
    t = event.type
    if t in DIGITS:
        ch = DIGITS[t]
    elif len(t) == 1 and t.isalpha():
        ch = t.lower()
    elif t == 'SPACE':
        ch = ' '
    else:
        return None
    if event.ctrl or event.oskey:
        if ch == 'z' and event.shift:
            return 'ctrl+shift+z'
        return 'ctrl+' + ch
    if event.shift and ch.isalpha():
        return ch.upper()
    return ch


def _tag_redraw_all():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('VIEW_3D', 'PROPERTIES'):
                area.tag_redraw()


def _draw_view_cb(op):
    try:
        ctx = bpy.context
        sess = op.session
        if sess is None or ctx.region_data is None:
            return
        obj = sess.obj
        if obj is None or sess.renderer is None:
            return
        view = BlenderView(ctx.region, ctx.region_data, obj)
        sess.renderer.draw_view(ctx, view, obj)
    except Exception:
        if not getattr(op, '_draw_error_reported', False):
            op._draw_error_reported = True
            traceback.print_exc()


def _draw_px_cb(op):
    try:
        ctx = bpy.context
        sess = op.session
        if sess is None or sess.renderer is None:
            return
        if ctx.area != op.area:
            return
        sess.renderer.draw_pixel(ctx)
    except Exception:
        if not getattr(op, '_draw_error_reported', False):
            op._draw_error_reported = True
            traceback.print_exc()


def numbered_autosave_name(fname):
    """<core>.<num>.xml -> <core>.<num+1>.xml (helper.hh decompose_xml_fname)."""
    base, ext = os.path.splitext(fname)
    if ext.lower() != '.xml':
        return fname
    core, dot, num = base.rpartition('.')
    if dot and num.isdigit():
        return '%s.%d.xml' % (core, int(num) + 1)
    return fname


def save_session_xml(sess, fname):
    obj = sess.obj
    verts, tris, _, uvs = session.mesh_arrays(obj)
    return fileio.save_xml(sess.engine, fname, verts, tris, uvs)


class SKETCHRETOPO_OT_sketch(bpy.types.Operator):
    """Start SketchRetopo on the base mesh: sketch curves on the surface to build quad patches"""
    bl_idname = 'sketch_retopo.sketch'
    bl_label = 'Start SketchRetopo'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        settings = context.scene.sketch_retopo
        obj = settings.base_object
        if obj is None and context.active_object is not None and context.active_object.type == 'MESH':
            obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, 'Select a mesh object (or set the Base Mesh) first')
            return {'CANCELLED'}
        if session.running_session() is not None:
            self.report({'WARNING'}, 'SketchRetopo is already running')
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            sess = session.get_or_create(obj, self.report)
        except Exception as ex:
            self.report({'ERROR'}, 'SketchRetopo: %s' % ex)
            return {'CANCELLED'}
        settings.base_object = obj
        self._start(context, sess)
        return {'RUNNING_MODAL'}

    # ---- lifecycle ---------------------------------------------------------------------------
    def _start(self, context, sess):
        self.session = sess
        sess.running = True
        sess.stop_requested = False
        self.area = context.area
        self.region = next(r for r in context.area.regions if r.type == 'WINDOW')
        self.rv3d = context.area.spaces.active.region_3d
        sess.renderer = Renderer(sess)
        self.camera = MayaCamera()
        self.buttons_down = set()
        self._draw_error_reported = False
        self._last_autocmpl = 0.0
        self._hidden_base = None
        self._hidden_outputs = {}
        apply_query_settings(sess.engine, context.scene.sketch_retopo)
        _autoload_patch_db(sess.engine, self.report)
        self._h3d = bpy.types.SpaceView3D.draw_handler_add(_draw_view_cb, (self,), 'WINDOW', 'POST_VIEW')
        self._h2d = bpy.types.SpaceView3D.draw_handler_add(_draw_px_cb, (self,), 'WINDOW', 'POST_PIXEL')
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        try:
            context.workspace.status_text_set(STATUS_TEXT)
        except Exception:
            pass
        self._update_view(context)
        _tag_redraw_all()

    def _finish(self, context):
        sess = self.session
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._h3d, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(self._h2d, 'WINDOW')
        except Exception:
            pass
        try:
            context.window_manager.event_timer_remove(self._timer)
        except Exception:
            pass
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass
        self._set_base_visibility(True)
        self._restore_output_visibility()
        if sess is not None:
            sess.running = False
            sess.stop_requested = False
            sess.renderer = None
        _tag_redraw_all()

    # ---- helpers -------------------------------------------------------------------------------
    def _update_view(self, context):
        eng = self.session.engine
        eng.view = BlenderView(self.region, self.rv3d, self.session.obj)
        eng.update_camera_distance()

    def _mouse(self, event):
        return event.mouse_x - self.region.x, event.mouse_y - self.region.y

    def _inside(self, event):
        r = self.region
        x, y = event.mouse_x, event.mouse_y
        if not (r.x <= x < r.x + r.width and r.y <= y < r.y + r.height):
            return False
        for other in self.area.regions:
            if other.type in OVERLAP_REGIONS and other.width > 1 and other.height > 1:
                if other.x <= x < other.x + other.width and other.y <= y < other.y + other.height:
                    return False
        return True

    def _set_base_visibility(self, visible):
        sess = self.session
        obj = sess.obj if sess is not None else None
        if obj is None:
            return
        try:
            if not visible and self._hidden_base is None:
                self._hidden_base = obj.hide_get()
                obj.hide_set(True)
            elif visible and self._hidden_base is not None:
                obj.hide_set(self._hidden_base)
                self._hidden_base = None
        except Exception:
            pass

    def _sync_output_visibility(self, context):
        # The exported retopo object coincides with the live quads drawn by the session and would
        # z-fight with them, so it is hidden while the session runs and shown again when it stops.
        out = context.scene.sketch_retopo.output_object
        if out is None or out.name in self._hidden_outputs or out.name == self.session.obj_name:
            return
        try:
            self._hidden_outputs[out.name] = out.hide_get()
            out.hide_set(True)
        except Exception:
            pass

    def _restore_output_visibility(self):
        for name, hidden in self._hidden_outputs.items():
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            try:
                obj.hide_set(hidden)
            except Exception:
                pass
        self._hidden_outputs.clear()

    def _sync_base_visibility(self):
        eng = self.session.engine
        want_hidden = (eng.configRender.mode == ConfigRender.QUADMESH_ONLY or bool(eng.basemesh.face_hidden.any()))
        self._set_base_visibility(not want_hidden)

    def _flush_messages(self):
        eng = self.session.engine
        for level, text in eng.messages:
            self.report({level}, text)
        eng.messages.clear()

    def _process_requests(self, context):
        eng = self.session.engine
        reqs = list(eng.requests)
        eng.requests.clear()
        for r in reqs:
            try:
                if r == 'ctrl+s':
                    fn = eng.configTemp.autoSave.filename
                    if fn:
                        if eng.configTemp.autoSave.unsaved:
                            eng.configTemp.autoSave.unsaved = False
                            eng.configTemp.autoSave.last_time = time.time()
                            save_session_xml(self.session, numbered_autosave_name(fn))
                            self.report({'INFO'}, 'Saved %s' % eng.configTemp.autoSave.filename)
                    else:
                        bpy.ops.sketch_retopo.save_xml('INVOKE_DEFAULT')
                elif r == 'ctrl+o':
                    bpy.ops.sketch_retopo.load_xml('INVOKE_DEFAULT')
                elif r == 'ctrl+i':
                    bpy.ops.sketch_retopo.import_basemesh('INVOKE_DEFAULT')
                elif r == 'ctrl+e':
                    bpy.ops.sketch_retopo.export_mesh()
                elif r == 'ctrl+f':
                    bpy.ops.wm.window_fullscreen_toggle()
                elif r == '0':
                    bm = eng.basemesh
                    center = self.session.obj.matrix_world @ Vector(bm.bbox_center())
                    MayaCamera.set_center(self.rv3d, center)
                elif r == '8':
                    bpy.ops.sketch_retopo.camera_params('INVOKE_DEFAULT')
                elif r == '6':
                    st = context.scene.sketch_retopo
                    st.turntable_mode = not st.turntable_mode
            except Exception as ex:
                self.report({'WARNING'}, 'SketchRetopo: %s' % ex)

    def _retarget(self, context, obj_name):
        """Switch the running session to another base object (after loading an XML or importing a mesh)."""
        sess = session.get(bpy.data.objects.get(obj_name))
        if sess is None:
            return
        self._set_base_visibility(True)
        self._restore_output_visibility()
        old = self.session
        old.running = False
        old.renderer = None
        self.session = sess
        sess.running = True
        sess.renderer = Renderer(sess)
        context.scene.sketch_retopo.base_object = sess.obj
        self._update_view(context)

    # ---- timer ---------------------------------------------------------------------------------
    def _on_timer(self, context):
        sess = self.session
        eng = sess.engine
        settings = context.scene.sketch_retopo
        redraw = False
        self._sync_output_visibility(context)
        if settings.turntable_mode:
            MayaCamera.turntable(self.rv3d, eng.configRender.turntable_speed)
            redraw = True
        if eng.state is eng.stateAutocmpl and eng.common_button_left_pressed:
            self._update_view(context)
            if eng.stateAutocmpl.update_preview():
                redraw = True
        if eng.state is eng.stateQueryDB and eng.stateQueryDB.update(0.02):
            redraw = True
            _tag_redraw_all()
        at = eng.configTemp.autoSave
        if at.enabled and at.filename and at.unsaved and time.time() - at.last_time > at.interval:
            at.last_time = time.time()
            try:
                save_session_xml(sess, numbered_autosave_name(at.filename))
            except Exception as ex:
                self.report({'WARNING'}, 'Autosave failed: %s' % ex)
        if redraw:
            self.area.tag_redraw()

    # ---- events --------------------------------------------------------------------------------
    def modal(self, context, event):
        sess = self.session
        try:
            if sess.stop_requested or sess.obj is None or self.area.type != 'VIEW_3D':
                self._finish(context)
                return {'FINISHED'}
        except ReferenceError:
            self._finish(context)
            return {'FINISHED'}
        if session.pending_switch:
            name = session.pending_switch
            session.pending_switch = None
            self._retarget(context, name)
            sess = self.session

        if event.type == 'TIMER':
            self._on_timer(context)
            return {'PASS_THROUGH'}

        eng = sess.engine
        mx, my = self._mouse(event)
        t = event.type
        v = event.value

        # camera dragging (Alt + mouse buttons) continues wherever the mouse goes
        if self.camera.drag_mode != MayaCamera.NONE:
            if t in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
                self.camera.upright = context.scene.sketch_retopo.camera_upright
                self.camera.mouse_move(self.rv3d, self.region, mx, my)
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if t in ('LEFTMOUSE', 'MIDDLEMOUSE', 'RIGHTMOUSE') and v == 'RELEASE':
                mode = self.camera.drag_mode
                if mode == MayaCamera.ROTATE and event.shift:
                    self.camera.snap_to_canonical(self.rv3d)
                if mode == MayaCamera.PAN and eng.configRender.auto_camera_center:
                    self._update_view(context)
                    hit = eng.intersect_mouse(self.region.width / 2, self.region.height / 2)
                    pn = eng.intersect_convert(hit)
                    if pn is not None:
                        MayaCamera.update_center(self.rv3d, sess.obj.matrix_world @ Vector(pn[:3]))
                self.camera.mouse_up()
                self._update_view(context)
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        inside = self._inside(event)
        if not inside and not self.buttons_down:
            if t in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE') and eng.common_pn_mouse is not None:
                eng.common_pn_mouse = None
                self.area.tag_redraw()
            return {'PASS_THROUGH'}

        if t == 'ESC' and v == 'PRESS':
            self._finish(context)
            return {'FINISHED'}
        if t in PASS_THROUGH_TYPES:
            return {'PASS_THROUGH'}
        if t == 'SPACE' and event.ctrl:
            return {'PASS_THROUGH'}

        try:
            self._update_view(context)
            if t in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
                eng.mouse_move(mx, my)
                if self.buttons_down or eng.state is eng.stateEdgeLoop:
                    eng.curvenetwork.invalidate_displist()
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if t in ('LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'):
                button = {'LEFTMOUSE': LEFT, 'RIGHTMOUSE': RIGHT, 'MIDDLEMOUSE': MIDDLE}[t]
                if v == 'PRESS':
                    if event.alt:
                        if button == LEFT:
                            mode = MayaCamera.PAN if event.ctrl else MayaCamera.ROTATE
                        elif button == MIDDLE:
                            mode = MayaCamera.PAN
                        else:
                            mode = MayaCamera.ZOOM
                        self.camera.mouse_down(mx, my, mode)
                        return {'RUNNING_MODAL'}
                    if button == MIDDLE:
                        return {'PASS_THROUGH'}      # Blender's own view navigation
                    self.buttons_down.add(t)
                    eng.mouse_down(mx, my, button, event.shift, event.ctrl, event.alt)
                elif v == 'RELEASE':
                    if t not in self.buttons_down:
                        return {'PASS_THROUGH'} if button == MIDDLE else {'RUNNING_MODAL'}
                    self.buttons_down.discard(t)
                    eng.mouse_up(mx, my, button, event.shift, event.ctrl, event.alt)
                else:
                    return {'RUNNING_MODAL'}
                self._after_edit(context)
                return {'RUNNING_MODAL'}

            key = event_key(event)
            if key is None:
                return {'RUNNING_MODAL'}
            if v == 'PRESS':
                eng.key_down(key)
                self._after_edit(context)
            elif v == 'RELEASE':
                eng.key_up(key.lower() if len(key) == 1 else key)
                if len(key) == 1:
                    eng.key_up(key.upper())
            return {'RUNNING_MODAL'}
        except Exception as ex:
            traceback.print_exc()
            self.report({'ERROR'}, 'SketchRetopo: %s' % ex)
            return {'RUNNING_MODAL'}

    def _after_edit(self, context):
        eng = self.session.engine
        eng.curvenetwork.invalidate_displist()
        self._flush_messages()
        self._process_requests(context)
        self._sync_base_visibility()
        self._sync_output_visibility(context)
        _tag_redraw_all()


class SKETCHRETOPO_OT_stop(bpy.types.Operator):
    """Stop the running SketchRetopo session (the sketch is kept)"""
    bl_idname = 'sketch_retopo.stop'
    bl_label = 'Stop SketchRetopo'

    def execute(self, context):
        sess = session.running_session()
        if sess is not None:
            sess.stop_requested = True
            _tag_redraw_all()
        return {'FINISHED'}


def _active_session(context):
    sess = session.running_session()
    if sess is None:
        sess = session.get(context.scene.sketch_retopo.base_object)
    return sess


def _autoload_patch_db(eng, report=None):
    """Loads the patch database of the preferences (or ~/patches.db) into an engine that has none."""
    lq = eng.learnquad
    if lq.is_loaded():
        return
    path = default_patch_db_path()
    if not path:
        return
    try:
        n = lq.load(path)
    except Exception as ex:
        if report:
            report({'WARNING'}, 'Could not load the patch database %s: %s' % (path, ex))
        return
    if report:
        report({'INFO'}, 'Patch database: %d templates (%s)' % (n, path))


class SKETCHRETOPO_OT_load_patch_db(bpy.types.Operator, ImportHelper):
    """Load the patch database (patches.db from the SketchRetopo2 demo) used by the query database mode and for
    patches with more than 6 corners"""
    bl_idname = 'sketch_retopo.load_patch_db'
    bl_label = 'Load Patch Database'
    filename_ext = '.db'
    filter_glob: StringProperty(default='*.db', options={'HIDDEN'})

    def execute(self, context):
        path = self.filepath
        from .core import learnquad
        try:
            db = learnquad.open_db(path)
        except Exception as ex:
            self.report({'ERROR'}, 'Not a patch database: %s' % ex)
            return {'CANCELLED'}
        for s in session.all_sessions():
            s.engine.learnquad.load(path)
        prefs = addon_prefs(context)
        if prefs is not None:
            prefs.patch_db_path = path
            context.preferences.is_dirty = True
        self.report({'INFO'}, 'Patch database: %d templates' % db.num_templates)
        _tag_redraw_all()
        return {'FINISHED'}


def _query_state(context):
    sess = _active_session(context)
    if sess is None or sess.engine.state is not sess.engine.stateQueryDB:
        return None
    return sess.engine.stateQueryDB


class SKETCHRETOPO_OT_query_stop(bpy.types.Operator):
    """Stop the database lookup, keeping the results found so far"""
    bl_idname = 'sketch_retopo.query_stop'
    bl_label = 'Stop'

    @classmethod
    def poll(cls, context):
        return _query_state(context) is not None

    def execute(self, context):
        _query_state(context).stop_lookup()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_query_restart(bpy.types.Operator):
    """Run the database lookup again for the selected patch and flows (e.g. after changing its settings)"""
    bl_idname = 'sketch_retopo.query_restart'
    bl_label = 'Restart'

    @classmethod
    def poll(cls, context):
        st = _query_state(context)
        return st is not None and st.selected_patch is not None

    def execute(self, context):
        sess = _active_session(context)
        apply_query_settings(sess.engine, context.scene.sketch_retopo)
        if not sess.engine.learnquad.is_loaded():
            self.report({'WARNING'}, 'No patch database loaded')
            return {'CANCELLED'}
        _query_state(context).restart()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_query_delete_flow(bpy.types.Operator):
    """Delete the last flow stroke (Restart to look up again)"""
    bl_idname = 'sketch_retopo.query_delete_flow'
    bl_label = 'Delete Flow'

    @classmethod
    def poll(cls, context):
        st = _query_state(context)
        return st is not None and bool(st.querydata.strokes)

    def execute(self, context):
        _query_state(context).delete_flow()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_set_mode(bpy.types.Operator):
    """Switch the SketchRetopo editing mode"""
    bl_idname = 'sketch_retopo.set_mode'
    bl_label = 'Set Mode'
    mode: EnumProperty(items=MODE_ITEMS)

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        sess = _active_session(context)
        sess.engine.state_set(int(self.mode))
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_undo(bpy.types.Operator):
    """Undo the last SketchRetopo edit (Ctrl+Z)"""
    bl_idname = 'sketch_retopo.undo'
    bl_label = 'Undo'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        _active_session(context).engine.memento_undo()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_redo(bpy.types.Operator):
    """Redo the last undone SketchRetopo edit (Ctrl+Y)"""
    bl_idname = 'sketch_retopo.redo'
    bl_label = 'Redo'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        _active_session(context).engine.memento_redo()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_clear(bpy.types.Operator):
    """Delete all curves and patches (undoable)"""
    bl_idname = 'sketch_retopo.clear'
    bl_label = 'Clear'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        eng = _active_session(context).engine
        eng.memento_store()
        eng.curvenetwork.clear()
        eng.state.init()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_toggle_symmetric(bpy.types.Operator):
    """Symmetry about the X=0 plane of the base mesh: the other side is shown as a mirrored preview and added to
    the exported mesh"""
    bl_idname = 'sketch_retopo.toggle_symmetric'
    bl_label = 'Symmetric'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        eng = _active_session(context).engine
        eng.set_symmetric(not eng.configSaved.symmetric)
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_snap_symmetry(bpy.types.Operator):
    """Move curve and quad vertices lying closer to the X=0 plane than the curve sampling interval onto it, so
    that the two halves join on export"""
    bl_idname = 'sketch_retopo.snap_symmetry'
    bl_label = 'Snap Symmetry'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        _active_session(context).engine.snap_symmetry()
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_save_xml(bpy.types.Operator, ExportHelper):
    """Save the sketch (base mesh, curves, quads) as a SketchRetopo XML file (Ctrl+S)"""
    bl_idname = 'sketch_retopo.save_xml'
    bl_label = 'Save XML'
    filename_ext = '.xml'
    filter_glob: StringProperty(default='*.xml', options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        sess = _active_session(context)
        try:
            fn = save_session_xml(sess, self.filepath)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        self.report({'INFO'}, 'Saved %s' % fn)
        return {'FINISHED'}


def _object_from_arrays(name, verts, tris, uvs=None, matrix=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in tris])
    if uvs is not None and len(uvs) == len(tris):
        uv_layer = mesh.uv_layers.new(name='UVMap')
        flat = np.asarray(uvs, dtype=float).reshape(-1, 2)
        if len(uv_layer.data) == len(flat):
            uv_layer.data.foreach_set('uv', flat.reshape(-1))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    if matrix is not None:
        obj.matrix_world = matrix
    bpy.context.scene.collection.objects.link(obj)
    return obj


class SKETCHRETOPO_OT_load_xml(bpy.types.Operator, ImportHelper):
    """Load a SketchRetopo XML file (also files saved by the original SketchRetopo program) (Ctrl+O)"""
    bl_idname = 'sketch_retopo.load_xml'
    bl_label = 'Load XML'
    filename_ext = '.xml'
    filter_glob: StringProperty(default='*.xml', options={'HIDDEN'})

    def execute(self, context):
        try:
            data = fileio.read_xml(self.filepath)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        name = os.path.splitext(os.path.basename(self.filepath))[0]
        if len(data['verts']) == 0:
            self.report({'ERROR'}, 'The XML file contains no base mesh')
            return {'CANCELLED'}
        obj = _object_from_arrays(name, data['verts'], data['tris'], data['uvs'])
        session.remove(obj.name)
        stale = bpy.data.texts.get(session.TEXT_PREFIX + obj.name)
        if stale is not None:
            bpy.data.texts.remove(stale)
        try:
            sess = session.get_or_create(obj)
            data['restore'](sess.engine)
            sess.engine.set_basemesh(sess.engine.basemesh, reset_curves=False)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        context.scene.sketch_retopo.base_object = obj
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if session.running_session() is not None:
            session.pending_switch = obj.name
        self.report({'INFO'}, 'Loaded %s' % self.filepath)
        _tag_redraw_all()
        return {'FINISHED'}


def _read_off(path):
    with open(path, 'r') as f:
        tokens = []
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                tokens.extend(line.split())
    if tokens and tokens[0].upper().endswith('OFF'):
        tokens = tokens[1:]
    nv, nf = int(tokens[0]), int(tokens[1])
    k = 3
    verts = []
    for _ in range(nv):
        verts.append((float(tokens[k]), float(tokens[k + 1]), float(tokens[k + 2])))
        k += 3
    faces = []
    for _ in range(nf):
        n = int(tokens[k])
        faces.append(tuple(int(x) for x in tokens[k + 1:k + 1 + n]))
        k += 1 + n
    return verts, faces


class SKETCHRETOPO_OT_import_basemesh(bpy.types.Operator, ImportHelper):
    """Import a mesh file (OBJ, OFF, PLY, STL) as the base mesh to retopologize (Ctrl+I)"""
    bl_idname = 'sketch_retopo.import_basemesh'
    bl_label = 'Import Base Mesh'
    filter_glob: StringProperty(default='*.obj;*.off;*.ply;*.stl', options={'HIDDEN'})

    def execute(self, context):
        path = self.filepath
        ext = os.path.splitext(path)[1].lower()
        before = set(bpy.data.objects)
        try:
            if ext == '.off':
                verts, faces = _read_off(path)
                mesh = bpy.data.meshes.new(os.path.splitext(os.path.basename(path))[0])
                mesh.from_pydata(verts, [], faces)
                mesh.update()
                obj = bpy.data.objects.new(mesh.name, mesh)
                context.scene.collection.objects.link(obj)
            elif ext == '.obj':
                bpy.ops.wm.obj_import(filepath=path)
            elif ext == '.ply':
                bpy.ops.wm.ply_import(filepath=path)
            elif ext == '.stl':
                try:
                    bpy.ops.wm.stl_import.get_rna_type()
                except KeyError:    # older Blender: legacy STL add-on
                    bpy.ops.import_mesh.stl(filepath=path)
                else:
                    bpy.ops.wm.stl_import(filepath=path)
            else:
                self.report({'ERROR'}, 'Unsupported format: %s' % ext)
                return {'CANCELLED'}
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        if not new:
            self.report({'ERROR'}, 'No mesh was imported')
            return {'CANCELLED'}
        obj = new[0]
        context.scene.sketch_retopo.base_object = obj
        if session.running_session() is not None:
            try:
                session.get_or_create(obj)
                session.pending_switch = obj.name
            except Exception as ex:
                self.report({'ERROR'}, str(ex))
        _tag_redraw_all()
        return {'FINISHED'}


class SKETCHRETOPO_OT_export_mesh(bpy.types.Operator):
    """Create (or update) a quad mesh object from the patches (Ctrl+E)"""
    bl_idname = 'sketch_retopo.export_mesh'
    bl_label = 'Export Retopo Mesh to Object'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        sess = _active_session(context)
        eng = sess.engine
        verts, faces = eng.retopo_mesh()
        if len(faces) == 0:
            self.report({'WARNING'}, 'There are no patches to export')
            return {'CANCELLED'}
        settings = context.scene.sketch_retopo
        out = settings.output_object
        base = sess.obj
        name = (base.name + '_retopo') if base is not None else 'Retopo'
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(f) for f in faces])
        mesh.update()
        if out is None or out.type != 'MESH' or out.name not in context.scene.objects:
            out = bpy.data.objects.new(name, mesh)
            context.scene.collection.objects.link(out)
            settings.output_object = out
        else:
            old = out.data
            out.data = mesh
            if old.users == 0:
                bpy.data.meshes.remove(old)
        if base is not None:
            out.matrix_world = base.matrix_world
        msg = 'Retopo mesh: %d vertices, %d quads -> %s' % (len(verts), len(faces), out.name)
        if session.running_session() is not None:
            msg += ' (shown when SketchRetopo stops)'
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SKETCHRETOPO_OT_export_file(bpy.types.Operator, ExportHelper):
    """Export the retopologized mesh to a file (OBJ, OFF, PLY, STL)"""
    bl_idname = 'sketch_retopo.export_file'
    bl_label = 'Export Retopo Mesh to File'
    filename_ext = '.obj'
    filter_glob: StringProperty(default='*.obj;*.off;*.ply;*.stl', options={'HIDDEN'})
    file_format: EnumProperty(name='Format', items=[('obj', 'OBJ', ''), ('off', 'OFF', ''), ('ply', 'PLY', ''),
                                                    ('stl', 'STL', '')], default='obj')

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def check(self, context):
        self.filename_ext = '.' + self.file_format
        return ExportHelper.check(self, context)

    def execute(self, context):
        eng = _active_session(context).engine
        verts, faces = eng.retopo_mesh()
        path = self.filepath
        if not path.lower().endswith('.' + self.file_format):
            path = os.path.splitext(path)[0] + '.' + self.file_format
        try:
            fileio.write_mesh(path, verts, faces)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        self.report({'INFO'}, 'Exported %s' % path)
        return {'FINISHED'}


def _apply_matcap(context, path):
    prefs = context.preferences
    name = os.path.basename(path)
    sl = None
    for s in prefs.studio_lights:
        if s.name == name and s.type == 'MATCAP':
            sl = s
            break
    if sl is None:
        sl = prefs.studio_lights.load(path, 'MATCAP')
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                shading = area.spaces.active.shading
                shading.type = 'SOLID'
                shading.light = 'MATCAP'
                shading.studio_light = sl.name


class SKETCHRETOPO_OT_load_material(bpy.types.Operator, ImportHelper):
    """Use a PNG image as the material (MatCap) of the base mesh; it is stored in the XML file"""
    bl_idname = 'sketch_retopo.load_material'
    bl_label = 'Load Material'
    filter_glob: StringProperty(default='*.png', options={'HIDDEN'})

    def execute(self, context):
        try:
            _apply_matcap(context, self.filepath)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        sess = _active_session(context)
        if sess is not None:
            with open(self.filepath, 'rb') as f:
                sess.engine.material_texture_png_base64 = base64.b64encode(f.read()).decode('ascii')
        return {'FINISHED'}


class SKETCHRETOPO_OT_default_material(bpy.types.Operator):
    """Shade the viewport with the SketchRetopo default material (installs it as a MatCap)"""
    bl_idname = 'sketch_retopo.default_material'
    bl_label = 'SketchRetopo MatCap'

    def execute(self, context):
        try:
            _apply_matcap(context, MATCAP_FILE)
        except Exception as ex:
            self.report({'ERROR'}, 'Error occurred: %s' % ex)
            return {'CANCELLED'}
        return {'FINISHED'}


class SKETCHRETOPO_OT_reset_center(bpy.types.Operator):
    """Reset the camera rotation center to the base mesh bounding box center (0)"""
    bl_idname = 'sketch_retopo.reset_center'
    bl_label = 'Reset Camera Center'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None and context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        sess = _active_session(context)
        center = sess.obj.matrix_world @ Vector(sess.engine.basemesh.bbox_center())
        MayaCamera.set_center(context.area.spaces.active.region_3d, center)
        return {'FINISHED'}


class SKETCHRETOPO_OT_camera_params(bpy.types.Operator):
    """Show or set the camera as text: eye(3) center(3) up(3) (8)"""
    bl_idname = 'sketch_retopo.camera_params'
    bl_label = 'Camera Parameters'
    params: StringProperty(name='Camera', description='eye_x eye_y eye_z center_x center_y center_z up_x up_y up_z')

    def _rv3d(self, context):
        if context.area and context.area.type == 'VIEW_3D':
            return context.area.spaces.active.region_3d
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    return area.spaces.active.region_3d
        return None

    def invoke(self, context, event):
        from .view import get_camera
        rv3d = self._rv3d(context)
        if rv3d is None:
            return {'CANCELLED'}
        eye, center, up = get_camera(rv3d)
        self.params = ' '.join('%g' % x for x in (*eye, *center, *up))
        return context.window_manager.invoke_props_dialog(self, width=500)

    def execute(self, context):
        from .view import set_camera
        try:
            vals = [float(x) for x in self.params.split()]
            eye, center, up = Vector(vals[0:3]), Vector(vals[3:6]), Vector(vals[6:9])
        except Exception:
            self.report({'ERROR'}, 'Expected 9 numbers: eye center up')
            return {'CANCELLED'}
        rv3d = self._rv3d(context)
        if rv3d is not None:
            set_camera(rv3d, eye, center, up)
        return {'FINISHED'}


class SKETCHRETOPO_OT_stats(bpy.types.Operator):
    """Report the number of base mesh triangles, quads and patches (7)"""
    bl_idname = 'sketch_retopo.stats'
    bl_label = 'Statistics'

    @classmethod
    def poll(cls, context):
        return _active_session(context) is not None

    def execute(self, context):
        self.report({'INFO'}, _active_session(context).engine.stats_report())
        return {'FINISHED'}


classes = (
    SKETCHRETOPO_OT_sketch,
    SKETCHRETOPO_OT_stop,
    SKETCHRETOPO_OT_set_mode,
    SKETCHRETOPO_OT_undo,
    SKETCHRETOPO_OT_redo,
    SKETCHRETOPO_OT_clear,
    SKETCHRETOPO_OT_toggle_symmetric,
    SKETCHRETOPO_OT_snap_symmetry,
    SKETCHRETOPO_OT_save_xml,
    SKETCHRETOPO_OT_load_xml,
    SKETCHRETOPO_OT_import_basemesh,
    SKETCHRETOPO_OT_export_mesh,
    SKETCHRETOPO_OT_export_file,
    SKETCHRETOPO_OT_load_material,
    SKETCHRETOPO_OT_default_material,
    SKETCHRETOPO_OT_reset_center,
    SKETCHRETOPO_OT_camera_params,
    SKETCHRETOPO_OT_stats,
    SKETCHRETOPO_OT_load_patch_db,
    SKETCHRETOPO_OT_query_stop,
    SKETCHRETOPO_OT_query_restart,
    SKETCHRETOPO_OT_query_delete_flow,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    sess = session.running_session()
    if sess is not None:
        sess.stop_requested = True
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
