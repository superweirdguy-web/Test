# Scene-level settings.  Most properties are views onto the engine of the current
# session (getter/setter pairs), so the sidebar always shows the live values,
# including the ones adjusted by right-dragging in the viewport.

import os

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty)

from . import session
from .core.config import ConfigRender, ConfigTemp
from .core.curvenetwork import Patch

_template = None


def _engine():
    s = session.running_session()
    if s is None:
        scene = bpy.context.scene
        obj = scene.sketch_retopo.base_object if scene else None
        s = session.get(obj)
    return s.engine if s is not None else None


class _Template:
    """Defaults shown while no session exists."""

    def __init__(self):
        self.configTemp = ConfigTemp()
        self.configRender = ConfigRender()


def _target():
    global _template
    eng = _engine()
    if eng is not None:
        return eng
    if _template is None:
        _template = _Template()
    return _template


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _invalidate():
    eng = _engine()
    if eng is not None:
        eng.curvenetwork.invalidate_displist()
        for p in eng.curvenetwork.patches:
            p.invalidate_displist()
    _redraw()


# ---- accessor factories -------------------------------------------------------------------------------

def _real_param(name):
    def get(self):
        return getattr(_target().configTemp, name).value

    def set(self, value):
        t = _target()
        getattr(t.configTemp, name).value = value
        if name == 'quadSize':
            Patch.quadSize = value
        _redraw()
    return get, set


def _attr(path, after=None, cast=None):
    owner_name, attr = path.split('.')

    def get(self):
        v = getattr(getattr(_target(), owner_name), attr)
        return cast(v) if cast else v

    def set(self, value):
        setattr(getattr(_target(), owner_name), attr, value)
        if after:
            after()
        _redraw()
    return get, set


def _proj_offset_get(self):
    eng = _engine()
    return eng.configSaved.projOffset if eng else 0.0


def _proj_offset_set(self, value):
    eng = _engine()
    if eng is None:
        return
    eng.configSaved.projOffset = value
    eng.reproject_all()
    _redraw()


def _autosave_enabled_get(self):
    return bool(_target().configTemp.autoSave.enabled)


def _autosave_enabled_set(self, value):
    _target().configTemp.autoSave.enabled = value


def _autosave_interval_get(self):
    return int(_target().configTemp.autoSave.interval)


def _autosave_interval_set(self, value):
    _target().configTemp.autoSave.interval = value


def _symmetric_get(self):
    eng = _engine()
    return bool(eng and eng.configSaved.symmetric)


def _undo_get(self):
    eng = _engine()
    return eng.memento.buffer_size() if eng else 150


def _undo_set(self, value):
    eng = _engine()
    if eng:
        eng.memento.init(value)


def _even_get(self):
    return Patch.use_even_num_subdiv


def _even_set(self, value):
    Patch.use_even_num_subdiv = value


_BASEMESH_DEFAULTS = {'normal_smooth_iter': 0, 'feature_smooth_iter': 0, 'use_feature_as_energy': True,
                      'negate_energy': True}


def _basemesh_attr(attr, recompute):
    def get(self):
        eng = _engine()
        if eng is None or eng.basemesh is None:
            return _BASEMESH_DEFAULTS[attr]
        return getattr(eng.basemesh, attr)

    def set(self, value):
        eng = _engine()
        if eng is None or eng.basemesh is None:
            return
        setattr(eng.basemesh, attr, value)
        if recompute == 'snake':
            eng.basemesh.compute_snake_energy()
        elif recompute == 'normal':
            obj = session.running_session().obj if session.running_session() else None
            if obj is not None:
                _, _, normals, _ = session.mesh_arrays(obj)
                eng.basemesh.smooth_normal(normals)
        _redraw()
    return get, set


def _render_mode_get(self):
    return int(_target().configRender.mode)


def _feature_type_get(self):
    eng = _engine()
    return int(eng.basemesh.snake_feature_type) if eng is not None and eng.basemesh is not None else 0


def _feature_type_set(self, value):
    eng = _engine()
    if eng is None or eng.basemesh is None:
        return
    eng.basemesh.snake_feature_type = value
    eng.basemesh.compute_snake_energy()


def _render_mode_set(self, value):
    _target().configRender.mode = value
    _redraw()


def _ortho_get(self):
    rv3d = _find_rv3d()
    return bool(rv3d and not rv3d.is_perspective)


def _ortho_set(self, value):
    rv3d = _find_rv3d()
    if rv3d is None:
        return
    rv3d.view_perspective = 'ORTHO' if value else 'PERSP'
    _target().configRender.use_ortho = value
    _redraw()


def _find_rv3d():
    ctx = bpy.context
    if ctx.space_data is not None and getattr(ctx.space_data, 'type', '') == 'VIEW_3D':
        return ctx.space_data.region_3d
    for window in ctx.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                return area.spaces.active.region_3d
    return None


def _mesh_poll(self, obj):
    return obj.type == 'MESH'


# ---- query database (SketchRetopo2's query_db group) --------------------------------------------------------

SORT_ITEMS = [('FLOW', 'By Flow', 'Sort by distance to the flow strokes, then singularities, then quad quality'),
              ('QUALITY', 'By Quality', 'Sort by quad quality, then singularities, then flow distance'),
              ('NUMSING', 'By Singularities', 'Sort by number and type of irregular vertices, then flow distance')]


def apply_query_settings(eng, st=None):
    """Copies the scene's query database settings to an engine (used by its next lookup)."""
    from .core import learnquad
    if st is None:
        st = bpy.context.scene.sketch_retopo
    lk = eng.learnquad.lookup
    lk.max_singularities = st.qdb_max_numsing
    lk.max_concavities = st.qdb_max_concave
    lk.min_valence = st.qdb_min_valence
    lk.max_valence = st.qdb_max_valence
    lk.min_size_threshold_ilp = 1 if st.qdb_ilp_flow else 1 << 62
    lk.flow_tolerance = st.qdb_ilp_flow_tlrnc
    lk.sort_criteria = {'FLOW': learnquad.SORT_BY_FLOW, 'QUALITY': learnquad.SORT_BY_QUALITY,
                        'NUMSING': learnquad.SORT_BY_NUMSING}[st.qdb_sort]
    rt = eng.learnquad.retopologizer
    rt.quality_threshold = st.qdb_quality_th
    rt.lap_weight = st.qdb_laplace_w


def _qdb_update(self, context):
    for s in session.all_sessions():
        apply_query_settings(s.engine, self)


def _qdb_state():
    eng = _engine()
    return eng.stateQueryDB if eng is not None else None


def _qdb_index_get(self):
    eng = _engine()
    return eng.stateQueryDB.lookup_index if eng is not None and eng.state is eng.stateQueryDB else -1


def _qdb_index_set(self, value):
    eng = _engine()
    if eng is not None and eng.state is eng.stateQueryDB:
        eng.stateQueryDB.set_index(value)
        _redraw()


def _qdb_freeze_get(self):
    st = _qdb_state()
    return bool(st and st.freeze)


def _qdb_freeze_set(self, value):
    st = _qdb_state()
    if st is not None:
        st.freeze = value


def addon_prefs(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def default_patch_db_path(context=None):
    """The database file to load: the one set in the add-on preferences, else patches.db in the home folder
    (where the SketchRetopo2 demo looks for it)."""
    prefs = addon_prefs(context)
    path = bpy.path.abspath(prefs.patch_db_path) if prefs is not None and prefs.patch_db_path else ''
    if path and os.path.isfile(path):
        return path
    home = os.path.join(os.path.expanduser('~'), 'patches.db')
    return home if os.path.isfile(home) else ''


class SketchRetopoPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    patch_db_path: StringProperty(name='Patch Database', subtype='FILE_PATH',
                                  description='patches.db of the SketchRetopo2 demo (data-driven patches: the query '
                                              'database mode and patches with more than 6 corners).  Loaded when '
                                              'SketchRetopo starts; patches.db in the home folder is used otherwise')

    def draw(self, context):
        self.layout.prop(self, 'patch_db_path')


_g = {}
for _n in ('snapSize', 'brushSize_spine', 'brushSize_autocmpl', 'brushSize_moveVertex', 'quadSize', 'segmentSize'):
    _g[_n] = _real_param(_n)


class SketchRetopoSettings(bpy.types.PropertyGroup):
    base_object: PointerProperty(name='Base Mesh', type=bpy.types.Object, poll=_mesh_poll,
                                 description='High-resolution mesh to retopologize')
    output_object: PointerProperty(name='Retopo Mesh', type=bpy.types.Object, poll=_mesh_poll,
                                   description='Object receiving the exported quad mesh')
    camera_upright: BoolProperty(name='Upright', default=False,
                                 description='Keep the camera upright (world Z up) during Alt+drag rotation')
    turntable_mode: BoolProperty(name='Turntable', default=False, description='Rotate the view continuously (key 6)')
    camera_params: StringProperty(name='Camera', default='')

    # sizes (relative ones are fractions of the camera distance, like the original)
    snap_size: FloatProperty(name='Snap Size', min=0.0, precision=4, step=0.1, get=_g['snapSize'][0], set=_g['snapSize'][1],
                             description='Snapping radius relative to the camera distance (Shift+Right-drag)')
    brush_spine: FloatProperty(name='Spine Brush', min=0.0, precision=4, step=0.1, get=_g['brushSize_spine'][0],
                               set=_g['brushSize_spine'][1], description='Width of patches made by spine sketching (Right-drag)')
    brush_autocmpl: FloatProperty(name='Auto-completion Brush', min=0.0, precision=4, step=0.1,
                                  get=_g['brushSize_autocmpl'][0], set=_g['brushSize_autocmpl'][1],
                                  description='Maximum size of suggested patches (Right-drag)')
    brush_move: FloatProperty(name='Move Brush', min=0.0, precision=4, step=0.1, get=_g['brushSize_moveVertex'][0],
                              set=_g['brushSize_moveVertex'][1], description='Vertex moving brush radius (Right-drag)')
    quad_size: FloatProperty(name='Default Quad Size', min=0.0, precision=5, step=0.01, get=_g['quadSize'][0],
                             set=_g['quadSize'][1], description='Default quad size of new patches (Ctrl+Right-drag)')
    segment_size: FloatProperty(name='Curve Sampling', min=0.0, precision=6, step=0.001, get=_g['segmentSize'][0],
                                set=_g['segmentSize'][1], description='Resampling interval of sketched curves (Ctrl+Shift+Right-drag)')
    proj_offset: FloatProperty(name='Projection Offset', min=0.0, precision=6, step=0.001,
                               get=_proj_offset_get, set=_proj_offset_set,
                               description='Offset of curves and quads above the base mesh (avoids z-fighting)')
    symmetric: BoolProperty(name='Symmetric', get=_symmetric_get, description='Base mesh is symmetric about X=0')

    autosave_enabled: BoolProperty(name='Autosave', get=_autosave_enabled_get, set=_autosave_enabled_set,
                                   description='Save the XML file automatically (once it has a file name)')
    autosave_interval: IntProperty(name='Autosave Interval', min=1,
                                   get=_autosave_interval_get, set=_autosave_interval_set,
                                   description='Autosave interval in seconds')
    undo_buffer_size: IntProperty(name='Undo Buffer Size', min=10, max=1000, get=_undo_get, set=_undo_set)
    normal_smooth_iter: IntProperty(name='Normal Smoothing Iterations', min=0, max=50,
                                    get=_basemesh_attr('normal_smooth_iter', 'normal')[0],
                                    set=_basemesh_attr('normal_smooth_iter', 'normal')[1])
    autocmpl_preview_interval: IntProperty(name='Auto-completion Preview Interval', min=100,
                                           get=_attr('configTemp.autocmpl_preview_interval')[0],
                                           set=_attr('configTemp.autocmpl_preview_interval')[1],
                                           description='Milliseconds between auto-completion previews')
    cylinder_num_div: IntProperty(name='Cylinder Curves', min=1, get=_attr('configTemp.cylinder_num_div')[0],
                                  set=_attr('configTemp.cylinder_num_div')[1],
                                  description='Number of curves generated by cylinder sketching between loops without corners')
    use_even_num_subdiv: BoolProperty(name='Use Even Subdivisions', get=_even_get, set=_even_set,
                                      description='Choose even edge subdivision counts for new patches')
    loop_threshold_area: FloatProperty(name='Loop Threshold (Area)', min=0, max=1, step=1,
                                       get=_attr('configTemp.loop_threshold_area')[0],
                                       set=_attr('configTemp.loop_threshold_area')[1])
    loop_threshold_normal: FloatProperty(name='Loop Threshold (Normal)', min=0, max=1, step=1,
                                         get=_attr('configTemp.loop_threshold_normal')[0],
                                         set=_attr('configTemp.loop_threshold_normal')[1])

    # snakes
    feature_smooth_iter: IntProperty(name='Feature Smoothing Iterations', min=0, max=50,
                                     get=_basemesh_attr('feature_smooth_iter', 'snake')[0],
                                     set=_basemesh_attr('feature_smooth_iter', 'snake')[1])
    feature_type: EnumProperty(name='Feature Type', items=[('0', 'Mean Curvature', ''), ('1', 'Normal Variation', ''),
                                                           ('2', 'Normal Curvature', '')],
                               get=_feature_type_get, set=_feature_type_set)
    use_feature_as_energy: BoolProperty(name='Feature as Energy', get=_basemesh_attr('use_feature_as_energy', 'snake')[0],
                                        set=_basemesh_attr('use_feature_as_energy', 'snake')[1])
    negate_energy: BoolProperty(name='Negate Energy', get=_basemesh_attr('negate_energy', 'snake')[0],
                                set=_basemesh_attr('negate_energy', 'snake')[1])
    snakes_internal1: FloatProperty(name='Internal (1st)', min=0, get=_attr('configTemp.snakes_internal1')[0],
                                    set=_attr('configTemp.snakes_internal1')[1])
    snakes_internal2: FloatProperty(name='Internal (2nd)', min=0, get=_attr('configTemp.snakes_internal2')[0],
                                    set=_attr('configTemp.snakes_internal2')[1])
    snakes_external: FloatProperty(name='External', min=0, get=_attr('configTemp.snakes_external')[0],
                                   set=_attr('configTemp.snakes_external')[1])
    snakes_damping: FloatProperty(name='Damping', min=0, max=1, step=10, get=_attr('configTemp.snakes_damping')[0],
                                  set=_attr('configTemp.snakes_damping')[1])
    snakes_on_ctrl_shift: BoolProperty(name='Ctrl+Shift+Drag Uses Snakes',
                                       description='Ctrl+Shift+drag smooths curves with geometric snakes (from the '
                                                   'earlier SketchRetopo) instead of plain smoothing',
                                       get=_attr('configTemp.snakes_on_ctrl_shift')[0],
                                       set=_attr('configTemp.snakes_on_ctrl_shift')[1])

    # camera
    use_ortho: BoolProperty(name='Ortho', get=_ortho_get, set=_ortho_set, description='Orthographic projection')
    auto_camera_center: BoolProperty(name='Auto Center', get=_attr('configRender.auto_camera_center')[0],
                                     set=_attr('configRender.auto_camera_center')[1],
                                     description='After Alt+pan, move the rotation center to the surface at the screen center')
    turntable_speed: FloatProperty(name='Turntable Speed', min=-1, max=1, step=0.1, precision=3,
                                   get=_attr('configRender.turntable_speed')[0], set=_attr('configRender.turntable_speed')[1])

    # render
    render_mode: EnumProperty(name='Render Mode', items=[('0', 'Default', 'Base mesh, patches and curves'),
                                                         ('1', 'Quad Mesh Only', 'Only the quad mesh (Space)'),
                                                         ('2', 'Base Mesh Only', 'Only the base mesh (Ctrl+B)')],
                              get=_render_mode_get, set=_render_mode_set)
    always_show_quads: BoolProperty(name='Always Show Quads', get=_attr('configRender.always_show_quads')[0],
                                    set=_attr('configRender.always_show_quads')[1])
    show_auxiliary: BoolProperty(name='Show Auxiliary', get=_attr('configRender.quadmesh_only_show_auxiliary')[0],
                                 set=_attr('configRender.quadmesh_only_show_auxiliary')[1],
                                 description='Show patch boundaries, corners and irregular vertices in quad-only mode')
    show_left_side: BoolProperty(name='Show Left Side', get=_attr('configRender.quadmesh_only_show_left_side')[0],
                                 set=_attr('configRender.quadmesh_only_show_left_side')[1])
    show_axis: BoolProperty(name='Axis', get=_attr('configRender.show_axis')[0], set=_attr('configRender.show_axis')[1])
    show_bbox: BoolProperty(name='Bounding Box', get=_attr('configRender.show_bbox')[0], set=_attr('configRender.show_bbox')[1])
    phong_tessellation: BoolProperty(name='Phong Tessellation', get=_attr('configRender.phong_tessellation')[0],
                                     set=_attr('configRender.phong_tessellation', _invalidate)[1])
    phong_subdiv: IntProperty(name='Phong Subdivision', min=1, max=6, get=_attr('configRender.phong_subdiv')[0],
                              set=_attr('configRender.phong_subdiv', _invalidate)[1])
    phong_weight: FloatProperty(name='Phong Weight', min=0, max=1, step=1, get=_attr('configRender.phong_weight')[0],
                                set=_attr('configRender.phong_weight', _invalidate)[1])
    corner_radius: FloatProperty(name='Corner Radius', min=0, step=0.1, precision=4,
                                 get=_attr('configRender.corner_radius')[0], set=_attr('configRender.corner_radius')[1])
    edge_width_interior: FloatProperty(name='Edge Width (Interior)', min=0, step=10,
                                       get=_attr('configRender.edge_width_interior')[0],
                                       set=_attr('configRender.edge_width_interior')[1])
    edge_width_boundary: FloatProperty(name='Edge Width (Boundary)', min=0, step=10,
                                       get=_attr('configRender.edge_width_boundary')[0],
                                       set=_attr('configRender.edge_width_boundary')[1])
    singularity_radius: FloatProperty(name='Singularity Radius', min=0, step=10,
                                      get=_attr('configRender.singularity_radius')[0],
                                      set=_attr('configRender.singularity_radius')[1])
    patch_alpha: FloatProperty(name='Patch Opacity', min=0, max=1, step=1, get=_attr('configRender.patch_color_alpha')[0],
                               set=_attr('configRender.patch_color_alpha')[1])

    show_help: BoolProperty(name='Show Controls', default=True)

    # query database (applied to the next lookup)
    qdb_index: IntProperty(name='Index', min=-1, get=_qdb_index_get, set=_qdb_index_set,
                           description='Result shown in the selected patch (Z / X in the viewport)')
    qdb_freeze: BoolProperty(name='Freeze', get=_qdb_freeze_get, set=_qdb_freeze_set,
                             description='Keep showing the current result while the lookup goes on')
    qdb_max_numsing: IntProperty(name='Max Singularities', default=16, min=0, update=_qdb_update,
                                 description='Skip templates with more irregular vertices')
    qdb_max_concave: IntProperty(name='Max Concave Corners', default=0, min=0, update=_qdb_update,
                                 description='Skip templates with more concave corners')
    qdb_min_valence: IntProperty(name='Min Valence', default=3, min=2, max=3, update=_qdb_update,
                                 description='Skip templates with irregular vertices of lower valence')
    qdb_max_valence: IntProperty(name='Max Valence', default=5, min=5, update=_qdb_update,
                                 description='Skip templates with irregular vertices of higher valence')
    qdb_ilp_flow: BoolProperty(name='ILP Flow', default=True, update=_qdb_update,
                               description='Require a polychord to start and end near each flow stroke\'s ends')
    qdb_ilp_flow_tlrnc: IntProperty(name='ILP Flow Tolerance', default=1, min=0, update=_qdb_update,
                                    description='How many edges a polychord end may be away from a flow end')
    qdb_quality_th: FloatProperty(name='Quality Threshold', default=1.0, min=0.0, max=1.0, step=1, update=_qdb_update,
                                  description='Skip results whose worst corner angle deviates more from 90 degrees '
                                              '(1 keeps all)')
    qdb_sort: EnumProperty(name='Sort', items=SORT_ITEMS, default='FLOW', update=_qdb_update)
    qdb_laplace_w: FloatProperty(name='Laplace Weight', default=0.8, min=0.01, max=0.99, step=1, update=_qdb_update,
                                 description='Smoothness against flow following when placing results')


classes = (SketchRetopoPreferences, SketchRetopoSettings)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.sketch_retopo = PointerProperty(type=SketchRetopoSettings)


def unregister():
    del bpy.types.Scene.sketch_retopo
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
