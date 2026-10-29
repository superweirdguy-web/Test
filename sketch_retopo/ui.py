# Sidebar panels (View3D > Sidebar > SketchRetopo), mirroring the original program's
# AntTweakBar: file, mode, config, snakes, camera and render groups, plus a controls reference.

import os

import bpy

from . import session
from .operators import MODE_ITEMS, MODE_NAMES, _MODE_ORDER_KEYS

CONTROLS = {
    0: ['Drag: draw a curve (closed loops of 2-6 corners become patches)',
        'Start/end near a green endpoint or a curve: snap (T-junction on a curve)',
        'Ctrl+drag from an endpoint: extend the curve (no corner)',
        'Hold C + drag: erase the curves crossed',
        'Hold X + drag from a curve into an empty region: create its patch',
        'Hold V + click: delete a patch',
        'Hold B + click a hole edge: curve along it (N: slightly inside)',
        'Hold Z + click: join the curves around the click',
        'G: erase all open-ended curves',
        'Shift+drag: smooth a curve'],
    5: ['Drag a corner / endpoint, or the middle of a curve',
        'Released corners / endpoints snap to nearby curves',
        'Ctrl+drag: oversketch between two points of a curve',
        'Shift+drag: smooth a curve'],
    3: ['Drag a line: cut the base mesh with the plane through it',
        'The traced curve starts at the line start point'],
    4: ['Drag a curve connecting two loops (e.g. laser loops)',
        'The loops need no corners or the same number of corners',
        'More curves are generated around the cylinder'],
    6: ['Drag from inside a patch to a corner: toggle corner / T-junction',
        'Ctrl+click: create or erase a corner on a curve'],
    7: ['Click a curve: select it (its number of edges is shown)',
        'Shift+click: toggle, Ctrl+click: every curve across the quad strip',
        'C / X: +2 / -2 edges, V / Z: +1 / -1 (patches need an even total)',
        'Click inside a patch: select it; 4-sided: drag to move the poles',
        'Other patches: click an arrow, Z X C V adjust it, B / Shift+B: pattern'],
    10: ['Click a patch to select it',
         'Drag inside it: a flow stroke starts the lookup',
         'Shift+drag: add another flow stroke',
         'Z / X: previous / next result from the database',
         'Needs patches.db (Query Database panel)'],
    9: ['Drag: move vertices (falloff brush), Shift+drag: smooth',
        'Ctrl+click a patch: relax it over the surface',
        'G: one step of global smoothing',
        'Right-drag: brush radius'],
    8: ['Hover a patch boundary: preview, click: insert an edge loop'],
    1: ['Drag: draw a spine, a patch strip is created around it',
        'Right-drag: brush (patch width)',
        'Snaps orthogonally / in parallel to nearby patch boundaries'],
    2: ['Drag along a patch boundary: preview (green)',
        'Release to freeze, then click anywhere to accept',
        'Z / X: browse alternatives, C: prefer a triangle',
        'Right-drag: brush (maximum patch size)'],
}

COMMON_CONTROLS = [
    'Alt+Left/Middle/Right drag: rotate / pan / zoom (Shift on release: snap view)',
    'Shift+Right-drag: snap size, Ctrl+Right-drag: default quad size',
    'Ctrl+Shift+Right-drag: curve sampling',
    'Space: quad mesh only, Ctrl+B: base mesh only',
    'Hold H + drag: hide the base mesh on one side, H+click: unhide',
    'Ctrl+Z / Ctrl+Y: undo / redo',
    'Ctrl+S save, Ctrl+O load, Ctrl+I import, Ctrl+E export',
    '0: reset camera center, 6: turntable, 7: statistics, 8: camera text',
    'Esc: stop',
]


def _session(context):
    sess = session.running_session()
    if sess is None:
        sess = session.get(context.scene.sketch_retopo.base_object)
    return sess


class SKETCHRETOPO_PT_main(bpy.types.Panel):
    bl_label = 'SketchRetopo'
    bl_idname = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        running = session.running_session()
        col = layout.column()
        col.prop(st, 'base_object')
        if running is None:
            col.operator('sketch_retopo.sketch', icon='GREASEPENCIL')
        else:
            row = col.row()
            row.alert = True
            row.operator('sketch_retopo.stop', icon='CANCEL')
        sess = _session(context)
        if sess is None:
            col.label(text='Pick a mesh and press Start', icon='INFO')
            return
        eng = sess.engine
        layout.label(text='Mode: %s' % eng.state.state_name)
        grid = layout.grid_flow(columns=2, even_columns=True, align=True)
        current = eng.state_get()
        for value, name, _ in MODE_ITEMS:
            i = int(value)
            op = grid.operator('sketch_retopo.set_mode', text='%s (%s)' % (name, _MODE_ORDER_KEYS[i]),
                               depress=(i == current))
            op.mode = value
        row = layout.row(align=True)
        row.operator('sketch_retopo.undo', icon='LOOP_BACK')
        row.operator('sketch_retopo.redo', icon='LOOP_FORWARDS')
        row = layout.row(align=True)
        row.operator('sketch_retopo.toggle_symmetric', text='Symmetric (X=0)', depress=eng.configSaved.symmetric,
                     icon='MOD_MIRROR')
        row.operator('sketch_retopo.snap_symmetry', icon='SNAP_ON')
        layout.operator('sketch_retopo.clear', icon='TRASH')
        layout.prop(st, 'render_mode', text='')
        layout.label(text=eng.stats_report())


class SKETCHRETOPO_PT_file(bpy.types.Panel):
    bl_label = 'File'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        col = layout.column(align=True)
        col.operator('sketch_retopo.import_basemesh', icon='IMPORT')
        col.operator('sketch_retopo.export_mesh', icon='MESH_GRID')
        col.prop(st, 'output_object')
        col.operator('sketch_retopo.export_file', icon='EXPORT')
        col = layout.column(align=True)
        col.operator('sketch_retopo.save_xml', icon='FILE_TICK')
        col.operator('sketch_retopo.load_xml', icon='FILE_FOLDER')
        sess = _session(context)
        if sess is not None and sess.engine.configTemp.autoSave.filename:
            col.label(text=sess.engine.configTemp.autoSave.filename + ('*' if sess.engine.configTemp.autoSave.unsaved else ''))
        row = layout.row(align=True)
        row.prop(st, 'autosave_enabled')
        row.prop(st, 'autosave_interval', text='')
        col = layout.column(align=True)
        col.operator('sketch_retopo.load_material', icon='MATERIAL')
        col.operator('sketch_retopo.default_material', icon='SHADING_SOLID')


class SKETCHRETOPO_PT_query_db(bpy.types.Panel):
    bl_label = 'Query Database'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        sess = _session(context)
        lq = sess.engine.learnquad if sess is not None else None
        col = layout.column(align=True)
        if lq is not None and lq.is_loaded():
            col.label(text='%d templates' % lq.num_templates(), icon='CHECKMARK')
            col.label(text=os.path.basename(lq.path))
        else:
            col.label(text='No database loaded', icon='ERROR')
        col.operator('sketch_retopo.load_patch_db', icon='FILE_FOLDER')
        querying = sess is not None and sess.engine.state is sess.engine.stateQueryDB
        col = layout.column(align=True)
        col.enabled = querying
        row = col.row(align=True)
        row.prop(st, 'qdb_index')
        row.prop(st, 'qdb_freeze', toggle=True)
        if querying and sess.engine.stateQueryDB.status_msg:
            for line in sess.engine.stateQueryDB.status_msg.split('\n'):
                col.label(text=line)
        col = layout.column(align=True)
        for p in ('qdb_max_numsing', 'qdb_max_concave', 'qdb_min_valence', 'qdb_max_valence'):
            col.prop(st, p)
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(st, 'qdb_ilp_flow', toggle=True)
        row.prop(st, 'qdb_ilp_flow_tlrnc', text='Tolerance')
        col.prop(st, 'qdb_quality_th')
        col.prop(st, 'qdb_laplace_w')
        layout.row(align=True).prop(st, 'qdb_sort', expand=True)
        row = layout.row(align=True)
        row.operator('sketch_retopo.query_stop', icon='PAUSE')
        row.operator('sketch_retopo.query_restart', icon='FILE_REFRESH')
        row.operator('sketch_retopo.query_delete_flow', icon='X')


class SKETCHRETOPO_PT_config(bpy.types.Panel):
    bl_label = 'Config'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        layout.enabled = _session(context) is not None
        col = layout.column(align=True)
        for p in ('proj_offset', 'snap_size', 'quad_size', 'segment_size', 'brush_spine', 'brush_autocmpl', 'brush_move'):
            col.prop(st, p)
        col = layout.column(align=True)
        for p in ('undo_buffer_size', 'normal_smooth_iter', 'autocmpl_preview_interval', 'cylinder_num_div',
                  'use_even_num_subdiv', 'loop_threshold_area', 'loop_threshold_normal'):
            col.prop(st, p)


class SKETCHRETOPO_PT_snakes(bpy.types.Panel):
    bl_label = 'Snakes'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        layout.enabled = _session(context) is not None
        col = layout.column(align=True)
        col.prop(st, 'snakes_on_ctrl_shift')
        col = layout.column(align=True)
        for p in ('feature_smooth_iter', 'feature_type', 'use_feature_as_energy', 'negate_energy', 'snakes_internal1',
                  'snakes_internal2', 'snakes_external', 'snakes_damping'):
            col.prop(st, p)


class SKETCHRETOPO_PT_camera(bpy.types.Panel):
    bl_label = 'Camera'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        row = layout.row(align=True)
        row.prop(st, 'camera_upright', toggle=True)
        row.prop(st, 'use_ortho', toggle=True)
        row.prop(st, 'auto_camera_center', toggle=True)
        row = layout.row(align=True)
        row.prop(st, 'turntable_mode', toggle=True)
        row.prop(st, 'turntable_speed')
        row = layout.row(align=True)
        row.operator('sketch_retopo.reset_center')
        row.operator('sketch_retopo.camera_params')


class SKETCHRETOPO_PT_render(bpy.types.Panel):
    bl_label = 'Render'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        st = context.scene.sketch_retopo
        col = layout.column(align=True)
        for p in ('always_show_quads', 'show_auxiliary', 'show_left_side', 'show_axis', 'show_bbox', 'phong_tessellation',
                  'phong_subdiv', 'phong_weight', 'patch_alpha', 'corner_radius', 'edge_width_interior',
                  'edge_width_boundary', 'singularity_radius'):
            col.prop(st, p)


class SKETCHRETOPO_PT_help(bpy.types.Panel):
    bl_label = 'Controls'
    bl_parent_id = 'SKETCHRETOPO_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SketchRetopo'

    def draw(self, context):
        layout = self.layout
        sess = _session(context)
        mode = sess.engine.state_get() if sess is not None else 0
        box = layout.box()
        box.label(text='%s (%s)' % (MODE_NAMES.get(mode, ''), _MODE_ORDER_KEYS.get(mode, '')), icon='HAND')
        for line in CONTROLS.get(mode, []):
            box.label(text=line)
        box = layout.box()
        box.label(text='All modes', icon='MOUSE_LMB_DRAG')
        for line in COMMON_CONTROLS:
            box.label(text=line)


classes = (
    SKETCHRETOPO_PT_main,
    SKETCHRETOPO_PT_file,
    SKETCHRETOPO_PT_query_db,
    SKETCHRETOPO_PT_config,
    SKETCHRETOPO_PT_snakes,
    SKETCHRETOPO_PT_camera,
    SKETCHRETOPO_PT_render,
    SKETCHRETOPO_PT_help,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
