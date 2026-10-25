"""Runs the add-on inside Blender's Python module (bpy from PyPI, background mode)."""
import os
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import sketch_retopo
from sketch_retopo import session
from helpers import TopView, stroke


def main():
    sketch_retopo.register()
    try:
        run()
    finally:
        sketch_retopo.unregister()


def run():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sketch_retopo.register() if not hasattr(bpy.types.Scene, 'sketch_retopo') else None
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    obj = bpy.context.active_object
    obj.location = (2.0, 0.0, 0.0)
    obj.scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    st = scene.sketch_retopo
    print('defaults without session: snap', round(st.snap_size, 4), 'render mode', st.render_mode)
    st.base_object = obj
    sess = session.get_or_create(obj)
    eng = sess.engine
    print('base mesh', len(eng.basemesh.verts), 'verts', len(eng.basemesh.tris), 'tris; quad size', round(st.quad_size, 4))
    eng.view = TopView(scale=0.003, distance=4.0)
    eng.update_camera_distance()
    for a, b in [((-0.4, -0.4), (0.4, -0.4)), ((0.4, -0.4), (0.4, 0.4)), ((0.4, 0.4), (-0.4, 0.4)), ((-0.4, 0.4), (-0.4, -0.4))]:
        stroke(eng, [a, b])
    print('patches', len(eng.curvenetwork.patches), 'quads', sum(p.n_faces() for p in eng.curvenetwork.patches))
    assert len(eng.curvenetwork.patches) == 1

    # property accessors talk to the engine
    st.snap_size = 0.03
    assert abs(eng.configTemp.snapSize.value - 0.03) < 1e-9
    st.render_mode = '1'
    assert eng.configRender.mode == 1
    st.render_mode = '0'
    st.phong_subdiv = 2
    assert eng.configRender.phong_subdiv == 2
    st.proj_offset = st.proj_offset * 2
    print('props ok; proj offset', round(st.proj_offset, 6), 'symmetric', st.symmetric)

    # export to an object
    res = bpy.ops.sketch_retopo.export_mesh()
    out = st.output_object
    print('export_mesh', res, out.name, len(out.data.vertices), 'verts', len(out.data.polygons), 'faces',
          'matrix copied', tuple(out.matrix_world.translation) == tuple(obj.matrix_world.translation))
    assert all(len(p.vertices) == 4 for p in out.data.polygons)

    # export to files
    tmp = tempfile.mkdtemp()
    for ext in ('obj', 'off', 'ply', 'stl'):
        path = os.path.join(tmp, 'retopo.' + ext)
        r = bpy.ops.sketch_retopo.export_file(filepath=path, file_format=ext)
        print('export', ext, r, os.path.getsize(path), 'bytes')

    # save / load XML through operators
    xml = os.path.join(tmp, 'sketch.xml')
    print('save_xml', bpy.ops.sketch_retopo.save_xml(filepath=xml), os.path.getsize(xml), 'bytes')
    print('load_xml', bpy.ops.sketch_retopo.load_xml(filepath=xml))
    loaded = st.base_object
    sess2 = session.get(loaded)
    print('loaded object', loaded.name, 'patches', len(sess2.engine.curvenetwork.patches))
    assert len(sess2.engine.curvenetwork.patches) == 1

    # persistence inside the .blend
    blend = os.path.join(tmp, 'scene.blend')
    st.base_object = obj
    obj_name = obj.name
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print('texts stored:', [t.name for t in bpy.data.texts])
    bpy.ops.wm.open_mainfile(filepath=blend)
    assert not session.all_sessions()
    obj = bpy.data.objects[obj_name]
    sess3 = session.get_or_create(obj, lambda lvl, msg: print('  report:', lvl, msg))
    print('restored from .blend: patches', len(sess3.engine.curvenetwork.patches))
    assert len(sess3.engine.curvenetwork.patches) == 1

    # operators: set_mode / undo / redo / clear / stats
    bpy.context.scene.sketch_retopo.base_object = obj
    bpy.ops.sketch_retopo.set_mode(mode='7')
    print('mode', sess3.engine.state.state_name)
    bpy.ops.sketch_retopo.clear()
    print('after clear: patches', len(sess3.engine.curvenetwork.patches))
    bpy.ops.sketch_retopo.undo()
    print('after undo: patches', len(sess3.engine.curvenetwork.patches))
    bpy.ops.sketch_retopo.stats()

    # base mesh changes are picked up (modifier) and curves are reprojected
    mod = obj.modifiers.new('sub', 'SUBSURF')
    mod.levels = 1
    sess4 = session.get_or_create(obj)
    print('after adding a modifier: base tris', len(sess4.engine.basemesh.tris), 'patches', len(sess4.engine.curvenetwork.patches))

    # the event-key translation used by the modal operator
    from sketch_retopo.operators import event_key

    class E:
        def __init__(self, type, shift=False, ctrl=False):
            self.type, self.shift, self.ctrl, self.oskey = type, shift, ctrl, False
    assert event_key(E('Q')) == 'q' and event_key(E('B', shift=True)) == 'B'
    assert event_key(E('Z', ctrl=True)) == 'ctrl+z' and event_key(E('Z', shift=True, ctrl=True)) == 'ctrl+shift+z'
    assert event_key(E('SPACE')) == ' ' and event_key(E('ZERO')) == '0' and event_key(E('LEFT_SHIFT')) is None
    print('event keys ok')


if __name__ == '__main__':
    main()
