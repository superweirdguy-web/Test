# Sessions tie a SketchRetopo engine to a Blender mesh object (the base mesh).
# Each session lives in memory while Blender runs and is stored in a Text
# datablock ("SketchRetopo:<object name>") when the .blend file is saved.

import bpy
import numpy as np

from .core.basemesh import BaseMesh
from .core.engine import Engine
from .core import fileio

TEXT_PREFIX = 'SketchRetopo:'

_sessions = {}

# name of a base object the running modal session should switch to (set by load/import operators)
pending_switch = None


class Session:
    def __init__(self, obj_name, engine, signature):
        self.obj_name = obj_name
        self.engine = engine
        self.signature = signature
        self.running = False
        self.stop_requested = False
        self.renderer = None

    @property
    def obj(self):
        return bpy.data.objects.get(self.obj_name)


def mesh_arrays(obj, depsgraph=None):
    """Evaluated (modifiers applied), triangulated base mesh in object-local coordinates."""
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        nv = len(mesh.vertices)
        verts = np.empty(nv * 3)
        mesh.vertices.foreach_get('co', verts)
        normals = np.empty(nv * 3)
        try:
            mesh.vertex_normals.foreach_get('vector', normals)
        except AttributeError:      # older API
            mesh.vertices.foreach_get('normal', normals)
        nt = len(mesh.loop_triangles)
        tris = np.empty(nt * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get('vertices', tris)
        uvs = None
        if mesh.uv_layers.active is not None and nt:
            loops = np.empty(nt * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get('loops', loops)
            luv = np.empty(len(mesh.loops) * 2)
            mesh.uv_layers.active.data.foreach_get('uv', luv)
            uvs = luv.reshape(-1, 2)[loops].reshape(nt, 3, 2)
    finally:
        eval_obj.to_mesh_clear()
    return verts.reshape(-1, 3), tris.reshape(-1, 3), normals.reshape(-1, 3), uvs


def signature_of(verts, tris):
    if len(verts) == 0:
        return (0, 0, 0.0)
    return (len(verts), len(tris), float(np.round(np.abs(verts).sum(), 6)))


def build_basemesh(obj):
    verts, tris, normals, uvs = mesh_arrays(obj)
    bm = BaseMesh(verts, tris, normals, uvs)
    return bm, signature_of(verts, tris)


def get(obj):
    if obj is None:
        return None
    return _sessions.get(obj.name)


def all_sessions():
    return list(_sessions.values())


def running_session():
    for s in _sessions.values():
        if s.running:
            return s
    return None


def get_or_create(obj, report=None):
    """Returns the session for obj, creating it (and restoring a stored sketch) if needed."""
    session = _sessions.get(obj.name)
    bm, sig = build_basemesh(obj)
    if len(bm.tris) == 0:
        raise RuntimeError('The base mesh has no faces')
    if session is not None:
        if session.signature != sig:
            # the base mesh changed: keep the sketch, rebuild acceleration data and reproject
            eng = session.engine
            eng.basemesh = bm
            session.signature = sig
            eng.set_basemesh(bm, reset_curves=False)
            eng.reproject_all()
        return session
    eng = Engine()
    eng.set_basemesh(bm)
    session = Session(obj.name, eng, sig)
    _sessions[obj.name] = session
    text = bpy.data.texts.get(TEXT_PREFIX + obj.name)
    if text is not None:
        try:
            data = fileio.read_xml_string(text.as_string())
            data['restore'](eng)
            eng.set_basemesh(bm, reset_curves=False)
            if report:
                report({'INFO'}, 'SketchRetopo: restored the sketch stored in this file')
        except Exception as ex:   # corrupted text: start fresh
            if report:
                report({'WARNING'}, 'SketchRetopo: could not restore stored sketch (%s)' % ex)
    return session


def replace_engine_basemesh(session, bm, sig):
    session.engine.set_basemesh(bm, reset_curves=False)
    session.signature = sig


def remove(obj_name):
    _sessions.pop(obj_name, None)


def clear_all():
    _sessions.clear()


def store_all():
    """Writes every session into its Text datablock (called before saving the .blend)."""
    for s in list(_sessions.values()):
        if s.obj is None:
            continue
        name = TEXT_PREFIX + s.obj_name
        text = bpy.data.texts.get(name)
        cn = s.engine.curvenetwork
        if cn.empty() and text is None:
            continue
        if text is None:
            text = bpy.data.texts.new(name)
        text.from_string(fileio.session_to_string(s.engine))
