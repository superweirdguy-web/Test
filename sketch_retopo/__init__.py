# SketchRetopo for Blender: a port of the SketchRetopo prototype by Kenshi Takayama,
# Daniele Panozzo, Alexander Sorkine-Hornung and Olga Sorkine-Hornung,
# "Sketch-Based Generation and Editing of Quad Meshes", ACM SIGGRAPH 2013.
# See README.md for licensing notes.

bl_info = {
    'name': 'SketchRetopo',
    'author': 'Port of SketchRetopo (Takayama et al., SIGGRAPH 2013)',
    'version': (1, 0, 0),
    'blender': (4, 2, 0),
    'location': 'View3D > Sidebar > SketchRetopo',
    'description': 'Sketch-based quad retopology: sketch curves on a mesh to generate and edit quad patches',
    'category': 'Mesh',
}

import bpy
from bpy.app.handlers import persistent

from . import props, operators, ui, session


@persistent
def _save_pre(*_args):
    try:
        session.store_all()
    except Exception as ex:  # never block saving the .blend
        print('SketchRetopo: could not store sessions:', ex)


@persistent
def _load_post(*_args):
    session.clear_all()


def register():
    props.register()
    operators.register()
    ui.register()
    if _save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_save_pre)
    if _load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post)


def unregister():
    if _save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_save_pre)
    if _load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post)
    ui.unregister()
    operators.unregister()
    props.unregister()
    session.clear_all()
