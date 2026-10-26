# Blender registration for the add-on (imported by __init__.py when running inside Blender).

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
