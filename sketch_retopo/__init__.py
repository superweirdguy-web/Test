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

try:
    import bpy  # noqa: F401
except ImportError:
    # Outside Blender only sketch_retopo.core is usable. Patch generation needs just numpy;
    # sketching on a base mesh also needs mathutils, which comes with the `bpy` package on PyPI.
    pass
else:
    from .addon import register, unregister  # noqa: F401
