# SketchRetopo for Blender

A Blender add-on that ports **SketchRetopo**, the sketch-based quad retopology prototype by
Kenshi Takayama, Daniele Panozzo, Alexander Sorkine-Hornung and Olga Sorkine-Hornung:

> *Sketch-Based Generation and Editing of Quad Meshes*, ACM Transactions on Graphics 32(4), SIGGRAPH 2013.
> Project page: <https://igl.ethz.ch/projects/sketch-retopo/>

You sketch curves directly on a high-resolution mesh. Whenever curves enclose a region with 2–6 corners,
a quad patch is generated immediately with the pattern-based quadrangulation of
*Pattern-Based Quadrangulation for N-Sided Patches* (Takayama et al., SGP 2014). You then refine the
topology (edge counts, irregular vertices, T-junctions, edge loops) and export a clean quad mesh.

The add-on is a line-by-line port of the original C++ source (SketchRetopo + patchgen). All ten
editing modes, the shortcuts, the snapping and T-junction rules, the patch patterns, undo/redo,
symmetry, and the XML file format behave like the original program.

## Installation

Requires **Blender 4.2 or newer** (it also runs on 4.0/4.1 when installed as a legacy add-on).
No third-party Python packages are needed.

1. Zip the `sketch_retopo` folder (the zip must contain the folder itself), or download the
   repository as a zip and extract `sketch_retopo` into its own zip.
2. In Blender: *Edit > Preferences > Add-ons*, open the drop-down menu (top right) and choose
   *Install from Disk…*, then pick the zip.
3. Enable **SketchRetopo**. The panel appears in the 3D View sidebar (`N`) under the **SketchRetopo** tab.

## Quick start

1. Select the high-resolution mesh you want to retopologize (or use *Import Base Mesh* for an
   OBJ/OFF/PLY/STL file), then press **Start SketchRetopo** in the sidebar. The status bar lists the shortcuts.
2. Draw four curves that cross or meet at corners (left-drag). A quad patch appears as soon as a
   region is closed.
3. Press `A` (edit topology) and click a curve, then press `C`/`X` to add/remove two edge subdivisions.
4. Press `Ctrl+E` (or *Export Retopo Mesh to Object*) to create the quad mesh object.
5. Press `Esc` to stop. The sketch stays in memory and is saved inside the `.blend` file
   (in a text block named `SketchRetopo:<object>`), so you can continue later.

## Controls

The mouse position is shown with a transparent sphere whose radius is the **snapping radius**.
Most operations use the left mouse button.

### Common to all modes

| Input | Action |
| --- | --- |
| `Alt` + left / middle / right drag | Rotate / pan / zoom the camera (as in Maya). `Alt+Ctrl`+left drag also pans. Hold `Shift` when releasing a rotation to snap to the nearest axis view. |
| Middle mouse, wheel, numpad | Blender's own navigation keeps working |
| `Shift` + right drag | Snapping radius |
| `Ctrl` + right drag | Default quad size for new patches |
| `Ctrl+Shift` + right drag | Curve sampling interval |
| `Shift` + drag *(curve modes)* | Smooth a curve |
| `Ctrl+Shift` + drag *(curve modes)* | Smooth a curve with geometric snakes |
| `Space` | Toggle quad-mesh-only rendering |
| `Ctrl+B` | Toggle base-mesh-only rendering |
| Hold `H` + drag | Hide the base mesh on one side of the stroke; `H` + click unhides |
| `Ctrl+Z` / `Ctrl+Y` (or `Ctrl+Shift+Z`) | Undo / redo |
| `Ctrl+S` / `Ctrl+O` | Save / load the XML file |
| `Ctrl+I` | Import a base mesh file |
| `Ctrl+E` | Export the retopo mesh (creates/updates a mesh object) |
| `Ctrl+F` | Toggle full screen |
| `0` | Reset the camera rotation center to the base mesh center |
| `6` | Turntable mode on/off |
| `7` | Statistics (triangles, quads, 3/4/other-sided patches) |
| `8` | Show/set the camera as text |
| `9` | Debug: random edge counts for a single patch |
| `Esc` | Stop the session |

### Modes

| Key | Mode | How to use |
| --- | --- | --- |
| `Q` | **Basic sketching** | Drag to draw a curve. Crossings become corners (red), ends close to green endpoints snap, ends close to a curve make a T-junction (blue). A region bounded by 2–6 corners becomes a patch (patches are colored by their number of sides). `Ctrl`+drag from an endpoint extends the curve without a corner. Hold `C` and drag to erase crossed curves. `G` erases every open-ended curve. Hold `Z` and click to rebuild the curve network around the clicked spot. |
| `W` | **Spine sketching** | Draw a single stroke; a strip of quads is created around it. Right-drag sets the width (blue sphere). Ends near a curve snap orthogonally; strokes running along a patch boundary snap in parallel (and may stop half way, creating T-junctions). |
| `E` | **Auto-completion** | Drag along the boundary of existing patches: a green preview appears. Release to freeze it, click anywhere to accept. `Z`/`X` browse alternatives (largest first), `C` prefers a triangle in concave areas. Right-drag sets the maximum size. |
| `R` | **Laser cutting** | Drag a line (purple). The base mesh is cut by the plane through the line and the eye; the intersection curve is traced from the line's start point. |
| `T` | **Cylinder sketching** | Draw a curve connecting two loops (typically from laser cuts). Three more curves are generated around the cylinder. If a loop has corners, the other must have none or the same number, and the stroke must end on corners. |
| `Y` | **Deform curves** | Drag a corner or endpoint, or drag the middle of a curve. Released corners/endpoints snap to nearby curves. `Ctrl`+drag oversketches a curve between two of its points. |
| `F` | **Corner editing** | Drag from inside a patch to one of its corners to toggle corner / T-junction. `Ctrl`+click on a curve creates a corner, or erases a corner with two curves. |
| `A` | **Topology editing** | Click a patch boundary curve to select it (red); `Shift`+click toggles, `Ctrl`+click selects every curve along the same quad strip. `X`/`C` change the number of edges by −2/+2, `Z`/`V` by −1/+1 (the total around a patch must stay even). Click inside a patch to select it: on 4-sided patches drag to move the irregular vertices; on other patches click the arrows and use `Z`/`X` (decrease) or `C`/`V` (increase) the selected parameter; `B`/`Shift+B` switch to the next/previous pattern. Patches that cannot be quadrangulated are black with blue points along their boundary. |
| `S` | **Edge loop insertion** | Hover a patch boundary to preview the edge loop (green), click to insert it. |
| `D` | **Move vertices** | Drag to move quad vertices with a smooth falloff (yellow sphere = brush, right-drag to resize). `Shift`+drag smooths. `G` applies one step of global smoothing. |

## Sidebar options

* **File** – import a base mesh, export to an object or to OBJ/OFF/PLY/STL, save/load XML,
  autosave (every *n* seconds once the sketch has a file name; `name.N.xml` files are numbered
  `name.N+1.xml` like the original), load a PNG MatCap as the base mesh material, or use the
  original SketchRetopo MatCap.
* **Symmetric (X=0)** – the base mesh is treated as symmetric about its local X=0 plane: only the
  X>0 side is edited and the other side is mirrored (and merged on export). Toggling discards all
  curves, so do it right after starting.
* **Config** – projection offset (distance of curves/quads above the surface), snapping radius, default
  quad size, curve sampling, brush sizes, undo buffer size, auto-completion preview interval, number
  of cylinder curves, even edge counts, loop detection thresholds.
* **Snakes** – energy used by `Ctrl+Shift`+drag (feature type, smoothing, weights).
* **Camera** – upright (turntable) rotation, orthographic projection, automatic rotation center after
  panning, turntable speed, reset center, camera parameters as text.
* **Render** – render mode, quads always visible, auxiliary data and mirrored side in quad-only mode,
  Phong tessellation of patches, opacity, corner/edge/singularity sizes, axis and bounding box.

## Files

* **XML** – the format of the original program (version 1): base mesh, configuration, curve network,
  patches and material. Files saved by the original SketchRetopo can be loaded (the base mesh is created
  as a new object), and files saved here can be opened by the original program.
* **.blend** – sessions are stored in text blocks named `SketchRetopo:<object>` when you save, and
  restored the next time you start SketchRetopo on that object.

All geometry is computed in the base object's local space, so the object may be moved, rotated or
scaled. Modifiers are applied (the evaluated mesh is used); if the base mesh changes between sessions,
the sketch is projected onto the new surface.

## Differences from the original program

The behavior follows the C++ source; these parts necessarily differ:

* **Viewport and camera.** The camera is Blender's 3D view. The original Maya-style `Alt`+drag camera is
  implemented on top of it; Blender's own navigation (middle mouse, wheel, numpad) also works. "Upright"
  keeps Blender's Z axis up (the original kept Y up).
* **Integer programming.** The original uses lp_solve to choose patch parameters. This port includes a
  small exact branch-and-bound solver. When several optimal parameter sets exist, it may pick a different
  (equally valid) one, so a patch can occasionally get a different but equivalent layout of its irregular
  vertices.
* **Geodesics.** Cylinder sketching (and the hidden harmonic re-parameterization, `Ctrl`+click in move-vertex
  mode) used an exact geodesic library. This port uses shortest edge paths straightened on the surface.
* **Rendering.** The base mesh is drawn by Blender (use *SketchRetopo MatCap* for the original look). The
  translucent overlays, colors and sizes follow the original. The "texture overlay" debug option, the
  exponential-map and snake-energy visualizations, and the background color ramp are not reproduced.
* **Export.** `Ctrl+E` creates or updates a Blender mesh object (*Export to File* writes OBJ/OFF/PLY/STL like
  the original).
* **Robustness.** A few situations that crash or hang the original (for example a boundary whose edge counts
  admit no pattern, or a laser plane passing exactly through mesh vertices) are handled gracefully.

## Development

The core is independent of Blender's UI and only needs `numpy` and `mathutils`:

```
sketch_retopo/core/   port of SketchRetopo (engine, curve network, states, base mesh, patchgen, ILP)
sketch_retopo/*.py    Blender integration (modal operator, drawing, panels, sessions)
tests/                tests (run with the `bpy` module from PyPI or inside Blender)
```

```
pip install bpy            # matching your Python version
python tests/test_patchgen.py
python tests/test_core_sketch.py
python tests/test_core_tools.py
python tests/test_blender_addon.py
```

`tests/gui_test.py` drives the modal operator inside a real Blender window with simulated mouse and
keyboard events and saves screenshots (Linux, needs `xvfb-run` and ImageMagick):

```
xvfb-run -a -s "-screen 0 1600x1000x24" blender --enable-event-simulate --python tests/gui_test.py -- /tmp/gui
```

## License and credits

SketchRetopo and patchgen are © Kenshi Takayama and co-authors (ETH Zurich, Interactive Geometry Lab).
This add-on is a derivative port of their source code, which its authors distribute for testing, research
and non-commercial use; the same terms apply to this port. Please refer to the project page for the
original license and cite the papers above when using it in research. The bundled MatCap image is the
default material of the original program.
