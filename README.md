# SketchRetopo for Blender

A Blender add-on that ports **SketchRetopo2**, the sketch-based quad retopology program by
Kenshi Takayama and co-authors (the demo distributed on the project page, version of October 2017):

> *Sketch-Based Generation and Editing of Quad Meshes*, Takayama, Panozzo, Sorkine-Hornung, Sorkine-Hornung,
> ACM Transactions on Graphics 32(4), SIGGRAPH 2013. Project page: <https://igl.ethz.ch/projects/sketch-retopo/>
>
> *Data-Driven Interactive Quadrangulation*, Marcias, Takayama, Pietroni, Panozzo, Sorkine-Hornung, Puppo,
> Cignoni, ACM Transactions on Graphics 34(4), SIGGRAPH 2015.

You sketch curves directly on a high-resolution mesh. Whenever curves enclose a region with 2–6 corners,
a quad patch is generated immediately with the pattern-based quadrangulation of
*Pattern-Based Quadrangulation for N-Sided Patches* (Takayama et al., SGP 2014). You then refine the
topology (edge counts, irregular vertices, T-junctions, edge loops) and export a clean quad mesh.

The add-on is a line-by-line port of the original C++ source (SketchRetopo2 + patchgen). The editing
modes, the shortcuts, the snapping and T-junction rules, the patch patterns, undo/redo, symmetry and the
XML file format behave like the original program, and its XML files open in SketchRetopo2 (and vice
versa). Spine sketching and auto-completion, which the earlier SketchRetopo release had, are kept on the
spare keys `Y` and `U`.

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
| `Ctrl+Shift` + drag *(curve modes)* | Smooth a curve (or with geometric snakes, if enabled in the *Snakes* panel) |
| `Space` | Toggle quad-mesh-only rendering |
| `Ctrl+B` | Toggle base-mesh-only rendering |
| Hold `H` + drag | Hide the base mesh on one side of the stroke; `H` + click unhides |
| `Ctrl+Z` / `Ctrl+Y` (or `Ctrl+Shift+Z`) | Undo / redo |
| `Ctrl+S` / `Ctrl+O` | Save / load the XML file |
| `Ctrl+I` | Import a base mesh file |
| `Ctrl+E` | Export the retopo mesh (creates/updates a mesh object; it stays hidden while the session runs, because the session already draws the same quads, and appears when you press `Esc`) |
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
| `Q` | **Basic sketching** | Drag to draw a curve. Crossings become corners (red), ends close to green endpoints snap, ends close to a curve make a T-junction (blue). A region bounded by 2–6 corners becomes a patch. `Ctrl`+drag from an endpoint extends the curve without a corner. Hold `C` and drag to erase the curves crossed. Hold `X` and drag from a curve into an empty region to create its patch (also for regions that are not filled automatically). Hold `V` and click a patch to delete it (its curves stay). Hold `B` and click near an edge of a hole in the base mesh to add a curve along it (`N`: slightly inside the hole, resampled). Hold `Z` and click to join the curves around the clicked spot. `G` erases every open-ended curve. |
| `W` | **Deform curves** | Drag a corner or endpoint, or drag the middle of a curve. Released corners/endpoints snap to nearby curves. `Ctrl`+drag oversketches a curve between two of its points. |
| `E` | **Laser sketching** | Drag a line. The base mesh is cut by the plane through the line and the eye; the intersection curve is traced from the line's start point (e.g. a loop around an arm). |
| `R` | **Cylinder sketching** | Draw a curve connecting two loops (typically from laser cuts). More curves are generated around the cylinder. If a loop has corners, the other must have none or the same number, and the stroke must end on corners. |
| `T` | **Corner editing** | Drag from inside a patch to one of its corners to toggle corner / T-junction for that patch (a corner shows red on the patches it is a corner of, blue where it is a T-junction). `Ctrl`+click on a curve creates a corner, or removes a corner joining two curves. |
| `A` | **Topology editing** | Click a curve to select it (red); its number of edges is shown. `Shift`+click toggles, `Ctrl`+click selects every curve across the same quad strip. `C`/`X` change the number of edges by +2/−2, `V`/`Z` by +1/−1 (a patch needs an even total, so a single ±1 leaves the region empty until another curve is changed and you sketch again or use `X`+drag). Click inside a patch to select it (it gets its default layout): on 4-sided patches drag to move the poles; on other patches click an arrow and use `Z`/`X` (decrease) or `C`/`V` (increase), `B`/`Shift+B` switch to the next/previous pattern. Curves without patches show their edges as blue points. |
| `S` | **Query database** | Click a patch, then drag flow strokes inside it (`Shift`+drag adds more). Layouts matching the boundary and the flow are looked up in the patch database (`patches.db`, see below); `Z`/`X` show the previous/next result. |
| `D` | **Move vertices** | Drag to move quad vertices with a smooth falloff (yellow sphere = brush, right-drag to resize). `Shift`+drag smooths. `Ctrl`+click a patch relaxes it over the surface. `G` applies one step of global smoothing. |
| `F` | **Edge loop insertion** | Hover a patch boundary to preview the edge loop, click to insert it. |
| `Y` | Spine sketching *(earlier SketchRetopo)* | Draw a single stroke; a strip of quads is created around it. Right-drag sets the width. |
| `U` | Auto-completion *(earlier SketchRetopo)* | Drag along the boundary of existing patches: a preview appears. Release to freeze it, click anywhere to accept. `Z`/`X` browse alternatives, `C` prefers a triangle. |

The keys only apply while a session runs and the mouse is over the 3D viewport; `Esc` ends the session
and Blender's own shortcuts apply again.

## Sidebar options

* **File** – import a base mesh, export to an object or to OBJ/OFF/PLY/STL, save/load XML,
  autosave (every *n* seconds once the sketch has a file name; `name.N.xml` files are numbered
  `name.N+1.xml` like the original), load a PNG MatCap as the base mesh material, or use the
  original SketchRetopo MatCap.
* **Symmetric (X=0)** – the other side of the base mesh's local X=0 plane is shown as a mirrored preview and
  added to the exported mesh. **Snap Symmetry** moves curve and quad vertices lying closer to X=0 than the
  curve sampling interval onto the plane, so the two halves join when exported.
* **Config** – projection offset (distance of curves/quads above the surface), snapping radius, default
  quad size, curve sampling, brush sizes, undo buffer size, auto-completion preview interval, number
  of cylinder curves, even edge counts, loop detection thresholds.
* **Snakes** – energy used by `Ctrl+Shift`+drag (feature type, smoothing, weights).
* **Camera** – upright (turntable) rotation, orthographic projection, automatic rotation center after
  panning, turntable speed, reset center, camera parameters as text.
* **Render** – render mode, quads always visible, auxiliary data and mirrored side in quad-only mode,
  Phong tessellation of patches, opacity, corner/edge/singularity sizes, axis and bounding box.

## Patch database (query mode and patches with more than 6 corners)

SketchRetopo2 fills loops of 2–6 corners from built-in patterns. For the query database mode (`S`), and for
loops with more corners (hold `X` and drag into the region), it looks layouts up in a database of patches
learned from artist-made quad meshes: `patches.db`, from
<https://github.com/kenshi84/files/releases/tag/sketch-retopo>. The *huge* version comes in two parts that are
joined into one file: on Windows, in a Command Prompt in the download folder,
`copy /b patches.huge.db.0 + patches.huge.db.1 patches.db`; on macOS or Linux,
`cat patches.huge.db.0 patches.huge.db.1 > patches.db`.

The two files complement each other. `patches.db` (174,000 templates, 740 MB) has more 4- to 8-sided layouts;
its templates with 9 or more corners are tiny, so it rarely fills such loops. The huge file (431,000 templates,
2.6 GB) has fewer 4-sided layouts but covers loops of up to 34 corners, so use it if you fill regions with many
corners.

* Set the file in the add-on preferences (*Patch Database*), or use **Load Patch Database** in the
  sidebar's *Query Database* panel. Like the demo, a `patches.db` in your home folder is also picked up.
  The database is loaded when SketchRetopo starts.
* **Query Database** panel: the result shown (*Index*, also `Z`/`X`), *Freeze* (keep the shown result while
  the lookup goes on), the lookup filters (maximum singularities and concave corners, singularity valence
  range), *ILP Flow* and its tolerance (a polychord must start and end within that many edges of each flow
  stroke's ends), the quality threshold, the Laplace weight used when placing results along the flows, the
  sort order (by flow, by quality, by singularities), and **Stop**, **Restart** and **Delete Flow**.
* The lookup runs in the background while you work; the best result so far is shown in the patch and the
  status is displayed at the top of the viewport. Settings take effect at the next lookup (Restart).
* Results appear within a second or two. A big boundary can match a very large number of layouts (a 12/8/12/8
  patch with one flow stroke finds about 105,000), and the full lookup then takes minutes, as in the original;
  Stop keeps what has been found.
* A flow stroke has to be followable: each result has a polychord that starts and ends within *Flow Tolerance*
  edges of the stroke's ends, so a stroke that crosses a regular patch at a steep angle finds layouts with
  irregular vertices that bend the edge flow.

## Files

* **XML** – SketchRetopo2's format: base mesh, configuration, curve network, patches and material. Files
  saved by SketchRetopo2 (and by the earlier SketchRetopo) can be loaded (the base mesh is created as a new
  object), and files saved here open in SketchRetopo2.
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
* **Patch database lookup.** The original runs the lookup on worker threads with the SCIP solver; this port
  runs it incrementally in Blender's event loop with its own exact enumeration of the integer solutions. It was
  checked against the original C++ lookup, compiled from the demo's source, on both real databases: every
  template decodes identically, and the same patches are found, in the same order (as the original run on one
  thread: with several threads it keeps an arbitrary one of two rotations of a symmetric template that give the
  same mesh), and placed identically (to about 1e-12). The exception is the placement along flow strokes of
  layouts with exactly or nearly parallelogram-shaped quads in the parameter domain (1–6% of the results in the
  tests):
  there the original's bilinear-coordinate formula is numerically unstable and its output depends on rounding
  noise, so it cannot be reproduced. Being free of SCIP's overhead, the lookup needs about a third of the
  original's CPU time; it runs on one core, so the original on four or more cores finishes big lookups sooner.
* **Sizes on screen.** The snapping radius, brushes and corner markers scale with the view as in the
  original (whose camera has a 40° field of view), so they look the same size on screen.

## Development

The core is independent of Blender's UI and only needs `numpy` and `mathutils`:

```
sketch_retopo/core/   port of SketchRetopo (engine, curve network, states, base mesh, patchgen, ILP,
                      learnquad = the patch database lookup)
sketch_retopo/*.py    Blender integration (modal operator, drawing, panels, sessions)
tests/                tests (run with the `bpy` module from PyPI or inside Blender)
```

```
pip install bpy            # matching your Python version
python tests/test_patchgen.py
python tests/test_core_sketch.py
python tests/test_core_tools.py
python tests/test_learnquad.py
python tests/test_blender_addon.py
```

`tests/gui_test.py` drives the modal operator inside a real Blender window with simulated mouse and
keyboard events and saves screenshots (Linux, needs `xvfb-run` and ImageMagick):

```
xvfb-run -a -s "-screen 0 1600x1000x24" blender --enable-event-simulate --python tests/gui_test.py -- /tmp/gui
```

`tests/gui_test_querydb.py` does the same for the query database mode with a real database (pass its path
after the output folder); it also saves animation frames.

## License and credits

SketchRetopo, SketchRetopo2 and patchgen are © Kenshi Takayama and co-authors (ETH Zurich, Interactive Geometry
Lab); the data-driven quadrangulation is © Giorgio Marcias and co-authors (ISTI-CNR).
This add-on is a derivative port of their source code, which its authors distribute for testing, research
and non-commercial use; the same terms apply to this port. Please refer to the project page for the
original license and cite the papers above when using it in research. The bundled MatCap image is the
default material of the original program.
