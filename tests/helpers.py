import math
import os
import sys

import numpy as np

try:
    import bpy  # noqa: F401  (makes mathutils importable when using the bpy wheel)
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketch_retopo.core.basemesh import BaseMesh
from sketch_retopo.core.engine import Engine


def grid_mesh(n=60, size=2.0, z_func=None):
    xs = np.linspace(-size / 2, size / 2, n + 1)
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = xs[i], xs[j]
            z = z_func(x, y) if z_func else 0.0
            verts.append((x, y, z))
    tris = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + n + 2
            d = a + n + 1
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.array(verts), np.array(tris)


def uv_sphere(nu=64, nv=32, r=1.0):
    verts = [(0, 0, r)]
    for j in range(1, nv):
        th = math.pi * j / nv
        for i in range(nu):
            ph = 2 * math.pi * i / nu
            verts.append((r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)))
    verts.append((0, 0, -r))
    tris = []
    for i in range(nu):
        tris.append((0, 1 + i, 1 + (i + 1) % nu))
    for j in range(nv - 2):
        for i in range(nu):
            a = 1 + j * nu + i
            b = 1 + j * nu + (i + 1) % nu
            c = 1 + (j + 1) * nu + (i + 1) % nu
            d = 1 + (j + 1) * nu + i
            tris.append((a, d, c))
            tris.append((a, c, b))
    last = len(verts) - 1
    base = 1 + (nv - 2) * nu
    for i in range(nu):
        tris.append((last, base + (i + 1) % nu, base + i))
    return np.array(verts, dtype=float), np.array(tris)


class TopView:
    """Orthographic camera looking down -Z; 1 pixel = `scale` units, pixel (0,0) at world (-w/2*scale, ...)."""

    def __init__(self, width=1000, height=1000, scale=0.0025, distance=3.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.distance = distance

    def to_world(self, x, y):
        return ((x - self.width / 2) * self.scale, (y - self.height / 2) * self.scale)

    def to_pixel(self, X, Y):
        return (X / self.scale + self.width / 2, Y / self.scale + self.height / 2)

    def camera_distance(self):
        return self.distance

    def eye(self):
        return np.array((0.0, 0.0, self.distance))

    def eye_to_center(self):
        return np.array((0.0, 0.0, -self.distance))

    def is_ortho(self):
        return True

    def ray(self, x, y):
        X, Y = self.to_world(x, y)
        return np.array((X, Y, 10.0)), np.array((0.0, 0.0, -1.0))

    def project(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return np.stack([pts[:, 0] / self.scale + self.width / 2, pts[:, 1] / self.scale + self.height / 2], axis=1)


def make_engine(verts, tris, view=None):
    eng = Engine()
    eng.view = view or TopView()
    eng.set_basemesh(BaseMesh(verts, tris))
    eng.update_camera_distance()
    return eng


def stroke(eng, world_pts, button='LEFT', shift=False, ctrl=False, alt=False, steps=None):
    """Drag along world-space XY points (interpolated)."""
    v = eng.view
    pix = [v.to_pixel(X, Y) for X, Y in world_pts]
    path = []
    for (x0, y0), (x1, y1) in zip(pix[:-1], pix[1:]):
        n = steps or max(2, int(math.hypot(x1 - x0, y1 - y0) / 3))
        for k in range(n):
            t = k / float(n)
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    path.append(pix[-1])
    eng.mouse_move(*path[0])
    eng.mouse_down(path[0][0], path[0][1], button, shift, ctrl, alt)
    for p in path[1:]:
        eng.mouse_move(*p)
    eng.mouse_up(path[-1][0], path[-1][1], button, shift, ctrl, alt)


def click(eng, X, Y, button='LEFT', shift=False, ctrl=False, alt=False):
    x, y = eng.view.to_pixel(X, Y)
    eng.mouse_move(x, y)
    eng.mouse_down(x, y, button, shift, ctrl, alt)
    eng.mouse_up(x, y, button, shift, ctrl, alt)


def hover(eng, X, Y):
    x, y = eng.view.to_pixel(X, Y)
    eng.mouse_move(x, y)
