# Adapter between Blender's 3D view and the engine's `view` protocol, plus the
# original program's Maya-style camera (kt84::CameraFree / CameraUpright) driven
# through the 3D view's RegionView3D.

import math

import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Quaternion, Vector


class BlenderView:
    """All positions are expressed in the base object's local space."""

    def __init__(self, region, rv3d, obj):
        self.region = region
        self.rv3d = rv3d
        self.width = region.width
        self.height = region.height
        M = obj.matrix_world.copy()
        self.M = M
        self.Minv = M.inverted_safe()
        self.Minv3 = self.Minv.to_3x3()
        s = M.to_scale()
        self.scale = (abs(s[0]) + abs(s[1]) + abs(s[2])) / 3.0 or 1.0
        self._mvp = np.array(rv3d.perspective_matrix @ M)

    def camera_distance(self):
        """The eye distance SketchRetopo2 would have for the same view.  Snapping radius, brush sizes and corner
        sizes are proportional to it, so they keep the same size on screen as in the original program (which
        uses a 40 degree perspective, or an orthographic view 2.5 x 0.1 x distance high)."""
        wm = self.rv3d.window_matrix
        k = abs(wm[1][1]) or 1.0
        if self.rv3d.is_perspective:
            half_height = self.rv3d.view_distance / k
            d = half_height / math.tan(math.radians(20.0))
        else:
            half_height = 1.0 / k
            d = half_height / 0.25
        return d / self.scale

    def eye_world(self):
        return self.rv3d.view_matrix.inverted().translation

    def eye(self):
        return np.array(self.Minv @ self.eye_world())

    def eye_to_center(self):
        d = self.rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
        return np.array(self.Minv3 @ d) * (self.rv3d.view_distance / self.scale)

    def is_ortho(self):
        return not self.rv3d.is_perspective

    def ray(self, x, y):
        co = (x, y)
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, co)
        o = self.Minv @ origin
        d = (self.Minv3 @ direction).normalized()
        return np.array(o), np.array(d)

    def project(self, pts):
        P = np.asarray(pts, dtype=float).reshape(-1, 3)
        H = np.hstack([P, np.ones((len(P), 1))]) @ self._mvp.T
        w = H[:, 3:4]
        w[w == 0] = 1e-12
        ndc = H[:, :2] / w
        return np.stack([(ndc[:, 0] + 1) * 0.5 * self.width, (ndc[:, 1] + 1) * 0.5 * self.height], axis=1)


# ---- camera ---------------------------------------------------------------------------------------

def get_camera(rv3d):
    center = Vector(rv3d.view_location)
    rot = rv3d.view_rotation
    eye = center + rot @ Vector((0.0, 0.0, rv3d.view_distance))
    up = rot @ Vector((0.0, 1.0, 0.0))
    return eye, center, up


def set_camera(rv3d, eye, center, up):
    z = eye - center
    dist = z.length
    if dist == 0:
        return
    z.normalize()
    x = up.cross(z)
    if x.length == 0:
        return
    x.normalize()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed()
    rv3d.view_rotation = m.to_quaternion()
    rv3d.view_location = center
    rv3d.view_distance = dist


class MayaCamera:
    """Alt + left/middle/right drag = rotate/pan/zoom, as in SketchRetopo (similar to Autodesk Maya)."""

    NONE, ROTATE, PAN, ZOOM = range(4)

    def __init__(self):
        self.drag_mode = MayaCamera.NONE
        self.prev = (0, 0)
        self.upright = False

    def mouse_down(self, x, y, mode):
        self.prev = (x, y)
        self.drag_mode = mode

    def mouse_up(self):
        self.drag_mode = MayaCamera.NONE

    def mouse_move(self, rv3d, region, x, y):
        if self.drag_mode == MayaCamera.NONE:
            return
        vs = (region.width + region.height) / 2.0
        dx = x - self.prev[0]
        dy = y - self.prev[1]
        eye, center, up = get_camera(rv3d)
        c2e = eye - center
        if self.upright:
            r = c2e.length
            theta = math.atan2(c2e.x, -c2e.y)
            phi = math.asin(max(-1.0, min(1.0, c2e.z / r))) if r > 0 else 0.0
            if up.z < 0:
                theta += math.pi
                phi = math.pi - phi
            if self.drag_mode == MayaCamera.ROTATE:
                theta -= 2 * math.pi * dx / vs
                phi += 2 * math.pi * dy / vs
            dir_ = Vector((math.cos(phi) * math.sin(theta), -math.cos(phi) * math.cos(theta), math.sin(phi)))
            new_up = Vector((-math.sin(phi) * math.sin(theta), math.sin(phi) * math.cos(theta), math.cos(phi)))
            if self.drag_mode == MayaCamera.PAN:
                right = Vector((math.cos(theta), math.sin(theta), 0.0))
                center = center - (r * dx / vs) * right
                center = center - (r * dy / vs) * new_up
            elif self.drag_mode == MayaCamera.ZOOM:
                r *= 1 - (dx + dy) / vs
                r = max(r, 1e-6)
            set_camera(rv3d, center + r * dir_, center, new_up)
        else:
            if self.drag_mode == MayaCamera.ROTATE:
                theta_x = 2 * math.pi * dx / vs
                theta_y = 2 * math.pi * dy / vs
                c2e = Matrix.Rotation(-theta_x, 3, up) @ c2e
                left = c2e.cross(up)
                if left.length > 0:
                    left.normalize()
                    rot_vrt = Matrix.Rotation(-theta_y, 3, left)
                    c2e = rot_vrt @ c2e
                    up = rot_vrt @ up
                eye = center + c2e
            elif self.drag_mode == MayaCamera.PAN:
                length = c2e.length
                trans_x = (-c2e).cross(up).normalized() * (-length * dx / vs)
                trans_y = up.normalized() * (-length * dy / vs)
                eye = eye + trans_x + trans_y
                center = center + trans_x + trans_y
            elif self.drag_mode == MayaCamera.ZOOM:
                eye = eye - (-c2e) * (dy / vs)
                if (eye - center).length < 1e-6:
                    return
            set_camera(rv3d, eye, center, up)
        self.prev = (x, y)

    def snap_to_canonical(self, rv3d):
        eye, center, up = get_camera(rv3d)
        d = eye - center
        dr = d.length
        if self.upright:
            r = dr
            theta = math.atan2(d.x, -d.y)
            phi = math.asin(max(-1.0, min(1.0, d.z / r))) if r > 0 else 0.0
            q = 0.5 * math.pi
            theta = round(theta / q) * q
            phi = round(phi / q) * q
            dir_ = Vector((math.cos(phi) * math.sin(theta), -math.cos(phi) * math.cos(theta), math.sin(phi)))
            new_up = Vector((-math.sin(phi) * math.sin(theta), math.sin(phi) * math.cos(theta), math.cos(phi)))
            set_camera(rv3d, center + r * dir_, center, new_up)
            return
        axis = max(range(3), key=lambda i: abs(d[i]))
        nd = Vector((0.0, 0.0, 0.0))
        nd[axis] = dr if d[axis] >= 0 else -dr
        axis_up = max(range(3), key=lambda i: abs(up[i]) if i != axis else -1)
        nu = Vector((0.0, 0.0, 0.0))
        nu[axis_up] = 1.0 if up[axis_up] >= 0 else -1.0
        set_camera(rv3d, center + nd, center, nu)

    @staticmethod
    def update_center(rv3d, center_new):
        """Moves the rotation center along the view axis to the point closest to center_new (eye fixed)."""
        eye, center, up = get_camera(rv3d)
        d = eye - center
        if d.length_squared == 0:
            return
        t = d.dot(center_new - center) / d.length_squared
        set_camera(rv3d, eye, center + t * d, up)

    @staticmethod
    def set_center(rv3d, center_new):
        """kt84::Camera::center = ... (the eye stays where it is)."""
        eye, center, up = get_camera(rv3d)
        if (eye - center_new).length == 0:
            return
        set_camera(rv3d, eye, center_new, up)

    @staticmethod
    def turntable(rv3d, speed):
        eye, center, up = get_camera(rv3d)
        rot = Matrix.Rotation(speed, 3, Vector((0.0, 0.0, 1.0)))
        set_camera(rv3d, center + rot @ (eye - center), center, rot @ up)
