# Port of kt84/geometry/PointNormal.hh and kt84/geometry/PolylineT.hh (Polyline_PointNormal).
#
# A "PointNormal" is a numpy float64 array of shape (6,): [px, py, pz, nx, ny, nz].

import math

import numpy as np


def pn_make(p, n):
    pn = np.empty(6)
    pn[:3] = p
    pn[3:] = n
    return pn


def pn_zero():
    return np.zeros(6)


def pn_norm(pn):
    x = pn[0]
    y = pn[1]
    z = pn[2]
    return math.sqrt(x * x + y * y + z * z)


def pn_normalize(pn):
    """In-place normalization of the normal part (Eigen::normalize leaves zero vectors untouched)."""
    n = math.sqrt(pn[3] * pn[3] + pn[4] * pn[4] + pn[5] * pn[5])
    if n > 0:
        pn[3:] /= n
    return pn


def pn_normalized(pn):
    return pn_normalize(np.array(pn, dtype=float))


def normalized(v):
    v = np.asarray(v, dtype=float)
    n = math.sqrt(float(np.dot(v, v)))
    if n > 0:
        return v / n
    return v.copy()


def norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


def orthonormalize(unit, p):
    """kt84::eigen_util::orthonormalize: returns p - (unit.p) unit, normalized."""
    p = np.asarray(p, dtype=float) - float(np.dot(unit, p)) * np.asarray(unit, dtype=float)
    return normalized(p)


def rotate90(xy):
    return np.array((-xy[1], xy[0]))


def angle2d(d0, d1):
    return math.atan2(float(np.dot(rotate90(d0), d1)), float(np.dot(d0, d1)))


def acos_clamped(x):
    return math.acos(max(-1.0, min(1.0, x)))


def rbf_wendland(r):
    """kt84::RBFKernel_Wendland with support 1."""
    if r > 1:
        return 0.0
    t1 = 1 - r
    return t1 * t1 * t1 * t1 * (4 * r + 1)


def project_to_line(v0, v1, p):
    d = v1 - v0
    r = float(np.dot(d, d))
    if r == 0:
        return None
    t1 = float(np.dot(p - v0, d)) / r
    return (1 - t1, t1)


def distance_to_line(v0, v1, p, do_clamp=False):
    t = project_to_line(v0, v1, p)
    if t is None:
        return None
    t0, t1 = t
    if do_clamp:
        t0 = min(max(t0, 0.0), 1.0)
        t1 = 1 - t0
    return norm3(t0 * v0 + t1 * v1 - p)


def project_to_triangle(x0, x1, x2, p):
    d01 = x1 - x0
    d02 = x2 - x0
    d0p = p - x0
    a = float(np.dot(d01, d01))
    b = float(np.dot(d01, d02))
    c = float(np.dot(d02, d02))
    det = a * c - b * b
    if det == 0:
        return None
    b0 = float(np.dot(d01, d0p))
    b1 = float(np.dot(d02, d0p))
    u = (c * b0 - b * b1) / det
    v = (-b * b0 + a * b1) / det
    return (1 - u - v, u, v)


def distance_to_triangle(x0, x1, x2, p, do_clamp=False):
    t = project_to_triangle(x0, x1, x2, p)
    if t is None:
        return None
    # NOTE: operator precedence of the original is kept: (do_clamp && t0 < 0) || t1 < 0 || t2 < 0
    if (do_clamp and t[0] < 0) or t[1] < 0 or t[2] < 0:
        d0 = distance_to_line(x0, x1, p, True)
        d1 = distance_to_line(x1, x2, p, True)
        d2 = distance_to_line(x2, x0, p, True)
        return min(d for d in (d0, d1, d2) if d is not None) if any(d is not None for d in (d0, d1, d2)) else None
    return norm3(t[0] * x0 + t[1] * x1 + t[2] * x2 - p)


class Polyline:
    """kt84::Polyline_PointNormal"""

    __slots__ = ('points', 'is_loop')

    def __init__(self, points=None, is_loop=False):
        self.points = [np.array(p, dtype=float) for p in points] if points is not None else []
        self.is_loop = is_loop

    def copy(self):
        return Polyline([p.copy() for p in self.points], self.is_loop)

    def __len__(self):
        return len(self.points)

    def size(self):
        return len(self.points)

    def empty(self):
        return not self.points

    def __getitem__(self, i):
        return self.points[i]

    def __setitem__(self, i, v):
        self.points[i] = v

    def __iter__(self):
        return iter(self.points)

    def clear(self):
        # NOTE: like the original, clear() does not reset is_loop
        self.points = []

    def push_back(self, pn):
        self.points.append(np.array(pn, dtype=float))

    def pop_back(self):
        self.points.pop()

    def front(self):
        return self.points[0]

    def back(self):
        return self.points[-1]

    def reverse(self):
        # NOTE: same (odd) behavior as the original Polyline_PointNormal::reverse()
        if self.is_loop:
            self.points.reverse()
        else:
            self.points.append(self.points[0])
            self.points.reverse()
            self.points.pop()

    def offset_front(self, target):
        offset = target - self.points[0]
        self.points[0] = np.array(target, dtype=float)
        n = len(self.points)
        for i in range(1, n - 1):
            self.points[i] = self.points[i] + offset * (n - 1 - i) / (n - 1.0)
            pn_normalize(self.points[i])

    def offset_back(self, target):
        offset = target - self.points[-1]
        self.points[-1] = np.array(target, dtype=float)
        n = len(self.points)
        for i in range(1, n - 1):
            self.points[i] = self.points[i] + offset * i / (n - 1.0)
            pn_normalize(self.points[i])

    def _segment_lengths(self):
        n = len(self.points)
        if n < 2:
            return np.zeros(0)
        arr = np.asarray(self.points)[:, :3]
        if self.is_loop:
            d = np.roll(arr, -1, axis=0) - arr
        else:
            d = arr[1:] - arr[:-1]
        return np.sqrt((d * d).sum(axis=1))

    def length(self):
        if len(self.points) < 2:
            return 0.0
        return float(self._segment_lengths().sum())

    def resample(self, target_num_points=None):
        if target_num_points is None:
            target_num_points = len(self.points)
        n = len(self.points)
        if n < 2 or target_num_points < 2:
            return
        if target_num_points == 2:
            self.points[1] = self.points[-1]
            del self.points[2:]
            return
        pts = list(self.points)
        if self.is_loop:
            pts.append(pts[0])
            n += 1
        arr = np.asarray(pts)
        seg = np.sqrt(((arr[1:, :3] - arr[:-1, :3]) ** 2).sum(axis=1))
        src_len_acc = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(src_len_acc[-1])
        tgt_len_segment = total / (target_num_points - 1.0)
        result = [None] * target_num_points
        result[0] = pts[0].copy()
        if tgt_len_segment <= 0:
            for i in range(1, target_num_points):
                result[i] = pts[0].copy()
        else:
            # same arc-length sampling as the original, written without its unbounded index walk
            for k in range(1, target_num_points - 1):
                tgt = k * tgt_len_segment
                src_i = int(np.searchsorted(src_len_acc, tgt, side='left'))
                src_i = min(max(src_i, 1), n - 1)
                l0 = src_len_acc[src_i - 1]
                l1 = src_len_acc[src_i]
                w1 = (tgt - l0) / (l1 - l0) if l1 > l0 else 0.0
                pn = pts[src_i - 1] * (1 - w1) + pts[src_i] * w1
                pn_normalize(pn)
                result[k] = pn
        result[-1] = pts[-1].copy()
        if self.is_loop:
            result.pop()
        self.points = result

    def resample_by_length(self, segment_length):
        if segment_length <= 0:
            return
        self.resample(max(int(self.length() / segment_length), 3))

    def insert_points_per_segment(self, num_inserted=3):
        n = len(self.points)
        if n < 2:
            return
        result = []
        pairs = n if self.is_loop else n - 1
        for i in range(n):
            result.append(self.points[i])
            if i < pairs:
                a = self.points[i]
                b = self.points[(i + 1) % n]
                d = (b - a) / (num_inserted + 1.0)
                p = a + d
                for _ in range(num_inserted):
                    result.append(pn_normalized(p))
                    p = p + d
        self.points = result

    def smooth(self, num_iter=1, weight_first_order=1.0, weight_second_order=0.0, damping=0.5):
        n = len(self.points)
        if n == 0:
            return
        P = np.asarray(self.points, dtype=float)
        idx = np.arange(n)
        # NOTE: operator precedence of the original is kept: (!is_loop && i == 0) || i == n - 1 are fixed,
        # and second order differences fall back to first order at (!is_loop && i == 1) || i == n - 2
        movable = ~(((not self.is_loop) & (idx == 0)) | (idx == n - 1))
        first_only = ((not self.is_loop) & (idx == 1)) | (idx == n - 2)
        wsum = weight_first_order + weight_second_order
        for _ in range(num_iter):
            p_prev = np.roll(P, 1, axis=0)
            p_next = np.roll(P, -1, axis=0)
            p_prev2 = np.roll(P, 2, axis=0)
            p_next2 = np.roll(P, -2, axis=0)
            p_first = (p_prev + p_next) / 2.0
            p_second = (-p_prev2 + 4.0 * p_prev + 4.0 * p_next - p_next2) / 6.0
            p_second[first_only] = p_first[first_only]
            target = (weight_first_order * p_first + weight_second_order * p_second) / wsum
            new = damping * P + (1 - damping) * target
            ln = np.linalg.norm(new[:, 3:], axis=1)
            ok = ln > 0
            new[ok, 3:] /= ln[ok, None]
            P = np.where(movable[:, None], new, P)
        self.points = [row.copy() for row in P]

    def point_at(self, t):
        if t < 0 or 1 < t:
            return pn_zero()
        if t == 0:
            return self.points[0].copy()
        if t == 1:
            return (self.points[0] if self.is_loop else self.points[-1]).copy()
        n = len(self.points)
        seg = self._segment_lengths()
        target = float(seg.sum()) * t
        acc = 0.0
        for i in range(len(seg)):
            s = float(seg[i])
            if acc <= target < acc + s:
                u = (target - acc) / s
                pn = (1 - u) * self.points[i] + u * self.points[(i + 1) % n]
                return pn_normalize(pn)
            acc += s
        return pn_zero()


class Polyline2d:
    """kt84::Polyline2d, used for the screen-space hiding stroke."""

    def __init__(self):
        self.points = []
        self.is_loop = False

    def clear(self):
        self.points = []

    def empty(self):
        return not self.points

    def push_back(self, p):
        self.points.append(np.array(p, dtype=float))

    def size(self):
        return len(self.points)

    def length(self):
        if len(self.points) < 2:
            return 0.0
        a = np.asarray(self.points)
        return float(np.sqrt(((a[1:] - a[:-1]) ** 2).sum(axis=1)).sum())

    def resample_by_length(self, segment_length):
        target = max(int(self.length() / segment_length), 3)
        n = len(self.points)
        if n < 2:
            return
        arr = np.asarray(self.points)
        seg = np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1))
        acc = np.concatenate(([0.0], np.cumsum(seg)))
        total = acc[-1]
        if total <= 0:
            return
        ts = np.linspace(0, total, target)
        xs = np.interp(ts, acc, arr[:, 0])
        ys = np.interp(ts, acc, arr[:, 1])
        self.points = [np.array((x, y)) for x, y in zip(xs, ys)]
