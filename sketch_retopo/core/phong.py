# Vectorized version of kt84/graphics/phong_tessellation.hh (Boubekeur & Alexa's Phong
# tessellation), used to render patches as smooth surfaces.

import numpy as np

_cache = {}


def _normalize_normals(X):
    n = np.linalg.norm(X[..., 3:], axis=-1, keepdims=True)
    n[n == 0] = 1
    X[..., 3:] /= n
    return X


def _project(p, q):
    """project(p, q): q moved onto the tangent plane of p (normal of p kept)."""
    d = ((q[..., :3] - p[..., :3]) * p[..., 3:]).sum(-1, keepdims=True)
    out = np.empty(np.broadcast(p, q).shape)
    out[..., :3] = q[..., :3] - d * p[..., 3:]
    out[..., 3:] = p[..., 3:]
    return out


def _grid(subdiv):
    if subdiv in _cache:
        return _cache[subdiv]
    s = subdiv
    pts = []
    index = {}
    for i in range(s + 1):
        for j in range(s + 1 - i):
            index[(i, j)] = len(pts)
            pts.append((1 - (i + j) / s, i / s, j / s))
    tris = []
    for i in range(s):
        for j in range(s - i):
            tris.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if i + j < s - 1:
                tris.append((index[(i, j + 1)], index[(i + 1, j)], index[(i + 1, j + 1)]))
    res = (np.array(pts), np.array(tris))
    _cache[subdiv] = res
    return res


def triangles(T, enabled=True, subdiv=3, weight=0.7):
    """T: (F, 3, 6) triangles of PointNormals.  Returns (positions (M, 3, 3), normals (M, 3, 3))."""
    T = np.asarray(T, dtype=float).reshape(-1, 3, 6)
    if len(T) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3, 3))
    if not enabled or subdiv <= 1:
        return T[:, :, :3].copy(), T[:, :, 3:].copy()
    bary, tris = _grid(subdiv)
    # p: barycentric interpolation (F, S, 6)
    p = np.einsum('sk,fkd->fsd', bary, T)
    _normalize_normals(p)
    q0 = _project(T[:, None, 0, :], p)
    q1 = _project(T[:, None, 1, :], p)
    q2 = _project(T[:, None, 2, :], p)
    q = bary[None, :, 0, None] * q0 + bary[None, :, 1, None] * q1 + bary[None, :, 2, None] * q2
    _normalize_normals(q)
    r = (1 - weight) * p + weight * q
    _normalize_normals(r)
    out = r[:, tris]               # (F, nt, 3, 6)
    out = out.reshape(-1, 3, 6)
    return out[:, :, :3], out[:, :, 3:]


def lines(L, enabled=True, subdiv=3, weight=0.7):
    """L: (E, 2, 6) segments.  Returns positions (M, 2, 3)."""
    L = np.asarray(L, dtype=float).reshape(-1, 2, 6)
    if len(L) == 0:
        return np.zeros((0, 2, 3))
    if not enabled or subdiv <= 1:
        return L[:, :, :3].copy()
    u = np.linspace(0, 1, subdiv + 1)
    p = (1 - u)[None, :, None] * L[:, None, 0] + u[None, :, None] * L[:, None, 1]
    _normalize_normals(p)
    q0 = _project(L[:, None, 0, :], p)
    q1 = _project(L[:, None, 1, :], p)
    q = (1 - u)[None, :, None] * q0 + u[None, :, None] * q1
    _normalize_normals(q)
    r = (1 - weight) * p + weight * q
    seg = np.stack([r[:, :-1, :3], r[:, 1:, :3]], axis=2)   # (E, s, 2, 3)
    return seg.reshape(-1, 2, 3)


def line_strip(P, enabled=True, subdiv=3, weight=0.7):
    P = np.asarray(P, dtype=float).reshape(-1, 6)
    if len(P) < 2:
        return np.zeros((0, 2, 3))
    return lines(np.stack([P[:-1], P[1:]], axis=1), enabled, subdiv, weight)


def fan(P, enabled=True, subdiv=5, weight=0.7):
    """Triangle fan of PointNormals (used for the auto-completion preview)."""
    P = np.asarray(P, dtype=float).reshape(-1, 6)
    if len(P) < 3:
        return np.zeros((0, 3, 3)), np.zeros((0, 3, 3))
    T = np.stack([np.repeat(P[:1], len(P) - 2, axis=0), P[1:-1], P[2:]], axis=1)
    return triangles(T, enabled, subdiv, weight)
