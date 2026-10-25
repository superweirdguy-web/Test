# Port of patchgen (Takayama et al., "Pattern-Based Quadrangulation for N-Sided
# Patches", SGP 2014) as bundled with SketchRetopo: the 20 topological patterns
# for 2- to 6-sided patches, default parameter selection, parameter adjustment,
# pattern switching and topology generation.

import math

import numpy as np

from .ilp import ILP, EQ, LE, GE
from .polymesh import insert_edgeloop, append_quad_strip, flip_faces

# ---- PatchVertexTag -------------------------------------------------------------------------
TAG_NONE = -1
C0, C1, C2, C3, C4, C5 = range(6)
V0, V1, V2, V3, V4, V5, V6, V7, V8 = range(6, 15)


def add_tagged_vertex(patch, index, is_corner):
    v = patch.add_vertex()
    patch.v_corner[v] = index if is_corner else -1
    patch.v_tag[v] = (C0 if is_corner else V0) + index
    return v


def find_tagged_vertex(patch, tag):
    for v in patch.vertices():
        if patch.v_tag[v] == tag:
            return v
    return -1


# ---- Permutation / PatchParam ----------------------------------------------------------------
class Permutation:
    __slots__ = ('num_sides', 'id')

    def __init__(self, num_sides=0, id_=0):
        self.num_sides = num_sides
        self.id = id_

    def init(self, num_sides, id_=0):
        self.num_sides = num_sides
        self.id = id_

    def copy(self):
        return Permutation(self.num_sides, self.id)

    def is_flipped(self):
        return self.id >= self.num_sides

    def __eq__(self, other):
        return self.num_sides == other.num_sides and self.id == other.id

    def next(self, is_forward=True):
        self.id += 1 if is_forward else -1

    def is_valid(self):
        return 0 <= self.id < 2 * self.num_sides

    def __getitem__(self, index):
        n = self.num_sides
        if self.id < n:
            return (index + self.id) % n
        return n - 1 - (index + self.id - n) % n

    def __call__(self, l):
        return [l[self[i]] for i in range(self.num_sides)]


class PatchParam:
    __slots__ = ('pattern_id', 'l', 'permutation', 'p', 'q', 'x', 'y', 'z', 'w')

    def __init__(self):
        self.pattern_id = -1
        self.l = []
        self.permutation = Permutation()
        self.p = []
        self.q = []
        self.x = -1
        self.y = -1
        self.z = -1
        self.w = -1

    def copy(self):
        r = PatchParam()
        r.pattern_id = self.pattern_id
        r.l = list(self.l)
        r.permutation = self.permutation.copy()
        r.p = list(self.p)
        r.q = list(self.q)
        r.x, r.y, r.z, r.w = self.x, self.y, self.z, self.w
        return r

    def get_num_sides(self):
        return len(self.l)

    def get_l_permuted(self):
        return self.permutation(self.l)


# ---- variable access -------------------------------------------------------------------------
def _var_get(param, spec):
    if isinstance(spec, tuple):
        return getattr(param, spec[0])[spec[1]]
    return getattr(param, spec)


def _var_set(param, spec, value):
    if isinstance(spec, tuple):
        getattr(param, spec[0])[spec[1]] = value
    else:
        setattr(param, spec, value)


def _unit(n, i):
    e = np.zeros(n)
    e[i] = 1
    return e


class Pattern:
    def __init__(self, num_sides, pattern_id, matrix, rhs_offset, variables, default, subtopology, indicators, param_names):
        self.num_sides = num_sides
        self.pattern_id = pattern_id
        self.matrix = np.array(matrix, dtype=float)
        self.rhs_offset = rhs_offset
        self.variables = variables
        self.default = default
        self.subtopology = subtopology
        self.indicators = indicators
        self.param_names = param_names

    def num_variables(self):
        return self.matrix.shape[1]

    def get_constraint_rhs(self, l):
        return np.array([l[i] - self.rhs_offset[i] for i in range(self.num_sides)], dtype=float)

    def var_spec(self, index):
        if index < len(self.variables):
            return self.variables[index]
        return self.variables[-1]

    def get_variable(self, param, index):
        return _var_get(param, self.var_spec(index))

    def set_variable(self, param, index, value):
        _var_set(param, self.var_spec(index), value)

    def get_default_parameter(self, l, param):
        A = self.matrix
        nv = A.shape[1]
        ilp = ILP(nv)
        ilp.add_constraints(A, EQ, self.get_constraint_rhs(l))
        kind = self.default[0]
        if kind == 'xmid':
            _, xi, extra, obj = self.default
            ilp.set_objective(_unit(nv, xi), False)
            if not ilp.solve():
                return False
            xmin = ilp.get_variables()[xi]
            ilp.refresh()
            ilp.set_objective(_unit(nv, xi), True)
            if not ilp.solve():
                return False
            xmax = ilp.get_variables()[xi]
            ilp.refresh()
            xmid = (xmin + xmax) // 2
            ilp.add_constraint(_unit(nv, xi), LE, xmid + 1)
            ilp.add_constraint(_unit(nv, xi), GE, xmid - 1)
            for row in extra:
                ilp.add_constraint(row, LE, 0)
            ilp.set_objective(obj, True)
            if not ilp.solve():
                return False
        elif kind == 'plain':
            _, extra, obj = self.default
            for row in extra:
                ilp.add_constraint(row, LE, 0)
            ilp.set_objective(obj, True)
            if not ilp.solve():
                return False
        else:
            if not ilp.solve():
                return False
        variables = ilp.get_variables()
        for i in range(nv):
            self.set_variable(param, i, variables[i])
        param.pattern_id = self.pattern_id
        return True

    def get_param_str(self, param):
        return '_'.join('%s=%d' % (name, _var_get(param, spec)) for name, spec in self.param_names)


# ---- subtopologies (patch.clear() is done by the caller) --------------------------------------
def _setup(patch, nc, nv):
    patch.clear()
    C = [add_tagged_vertex(patch, i, True) for i in range(nc)]
    V = [add_tagged_vertex(patch, i, False) for i in range(nv)]
    return C, V


def _insert(patch, h, count):
    for _ in range(count):
        insert_edgeloop(patch, h)


def _topo_2_0(param, patch):
    C, V = _setup(patch, 2, 2)
    patch.add_face([C[0], V[0], V[1], C[1]])
    _insert(patch, patch.halfedge_handle(V[1]), param.y)


def _topo_2_1(param, patch):
    C, V = _setup(patch, 2, 2)
    patch.add_face([C[0], V[0], C[1], V[1]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(C[1])
    _insert(patch, hy, param.y)


def _topo_3_0(param, patch):
    C, V = _setup(patch, 3, 1)
    patch.add_face([C[0], V[0], C[1], C[2]])


def _topo_3_1(param, patch):
    C, V = _setup(patch, 3, 4)
    patch.add_face([C[0], V[0], V[3], C[2]])
    patch.add_face([V[0], V[1], V[2], V[3]])
    patch.add_face([C[2], V[3], V[2], C[1]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq1 = patch.halfedge_handle(V[2])
    _insert(patch, hq1, param.q[1])
    hq2 = patch.halfedge_handle(V[1])
    _insert(patch, hq2, param.q[2])


def _topo_3_2(param, patch):
    C, V = _setup(patch, 3, 5)
    patch.add_face([C[0], V[0], V[3], C[2]])
    patch.add_face([V[0], V[1], V[4], V[3]])
    patch.add_face([C[1], V[2], V[4], V[1]])
    patch.add_face([C[2], V[3], V[4], V[2]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq2 = patch.halfedge_handle(V[1])
    _insert(patch, hq2, param.q[2])


def _topo_3_3(param, patch):
    C, V = _setup(patch, 3, 7)
    patch.add_face([C[0], V[0], V[1], V[5]])
    patch.add_face([V[1], V[2], C[1], V[5]])
    patch.add_face([C[1], V[3], V[6], V[5]])
    patch.add_face([V[3], C[2], V[4], V[6]])
    patch.add_face([V[4], C[0], V[5], V[6]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq2 = patch.halfedge_handle(V[1])
    _insert(patch, hq2, param.q[2])
    hq1 = patch.halfedge_handle(V[2])
    _insert(patch, hq1, param.q[1])


def _topo_4_0(param, patch):
    C, V = _setup(patch, 4, 0)
    patch.add_face([C[0], C[1], C[2], C[3]])


def _topo_4_1(param, patch):
    C, V = _setup(patch, 4, 3)
    patch.add_face([C[0], V[0], V[2], C[3]])
    patch.add_face([V[0], C[1], V[1], V[2]])
    patch.add_face([V[1], C[2], C[3], V[2]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)


def _topo_4_2(param, patch):
    C, V = _setup(patch, 4, 2)
    patch.add_face([C[0], V[0], C[2], C[3]])
    patch.add_face([V[0], V[1], C[1], C[2]])
    hx = patch.halfedge_handle(V[1])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(C[1])
    _insert(patch, hy, param.y)


def _topo_4_3(param, patch):
    C, V = _setup(patch, 4, 4)
    patch.add_face([C[0], V[0], V[3], C[3]])
    patch.add_face([V[0], V[1], V[2], V[3]])
    patch.add_face([V[1], C[1], C[2], V[2]])
    patch.add_face([C[2], C[3], V[3], V[2]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq1 = patch.halfedge_handle(V[1])
    _insert(patch, hq1, param.q[1])


def _topo_4_4(param, patch):
    C, V = _setup(patch, 4, 7)
    patch.add_face([C[0], V[0], V[6], C[3]])
    patch.add_face([V[0], V[1], V[5], V[6]])
    patch.add_face([V[1], V[2], V[4], V[5]])
    patch.add_face([V[2], C[1], V[3], V[4]])
    patch.add_face([V[3], C[2], V[5], V[4]])
    patch.add_face([C[2], C[3], V[6], V[5]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(V[2])
    _insert(patch, hy, param.y)
    hq1 = patch.halfedge_handle(V[1])
    _insert(patch, hq1, param.q[1])


def _topo_5_0(param, patch):
    C, V = _setup(patch, 5, 1)
    patch.add_face([V[0], C[3], C[4], C[0]])
    patch.add_face([V[0], C[1], C[2], C[3]])


def _topo_5_1(param, patch):
    C, V = _setup(patch, 5, 1)
    patch.add_face([C[0], V[0], C[1], C[2]])
    patch.add_face([C[0], C[2], C[3], C[4]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq4 = patch.halfedge_handle(C[1])
    _insert(patch, hq4, param.q[4])


def _topo_5_2(param, patch):
    C, V = _setup(patch, 5, 5)
    patch.add_face([C[0], V[0], V[3], V[4]])
    patch.add_face([C[1], V[4], V[3], V[2]])
    patch.add_face([V[0], V[1], V[2], V[3]])
    patch.add_face([V[4], C[3], C[4], C[0]])
    patch.add_face([V[4], C[1], C[2], C[3]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hq0 = patch.halfedge_handle(C[0])
    _insert(patch, hq0, param.q[0])
    hq1 = patch.halfedge_handle(V[1])
    _insert(patch, hq1, param.q[1])
    hq4 = patch.halfedge_handle(V[2])
    _insert(patch, hq4, param.q[4])


def _topo_5_3(param, patch):
    C, V = _setup(patch, 5, 9)
    patch.add_face([C[0], V[0], V[5], C[4]])
    patch.add_face([V[0], V[1], V[6], V[5]])
    patch.add_face([V[1], V[2], V[7], V[6]])
    patch.add_face([V[2], V[3], V[8], V[7]])
    patch.add_face([V[3], C[1], V[4], V[8]])
    patch.add_face([V[4], C[2], V[7], V[8]])
    patch.add_face([C[2], C[3], V[6], V[7]])
    patch.add_face([C[3], C[4], V[5], V[6]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(V[3])
    _insert(patch, hy, param.y)
    hq1 = patch.halfedge_handle(V[2])
    _insert(patch, hq1, param.q[1])
    hq4 = patch.halfedge_handle(V[1])
    _insert(patch, hq4, param.q[4])


def _topo_5_4(param, patch):
    C, V = _setup(patch, 5, 8)
    patch.add_face([C[0], V[0], V[7], V[6]])
    patch.add_face([V[0], V[1], C[1], V[2]])
    patch.add_face([V[2], V[3], V[7], V[0]])
    patch.add_face([V[3], C[2], V[4], V[7]])
    patch.add_face([V[4], V[5], V[6], V[7]])
    patch.add_face([V[5], C[3], C[4], V[6]])
    hx = patch.halfedge_handle(V[1])
    _insert(patch, hx, param.x)
    hq0 = patch.halfedge_handle(V[3])
    _insert(patch, hq0, param.q[0])
    hq1 = patch.halfedge_handle(V[0])
    _insert(patch, hq1, param.q[1])


def _topo_6_0(param, patch):
    C, V = _setup(patch, 6, 0)
    patch.add_face([C[0], C[1], C[2], C[5]])
    patch.add_face([C[2], C[3], C[4], C[5]])
    hx = patch.halfedge_handle(C[1])
    _insert(patch, hx, param.x)


def _topo_6_1(param, patch):
    C, V = _setup(patch, 6, 4)
    patch.add_face([C[0], V[0], V[2], V[3]])
    patch.add_face([C[1], V[1], V[2], V[0]])
    patch.add_face([V[1], C[2], V[3], V[2]])
    patch.add_face([C[2], C[3], C[4], V[3]])
    patch.add_face([C[4], C[5], C[0], V[3]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(C[1])
    _insert(patch, hy, param.y)
    hz = patch.halfedge_handle(V[1])
    _insert(patch, hz, param.z)
    hw = patch.halfedge_handle(C[3])
    _insert(patch, hw, param.w)


def _topo_6_2(param, patch):
    C, V = _setup(patch, 6, 6)
    patch.add_face([C[0], V[0], V[2], V[4]])
    patch.add_face([V[0], V[1], V[3], V[2]])
    patch.add_face([V[1], C[1], V[5], V[3]])
    patch.add_face([C[1], C[2], C[3], V[5]])
    patch.add_face([C[3], C[4], V[4], V[5]])
    patch.add_face([C[4], C[5], C[0], V[4]])
    patch.add_face([V[2], V[3], V[5], V[4]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(V[1])
    _insert(patch, hy, param.y)
    hq0 = patch.halfedge_handle(C[0])
    _insert(patch, hq0, param.q[0])
    hq3 = patch.halfedge_handle(C[3])
    _insert(patch, hq3, param.q[3])


def _topo_6_3(param, patch):
    C, V = _setup(patch, 6, 9)
    patch.add_face([C[0], V[0], V[6], C[5]])
    patch.add_face([V[0], V[1], V[5], V[6]])
    patch.add_face([V[1], V[2], V[4], V[5]])
    patch.add_face([V[2], C[1], V[3], V[4]])
    patch.add_face([V[3], C[2], V[5], V[4]])
    patch.add_face([C[2], C[3], V[8], V[5]])
    patch.add_face([C[3], C[4], V[7], V[8]])
    patch.add_face([C[4], C[5], V[6], V[7]])
    patch.add_face([V[5], V[8], V[7], V[6]])
    hx = patch.halfedge_handle(V[0])
    _insert(patch, hx, param.x)
    hy = patch.halfedge_handle(V[2])
    _insert(patch, hy, param.y)
    hz = patch.halfedge_handle(V[1])
    _insert(patch, hz, param.z)
    hq3 = patch.halfedge_handle(C[3])
    _insert(patch, hq3, param.q[3])


def _p(n):
    return [('p', i) for i in range(n)]


def _row(*values):
    return np.array(values, dtype=float)


PATTERNS = {
    (2, 0): Pattern(
        2, 0,
        [[0, 2, 1],
         [2, 0, 1]],
        [3, 1],
        _p(2) + ['y'],
        ('xmid', 2, [], _row(1, 1, 0)),
        _topo_2_0,
        [[(V0, V1)]],
        [('y', 'y')]),
    (2, 1): Pattern(
        2, 1,
        [[0, 2, 1, 1],
         [2, 0, 1, 1]],
        [2, 2],
        _p(2) + ['x', 'y'],
        ('xmid', 2, [], _row(1, 1, 0, 0)),
        _topo_2_1,
        [[(C0, V0), (C1, V1)],
         [(C0, V1), (C1, V0)]],
        [('x', 'x'), ('y', 'y')]),
    (3, 0): Pattern(
        3, 0,
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]],
        [2, 1, 1],
        _p(3),
        ('simple',),
        _topo_3_0,
        [],
        []),
    (3, 1): Pattern(
        3, 1,
        [[0, 1, 1, 1, 1, 2],
         [1, 0, 1, 0, 1, 0],
         [1, 1, 0, 1, 0, 0]],
        [4, 1, 1],
        _p(3) + [('q', 1), ('q', 2), 'x'],
        ('xmid', 5, [_row(0, 1, 0, -1, 0, 0), _row(0, 0, 1, 0, -1, 0)], _row(1, 1, 1, 0, 0, 0)),
        _topo_3_1,
        [[(C0, C2), (V0, V3), (V1, V2)],
         [(C1, C2), (V2, V3), (V0, V1)],
         [(C0, V0), (C1, V2), (C2, V3)]],
        [('q1', ('q', 1)), ('q2', ('q', 2)), ('x', 'x')]),
    (3, 2): Pattern(
        3, 2,
        [[0, 1, 1, 1, 2],
         [1, 0, 1, 1, 0],
         [1, 1, 0, 0, 0]],
        [3, 2, 1],
        _p(3) + [('q', 2), 'x'],
        ('xmid', 4, [_row(0, 0, 1, -1, 0)], _row(1, 1, 1, 0, 0)),
        _topo_3_2,
        [[(V0, V1), (V3, V4), (C2, V2)],
         [(C0, V0), (C2, V3), (V2, V4), (C1, V1)]],
        [('q2', ('q', 2)), ('x', 'x')]),
    (3, 3): Pattern(
        3, 3,
        [[0, 1, 1, 1, 1, 2],
         [1, 0, 1, 0, 1, 0],
         [1, 1, 0, 1, 0, 0]],
        [4, 2, 2],
        _p(3) + [('q', 1), ('q', 2), 'x'],
        ('xmid', 5, [_row(0, 1, 0, -1, 0, 0), _row(0, 0, 1, 0, -1, 0)], _row(1, 1, 1, 0, 0, 0)),
        _topo_3_3,
        [[(V1, V2), (V5, C1), (V6, V3), (V4, C2)],
         [(V0, V1), (C0, V5), (V4, V6), (C2, V3)],
         [(C0, V0), (V5, V1), (C1, V2)]],
        [('q1', ('q', 1)), ('q2', ('q', 2)), ('x', 'x')]),
    (4, 0): Pattern(
        4, 0,
        [[0, 1],
         [1, 0],
         [0, 1],
         [1, 0]],
        [1, 1, 1, 1],
        _p(2),
        ('simple',),
        _topo_4_0,
        [],
        []),
    (4, 1): Pattern(
        4, 1,
        [[0, 1, 0, 1, 1],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 0]],
        [2, 2, 1, 1],
        _p(4) + ['x'],
        ('plain', [_row(1, 0, -1, 0, 0), _row(0, 1, 0, -1, 0)], _row(1, 1, 0, 0, 0)),
        _topo_4_1,
        [[]],
        [('p0', ('p', 0)), ('p1', ('p', 1))]),
    (4, 2): Pattern(
        4, 2,
        [[0, 1, 0, 1, 1, 1],
         [1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 0],
         [1, 0, 1, 0, 0, 1]],
        [3, 1, 1, 1],
        _p(4) + ['x', 'y'],
        ('plain', [_row(1, 0, -1, 0, 0, 0), _row(0, 1, 0, -1, 0, 0)], _row(1, 1, 0, 0, 0, 0)),
        _topo_4_2,
        [[], []],
        [('p0', ('p', 0)), ('p1', ('p', 1))]),
    (4, 3): Pattern(
        4, 3,
        [[0, 1, 0, 1, 1, 2],
         [1, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 1, 0],
         [1, 0, 1, 0, 0, 0]],
        [3, 1, 1, 1],
        _p(4) + [('q', 1), 'x'],
        ('plain', [_row(1, 0, -1, 0, 0, 0), _row(0, 1, 0, -1, 0, 0), _row(0, 0, 0, 1, -1, 0)], _row(1, 1, 0, 0, 0, 0)),
        _topo_4_3,
        [[(V0, V1), (V2, V3), (C2, C3)],
         []],
        [('p0', ('p', 0)), ('p1', ('p', 1)), ('q1', ('q', 1))]),
    (4, 4): Pattern(
        4, 4,
        [[0, 1, 0, 1, 1, 2, 1],
         [1, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 1, 1, 0, 0],
         [1, 0, 1, 0, 0, 0, 0]],
        [4, 2, 1, 1],
        _p(4) + [('q', 1), 'x', 'y'],
        ('plain', [_row(1, 0, -1, 0, 0, 0, 0), _row(0, 1, 0, -1, 0, 0, 0), _row(0, 0, 0, 1, -1, 0, 0)], _row(1, 1, 0, 0, 0, 0, 0)),
        _topo_4_4,
        [[(V0, V1), (V5, V6), (C2, C3)],
         [],
         []],
        [('p0', ('p', 0)), ('p1', ('p', 1)), ('q1', ('q', 1))]),
    (5, 0): Pattern(
        5, 0,
        [[0, 1, 0, 0, 1],
         [1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 1],
         [1, 0, 0, 1, 0]],
        [2, 1, 1, 1, 1],
        _p(5),
        ('simple',),
        _topo_5_0,
        [],
        []),
    (5, 1): Pattern(
        5, 1,
        [[0, 1, 0, 0, 1, 1, 1],
         [1, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 1, 0],
         [1, 0, 0, 1, 0, 0, 0]],
        [2, 1, 1, 1, 1],
        _p(5) + [('q', 4), 'x'],
        ('xmid', 6, [_row(0, 0, 0, 0, 1, -1, 0)], _row(1, 1, 1, 1, 1, 0, 0)),
        _topo_5_1,
        [[(C1, V0), (C0, C2), (C3, C4)],
         [(C0, V0), (C1, C2)]],
        [('q4', ('q', 4)), ('x', 'x')]),
    (5, 2): Pattern(
        5, 2,
        [[0, 1, 0, 0, 1, 0, 1, 1, 2],
         [1, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0]],
        [4, 1, 1, 1, 1],
        _p(5) + [('q', 0), ('q', 1), ('q', 4), 'x'],
        ('xmid', 8, [_row(1, 0, 0, 0, 0, -1, 0, 0, 0), _row(0, 1, 0, 0, 0, 0, -1, 0, 0), _row(0, 0, 0, 0, 1, 0, 0, -1, 0)],
         _row(1, 1, 1, 1, 1, 0, 0, 0, 0)),
        _topo_5_2,
        [[(C0, C4), (C1, C2), (C3, V4)],
         [(V0, V1), (V2, V3), (C1, V4), (C2, C3)],
         [(V2, V1), (V0, V3), (C0, V4), (C4, C3)],
         [(C0, V0), (C1, V2), (V3, V4)]],
        [('q0', ('q', 0)), ('q1', ('q', 1)), ('q4', ('q', 4)), ('x', 'x')]),
    (5, 3): Pattern(
        5, 3,
        [[0, 1, 0, 0, 1, 1, 1, 2, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 1, 0, 0],
         [1, 0, 0, 1, 0, 0, 0, 0, 0]],
        [5, 2, 1, 1, 1],
        _p(5) + [('q', 1), ('q', 4), 'x', 'y'],
        ('xmid', 7, [_row(0, 1, 0, 0, 0, -1, 0, 0, 0), _row(0, 0, 0, 0, 1, 0, -1, 0, 0)], _row(1, 1, 1, 1, 1, 0, 0, 0, 0)),
        _topo_5_3,
        [[(V1, V2), (V6, V7), (C2, C3)],
         [(V0, V1), (V5, V6), (C3, C4)],
         [(C0, V0), (C4, V5), (C3, V6), (C2, V7), (V4, V8), (C1, V3)],
         [(V2, V3), (V7, V8), (C2, V4)]],
        [('q1', ('q', 1)), ('q4', ('q', 4)), ('x', 'x'), ('y', 'y')]),
    (5, 4): Pattern(
        5, 4,
        [[0, 1, 0, 0, 1, 0, 1, 1],
         [1, 0, 1, 0, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0]],
        [3, 3, 3, 1, 2],
        _p(5) + [('q', 0), ('q', 1), 'x', 'y'],
        ('xmid', 7, [_row(0, 1, 0, 0, 0, -1, 0, 0), _row(0, 0, 0, 0, 1, 0, -1, 0)], _row(1, 1, 1, 1, 1, 0, 0, 0)),
        _topo_5_4,
        [[(V2, V3), (V0, V7), (C0, V6)],
         [(C0, V0), (V6, V7), (V5, V4)],
         [(V0, V1), (V2, C1)]],
        [('q0', ('q', 0)), ('q1', ('q', 1)), ('x', 'x')]),
    (6, 0): Pattern(
        6, 0,
        [[0, 1, 0, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1, 0, 0]],
        [1, 1, 1, 1, 1, 1],
        _p(6) + ['x'],
        ('xmid', 6, [], _row(1, 1, 1, 1, 1, 1, 0)),
        _topo_6_0,
        [[(C0, C1), (C5, C2), (C4, C3)]],
        [('x', 'x')]),
    (6, 1): Pattern(
        6, 1,
        [[0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
         [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
        [2, 2, 1, 1, 1, 1],
        _p(6) + ['x', 'y', 'z', 'w'],
        ('xmid', 6, [], _row(1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        _topo_6_1,
        [[(C0, V0), (V3, V2), (C2, V1)],
         [(C1, V0), (V1, V2), (C2, V3), (C3, C4)],
         [(C1, V1), (V0, V2), (C0, V3), (C5, C4)],
         [(C0, C5), (V3, C4), (C2, C3)]],
        [('x', 'x'), ('y', 'y'), ('z', 'z'), ('w', 'w')]),
    (6, 2): Pattern(
        6, 2,
        [[0, 1, 0, 0, 0, 1, 0, 0, 2, 1],
         [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
         [1, 0, 0, 0, 1, 0, 1, 0, 0, 0]],
        [3, 1, 1, 1, 1, 1],
        _p(6) + [('q', 0), ('q', 3), 'x', 'y'],
        ('xmid', 8, [_row(1, 0, 0, 0, 0, 0, -1, 0, 0, 0), _row(0, 0, 0, 1, 0, 0, 0, -1, 0, 0)],
         _row(1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        _topo_6_2,
        [[(C0, C5), (V4, C4), (V5, C3), (C1, C2)],
         [(C4, C5), (V4, C0), (V2, V0), (V3, V1), (V5, C1), (C3, C2)],
         [(C0, V0), (V4, V2), (V5, V3), (C1, V1)],
         [(V0, V1), (V2, V3), (V4, V5), (C4, C3)]],
        [('q0', ('q', 0)), ('q3', ('q', 3)), ('x', 'x'), ('y', 'y')]),
    (6, 3): Pattern(
        6, 3,
        [[0, 1, 0, 0, 0, 1, 0, 2, 1, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        [4, 2, 1, 1, 1, 1],
        _p(6) + [('q', 3), 'x', 'y', 'z'],
        ('xmid', 7, [], _row(1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        _topo_6_3,
        [[(C4, C5), (V7, V6), (V8, V5), (C3, C2)],
         [(C0, V0), (C5, V6), (C4, V7), (C3, V8), (C2, V5), (V3, V4), (C1, V2)],
         [(V1, V2), (V5, V4), (C2, V3)],
         [(V0, V1), (V6, V5), (V7, V8), (C4, C3)]],
        [('q3', ('q', 3)), ('x', 'x'), ('y', 'y'), ('z', 'z')]),
}

NUM_PATTERNS = [0, 0, 2, 4, 5, 5, 4]

# patterns tried (in order) by get_default_parameter; note 3_3 and 5_4 are only reachable via switch_pattern
DEFAULT_PATTERN_ORDER = {
    2: [0, 1],
    3: [0, 1, 2],
    4: [0, 1, 2, 3, 4],
    5: [0, 1, 2, 3],
    6: [0, 1, 2, 3],
}


def get_pattern(num_sides, pattern_id):
    return PATTERNS[(num_sides, pattern_id)]


def get_num_variables(num_sides, pattern_id):
    return PATTERNS[(num_sides, pattern_id)].num_variables()


def get_variable(num_sides, pattern_id, param, index):
    return PATTERNS[(num_sides, pattern_id)].get_variable(param, index)


def set_variable(num_sides, pattern_id, param, index, value):
    PATTERNS[(num_sides, pattern_id)].set_variable(param, index, value)


def get_variable_indicators(num_sides, pattern_id):
    return PATTERNS[(num_sides, pattern_id)].indicators


def get_param_str(num_sides, pattern_id, param):
    return PATTERNS[(num_sides, pattern_id)].get_param_str(param)


def get_constraint_matrix(num_sides, pattern_id):
    return PATTERNS[(num_sides, pattern_id)].matrix


def get_constraint_rhs(num_sides, pattern_id, l):
    return PATTERNS[(num_sides, pattern_id)].get_constraint_rhs(l)


def get_default_parameter(l):
    """Returns a PatchParam, or None when no pattern admits the boundary subdivision l."""
    num_sides = len(l)
    param = PatchParam()
    param.l = list(l)
    param.p = [-1] * num_sides
    param.q = [-1] * num_sides
    for pattern_id in DEFAULT_PATTERN_ORDER.get(num_sides, []):
        pattern = PATTERNS[(num_sides, pattern_id)]
        param.permutation.init(num_sides)
        while param.permutation.is_valid():
            if pattern.get_default_parameter(param.get_l_permuted(), param):
                return param
            param.permutation.next()
    return None


def adjust_parameter(param, variable_index, is_increase):
    num_sides = param.get_num_sides()
    pattern = PATTERNS[(num_sides, param.pattern_id)]
    A = pattern.matrix
    b = pattern.get_constraint_rhs(param.get_l_permuted())
    nv = A.shape[1]
    nv2 = nv * 2
    if variable_index >= nv:
        return False
    A2 = np.zeros((num_sides, nv2))
    A2[:, :nv] = A
    ilp = ILP(nv2)
    ilp.add_constraints(A2, EQ, b)
    ilp.add_constraint(_unit(nv2, variable_index), GE if is_increase else LE,
                       pattern.get_variable(param, variable_index) + (1 if is_increase else -1))
    for i in range(nv):
        row = np.zeros(nv2)
        row[nv + i] = -1
        old_value = pattern.get_variable(param, i)
        row[i] = 1
        ilp.add_constraint(row.copy(), LE, old_value)
        row[i] = -1
        ilp.add_constraint(row.copy(), LE, -old_value)
    obj = np.zeros(nv2)
    obj[nv:] = 1
    ilp.set_objective(obj, False)
    if not ilp.solve():
        return False
    variables = ilp.get_variables()
    for i in range(nv):
        pattern.set_variable(param, i, variables[i])
    return True


def switch_pattern(param, is_forward):
    param_old = param.copy()
    num_sides = param.get_num_sides()
    param.permutation.next(is_forward)
    while True:
        if not param.permutation.is_valid():
            n_pat = NUM_PATTERNS[num_sides]
            param.pattern_id = (param.pattern_id + (1 if is_forward else -1)) % n_pat
            param.permutation.init(num_sides, 0 if is_forward else 2 * num_sides - 1)
        if param.pattern_id == param_old.pattern_id and param.permutation == param_old.permutation:
            _restore_param(param, param_old)
            return False
        pattern = PATTERNS[(num_sides, param.pattern_id)]
        if pattern.get_default_parameter(param.get_l_permuted(), param):
            return True
        param.permutation.next(is_forward)


def _restore_param(param, src):
    param.pattern_id = src.pattern_id
    param.l = list(src.l)
    param.permutation = src.permutation.copy()
    param.p = list(src.p)
    param.q = list(src.q)
    param.x, param.y, param.z, param.w = src.x, src.y, src.z, src.w


# ---- topology generation ---------------------------------------------------------------------
def add_padding(patch, param):
    num_sides = param.get_num_sides()
    # IMPORTANT NOTE: boundary halfedges are oriented CLOCKWISE!
    h_boundary_front = -1
    for v in patch.vertices():
        if patch.v_corner[v] == 0:
            h_boundary_front = patch.halfedge_handle(v)
            break
    for i in range(num_sides - 1, -1, -1):
        h_boundary_back = h_boundary_front
        while patch.v_corner[patch.to_vertex_handle(h_boundary_back)] == -1:
            h_boundary_back = patch.next_halfedge_handle(h_boundary_back)
        for _ in range(param.p[i]):
            patch.v_corner[patch.from_vertex_handle(h_boundary_front)] = -1
            patch.v_corner[patch.to_vertex_handle(h_boundary_back)] = -1
            h_boundary_front, h_boundary_back = append_quad_strip(patch, h_boundary_front, h_boundary_back)
            patch.v_corner[patch.from_vertex_handle(h_boundary_front)] = (i + 1) % num_sides
            patch.v_corner[patch.to_vertex_handle(h_boundary_back)] = i
        h_boundary_front = patch.next_halfedge_handle(h_boundary_back)


def generate_topology(param, patch):
    num_sides = param.get_num_sides()
    PATTERNS[(num_sides, param.pattern_id)].subtopology(param, patch)
    add_padding(patch, param)
    flipped = param.permutation.is_flipped()
    for v in patch.vertices():
        ci = patch.v_corner[v]
        if ci == -1:
            continue
        ci = param.permutation[ci]
        if flipped:
            ci = (ci + 1) % num_sides
        patch.v_corner[v] = ci
    if flipped:
        flip_faces(patch)


def get_boundary_geometry(num_sides, t):
    pi = math.pi
    if num_sides == 1:
        theta = t * 2 * pi
        return np.array((math.cos(theta), math.sin(theta)))
    if num_sides == 2:
        if t < 1:
            theta = (2 * t - 1) * 3.14159 / 3
            return np.array((math.sin(theta), -math.cos(theta) + 0.5))
        t -= 1
        theta = (2 * t - 1) * 3.14159 / 3
        return np.array((-math.sin(theta), math.cos(theta) - 0.5))
    side_angle = 2 * pi / num_sides
    corners = []
    for i in range(num_sides):
        theta = side_angle * i - (0.5 * pi + 0.5 * side_angle)
        corners.append(np.array((math.cos(theta), math.sin(theta))))
    if t == num_sides:
        return corners[0]
    i1 = int(t)
    i2 = (i1 + 1) % num_sides
    s = t - i1
    return (1 - s) * corners[i1] + s * corners[i2]
