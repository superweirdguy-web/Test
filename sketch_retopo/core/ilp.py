# Minimal integer linear programming solver replacing lp_solve (patchgen/ILP.hh).
#
# Semantics follow lp_solve's defaults as used by patchgen: all variables are
# integers with lower bound 0 and no upper bound; the default objective is
# "maximize the sum of all variables"; solve() succeeds only for an optimal
# solution (infeasible or unbounded problems fail).
#
# The problems are tiny (at most 20 variables and ~30 rows), so a dense
# two-phase simplex (Bland's rule) inside a depth-first branch and bound is
# plenty.  When several integer optima exist the one returned may differ from
# lp_solve's choice; every optimum is a valid patch parameter.

import math

import numpy as np

EQ = 'EQ'
LE = 'LE'
GE = 'GE'

_EPS = 1e-9
_INT_EPS = 1e-6
_MAX_NODES = 20000


def _simplex(A, types, b, c, maximize):
    """Solves max/min c.x s.t. A x (types) b, x >= 0.  Returns (status, x, obj).
    status: 'optimal', 'infeasible', 'unbounded'."""
    m, n = A.shape
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    types = list(types)
    for i in range(m):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]
            types[i] = {LE: GE, GE: LE, EQ: EQ}[types[i]]

    n_slack = sum(1 for t in types if t != EQ)
    n_art = sum(1 for t in types if t != LE)
    ncols = n + n_slack + n_art
    T = np.zeros((m + 1, ncols + 1))
    T[:m, :n] = A
    T[:m, -1] = b
    basis = [0] * m
    si = n
    ai = n + n_slack
    art_cols = []
    for i, t in enumerate(types):
        if t == LE:
            T[i, si] = 1.0
            basis[i] = si
            si += 1
        elif t == GE:
            T[i, si] = -1.0
            si += 1
            T[i, ai] = 1.0
            basis[i] = ai
            art_cols.append(ai)
            ai += 1
        else:
            T[i, ai] = 1.0
            basis[i] = ai
            art_cols.append(ai)
            ai += 1

    def pivot(r, col):
        T[r] /= T[r, col]
        for i in range(T.shape[0]):
            if i != r and T[i, col] != 0.0:
                T[i] -= T[i, col] * T[r]
        basis[r] = col

    def run(allowed):
        # minimizes the objective stored in the last row (reduced costs, rhs = -objective)
        for _ in range(5000):
            col = -1
            for j in allowed:
                if T[-1, j] < -_EPS:
                    col = j
                    break
            if col < 0:
                return 'optimal'
            r = -1
            best = None
            for i in range(len(basis)):
                a = T[i, col]
                if a > _EPS:
                    ratio = T[i, -1] / a
                    if best is None or ratio < best - _EPS or (abs(ratio - best) <= _EPS and basis[i] < basis[r]):
                        best = ratio
                        r = i
            if r < 0:
                return 'unbounded'
            pivot(r, col)
        return 'optimal'

    # phase 1
    if art_cols:
        T[-1, :] = 0.0
        for j in art_cols:
            T[-1, j] = 1.0
        for i in range(m):
            if basis[i] in art_cols:
                T[-1] -= T[i]
        run(list(range(ncols)))
        if -T[-1, -1] > 1e-7:
            return 'infeasible', None, None
        # drive artificial variables out of the basis
        art_set = set(art_cols)
        rows_to_drop = []
        for i in range(m):
            if basis[i] in art_set:
                for j in range(n + n_slack):
                    if abs(T[i, j]) > _EPS:
                        pivot(i, j)
                        break
                else:
                    rows_to_drop.append(i)
        if rows_to_drop:
            keep = [i for i in range(m) if i not in rows_to_drop]
            T = np.vstack([T[keep], T[-1:]])
            basis = [basis[i] for i in keep]
            m = len(keep)
        T = np.delete(T, art_cols, axis=1)
        ncols -= n_art
        basis = [bi for bi in basis]

    # phase 2
    cost = np.zeros(ncols)
    cost[:n] = -c if maximize else c
    T[-1, :] = 0.0
    T[-1, :ncols] = cost
    for i in range(m):
        if T[-1, basis[i]] != 0.0:
            T[-1] -= T[-1, basis[i]] * T[i]
    status = run(list(range(ncols)))
    if status == 'unbounded':
        return 'unbounded', None, None
    x = np.zeros(ncols)
    for i in range(m):
        x[basis[i]] = T[i, -1]
    x = x[:n]
    obj = float(np.dot(c, x))
    return 'optimal', x, obj


class ILP:
    def __init__(self, num_variables):
        self.num_variables = num_variables
        self.rows = []
        self.types = []
        self.rhs = []
        self.objective = np.ones(num_variables)
        self.is_maxim = True
        self.solution = None

    def add_constraint(self, row, constr_type, rhs):
        row = np.asarray(row, dtype=float)
        assert row.shape == (self.num_variables,)
        self.rows.append(row)
        self.types.append(constr_type)
        self.rhs.append(float(rhs))

    def add_constraints(self, rows, constr_type, rhs):
        rows = np.asarray(rows, dtype=float)
        for i in range(rows.shape[0]):
            self.add_constraint(rows[i], constr_type, rhs[i])

    def set_objective(self, row, is_maxim):
        self.objective = np.asarray(row, dtype=float)
        self.is_maxim = is_maxim

    def refresh(self):
        pass

    def get_variables(self):
        return list(self.solution)

    def solve(self):
        self.solution = None
        n = self.num_variables
        A0 = np.array(self.rows, dtype=float).reshape(len(self.rows), n)
        b0 = np.array(self.rhs, dtype=float)
        types0 = list(self.types)
        c = self.objective
        maximize = self.is_maxim
        integral_obj = np.all(np.abs(c - np.round(c)) < 1e-12)

        best = [None, None]   # objective, x
        nodes = [0]

        def better(obj):
            if best[0] is None:
                return True
            return obj > best[0] + 1e-9 if maximize else obj < best[0] - 1e-9

        def bound_ok(lp_obj):
            if best[0] is None:
                return True
            if integral_obj:
                bound = math.floor(lp_obj + 1e-7) if maximize else math.ceil(lp_obj - 1e-7)
            else:
                bound = lp_obj
            return better(bound)

        def branch(extra):
            nodes[0] += 1
            if nodes[0] > _MAX_NODES:
                return
            if extra:
                A = np.vstack([A0] + [r for r, _, _ in extra])
                b = np.concatenate([b0, [v for _, _, v in extra]])
                types = types0 + [t for _, t, _ in extra]
            else:
                A, b, types = A0, b0, types0
            status, x, obj = _simplex(A, types, b, c, maximize)
            if status != 'optimal':
                if status == 'unbounded' and not extra:
                    best[0] = 'unbounded'
                return
            if not bound_ok(obj):
                return
            frac_j = -1
            for j in range(n):
                if abs(x[j] - round(x[j])) > _INT_EPS:
                    frac_j = j
                    break
            if frac_j < 0:
                xi = [int(round(v)) for v in x]
                if self._feasible(xi):
                    o = float(np.dot(c, xi))
                    if better(o):
                        best[0] = o
                        best[1] = xi
                return
            v = x[frac_j]
            e = np.zeros(n)
            e[frac_j] = 1.0
            # ceiling branch first, then floor branch
            branch(extra + [(e, GE, math.ceil(v))])
            if best[0] == 'unbounded':
                return
            branch(extra + [(e, LE, math.floor(v))])

        branch([])
        if best[0] is None or best[0] == 'unbounded':
            return False
        self.solution = best[1]
        return True

    def _feasible(self, xi):
        if any(v < 0 for v in xi):
            return False
        for row, t, r in zip(self.rows, self.types, self.rhs):
            s = float(np.dot(row, xi))
            if t == EQ and abs(s - r) > 1e-6:
                return False
            if t == LE and s > r + 1e-6:
                return False
            if t == GE and s < r - 1e-6:
                return False
        return True
