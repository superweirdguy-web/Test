"""Checks the patchgen port: every generated patch is all-quad, and boundary side i has exactly l[i] edges."""
import itertools
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketch_retopo.core import patchgen
from sketch_retopo.core.polymesh import PolyMesh


class TestPatch(PolyMesh):
    def clear(self):
        self.clear_mesh()


def side_lengths(patch, n):
    # walk the boundary (boundary halfedges are oriented clockwise) starting from corner 0
    v0 = next(v for v in patch.vertices() if patch.v_corner[v] == 0)
    h = patch.halfedge_handle(v0)
    assert patch.is_boundary_h(h)
    # sides in clockwise order starting at corner 0: walk prev to go counterclockwise
    h = patch.prev_halfedge_handle(h)
    lens = []
    count = 0
    start = h
    corners_seen = []
    while True:
        count += 1
        v_from = patch.from_vertex_handle(h)
        if patch.v_corner[v_from] != -1:
            lens.append(count)
            corners_seen.append(patch.v_corner[v_from])
            count = 0
        h = patch.prev_halfedge_handle(h)
        if h == start:
            break
    return lens, corners_seen


def check(l):
    param = patchgen.get_default_parameter(l)
    if param is None:
        return None
    patch = TestPatch()
    patchgen.generate_topology(param, patch)
    for f in patch.faces():
        assert patch.face_valence(f) == 4, (l, 'non-quad')
    lens, corners = side_lengths(patch, len(l))
    assert corners == list(range(1, len(l))) + [0], (l, corners)
    assert lens == list(l), (l, lens, param.pattern_id)
    return param


def main():
    random.seed(1)
    t0 = time.time()
    n_ok = n_none = 0
    failures = {}
    for n in range(2, 7):
        rng = range(1, 9) if n <= 4 else range(1, 7)
        cases = [l for l in itertools.product(rng, repeat=n) if sum(l) % 2 == 0]
        if len(cases) > 400:
            cases = random.sample(cases, 400)
        for l in cases:
            p = check(list(l))
            if p is None:
                n_none += 1
                failures.setdefault(n, []).append(l)
            else:
                n_ok += 1
    # larger random cases
    for _ in range(150):
        n = random.randint(2, 6)
        l = [random.randint(1, 30) for _ in range(n)]
        if sum(l) % 2:
            l[0] += 1
        if check(l) is None:
            n_none += 1
            failures.setdefault(n, []).append(tuple(l))
        else:
            n_ok += 1
    print('ok:', n_ok, 'no pattern:', n_none, 'time: %.1fs' % (time.time() - t0))
    for n, ls in failures.items():
        print('  %d-sided without pattern (first few):' % n, ls[:8], 'count', len(ls))

    # adjust / switch
    param = patchgen.get_default_parameter([6, 4, 6, 4])
    assert param.pattern_id == 0
    param = patchgen.get_default_parameter([8, 4, 4, 4, 6])
    n_var = patchgen.get_num_variables(5, param.pattern_id)
    print('5-sided pattern', param.pattern_id, patchgen.get_param_str(5, param.pattern_id, param))
    for i in range(5, n_var):
        q = param.copy()
        if patchgen.adjust_parameter(q, i, True):
            patch = TestPatch()
            patchgen.generate_topology(q, patch)
            assert side_lengths(patch, 5)[0] == [8, 4, 4, 4, 6]
    q = param.copy()
    seen = set()
    for _ in range(12):
        if not patchgen.switch_pattern(q, True):
            break
        patch = TestPatch()
        patchgen.generate_topology(q, patch)
        assert side_lengths(patch, 5)[0] == [8, 4, 4, 4, 6]
        seen.add((q.pattern_id, q.permutation.id))
    print('switch_pattern visited', len(seen), 'configurations')


if __name__ == '__main__':
    main()
