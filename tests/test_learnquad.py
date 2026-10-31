import itertools
import math
import os
import tempfile
import time

import numpy as np

from helpers import grid_mesh, make_engine, stroke, click
from sketch_retopo.core import learnquad as lq, patchgen
from sketch_retopo.core.curvenetwork import Patch, delete_patch


def pattern_faces(l):
    param = patchgen.get_default_parameter(list(l))
    if param is None:
        return None
    p = Patch(0)
    patchgen.generate_topology(param, p)
    return [list(p.fv(f))[:4] for f in p.faces()]


def ddq_sides(faces):
    """Vertices of each side between the database's corners (boundary vertices on a single face)."""
    m = lq.mesh_from_faces(faces)
    start = next(lq.Pos(m, f, z, m.fv[f][z]) for f in range(m.nf) for z in range(4) if m.ffp[f][z] == f)
    _, corners = lq.find_polygon_corners(m, start)
    sides = []
    for i, c in enumerate(corners):
        run = c.copy()
        vs = [run.v]
        while True:
            run.flip_v()
            run.flip_e()
            vs.append(run.v)
            while not run.is_border():
                run.flip_f()
                run.flip_e()
            if run == corners[(i + 1) % len(corners)]:
                break
        sides.append(vs)
    return sides


def glue(fa, fb, ia, ib):
    """Glues side ia of patch fa to side ib of patch fb (same number of edges)."""
    sa, sb = ddq_sides(fa)[ia], ddq_sides(fb)[ib]
    assert len(sa) == len(sb)
    off = max(max(f) for f in fa) + 1
    ident = {v: sa[len(sa) - 1 - t] for t, v in enumerate(sb)}
    faces = fa + [[ident.get(v, v + off) for v in f] for f in fb]
    used = sorted({v for f in faces for v in f})
    remap = {v: k for k, v in enumerate(used)}
    return [[remap[v] for v in f] for f in faces]


def build_test_db(path):
    rows = {}

    def add(faces):
        row = lq.encode_patch(faces)
        rows.setdefault(row[0], row)

    for ns in (3, 4, 5):
        for l in itertools.product(range(1, 5), repeat=ns):
            if sum(l) % 2 == 0:
                faces = pattern_faces(l)
                if faces:
                    add(faces)
    pent = pattern_faces([2, 2, 2, 2, 2])
    hexa = glue(pent, pent, 0, 0)               # 6 corners, 2 singularities
    hepta = glue(hexa, pent, 1, 0)              # 7 corners, 3 singularities
    octa = glue(hexa, hexa, 1, 1)               # 8 corners
    for f in (hexa, hepta, octa):
        add(f)
    rows = list(rows.values())
    lq.create_db(path, rows)
    return rows


_db_dir = tempfile.mkdtemp(prefix='sr_lq_')
DB_PATH = os.path.join(_db_dir, 'patches.db')
ROWS = build_test_db(DB_PATH)


def side_lengths(m, start):
    corners = lq.corners_from_start(m, start)
    out = []
    for i in range(len(corners)):
        run = corners[i].copy()
        n = 0
        while True:
            run.flip_v()
            run.flip_e()
            n += 1
            while not run.is_border():
                run.flip_f()
                run.flip_e()
            if run == corners[(i + 1) % len(corners)]:
                break
        out.append(n)
    return out


def test_encode_decode_roundtrip():
    for i, row in enumerate(ROWS):
        t = lq.Template(i, row)
        t.ensure_mesh()
        assert lq.build_topology_str(t.mesh, t.start) == row[0]
        assert t.n_vars * 2 == t.perimeter
    nc = sorted({r[4] for r in ROWS})
    print('templates', len(ROWS), 'corner counts', nc)
    assert {6, 7, 8} <= set(nc)


def test_enumerate_ilp():
    # x0 + x1 = 4, x1 + x2 = 3, x >= 1
    sols = list(lq.enumerate_ilp([[1, 1, 0], [0, 1, 1]], [4, 3], [1, 1, 1], [lq.INF] * 3))
    assert sorted(map(tuple, sols)) == [(2, 2, 1), (3, 1, 2)]
    # bounded dummy variables with negative coefficients
    sols = list(lq.enumerate_ilp([[1, -1, 0], [0, 1, 1]], [0, 2], [0, -1, 0], [lq.INF, 1, lq.INF]))
    assert sorted(map(tuple, sols)) == [(0, 0, 2), (1, 1, 1)]


def test_lookup_side_lengths():
    db = lq.PatchDB(DB_PATH)
    for L in ([5, 3, 3, 3], [4, 4, 6, 6], [3, 3, 3, 3, 4], [5, 5, 4], [4, 8, 8, 4, 4, 12, 4],
              [2, 4, 6, 2, 2, 4, 6, 2]):
        lk = lq.Lookup(db)
        lk.start(len(L), L)
        lk.run(time_limit=30)
        assert lk.status == 'complete'
        hashes = set()
        for r in lk.results:
            m, start = lk.mesh_of(r)
            assert side_lengths(m, start) == L
            hashes.add(lq.build_topology_str(m, start))
        assert len(hashes) == lk.num_results()
        keys = [r.sort_key(lk.sort_criteria) for r in lk.results]
        assert keys == sorted(keys)
        print(L, 'results', lk.num_results())
        if len(L) >= 7:
            assert lk.num_results() > 0


def test_expanded_topology_matches_mesh():
    """The uniqueness key computed from the expansion's adjacency equals the string of the expanded mesh."""
    db = lq.PatchDB(DB_PATH)
    lk = lq.Lookup(db)
    lk.start(4, [6, 4, 6, 4])
    lk.run()
    strings = {}
    for r in lk.results:
        ex, (cell, z, v, d) = r.template.expand_adjacency(r.var, r.rot)
        s = lq._topology_str_adj(ex.ffp, ex.ffi, cell, z, d)
        assert s == lq.build_topology_str(*lk.mesh_of(r))
        strings[lq._topology_key_adj(ex.ffp, ex.ffi, cell, z, d)] = s
    assert len(strings) == len(set(strings.values())) == lk.num_results()


def _polychord_points_reference(start, P):
    """PolychordPoints::polychordToPoints written with Pos, as in the original."""
    run = start.copy()
    pts = []
    while True:
        pts.append((P[run.v] + P[run.vflip()]) / 2.0)
        run.flip_e()
        run.flip_v()
        run.flip_e()
        pts.append((pts[-1] + (P[run.v] + P[run.vflip()]) / 2.0) / 2.0)
        run.flip_f()
        if run == start or run.is_border():
            break
    pts.append((P[run.v] + P[run.vflip()]) / 2.0)
    return np.array(pts)


def test_polychord_points():
    rng = np.random.default_rng(5)
    lk = lq.Lookup(lq.PatchDB(DB_PATH))
    lk.start(4, [4, 4, 6, 6])
    lk.run()
    n = 0
    for r in lk.results:
        m, start = lk.mesh_of(r)
        P = rng.random((m.nv, 3))
        positions = [lq.Pos(m, f, zz, m.fv[f][zz]) for f in range(m.nf) for zz in range(4)]
        positions += [lq.Pos(m, f, zz, m.fv[f][(zz + 1) & 3]) for f in range(m.nf) for zz in range(4)]
        for pos in positions:
            pp = lq.PolychordPoints()
            pp.start = pos
            pp.polychord_to_points(P)
            assert np.array_equal(pp.points, _polychord_points_reference(pos, P))
            n += 1
    print('polychords compared', n)


def test_uv_locator_and_solver():
    rng = np.random.default_rng(3)
    V, T = grid_mesh(30, 2.0)
    uv = V[:, :2] + 0.01 * rng.standard_normal((len(V), 2))
    nrm = np.tile((0.0, 0.0, 1.0), (len(V), 1))
    loc = lq.UVLocator(V, nrm, uv, T)
    pts = rng.uniform(-1.2, 1.2, (300, 2))
    tri, bc = loc.locate_all(pts)
    for p, t, b in zip(pts, tri, bc):
        # the first triangle of the point's grid cell that contains it, as UVGrid; else the closest one
        res = None
        if np.all(p >= loc.lo) and np.all(p <= loc.hi):
            c = np.clip(np.floor((p - loc.lo) / loc.edge).astype(int), 0, loc.size - 1)
            cid = c[0] * loc.size + c[1]
            for cand in loc.cell_tris[loc.cell_ptr[cid]:loc.cell_ptr[cid + 1]]:
                bb = loc._bary(np.array([cand]), p)[0]
                if np.all((bb >= -1e-4) & (bb <= 1 + 1e-4)) and loc.area[cand] > 1e-5:
                    res = (cand, bb)
                    break
        if res is None:
            res = loc._closest(p)
        assert res[0] == t and np.allclose(res[1], b)
    # conjugate gradients agree with the dense solve
    lk = lq.Lookup(lq.PatchDB(DB_PATH))
    lk.start(4, [4, 4, 6, 6])
    lk.run()
    m = lk.mesh_of(lk.results[-1])[0]
    m.P = rng.random((m.nv, 3))
    border = m.border_vertices()
    rows, cols, vals = lq._laplacian_triplets(m, lq.SMOOTH_LAMBDA)
    b = np.flatnonzero(border)
    rows, cols, vals = (np.concatenate(rows + [b]), np.concatenate(cols + [b]),
                        np.concatenate(vals + [np.full(len(b), lq.PENALTY)]))
    rhs = np.zeros((m.nv, 3))
    rhs[b] = lq.PENALTY * m.P[b]
    saved = lq.DENSE_SOLVE_MAX
    try:
        lq.DENSE_SOLVE_MAX = 10 ** 9
        dense = lq._solve_spd(rows, cols, vals, m.nv, rhs)
        lq.DENSE_SOLVE_MAX = 0
        cg = lq._solve_spd(rows, cols, vals, m.nv, rhs)
    finally:
        lq.DENSE_SOLVE_MAX = saved
    print('solver: n', m.nv, 'max difference dense/cg %.1e' % np.abs(dense - cg).max())
    assert np.abs(dense - cg).max() < 1e-8


def _square(eng, h=0.5):
    pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
    for i in range(4):
        stroke(eng, [pts[i], pts[(i + 1) % 4]])


def test_query_database_mode():
    V, T = grid_mesh(60, 2.0, z_func=lambda x, y: 0.15 * math.sin(2 * x) * math.cos(y))
    eng = make_engine(V, T)
    _square(eng)
    assert len(eng.curvenetwork.patches) == 1
    # 6 edges per side keeps the lookup short (the test database holds every small grid, all expanding alike)
    delete_patch(eng.curvenetwork.patches[0])
    for e in eng.curvenetwork.edgechains:
        e.num_subdiv = 6
    eng.generate_patches()
    patch = eng.curvenetwork.patches[0]
    ns = patch.num_subdiv_all()
    assert ns == [6, 6, 6, 6]
    print('query boundary', ns)
    assert not eng.learnquad.is_loaded()
    n = eng.learnquad.load(DB_PATH)
    assert n == len(ROWS)
    eng.key_down('s')
    st = eng.stateQueryDB
    assert eng.state is st
    click(eng, 0.05, 0.02)
    assert st.selected_patch is patch and len(st.querydata.boundary_xyz) == sum(ns)
    # a flow stroke across the patch starts the lookup
    stroke(eng, [(-0.45, 0.02), (0.0, 0.03), (0.45, 0.02)])
    assert st.status == 'started' and len(st.querydata.strokes) == 1
    flows = eng.learnquad.retopologizer.flow_map_uv
    print('flows', {k: [(f.start_edge, f.end_edge) for f in v] for k, v in flows.items()})
    assert list(flows) == [(1, 3)] or list(flows) == [(0, 2)] or len(flows) == 1
    t0 = time.time()
    while st.status != 'complete' and time.time() - t0 < 120:
        st.update(0.05)
    print('lookup: %d results in %.1fs, status %r' % (eng.learnquad.num_results(), time.time() - t0, st.status_msg))
    assert st.status == 'complete' and eng.learnquad.num_results() > 0
    assert 'lookup complete!' in st.status_msg
    res = eng.learnquad.lookup.results
    fd = [r.flow_distance for r in res]
    assert fd == sorted(fd)
    # the shown result replaced the patch: same corners and boundary, quads on the surface
    assert patch.num_subdiv_all() == ns
    corners = sorted(patch.v_corner[v] for v in patch.vertices() if patch.v_corner[v] >= 0)
    assert corners == [0, 1, 2, 3]
    assert patch.n_faces() == eng.learnquad.lookup.mesh_of(res[0])[0].nf
    for v in patch.vertices():
        p = patch.pn[v][:3]
        assert abs(p[0]) <= 0.52 and abs(p[1]) <= 0.52
        assert abs(p[2] - 0.15 * math.sin(2 * p[0]) * math.cos(p[1])) < 0.01
    # faces keep the orientation of the surface (+Z here)
    up = 0
    for f in patch.faces():
        vs = patch.fv(f)
        P = [patch.pn[v][:3] for v in vs]
        nrm = np.cross(P[1] - P[0], P[2] - P[0]) + np.cross(P[2] - P[0], P[3] - P[0])
        up += nrm[2] > 0
    assert up == patch.n_faces()
    # browse the results
    eng.key_down('x')
    assert st.lookup_index == min(1, eng.learnquad.num_results() - 1)
    eng.key_down('z')
    assert st.lookup_index == 0
    # shift adds a second flow; delete_flow removes it
    stroke(eng, [(0.02, -0.45), (0.03, 0.0), (0.02, 0.45)], shift=True)
    assert len(st.querydata.strokes) == 2
    st.stop_lookup()
    assert st.status == 'complete'
    st.delete_flow()
    assert len(st.querydata.strokes) == 1 and st.status == 'idle'
    st.restart()
    assert st.status == 'started'
    st.stop_lookup()
    # leaving the mode stops the lookup
    eng.key_down('q')
    assert not eng.learnquad.lookup.running
    return eng


def test_generate_patch_more_than_6_corners():
    V, T = grid_mesh(60, 2.0)
    eng = make_engine(V, T)
    pts = [(0.6 * math.cos(2 * math.pi * k / 7 + 0.3), 0.6 * math.sin(2 * math.pi * k / 7 + 0.3)) for k in range(7)]
    for i in range(7):
        stroke(eng, [pts[i], pts[(i + 1) % 7]])
    cn = eng.curvenetwork
    # loops with more than 6 corners are not filled automatically; X + drag asks for a patch (the database)
    mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    inner = (mid[0] * 0.6, mid[1] * 0.6)
    eng.key_down('x')
    stroke(eng, [mid, inner])
    eng.key_up('x')
    print('heptagon without database: patches', len(cn.patches))
    assert len(cn.patches) == 0
    eng.learnquad.load(DB_PATH)
    # spans the 7-corner template of the test database can realize (any rotation, either orientation)
    for e, n in zip(cn.edgechains, [4, 8, 8, 4, 4, 12, 4]):
        e.num_subdiv = n
    eng.key_down('x')
    stroke(eng, [mid, inner])
    eng.key_up('x')
    print('with database: patches', [p.num_subdiv_all() for p in cn.patches])
    assert len(cn.patches) == 1
    p = cn.patches[0]
    assert p.num_corners() == 7 and p.n_faces() > 0
    corners = sorted(p.v_corner[v] for v in p.vertices() if p.v_corner[v] >= 0)
    assert corners == list(range(7))
    zmax = max(abs(p.pn[v][2]) for v in p.vertices())
    print("max |z|", zmax, "faces", p.n_faces(), "nv", p.n_vertices())
    assert zmax < 0.01      # projection offset
    return eng


if __name__ == '__main__':
    for fn in (test_encode_decode_roundtrip, test_enumerate_ilp, test_lookup_side_lengths,
               test_expanded_topology_matches_mesh, test_polychord_points, test_uv_locator_and_solver,
               test_query_database_mode,
               test_generate_patch_more_than_6_corners):
        t0 = time.time()
        print('==', fn.__name__)
        fn()
        print('   (%.2fs)' % (time.time() - t0))
