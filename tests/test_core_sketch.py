import time

from helpers import grid_mesh, make_engine, stroke, click, hover
from sketch_retopo.core.engine import SKETCH, EDITTOPOLOGY, EDGELOOP


def quad_count(eng):
    return sum(p.n_faces() for p in eng.curvenetwork.patches)


def main():
    V, T = grid_mesh(60, 2.0)
    eng = make_engine(V, T)
    t0 = time.time()
    # the grid boundary is traced as an imaginary-boundary curve; draw a square with 4 strokes
    stroke(eng, [(-0.5, -0.5), (0.5, -0.5)])
    stroke(eng, [(0.5, -0.5), (0.5, 0.5)])
    stroke(eng, [(0.5, 0.5), (-0.5, 0.5)])
    stroke(eng, [(-0.5, 0.5), (-0.5, -0.5)])
    cn = eng.curvenetwork
    print('patches', len(cn.patches), 'quads', quad_count(eng), 'vertices', len(cn.vertices),
          'time %.2fs' % (time.time() - t0))
    assert len(cn.patches) == 1, len(cn.patches)
    p = cn.patches[0]
    print('patch sides', p.num_corners(), 'l', p.param.l, 'pattern', p.param.pattern_id)
    assert p.num_corners() == 4

    # a crossing stroke splits the square into two patches
    stroke(eng, [(0.0, -0.7), (0.0, 0.7)])
    print('after split: patches', len(cn.patches), [pp.param.l for pp in cn.patches])
    assert len(cn.patches) == 2

    # undo / redo
    eng.key_down('ctrl+z')
    print('after undo: patches', len(eng.curvenetwork.patches))
    assert len(eng.curvenetwork.patches) == 1
    eng.key_down('ctrl+y')
    print('after redo: patches', len(eng.curvenetwork.patches))
    assert len(eng.curvenetwork.patches) == 2

    # erase all open-ended curves with 'g' (the parts of the vertical stroke sticking out)
    n_before = len(eng.curvenetwork.edgechains)
    eng.key_down('g')
    print('edgechains', n_before, '->', len(eng.curvenetwork.edgechains), 'patches', len(eng.curvenetwork.patches))
    assert len(eng.curvenetwork.patches) == 2

    # edit topology: select the middle curve and add 2 subdivisions
    eng.key_down('a')
    click(eng, 0.0, 0.0)
    sel = [e for e in eng.curvenetwork.edgechains if e.flag]
    print('selected curves', len(sel), [e.num_subdiv for e in sel])
    q0 = quad_count(eng)
    eng.key_down('c')
    print('quads', q0, '->', quad_count(eng), [pp.param.l for pp in eng.curvenetwork.patches])

    # edge loop insertion
    eng.key_down('s')
    hover(eng, -0.21, -0.5)
    preview = eng.stateEdgeLoop.edgeLoop_preview.size()
    q1 = quad_count(eng)
    click(eng, -0.21, -0.5)
    print('edge loop preview points', preview, 'quads', q1, '->', quad_count(eng))
    assert quad_count(eng) > q1

    # eraser stroke while holding 'c' removes the crossing curve -> back to one patch
    eng.key_down('q')
    eng.key_down('c')
    stroke(eng, [(-0.1, 0.0), (0.1, 0.05)])
    eng.key_up('c')
    print('after erase: patches', len(eng.curvenetwork.patches))
    print('total time %.2fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
