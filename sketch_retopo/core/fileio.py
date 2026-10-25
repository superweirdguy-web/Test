# Port of SketchRetopo_fileio.cc: the XML session format (version 1) of the
# original program (base mesh, configuration, curve network with patches,
# material texture), plus simple OBJ/OFF/PLY/STL writers for the retopo mesh.

import xml.etree.ElementTree as ET

import numpy as np

from .curvenetwork import Core, Patch, IMAGINARY_SYMMETRY, IMAGINARY_BOUNDARY, delete_patch
from .config import ConfigSaved
from . import patchgen

XML_VERSION = 1


def _fmt(x):
    return repr(float(x))


def save_xml(engine, fname, basemesh_verts, basemesh_tris, basemesh_uvs=None):
    """basemesh_uvs: optional (F, 3, 2) per-corner texture coordinates."""
    root = build_xml(engine, basemesh_verts, basemesh_tris, basemesh_uvs)
    tree = ET.ElementTree(root)
    if not fname.lower().endswith('.xml'):
        fname += '.xml'
    tree.write(fname, encoding='utf-8', xml_declaration=True)
    engine.configTemp.autoSave.filename = fname
    engine.configTemp.autoSave.unsaved = False
    return fname


def session_to_string(engine, basemesh_verts=None, basemesh_tris=None):
    """Same format as save_xml; without base mesh arrays the <basemesh> element is left empty
    (used to store a session inside a .blend file next to its base object)."""
    if basemesh_verts is None:
        basemesh_verts = np.zeros((0, 3))
        basemesh_tris = np.zeros((0, 3), dtype=np.int64)
    root = build_xml(engine, basemesh_verts, basemesh_tris, None)
    return ET.tostring(root, encoding='unicode')


def build_xml(engine, basemesh_verts, basemesh_tris, basemesh_uvs=None):
    root = ET.Element('sketchretopo', {'version': str(XML_VERSION)})
    has_texture = basemesh_uvs is not None
    xb = ET.SubElement(root, 'basemesh', {'has_texture': '1' if has_texture else '0'})
    xv = ET.SubElement(xb, 'vertices', {'num': str(len(basemesh_verts))})
    xv.text = '\n' + ''.join('%s %s %s\n' % (_fmt(p[0]), _fmt(p[1]), _fmt(p[2])) for p in basemesh_verts)
    xf = ET.SubElement(xb, 'faces', {'num': str(len(basemesh_tris))})
    lines = []
    for i, f in enumerate(basemesh_tris):
        s = '%d %d %d ' % (f[0], f[1], f[2])
        if has_texture:
            # OpenMesh fh_range of a triangle starts at the halfedge pointing to v0
            for k in range(3):
                uv = basemesh_uvs[i][k]
                s += '%s %s ' % (_fmt(uv[0]), _fmt(uv[1]))
        lines.append(s)
    xf.text = '\n' + '\n'.join(lines) + '\n'

    cs = engine.configSaved
    ET.SubElement(root, 'config', {'projoffset': _fmt(cs.projOffset), 'symmetric': 'true' if cs.symmetric else 'false'})

    cn = engine.curvenetwork

    def rid(x):
        return x.id if x is not None else -1

    xc = ET.SubElement(root, 'curvenetwork')
    e = ET.SubElement(xc, 'vertices', {'num': str(len(cn.vertices))})
    e.text = '\n' + ''.join('%d %d %s\n' % (v.id, rid(v.halfedge), ' '.join(_fmt(x) for x in v.pn)) for v in cn.vertices)
    e.append(ET.Comment('id halfedge px py pz nx ny nz'))
    e = ET.SubElement(xc, 'halfedges', {'num': str(len(cn.halfedges))})
    e.text = '\n' + ''.join('%d %d %d %d %d %d %d %d\n' % (
        h.id, rid(h.vertex), rid(h.next), rid(h.prev), rid(h.opposite), rid(h.halfchain), 1 if h.is_corner else 0,
        h.imaginary_patch.id if h.imaginary_patch is not None else -1) for h in cn.halfedges)
    e.append(ET.Comment('id vertex next prev opposite halfchain is_corner imaginary_patch_id'))
    e = ET.SubElement(xc, 'halfchains', {'num': str(len(cn.halfchains))})
    e.text = '\n' + ''.join('%d %d %d %d %d\n' % (c.id, rid(c.halfedge_front), rid(c.halfedge_back), rid(c.patch),
                                                  rid(c.edgechain)) for c in cn.halfchains)
    e.append(ET.Comment('id halfedge_front halfedge_back patch edgechain'))
    e = ET.SubElement(xc, 'edgechains', {'num': str(len(cn.edgechains))})
    e.text = '\n' + ''.join('%d %d %d %d\n' % (ec.id, rid(ec.halfchain[0]), rid(ec.halfchain[1]), ec.num_subdiv)
                            for ec in cn.edgechains)
    e.append(ET.Comment('id halfchain_0 halfchain_1 num_subdiv'))
    xp = ET.SubElement(xc, 'patches', {'num': str(len(cn.patches))})
    for p in cn.patches:
        xpp = ET.SubElement(xp, 'patch', {'id': str(p.id), 'halfchain': str(rid(p.halfchain)),
                                          'is_failure': '1' if p.is_failure else '0'})
        param = p.param
        ns = param.get_num_sides()
        xparam = ET.SubElement(xpp, 'param', {'num_sides': str(ns), 'pattern_id': str(param.pattern_id),
                                              'permutation_id': str(param.permutation.id)})
        ET.SubElement(xparam, 'l').text = ''.join('%d ' % x for x in param.l)
        nvar = patchgen.get_num_variables(ns, param.pattern_id) if 2 <= ns <= 6 and param.pattern_id >= 0 else 0
        ET.SubElement(xparam, 'variable').text = ''.join(
            '%d ' % patchgen.get_variable(ns, param.pattern_id, param, i) for i in range(nvar))
        xpv = ET.SubElement(xpp, 'patch_vertices', {'num': str(p.n_vertices())})
        xpv.text = '\n' + ''.join('%s %d %d\n' % (' '.join(_fmt(x) for x in p.pn[v]), p.v_corner[v], p.v_tag[v])
                                  for v in p.vertices())
        xpv.append(ET.Comment('px py pz nx ny nz corner_index tag'))
        xpf = ET.SubElement(xpp, 'patch_faces', {'num': str(p.n_faces())})
        xpf.text = '\n' + ''.join(' '.join(str(v) for v in p.fv(f)) + ' \n' for f in p.faces())
    ET.SubElement(root, 'texture').text = engine.material_texture_png_base64 or ''
    return root


def _numbers(text, cast=float):
    if not text:
        return []
    return [cast(x) for x in text.split()]


def read_xml(fname):
    """Parses an XML session.  Returns a dict with the base mesh arrays, config and a restore()
    callable that loads the curve network into an engine (after its base mesh was set)."""
    return parse_xml_root(ET.parse(fname).getroot(), fname)


def read_xml_string(text):
    return parse_xml_root(ET.fromstring(text), '')


def parse_xml_root(root, fname):
    if root.tag != 'sketchretopo':
        raise ValueError('not a SketchRetopo XML file')
    version = int(root.get('version', '0'))
    xb = root.find('basemesh')
    if xb is None:
        raise ValueError('missing <basemesh>')
    has_texture = int(xb.get('has_texture', '0')) != 0
    xv = xb.find('vertices')
    nv = int(xv.get('num'))
    verts = np.array(_numbers(xv.text), dtype=float)[:3 * nv].reshape(nv, 3)
    xf = xb.find('faces')
    nf = int(xf.get('num'))
    vals = _numbers(xf.text)
    tris = []
    uvs = [] if has_texture else None
    k = 0
    for _ in range(nf):
        tris.append((int(vals[k]), int(vals[k + 1]), int(vals[k + 2])))
        k += 3
        if has_texture:
            uvs.append([(vals[k], vals[k + 1]), (vals[k + 2], vals[k + 3]), (vals[k + 4], vals[k + 5])])
            k += 6
    xconf = root.find('config')
    if xconf is None:
        raise ValueError('missing <config>')
    config = ConfigSaved()
    config.projOffset = float(xconf.get('projoffset', '0'))
    config.symmetric = xconf.get('symmetric', 'false') == 'true'
    xc = root.find('curvenetwork')
    if xc is None:
        raise ValueError('missing <curvenetwork>')
    texture = root.find('texture')
    texture_b64 = texture.text.strip() if texture is not None and texture.text else ''

    def restore(engine):
        cn = Core()
        vmap, hmap, cmap, emap, pmap = {}, {}, {}, {}, {}
        e = xc.find('vertices')
        n = int(e.get('num'))
        vals_ = _numbers(e.text)
        from .curvenetwork import Vertex, Halfedge, Halfchain, Edgechain
        refs_v = []
        for i in range(n):
            t = vals_[8 * i: 8 * i + 8]
            v = Vertex(int(t[0]))
            v.pn = np.array(t[2:8], dtype=float)
            cn.vertices.append(v)
            vmap[v.id] = v
            refs_v.append((v, int(t[1])))
        e = xc.find('halfedges')
        n = int(e.get('num'))
        vals_ = _numbers(e.text, int)
        refs_h = []
        for i in range(n):
            t = vals_[8 * i: 8 * i + 8]
            h = Halfedge(t[0])
            h.is_corner = t[6] != 0
            if t[7] == -2:
                h.imaginary_patch = IMAGINARY_SYMMETRY
            elif t[7] == -3:
                h.imaginary_patch = IMAGINARY_BOUNDARY
            cn.halfedges.append(h)
            hmap[h.id] = h
            refs_h.append((h, t))
        e = xc.find('halfchains')
        n = int(e.get('num'))
        vals_ = _numbers(e.text, int)
        refs_c = []
        for i in range(n):
            t = vals_[5 * i: 5 * i + 5]
            c = Halfchain(t[0])
            cn.halfchains.append(c)
            cmap[c.id] = c
            refs_c.append((c, t))
        e = xc.find('edgechains')
        n = int(e.get('num'))
        vals_ = _numbers(e.text, int)
        refs_e = []
        for i in range(n):
            t = vals_[4 * i: 4 * i + 4]
            ec = Edgechain(t[0])
            ec.num_subdiv = t[3]
            cn.edgechains.append(ec)
            emap[ec.id] = ec
            refs_e.append((ec, t))
        xps = xc.find('patches')
        refs_p = []
        if xps is not None:
            for xpp in xps.findall('patch'):
                p = Patch(int(xpp.get('id')))
                p.is_failure = int(xpp.get('is_failure', '0')) != 0
                if version >= 1:
                    xparam = xpp.find('param')
                    ns = int(xparam.get('num_sides'))
                    p.param.pattern_id = int(xparam.get('pattern_id'))
                    p.param.permutation.init(ns, int(xparam.get('permutation_id')))
                    p.param.l = _numbers(xparam.find('l').text, int)[:ns]
                    p.param.p = [-1] * ns
                    p.param.q = [-1] * ns
                    if 2 <= ns <= 6 and p.param.pattern_id >= 0:
                        var = _numbers(xparam.find('variable').text, int)
                        for i in range(patchgen.get_num_variables(ns, p.param.pattern_id)):
                            patchgen.set_variable(ns, p.param.pattern_id, p.param, i, var[i])
                xpv = xpp.find('patch_vertices')
                npv = int(xpv.get('num'))
                pv = _numbers(xpv.text) if npv else []
                for i in range(npv):
                    t = pv[8 * i: 8 * i + 8]
                    v = p.add_vertex(np.array(t[:6]))
                    p.v_corner[v] = int(t[6])
                    p.v_tag[v] = int(t[7])
                xpf = xpp.find('patch_faces')
                npf = int(xpf.get('num'))
                pf = _numbers(xpf.text, int) if npf else []
                for i in range(npf):
                    p.add_face(pf[4 * i: 4 * i + 4])
                cn.patches.append(p)
                pmap[p.id] = p
                refs_p.append((p, int(xpp.get('halfchain'))))

        def patch_ref(pid):
            if pid == -2:
                return IMAGINARY_SYMMETRY
            if pid == -3:
                return IMAGINARY_BOUNDARY
            return pmap.get(pid)

        for v, hid in refs_v:
            v.halfedge = hmap.get(hid)
        for h, t in refs_h:
            h.vertex = vmap.get(t[1])
            h.next = hmap.get(t[2])
            h.prev = hmap.get(t[3])
            h.opposite = hmap.get(t[4])
            h.halfchain = cmap.get(t[5])
        for c, t in refs_c:
            c.halfedge_front = hmap.get(t[1])
            c.halfedge_back = hmap.get(t[2])
            c.patch = patch_ref(t[3])
            c.edgechain = emap.get(t[4])
        for ec, t in refs_e:
            ec.halfchain = [cmap.get(t[1]), cmap.get(t[2])]
        for p, cid in refs_p:
            p.halfchain = cmap.get(cid)
            p.set_halfedgeData()
        engine.curvenetwork = cn
        if version < 1:
            for p in cn.patches:
                delete_patch(p)
            cn.garbage_collect()
            engine.generate_patches()
        engine.memento.init()
        engine.configSaved = config
        if texture_b64:
            engine.material_texture_png_base64 = texture_b64
        engine.configTemp.autoSave.filename = fname
        engine.configTemp.autoSave.unsaved = False
        cn.invalidate_displist()

    return {'verts': verts, 'tris': np.array(tris, dtype=np.int64).reshape(-1, 3), 'uvs': uvs,
            'config': config, 'restore': restore, 'texture': texture_b64, 'version': version}


def write_mesh(fname, verts, faces):
    """Writes OBJ / OFF / PLY (ascii) / STL (ascii, fan-triangulated) depending on the extension."""
    ext = fname.lower().rsplit('.', 1)[-1] if '.' in fname else 'obj'
    with open(fname, 'w', encoding='ascii') as f:
        if ext == 'off':
            f.write('OFF\n%d %d 0\n' % (len(verts), len(faces)))
            for p in verts:
                f.write('%r %r %r\n' % (float(p[0]), float(p[1]), float(p[2])))
            for fc in faces:
                f.write('%d %s\n' % (len(fc), ' '.join(str(v) for v in fc)))
        elif ext == 'ply':
            f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
                    'element face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(verts), len(faces)))
            for p in verts:
                f.write('%r %r %r\n' % (float(p[0]), float(p[1]), float(p[2])))
            for fc in faces:
                f.write('%d %s\n' % (len(fc), ' '.join(str(v) for v in fc)))
        elif ext == 'stl':
            f.write('solid retopo\n')
            V = np.asarray(verts, dtype=float)
            for fc in faces:
                for k in range(1, len(fc) - 1):
                    a, b, c = V[fc[0]], V[fc[k]], V[fc[k + 1]]
                    n = np.cross(b - a, c - a)
                    ln = np.linalg.norm(n)
                    n = n / ln if ln > 0 else n
                    f.write('facet normal %r %r %r\n outer loop\n' % tuple(float(x) for x in n))
                    for p in (a, b, c):
                        f.write('  vertex %r %r %r\n' % tuple(float(x) for x in p))
                    f.write(' endloop\nendfacet\n')
            f.write('endsolid retopo\n')
        else:
            for p in verts:
                f.write('v %r %r %r\n' % (float(p[0]), float(p[1]), float(p[2])))
            for fc in faces:
                f.write('f %s\n' % ' '.join(str(v + 1) for v in fc))
