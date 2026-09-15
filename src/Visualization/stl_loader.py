"""Load STL files (binary or ASCII) into Panda3D NodePath objects."""

import os
import struct
import re
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomNode, GeomTriangles, NodePath, LVecBase3f,
)

_VERTEX_RE = re.compile(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
_NORMAL_RE = re.compile(r"facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def _is_binary_stl(filepath: str) -> bool:
    size = os.path.getsize(filepath)
    if size < 84:
        return False
    with open(filepath, "rb") as f:
        f.seek(80)
        num_tris = struct.unpack("<I", f.read(4))[0]
    return size == 84 + num_tris * 50


def _load_binary(filepath: str):
    verts = []
    normals = []
    with open(filepath, "rb") as f:
        f.read(80)
        num_tris = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_tris):
            data = struct.unpack("<12fH", f.read(50))
            nx, ny, nz = data[0], data[1], data[2]
            n = LVecBase3f(nx, ny, nz)
            for vi in range(3):
                base = 3 + vi * 3
                verts.append((data[base], data[base + 1], data[base + 2]))
                normals.append(n)
    return verts, normals


def _load_ascii(filepath: str):
    verts = []
    normals = []
    cur_normal = LVecBase3f(0, 0, 1)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            m = _NORMAL_RE.match(line)
            if m:
                cur_normal = LVecBase3f(float(m[1]), float(m[2]), float(m[3]))
                continue
            m = _VERTEX_RE.match(line)
            if m:
                verts.append((float(m[1]), float(m[2]), float(m[3])))
                normals.append(cur_normal)
    return verts, normals


def load_stl(filepath: str, name: str = "stl") -> NodePath:
    if _is_binary_stl(filepath):
        verts, normals = _load_binary(filepath)
    else:
        verts, normals = _load_ascii(filepath)

    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(len(verts))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")

    for v, n in zip(verts, normals):
        vw.addData3(*v)
        nw.addData3(n)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(0, len(verts), 3):
        tris.addVertices(i, i + 1, i + 2)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)
