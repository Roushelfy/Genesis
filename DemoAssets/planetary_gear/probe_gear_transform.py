"""Probe Genesis gear body transforms to determine the correct rendering formula."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np

_HERE  = Path(__file__).resolve().parent
ASSETS = _HERE / "assets"

MESH_SCALE = 0.0012
GEAR_ROTATION_DEG = 270.0
CX, CY, CZ = 0.5, 0.0, 0.79

def _load_obj_verts(path):
    vs = []
    for l in open(path, errors='replace'):
        if l.startswith('v '):
            vs.append([float(x) for x in l.split()[1:4]])
    return np.array(vs, np.float64)

import genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Box(size=(0.1,0.1,0.001), pos=(0,0,0), fixed=True))

entities = {}
entities["ring_gear"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"ring_gear.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,GEAR_ROTATION_DEG), scale=MESH_SCALE,
                   fixed=True, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

entities["sun_gear"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"sun_gear_handle.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,GEAR_ROTATION_DEG), scale=MESH_SCALE,
                   fixed=False, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

carrier_tz = -(12.0/2)*MESH_SCALE
entities["carrier"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"carrier.obj"), pos=(CX,CY,CZ+carrier_tz),
                   euler=(0,0,GEAR_ROTATION_DEG), scale=MESH_SCALE,
                   fixed=False, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

entities["planet_0"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"planet_gear.obj"),
                   pos=(CX, CY-0.0378, CZ),
                   euler=(0,0,270.0), scale=MESH_SCALE,
                   fixed=False, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

pin_tz = -12.0*MESH_SCALE
entities["support_pin"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"support_pin.obj"), pos=(CX,CY,CZ+pin_tz),
                   scale=MESH_SCALE, fixed=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

scene.build(n_envs=0)

OBJ_FILES = {
    "ring_gear":   "ring_gear.obj",
    "sun_gear":    "sun_gear_handle.obj",
    "carrier":     "carrier.obj",
    "planet_0":    "planet_gear.obj",
    "support_pin": "support_pin.obj",
}

print("="*70)
print("Genesis gear body transforms at t=0")
print("="*70)

for name, ent in entities.items():
    pos  = np.array(ent.get_pos())
    quat = np.array(ent.get_quat())   # Genesis wxyz scalar-first
    qw,qx,qy,qz = quat
    R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                  [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
                  [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]])

    V_obj = _load_obj_verts(str(ASSETS / OBJ_FILES[name]))
    bbox_ctr  = (V_obj.min(0) + V_obj.max(0)) / 2
    vert_mean = V_obj.mean(0)

    print(f"\n--- {name} ---")
    print(f"  pos  = {np.round(pos,5)}  quat(wxyz) = {np.round(quat,5)}")
    print(f"  OBJ Z range: [{V_obj[:,2].min():.2f}, {V_obj[:,2].max():.2f}] mm  bbox_ctr={np.round(bbox_ctr,3)}")

    # If Genesis uses OBJ-origin-at-pos: OBJ(0,0,0) -> pos, OBJ bbox_ctr -> pos + R@(bbox_ctr*MS)
    w_origin_f1 = pos
    w_ctr_f1    = pos + R @ (bbox_ctr * MESH_SCALE)
    print(f"  [F1: OBJ-origin=pos] OBJ(0,0,0)->world: {np.round(w_origin_f1,5)}")
    print(f"                        OBJ bbox_ctr->world: {np.round(w_ctr_f1,5)}")

    # If Genesis uses bbox-ctr-at-pos: OBJ(0,0,0) -> pos - R@(bbox_ctr*MS)
    w_origin_f2 = pos - R @ (bbox_ctr * MESH_SCALE)
    print(f"  [F2: bbox_ctr=pos  ] OBJ(0,0,0)->world: {np.round(w_origin_f2,5)}")
    print(f"                        OBJ bbox_ctr->world: {np.round(pos,5)}")

    # Try to get actual visual mesh vertex positions from Genesis
    got_verts = False
    for attr in ['get_verts', 'morph.vmesh.verts_cache', 'verts']:
        try:
            if '.' in attr:
                parts = attr.split('.')
                obj = ent
                for p in parts: obj = getattr(obj, p)
                verts_w = np.array(obj)
            else:
                verts_w = np.array(getattr(ent, attr)())
            zmin, zmax = verts_w[:,2].min(), verts_w[:,2].max()
            print(f"  [GENESIS VISUAL VERTS via {attr}]  world Z: [{zmin:.5f}, {zmax:.5f}]")
            # find vertex closest to OBJ (0,0,0) scaled
            got_verts = True
            break
        except Exception as e:
            pass
    if not got_verts:
        print(f"  [GENESIS VISUAL VERTS] not accessible via known API")

    # Also try: does the visual link expose vertices?
    try:
        for link in ent.links:
            for geom in link.geoms:
                try:
                    v = np.array(geom.get_verts())
                    print(f"  [link.geom.get_verts()]  world Z: [{v[:,2].min():.5f}, {v[:,2].max():.5f}]  n={len(v)}")
                    break
                except: pass
    except: pass

print("\n" + "="*70)
print("SUMMARY - what world Z range should each gear occupy?")
print(f"  Genesis initial ring/sun/planet: pos_z={CZ}, OBJ Z [0,12]mm")
print(f"  F1 (OBJ-origin=pos): world Z [{CZ:.4f}, {CZ+12*MESH_SCALE:.4f}]")
print(f"  F2 (bbox-ctr=pos):   world Z [{CZ-6*MESH_SCALE:.4f}, {CZ+6*MESH_SCALE:.4f}]")
print(f"  carrier pos_z={CZ+carrier_tz:.4f}  OBJ Z [-6,21]mm")
print(f"  F1 carrier: world Z [{CZ+carrier_tz-6*MESH_SCALE:.4f}, {CZ+carrier_tz+21*MESH_SCALE:.4f}]")
print(f"  F2 carrier: world Z [{CZ+carrier_tz-bbox_ctr[2]*MESH_SCALE:.4f}, ...]")
