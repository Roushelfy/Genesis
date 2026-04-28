"""Probe Genesis: extract the EXACT body-local frame by comparing
entity.get_verts() (world) with OBJ vertices (local).

Solves:  V_world = R @ V_local + t   via SVD on matched vertex pairs.
Then decomposes to understand what V_local is relative to V_obj.
"""
import numpy as np
from pathlib import Path

_HERE  = Path(__file__).resolve().parent
ASSETS = _HERE / "assets"
MESH_SCALE = 0.0012

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

CX, CY, CZ = 0.5, 0.0, 0.79

entities = {}
entities["sun_gear"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"sun_gear_handle.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MESH_SCALE,
                   fixed=False, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

entities["ring_gear"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"ring_gear.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MESH_SCALE,
                   fixed=True, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

entities["planet_0"] = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"planet_gear.obj"),
                   pos=(CX, CY-0.0378, CZ), euler=(0,0,270), scale=MESH_SCALE,
                   fixed=False, file_meshes_are_zup=True, convexify=False, decimate=False),
    material=gs.materials.Rigid(rho=1000.0))

scene.build(n_envs=0)

OBJ_FILES = {
    "sun_gear":  "sun_gear_handle.obj",
    "ring_gear": "ring_gear.obj",
    "planet_0":  "planet_gear.obj",
}

for name, ent in entities.items():
    V_obj = _load_obj_verts(str(ASSETS / OBJ_FILES[name]))
    V_world = np.array(ent.get_verts())
    pos  = np.array(ent.get_pos())
    quat = np.array(ent.get_quat())

    n = min(len(V_obj), len(V_world))
    V_obj_n = V_obj[:n]
    V_w_n   = V_world[:n]

    # Method 1: V_world = R @ (V_obj * scale) + t  — solve via SVD
    V_scaled = V_obj_n * MESH_SCALE
    c_s = V_scaled.mean(0)
    c_w = V_w_n.mean(0)
    H = (V_scaled - c_s).T @ (V_w_n - c_w)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R_fit = Vt.T @ D @ U.T
    t_fit = c_w - R_fit @ c_s

    residual = np.linalg.norm(V_w_n - (R_fit @ V_scaled.T).T - t_fit, axis=1)

    print(f"\n{'='*60}")
    print(f"{name}  ({n} verts matched)")
    print(f"  get_pos  = {np.array2string(pos, precision=6)}")
    print(f"  get_quat = {np.array2string(quat, precision=6)} (wxyz)")
    print(f"  Fitted R:\n{np.array2string(R_fit, precision=6, suppress_small=True)}")
    print(f"  Fitted t: {np.array2string(t_fit, precision=6)}")
    print(f"  Residual: max={residual.max():.2e}  mean={residual.mean():.2e}")
    print(f"  R @ [0,0,1] = {np.round(R_fit @ [0,0,1], 4)}  (Z-axis in world)")
    print(f"  det(R) = {np.linalg.det(R_fit):.6f}")

    # Compare fitted t with get_pos
    print(f"  t - get_pos = {np.array2string(t_fit - pos, precision=6)}")

    # Check: is R_fit = R(quat)?
    qw,qx,qy,qz = quat
    R_quat = np.array([
        [1-2*(qy**2+qz**2),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),1-2*(qx**2+qz**2),2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx**2+qy**2)]])
    R_diff = np.linalg.norm(R_fit - R_quat)
    print(f"  ||R_fit - R(quat)|| = {R_diff:.6f}")

    # If R_fit != R(quat), what is R_fit @ R(quat).T?  (= R_body_local)
    if R_diff > 0.01:
        R_local = R_fit @ R_quat.T
        print(f"  R_body_local = R_fit @ R(quat).T:")
        print(f"  {np.array2string(R_local, precision=4, suppress_small=True)}")
        # Check if R_local is R_z(270)
        import math
        a = math.radians(270)
        Rz270 = np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
        print(f"  ||R_local - Rz(270)|| = {np.linalg.norm(R_local - Rz270):.6f}")
