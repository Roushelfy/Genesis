"""M1 environment smoke test for the ipc_monolithic coupling mode.

Exercises exactly the libuipc surfaces the new mode will use:
  - AffineBodyConstitution (ABD links)
  - AffineBodyRevoluteJoint (revolute DOF)
  - AffineBodyRevoluteJointExternalForce (TORQUE actuation = the B1 path)
  - AffineBodyStateAccessorFeature.copy_to (readback = the C1 path)
  - World.advance()/retrieve() on the 'cuda' backend

Scene: two ABD cubes joined by one revolute joint about +z, one cube fixed.
A constant external torque is applied to the joint; we advance and check that
the free cube rotates (i.e. the torque actually drives the articulation) and the
world stays valid.

Run with the gs-gym-internal py3.10 venv:
  /home/zhaofeng/work/gs-gym-internal/.venv/bin/python \
      genesis/engine/couplers/ipc_coupler/docs/development/m1_uipc_smoke.py
"""

import math
import os
import tempfile

import numpy as np

from uipc import Logger, Matrix4x4, Engine, World, Scene, view, builtin
from uipc.geometry import (
    SimplicialComplexIO,
    label_surface,
    label_triangle_orient,
    flip_inward_triangles,
    linemesh,
)
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyRevoluteJoint,
    AffineBodyRevoluteJointExternalForce,
)
from uipc.core import AffineBodyStateAccessorFeature

CUBE_MSH = "/home/zhaofeng/work/gs-gym-internal/third_party/libuipc/assets/sim_data/tetmesh/cube.msh"
TORQUE = 50.0          # N·m about the joint axis
N_STEPS = 40


def prep_surface(mesh):
    label_surface(mesh)
    label_triangle_orient(mesh)
    return flip_inward_triangles(mesh)


def body_xy_angle(T):
    """In-plane rotation angle (rad) of a 4x4 transform's x-axis in the xy plane."""
    return math.atan2(T[1, 0], T[0, 0])


def main():
    Logger.set_level(Logger.Level.Error)

    workspace = tempfile.mkdtemp(prefix="m1_uipc_smoke_")
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    # default integrator is bdf1 (required by the mode); assert it
    integ = str(config.get("integrator", {}).get("type", "bdf1"))
    print(f"[cfg] integrator/type = {integ!r}  dt = {config['dt']}")
    scene = Scene(config)

    abd = AffineBodyConstitution()

    pre = Matrix4x4.Identity()
    pre[0, 0] = pre[1, 1] = pre[2, 2] = 0.2  # scale the unit cube down
    io = SimplicialComplexIO(pre)
    link = io.read(CUBE_MSH)
    link = prep_surface(link)
    link.instances().resize(2)
    abd.apply_to(link, 1.0e8)

    T = view(link.transforms())
    t0 = Matrix4x4.Identity(); t0[0:3, 3] = np.array([0.30, 0.0, 0.0]); T[0] = t0   # free
    t1 = Matrix4x4.Identity(); t1[0:3, 3] = np.array([-0.30, 0.0, 0.0]); T[1] = t1  # fixed

    is_fixed = view(link.instances().find(builtin.is_fixed))
    is_fixed[:] = 0
    is_fixed[1] = 1

    links = scene.objects().create("links")
    link_slot = links.geometries().create(link)[0]

    # Revolute joint about +z at the origin between body0 (free) and body1 (fixed)
    revolute = AffineBodyRevoluteJoint()
    joint = linemesh(
        np.array([[0.0, 0.0, -0.25], [0.0, 0.0, 0.25]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int32),
    )
    revolute.apply_to(joint, [link_slot], [0], [link_slot], [1], [100.0])

    # TORQUE actuation (B1): external torque about the joint axis
    ext = AffineBodyRevoluteJointExternalForce()
    ext.apply_to(joint, float(TORQUE))

    joints = scene.objects().create("joints")
    joints.geometries().create(joint)

    world.init(scene)
    assert world.is_valid(), "world invalid right after init"

    # Readback path (C1): AffineBodyStateAccessorFeature.copy_to
    accessor = world.features().find(AffineBodyStateAccessorFeature)
    assert accessor is not None, "AffineBodyStateAccessorFeature missing"
    print(f"[accessor] body_count = {accessor.body_count()}")
    state_geo = accessor.create_geometry()
    state_geo.instances().create(builtin.transform, np.eye(4, dtype=np.float64))

    def read_free_angle_via_accessor():
        accessor.copy_to(state_geo)
        Ts = state_geo.instances().find(builtin.transform).view()
        return body_xy_angle(np.asarray(Ts[0]))

    geom_T = link_slot.geometry().transforms()
    a0 = body_xy_angle(np.asarray(view(geom_T)[0]))
    acc0 = read_free_angle_via_accessor()
    print(f"[init] free-body angle: geom={a0:+.5f} rad  accessor={acc0:+.5f} rad")

    for i in range(N_STEPS):
        world.advance()
        if not world.is_valid():
            raise RuntimeError(f"world invalid after advance at step {i+1}")
        world.retrieve()

    a1 = body_xy_angle(np.asarray(view(geom_T)[0]))
    acc1 = read_free_angle_via_accessor()
    dangle = a1 - a0
    print(f"[after {N_STEPS} steps] free-body angle: geom={a1:+.5f} rad  accessor={acc1:+.5f} rad")
    print(f"[delta] rotated {dangle:+.5f} rad  | accessor agrees: {abs(acc1 - a1) < 1e-6}")

    ok = world.is_valid() and abs(dangle) > 1e-3 and abs(acc1 - a1) < 1e-6
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
