import numpy as np, os

base = "c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo"
candidates = [
    os.path.join(base, "DemoAssets/planetary_with_teleop/ik_sequence_50.npz"),
    os.path.join(base, "ik_sequence_50.npz"),
]

p2_path = None
for p in candidates:
    if os.path.exists(p):
        p2_path = p
        break

if p2_path is None:
    print("ik_sequence_50.npz not found!")
    raise SystemExit(1)

d = np.load(p2_path)
n = len(d["sim_time"])
print(f"Phase 2: {p2_path}  ({n} frames)")

conv = np.load(os.path.join(base, "DemoAssets/planetary_gear/trajectory_gear_sharpa_objorigin.npz"))

keys = ["rigid_ring_gear", "rigid_sun_gear", "rigid_carrier",
        "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2"]

print("\n=== Phase 1 last 3 frames ===")
for k in keys:
    if k not in conv:
        continue
    for i in [-3, -2, -1]:
        pos = conv[k][i, :3]
        print(f"  P1[{i:3d}] {k:30s} pos={np.round(pos, 5)}")

print("\n=== Phase 2 first 5 frames ===")
for k in keys:
    if k not in d:
        continue
    for i in range(min(5, n)):
        pos = d[k][i, :3]
        print(f"  P2[{i:3d}] {k:30s} pos={np.round(pos, 5)}")

print("\n=== Jump: P1[-1] -> P2[0] ===")
for k in keys:
    if k not in conv or k not in d:
        continue
    p1_last = conv[k][-1, :3]
    p2_first = d[k][0, :3]
    diff = p2_first - p1_last
    print(f"  {k:30s}  P1[-1]_Z={p1_last[2]:.5f}  P2[0]_Z={p2_first[2]:.5f}  dZ={diff[2]:.5f}  |d|={np.linalg.norm(diff):.5f}")
