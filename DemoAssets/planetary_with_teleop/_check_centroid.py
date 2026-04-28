import numpy as np

def load_obj(path):
    vs, fs = [], []
    for l in open(path, errors='replace'):
        if l.startswith('v '):
            vs.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '):
            idx=[int(t.split('/')[0])-1 for t in l.split()[1:]]
            if len(idx)==3: fs.append(idx)
            elif len(idx)==4: fs.append(idx[:3]); fs.append([idx[0],idx[2],idx[3]])
    return np.array(vs), np.array(fs)

def vol_centroid(V, F):
    tv=0.0; c=np.zeros(3)
    for t in F:
        v0,v1,v2=V[t[0]],V[t[1]],V[t[2]]
        vol=np.dot(v0,np.cross(v1,v2))/6
        tv+=vol; c+=vol*(v0+v1+v2)/4
    return c/tv

MS = 0.0012
assets = "c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo/DemoAssets/planetary_gear/assets/"

print("=== v1 vs v2 centroid comparison ===")
pairs = [
    ("sun_gear_handle.obj", "sun_gear_handle_v2.obj"),
    ("planet_gear.obj", "planet_gear_v2.obj"),
]
for v1, v2 in pairs:
    V1, F1 = load_obj(assets + v1)
    V2, F2 = load_obj(assets + v2)
    c1 = vol_centroid(V1, F1)
    c2 = vol_centroid(V2, F2)
    print(f"  {v1:25s}  centroid_mm={np.round(c1,3)}  nverts={len(V1)}")
    print(f"  {v2:25s}  centroid_mm={np.round(c2,3)}  nverts={len(V2)}")
    diff_mm = c2 - c1
    print(f"  diff_mm = {np.round(diff_mm, 4)}")
    print(f"  diff_world_m = {np.round(diff_mm * MS, 6)}")
    print()

print("=== Check: does UIPC report centroid or OBJ-origin? ===")
uipc = np.load("c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo/DemoAssets/planetary_with_teleop/ik_sequence_50.npz")

# Planet_0: placement = [0.5, -0.0378, 0.79]
# If OBJ-origin: UIPC[0] Z ~ 0.79 (minus settle)
# If centroid:   UIPC[0] Z ~ 0.79 + 6*0.0012 = 0.7972 (minus settle)
p0_z = uipc["rigid_planet_gear_0"][0, 2]
print(f"  planet_0 UIPC[0] Z = {p0_z:.5f}")
print(f"    if OBJ-origin: expect ~0.790 (- settle)")
print(f"    if centroid:   expect ~0.797 (- settle)")
print(f"    --> looks like {'centroid' if p0_z > 0.793 else 'OBJ-origin'}")

# Carrier: placement = [0.5, 0, 0.7828]
# Carrier centroid Z = -0.6083 mm -> centroid_scaled = -0.00073
# If OBJ-origin: Z ~ 0.7828 (- settle)
# If centroid:   Z ~ 0.7828 + (-0.00073) = 0.78207 (- settle)
c_z = uipc["rigid_carrier"][0, 2]
print(f"  carrier UIPC[0] Z = {c_z:.5f}")
print(f"    if OBJ-origin: expect ~0.7828 (- settle)")
print(f"    if centroid:   expect ~0.7821 (- settle)")
print(f"    --> looks like {'centroid' if c_z < 0.7824 else 'OBJ-origin'}")

# Sun: placement = [0.5, 0, 0.79]
# Sun centroid Z = 8.1667 mm -> centroid_scaled = 0.0098
# If OBJ-origin: Z ~ 0.79 (- settle)
# If centroid:   Z ~ 0.79 + 0.0098 = 0.7998 (- settle)
s_z = uipc["rigid_sun_gear"][0, 2]
print(f"  sun_gear UIPC[0] Z = {s_z:.5f}")
print(f"    if OBJ-origin: expect ~0.790 (- settle)")
print(f"    if centroid:   expect ~0.800 (- settle)")
print(f"    --> looks like {'centroid' if s_z > 0.795 else 'OBJ-origin'}")
