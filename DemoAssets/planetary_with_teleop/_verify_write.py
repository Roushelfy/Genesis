import numpy as np

uipc = np.load("c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo/DemoAssets/planetary_with_teleop/ik_sequence.npz")
conv = np.load("c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo/DemoAssets/planetary_gear/trajectory_gear_sharpa_objorigin.npz")

n = len(uipc["sim_time"])
print("UIPC frames:", n)
print("sim_time:", uipc["sim_time"][:6])
print()

print("=== Frame-by-frame carrier Z ===")
for i in range(n):
    z = uipc["rigid_carrier"][i, 2]
    print("  UIPC[%d] carrier_Z=%.6f" % (i, z))

print()
print("Genesis[488] carrier_Z=%.6f" % conv["rigid_carrier"][488, 2])

print()
print("=== Byte-exact comparison: UIPC[0] vs Genesis[488] ===")
keys = ["rigid_carrier", "rigid_sun_gear", "rigid_ring_gear",
        "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2"]
for k in keys:
    a = conv[k][488]
    b = uipc[k][0]
    if np.array_equal(a, b):
        print("  %s: EXACT MATCH" % k)
    else:
        diff = b.astype(np.float64) - a.astype(np.float64)
        print("  %s: diff = %s  max_abs=%.2e" % (k, np.round(diff, 8), np.max(np.abs(diff))))
