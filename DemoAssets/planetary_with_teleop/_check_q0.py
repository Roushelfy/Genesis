import numpy as np
from scipy.spatial.transform import Rotation
d = np.load("c:/Users/81946/Projects/GenesisIPCDemo/Genesis_IPC_demo/DemoAssets/planetary_gear/trajectory_gear_sharpa.npz")

fmt = 'ZYX'
for k in ['rigid_ring_gear','rigid_sun_gear','rigid_carrier',
          'rigid_planet_gear_0','rigid_planet_gear_1','rigid_planet_gear_2']:
    q0 = d[k][0, 3:]
    ql = d[k][-1, 3:]
    r0 = Rotation.from_quat([q0[1],q0[2],q0[3],q0[0]])
    rl = Rotation.from_quat([ql[1],ql[2],ql[3],ql[0]])
    diff = (r0.inv() * rl).magnitude() * 180/np.pi
    e0 = r0.as_euler(fmt, degrees=True)
    el = rl.as_euler(fmt, degrees=True)
    print(k)
    print("  frame  0: pos=%s  eulerZ=%.1f" % (np.round(d[k][0,:3],4), e0[0]))
    print("  frame -1: pos=%s  eulerZ=%.1f" % (np.round(d[k][-1,:3],4), el[0]))
    print("  angle(q0, q_last) = %.1f deg" % diff)
    print()
