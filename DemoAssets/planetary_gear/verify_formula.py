import numpy as np
MS = 0.0012
gears = {
    'ring_gear': {'COM_z': 0.79,    'q0': [-0.7071,0,0,0.7071],          'c_z': None,  'fixed': True},
    'sun_gear':  {'COM_z': 0.7998,  'q0': [-0.4937,-0.0793,0.1373,0.8551],'c_z': 8.17, 'fixed': False},
    'carrier':   {'COM_z': 0.78207, 'q0': [-0.7071,0,0,0.7071],           'c_z': -0.61,'fixed': False},
    'planet_0':  {'COM_z': 0.7972,  'q0': [-0.7071,0,0,0.7071],           'c_z': 6.0,  'fixed': False},
    'pin':       {'COM_z': 0.7756,  'q0': [1,0,0,0],                      'c_z': None,  'fixed': True},
}
Rz = np.array([[0,1,0],[-1,0,0],[0,0,1]], float)  # R_z(270)
def qmat(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
expected = {
    'ring_gear': [0.79000, 0.80440],
    'sun_gear':  [0.79000, 0.84038],
    'carrier':   [0.77560, 0.80800],
    'planet_0':  [0.79000, 0.80440],
    'pin':       [0.77560, 0.79960],
}
print('Gear         Zmin_calc  Zmax_calc  Zmin_exp   Zmax_exp   OK')
for k, g in gears.items():
    qw,qx,qy,qz = g['q0']
    R_q0 = qmat(qw,qx,qy,qz)
    combined = R_q0.T @ Rz
    c_z = (g['c_z'] or 0.0) * MS
    obj_zmin = -6*MS if k == 'carrier' else 0.0
    obj_zmax = 21*MS if k == 'carrier' else (42*MS if 'sun' in k else 20*MS if 'pin' in k else 12*MS)
    vl_zmin = (combined @ np.array([0, 0, obj_zmin - c_z]))[2]
    vl_zmax = (combined @ np.array([0, 0, obj_zmax - c_z]))[2]
    zmin = vl_zmin + g['COM_z']
    zmax = vl_zmax + g['COM_z']
    ok = abs(zmin - expected[k][0]) < 0.001 and abs(zmax - expected[k][1]) < 0.001
    print(f'{k:12s}  {zmin:.5f}    {zmax:.5f}   {expected[k][0]:.5f}    {expected[k][1]:.5f}    {"OK" if ok else "FAIL"}')
