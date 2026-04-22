import numpy as np
from pathlib import Path

traj = np.load('trajectory.npz')
pan  = traj['rigid_pan']
spat = traj['rigid_spatula']

print('rigid_pan frame0:', pan[0].round(4))
print('rigid_spatula frame0:', spat[0].round(4))
print('|pan quat [3:]|:', np.linalg.norm(pan[0, 3:]))
print()
print('Pan pos [0:3]:', pan[0, :3])
print('Spat pos [0:3]:', spat[0, :3])
print()

# Check marvin_gss URDF joint names
import xml.etree.ElementTree as ET
gss_urdf = Path('../../DemoAssets/marvin_gss/marvin_gss.urdf')
if gss_urdf.exists():
    tree = ET.parse(str(gss_urdf))
    joints = [(j.attrib['name'], j.attrib['type'])
              for j in tree.findall('.//joint')
              if j.attrib.get('type') in ('revolute', 'prismatic', 'continuous')]
    print(f'marvin_gss joints ({len(joints)}):')
    for name, jtype in joints:
        print(f'  {name}  [{jtype}]')
else:
    print('marvin_gss.urdf not found')
