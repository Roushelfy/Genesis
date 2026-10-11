"""Publish the exact first failed original component as a normal saved contact regression."""
from pathlib import Path
import json
import hashlib
import numpy as np

root=Path(__file__).parent
original=root/'b1024-stage-failure-v1/results/first-error.npz'
destination=Path('tests/rigid/assets/nonconvex_mesh64_tick517.npz')
assert not destination.exists()
with np.load(original) as saved:
    assert np.all(saved['chunk_sources.free_moment'][0,0]==0)
    np.savez_compressed(destination,position=saved['chunk_contacts.position'][0,0],
                        anchor=saved['chunk_contacts.center'][0,0],force=saved['chunk_contacts.force'][0,0],
                        radius=saved['chunk_contacts.radius'][0,0],friction=saved['chunk_contacts.friction'][0,0])
print(json.dumps({'original':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                  'destination':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}),flush=True)
