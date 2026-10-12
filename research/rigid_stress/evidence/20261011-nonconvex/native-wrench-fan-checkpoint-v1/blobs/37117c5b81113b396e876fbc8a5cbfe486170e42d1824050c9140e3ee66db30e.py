"""Publish small byte-exact original single-source regression inputs from immutable capture files."""
from pathlib import Path
import hashlib
import numpy as np

root = Path(__file__).parent
destination = Path("tests/rigid/assets")
for case, label in (("varied2048", "b2048-623102"), ("varied4096", "b4096-623103")):
    capture = root / f"native-body-failure-capture-v1/{label}/failure/before-rigid-integration.npz"
    with np.load(capture) as saved:
        selected = np.flatnonzero(saved["fit_contacts.valid"][0])
        assert len(selected) == 1
        values = {key: saved[f"fit_contacts.{original}"][0, selected].copy()
                  for key, original in (("anchor", "center"), ("point", "position"), ("force", "force"),
                                        ("radius", "radius"), ("friction", "friction"), ("anchor_face", "anchor_face"))}
        values["moment"] = saved["fit_sources.anchor_moment"][0, selected].copy()
    path = destination / f"nonconvex_{case}_single_source.npz"
    assert not path.exists()
    np.savez_compressed(path, **values)
    with np.load(path) as checked:
        assert all(np.array_equal(checked[key], value) for key, value in values.items())
    print(path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
