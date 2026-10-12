"""Exact versus doubled native contact workspace growth after seed reuse and sample-major scheduling."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts

mode = sys.argv.pop(1)
assert mode in ("on", "off")
original_allocate = IntrinsicContacts._allocate_workspace
capacities = {}
growths = []


def selected_allocate(contacts, surface_info, options, n_chunk, n_envs):
    previous = capacities.get(contacts)
    selected = n_chunk
    if mode == "off" and previous is not None:
        selected = min(contacts.raw_sources.status.shape[0], max(n_chunk, 2 * previous))
    original_allocate(contacts, surface_info, options, selected, n_envs)
    capacities[contacts] = selected
    growths.append({"requested_contacts": n_chunk, "allocated_contacts": selected, "environments": n_envs})


with patch.object(IntrinsicContacts, "_allocate_workspace", selected_allocate):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["exact_contact_growth"] = mode == "on"
result["contact_growths"] = growths
result["contact_growth_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
