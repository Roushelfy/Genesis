"""Load preserved historical scripts only in independent validation commands."""

import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference"
sys.path[:0] = [
    str(REFERENCE / "output"),
    str(REFERENCE / "output/arbitrary_contact_cpu"),
    str(REFERENCE / "output/smooth_friction_cpu"),
    str(REFERENCE / "tmp/batch_history_validation"),
]

import arbitrary_contact_test as reference  # noqa: E402, F401
from general_peak import baseline_numpy  # noqa: E402, F401
