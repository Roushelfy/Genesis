"""Every-step audit of the recorded integrated-wrench numerical override."""

import hashlib
import json
import sys
from pathlib import Path

from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from research.rigid_stress import native_validity_probe as audit
from research.rigid_stress.probe_wrench_relief import create_cache, generate


def main():
    command = [sys.executable, *sys.argv]
    module, generated = generate()
    original = RigidStressRecovery.__init__

    def initialize(recovery, *args, **kwargs):
        original(recovery, *args, **kwargs)
        create_cache(module, recovery)

    RigidStressRecovery.__init__ = initialize
    RigidStressRecovery.recover = module.recover
    audit.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.with_suffix(".variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "probe_sha256": hashlib.sha256(Path("research/rigid_stress/probe_wrench_relief.py").read_bytes()).hexdigest(),
                "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "generated_source": generated,
                "additional_immutable_bytes": 288,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
