"""Record installed IPC collision constructors and force-magnitude methods before extending a reference."""

import argparse
import json
from pathlib import Path

import ipctk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    classes = [
        ipctk.VertexVertexNormalCollision,
        ipctk.EdgeVertexNormalCollision,
        ipctk.EdgeEdgeNormalCollision,
        ipctk.FaceVertexNormalCollision,
    ]
    result = {cls.__name__: cls.__init__.__doc__ for cls in classes}
    result["force_magnitude"] = ipctk.BarrierPotential.force_magnitude.__doc__
    result["compute_distance"] = ipctk.EdgeEdgeNormalCollision.compute_distance.__doc__
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
