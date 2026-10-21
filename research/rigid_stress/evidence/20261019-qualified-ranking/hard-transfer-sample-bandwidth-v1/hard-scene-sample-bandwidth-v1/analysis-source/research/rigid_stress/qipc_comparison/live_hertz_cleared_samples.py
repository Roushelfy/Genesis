"""Paired scene control with the original inactive-sample geometry clearing."""

from .live_hertz_inputs import main
from .live_hertz_scene import SceneObserver


class ClearedSampleObserver(SceneObserver):
    def __init__(self, solver, check_grouping=False):
        super().__init__(solver, check_grouping)
        self.transfer.clear_inactive = True


if __name__ == "__main__":
    main(
        ClearedSampleObserver,
        ("max", "full", "rigid"),
        "Paired original-clearing control for the same native Hertz transfer and coarse max/full scene. Geometry for every inactive sample is cleared; all active sampling, force/moment/facet-cone admission, rigid conditions and elastic output remain unchanged. Proportional shear and unresolved near-field/shell accuracy prevent physical estimator qualification.",
    )
