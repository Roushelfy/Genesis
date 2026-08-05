import math
from typing import Annotated

from typing_extensions import Self

from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import PositiveFloat, ValidFloat

from .cloth import Cloth


class SealedGasShell(Cloth):
    """Thin elastic shell containing sealed ideal gas, simulated by QIPC.

    The shell mesh must be a closed, consistently wound triangle manifold. Its
    authored position at `scene.build()` defines the initial gas volume; a
    separate elastic rest geometry, when configured, does not change that
    volume.

    Args:
        p_gauge0: Initial pressure relative to `p_atm` in Pa. Defaults to 0.
        p_atm: Ambient absolute pressure in Pa. Defaults to 101325.
        gamma: Polytropic exponent. Use 1 for isothermal gas. Defaults to 1.
        v_min_rel: Solver volume floor as a fraction of the authored initial
            gas volume. Defaults to 1e-4.
        auto_flip: Accept a consistently inward-wound mesh without changing its
            shared topology. Defaults to True.

    This material requires `QIPCCouplerOptions`. Gas state can be adjusted after
    build with `FEMEntity.set_gas_state`.
    """

    p_gauge0: ValidFloat = 0.0
    p_atm: PositiveFloat = 101325.0
    gamma: PositiveFloat = 1.0
    v_min_rel: Annotated[ValidFloat, Field(gt=0.0, lt=1.0)] = 1e-4
    auto_flip: StrictBool = True

    @model_validator(mode="after")
    def _validate_initial_absolute_pressure(self) -> Self:
        p0 = self.p_atm + self.p_gauge0
        if not math.isfinite(p0) or p0 <= 0.0:
            gs.raise_exception("SealedGasShell requires p_atm + p_gauge0 to be finite and positive.")
        return self
