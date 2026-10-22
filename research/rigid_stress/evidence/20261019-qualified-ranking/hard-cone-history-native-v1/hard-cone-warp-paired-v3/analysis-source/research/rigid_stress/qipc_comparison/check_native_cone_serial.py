"""Paired serial control for the same tuned cone-memory equations."""

from . import check_native_cone_vectors
from .native_hertz_cone_history import NativeConeHistory


class SerialConeHistory(NativeConeHistory):
    def __init__(self, count, intervals=16, maximum_iterations=2048):
        super().__init__(count, intervals, maximum_iterations, penalty=0.1, cooperative=False)


if __name__ == "__main__":
    check_native_cone_vectors.NativeConeHistory = SerialConeHistory
    check_native_cone_vectors.main()
