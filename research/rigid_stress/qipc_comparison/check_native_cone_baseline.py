"""Same vector histories and resident benchmark with the original ADMM penalty."""

from . import check_native_cone_vectors
from .native_hertz_cone_history import NativeConeHistory


class BaselineConeHistory(NativeConeHistory):
    def __init__(self, count, intervals=16, maximum_iterations=2048):
        super().__init__(count, intervals, maximum_iterations, penalty=1.0, cooperative=False)


if __name__ == "__main__":
    check_native_cone_vectors.NativeConeHistory = BaselineConeHistory
    check_native_cone_vectors.main()
