"""Schedule public manual observations after the workload's input mutations."""


def sampled(tick, stride):
    """Whether a zero-based workload tick requests a fresh stress observation."""
    return (tick + 1) % stride == 0


class ObservationCadence:
    def __init__(self, workload, stride):
        if stride < 1:
            raise ValueError("Observation stride must be positive")
        self.workload = workload
        self.stride = stride
        self.requests = 0
        if workload.stress and stride > 1:
            if workload.link.stress_options.update_mode != "manual":
                raise ValueError("Sparse observation requires pre-build manual mode")
            workload.scene.register_pre_substep_callback(self.before_substep)

    def before_substep(self, substep):
        # FrankaEgg.step applies controls, changes radii and performs any
        # partial reset before scene.step. Request only after those inputs.
        if substep == 0 and sampled(self.workload.tick, self.stride):
            self.workload.link.request_stress_update()
            self.requests += 1
