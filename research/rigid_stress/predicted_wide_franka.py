"""Use a CCD-qualified motion predictor only as the initial Newton iterate."""

import ipctk
import numpy as np

from .wide_coordinate_franka import WideArticulatedContact


class PredictedWideContact(WideArticulatedContact):
    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
        if self.dynamic:
            old_body = np.asarray(body_u, dtype=np.longdouble)
            increment = self.contact.predictor - old_body
            start = self.contact.mesh.vertices(self.contact.rest + previous.reshape(-1, 3))
            accepted = False
            for exponent in range(21):
                alpha = 0.5**exponent
                trial_body = old_body + alpha * increment
                trial_variable = alpha * variable
                trial_q = self.previous_q + np.longdouble(self.dt) * trial_variable
                full = self.pack(trial_body, trial_q)[0]
                end = self.contact.mesh.vertices(self.contact.rest + full.reshape(-1, 3))
                safe = ipctk.compute_collision_free_stepsize(self.contact.mesh, start, end)
                if safe >= 1 - 1e-12 and not ipctk.has_intersections(self.contact.mesh, end):
                    body_u, variable = trial_body, trial_variable
                    accepted = True
                    break
            if not accepted:
                body_u, variable = old_body, np.zeros_like(variable)
            if not hasattr(self, "initial_guess_records"):
                self.initial_guess_records = []
            self.initial_guess_records.append({"phase": phase, "predictor_scale": alpha if accepted else 0.0})
        return super().solve(body_u, variable, previous, hold_body, phase, max_iterations)
