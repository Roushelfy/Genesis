"""Experimental fast cooperative fit with device-masked scalar recovery; no host status reads."""

from traction_original_point_candidate_v10 import kernel_fit_tractions as kernel_fit_cooperative_tractions
from traction_scalar_fallback_candidate_v1 import kernel_fit_failed_tractions


def kernel_fit_tractions(
    contact_state, source_state, traction_state, traction_info, epsilon, fit_tolerance, max_evaluations
):
    kernel_fit_cooperative_tractions(
        contact_state, source_state, traction_state, traction_info, epsilon, fit_tolerance, max_evaluations
    )
    kernel_fit_failed_tractions(
        contact_state, source_state, traction_state, traction_info, epsilon, fit_tolerance, max_evaluations
    )
