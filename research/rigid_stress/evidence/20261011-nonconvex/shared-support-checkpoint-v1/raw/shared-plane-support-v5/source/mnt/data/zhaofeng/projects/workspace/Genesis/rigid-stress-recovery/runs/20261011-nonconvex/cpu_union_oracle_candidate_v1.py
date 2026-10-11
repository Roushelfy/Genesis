"""Independent FP64 convex-dual solve followed by a residual root polish; no native multipliers are inputs."""
import json
import numpy as np
from scipy.optimize import minimize, root
from research.rigid_stress.wrench import WrenchFit, WrenchPatch, project_coulomb
from tests.rigid.test_stress_surface_fan import cpu_dual_energy_gradient, cpu_dual_hessian


def cpu_resolved_wrench_fit(positions, profiles, normals, patch):
    scale = np.linalg.norm(patch.force_n)
    normalized = WrenchPatch(patch.center_m, patch.force_n/scale, patch.radius_m, patch.friction, patch.moment_nm/scale)
    fitter = WrenchFit(positions, profiles, normals, normalized)
    gram = np.einsum('qij,qkj,q->ik', fitter.operator, fitter.operator, fitter.weights)
    initial = np.einsum('qij,j,q->i', fitter.operator, normalized.force_n, fitter.weights)
    initial = np.linalg.solve(gram, initial-fitter.target)
    result = minimize(cpu_dual_energy_gradient, initial, args=(fitter,), jac=True,
        hess=cpu_dual_hessian, method='trust-exact',
        options={'gtol': 1e-11, 'maxiter': 1024, 'initial_trust_radius': 1.0, 'max_trust_radius': 1e9})
    candidates = [result.x]
    diagnostics = [{'method': 'convex-dual-trust-exact', 'success': bool(result.success),
        'message': str(result.message), 'residual': fitter.residual(result.x).tolist()}]
    if np.linalg.norm(fitter.residual(result.x))*fitter.scale > 1e-11:
        polished = root(fitter.residual, result.x, jac=fitter.jacobian,
            method='hybr', options={'xtol': 1e-12, 'maxfev': 4096})
        candidates.append(polished.x)
        diagnostics.append({'method': 'hybr-root-polish', 'success': bool(polished.success),
            'message': str(polished.message), 'residual': fitter.residual(polished.x).tolist()})
    multiplier = min(candidates, key=lambda value: np.linalg.norm(fitter.residual(value)))
    vectors = normalized.force_n-np.einsum('qij,i->qj', fitter.operator, multiplier)
    projected, _ = project_coulomb(vectors, normals, patch.friction)
    force = scale*fitter.weights[:, None]*projected
    mismatch = np.einsum('qij,qj->i', fitter.operator, force)-scale*fitter.target
    force_error, moment_error = (float(np.linalg.norm(mismatch[:3])/scale), float(np.linalg.norm(mismatch[3:])/scale))
    print(json.dumps({'oracle_stages': diagnostics, 'dual_norm': float(np.linalg.norm(multiplier)),
        'force_error': force_error, 'moment_error': moment_error}), flush=True)
    assert force_error <= 1e-8 and moment_error <= 1e-8
    return force
