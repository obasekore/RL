"""Per-frame clamped damped-least-squares (DLS) IK with a null-space rest-pose bias.

Per-iteration update:
    J          = numeric_jacobian(...)                                # (3m, J)
    M          = J @ J.T + damping_lambda**2 * I_3m
    dls_step   = J.T @ solve(M, e)                                     # damped primary task
    J_pinv     = np.linalg.pinv(J)                                      # EXACT (SVD-based) pseudo-inverse
    null_proj  = I_J - J_pinv @ J                                        # exact null-space projector
    bias_step  = rest_pose_gain * (null_proj @ (rest_rad - theta))
    step       = clip(dls_step + bias_step, -max_step_rad, max_step_rad)
    theta      = clip(theta + step, bounds_rad[:,0], bounds_rad[:,1])     # per-iteration limit clamp

The null-space projector deliberately uses the *exact* pseudo-inverse, not the damped
one used for the primary task: reusing the damped J^T(JJ^T+lambda^2 I)^-1 for the
projector leaks a permanent steady-state bias into the primary task and prevents full
convergence (verified independently - damped-everywhere plateaus well above tolerance;
exact-projector converges cleanly). Damped primary task + exact null-space projector is
standard robotics redundancy-resolution practice, not a novel technique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coppelia_rl.retargeting.fk import effector_positions
from coppelia_rl.retargeting.jacobian import numeric_jacobian
from coppelia_rl.skeleton.schema import SkeletonSpec


@dataclass
class FrameIKResult:
    theta_rad: np.ndarray  # (J,), always clamped within bounds_rad
    converged: bool
    unreachable_bones: list[str] = field(default_factory=list)
    iterations: int = 0
    residuals_m: dict[str, float] = field(default_factory=dict)


def _residuals(
    skeleton: SkeletonSpec, dof_names: list[str], theta: np.ndarray, target_bones: list[str], target_vec: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    current = effector_positions(skeleton, dof_names, theta, target_bones).reshape(-1)
    e = target_vec - current
    residuals_m = {bone: float(np.linalg.norm(e[3 * i : 3 * i + 3])) for i, bone in enumerate(target_bones)}
    return e, residuals_m


def solve_frame_ik(
    skeleton: SkeletonSpec,
    dof_names: list[str],
    bounds_rad: np.ndarray,
    rest_rad: np.ndarray,
    targets: dict[str, np.ndarray],  # bone -> (3,), root-relative
    initial_guess_rad: np.ndarray,
    *,
    max_iterations: int = 100,
    tolerance_m: float = 1e-4,
    damping_lambda: float = 0.05,
    rest_pose_gain: float = 0.3,
    max_step_rad: float = 0.5,
) -> FrameIKResult:
    n_dof = len(dof_names)
    theta = np.clip(initial_guess_rad.copy(), bounds_rad[:, 0], bounds_rad[:, 1])
    target_bones = sorted(targets.keys())
    m = len(target_bones)

    if m == 0:
        # No active target this frame (undriven limb, per Doc 4) - decay geometrically
        # toward the rest pose. Also handles the *partial*-target case for free at the
        # per-DOF level via the null-space projector below: an untargeted sibling
        # limb's structurally-zero Jacobian columns get pulled toward rest_rad with no
        # special-casing.
        for _ in range(max_iterations):
            theta = np.clip(theta + rest_pose_gain * (rest_rad - theta), bounds_rad[:, 0], bounds_rad[:, 1])
        return FrameIKResult(theta_rad=theta, converged=True, unreachable_bones=[], iterations=max_iterations)

    target_vec = np.concatenate([targets[bone] for bone in target_bones])
    iterations = 0
    residuals_m: dict[str, float] = {}

    for step_num in range(max_iterations):
        e, residuals_m = _residuals(skeleton, dof_names, theta, target_bones, target_vec)
        iterations = step_num + 1
        if all(r <= tolerance_m for r in residuals_m.values()):
            break

        jac = numeric_jacobian(skeleton, dof_names, theta, target_bones)
        gram = jac @ jac.T + damping_lambda**2 * np.eye(3 * m)
        dls_step = jac.T @ np.linalg.solve(gram, e)

        jac_pinv = np.linalg.pinv(jac)
        null_proj = np.eye(n_dof) - jac_pinv @ jac
        bias_step = rest_pose_gain * (null_proj @ (rest_rad - theta))

        step = np.clip(dls_step + bias_step, -max_step_rad, max_step_rad)
        theta = np.clip(theta + step, bounds_rad[:, 0], bounds_rad[:, 1])
    else:
        # Ran out of iterations without breaking - residuals_m above reflects theta
        # from before the loop's final update; recompute for the theta actually returned.
        _, residuals_m = _residuals(skeleton, dof_names, theta, target_bones, target_vec)

    unreachable_bones = [bone for bone, residual in residuals_m.items() if residual > tolerance_m]
    return FrameIKResult(
        theta_rad=theta,
        converged=len(unreachable_bones) == 0,
        unreachable_bones=unreachable_bones,
        iterations=iterations,
        residuals_m=residuals_m,
    )
