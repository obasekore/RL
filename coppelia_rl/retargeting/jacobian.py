"""Numeric Jacobian of end-effector positions w.r.t. joint angles.

Central-difference, full-FK-recompute-per-perturbed-DOF - simple and correct at the
12-22 DOF scale of the current skeleton fixtures; performance optimization is
explicitly deferred per the retargeting spec's "explicitly deferred" section.
"""

from __future__ import annotations

import numpy as np

from coppelia_rl.retargeting.fk import effector_positions
from coppelia_rl.skeleton.schema import SkeletonSpec

_EPSILON_RAD = 1e-6


def numeric_jacobian(
    skeleton: SkeletonSpec, dof_names: list[str], theta_rad: np.ndarray, target_bones: list[str]
) -> np.ndarray:
    """(3*len(target_bones), len(dof_names)) - row block k*3:(k+1)*3 is target_bones[k]'s
    position derivative w.r.t. each DOF (column)."""
    n_dof = len(dof_names)
    jac = np.zeros((3 * len(target_bones), n_dof))
    for k in range(n_dof):
        theta_plus = theta_rad.copy()
        theta_plus[k] += _EPSILON_RAD
        theta_minus = theta_rad.copy()
        theta_minus[k] -= _EPSILON_RAD

        pos_plus = effector_positions(skeleton, dof_names, theta_plus, target_bones)
        pos_minus = effector_positions(skeleton, dof_names, theta_minus, target_bones)

        jac[:, k] = ((pos_plus - pos_minus) / (2 * _EPSILON_RAD)).reshape(-1)
    return jac
