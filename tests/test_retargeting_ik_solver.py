from pathlib import Path

import numpy as np

from coppelia_rl.retargeting.fk import dof_bounds_rad, dof_rest_angles_rad, effector_positions
from coppelia_rl.retargeting.ik_solver import solve_frame_ik
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml

_SKELETONS_DIR = Path(__file__).resolve().parents[1] / "skeletons"
_QUADRUPED = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
_QUADRUPED_DOF_NAMES = ordered_dof_names(_QUADRUPED)
_QUADRUPED_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

_BIPED = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
_BIPED_DOF_NAMES = ordered_dof_names(_BIPED)
_BIPED_END_EFFECTORS = ["L_foot", "R_foot", "L_hand", "R_hand"]


def _fk_then_ik_round_trip(skeleton, dof_names, bone_names, seed):
    bounds_rad = dof_bounds_rad(skeleton, dof_names)
    rest_rad = dof_rest_angles_rad(skeleton, dof_names)
    true_theta = rest_rad  # the fixture's own authored rest pose is the ground truth

    target_positions = effector_positions(skeleton, dof_names, true_theta, bone_names)
    targets = dict(zip(bone_names, target_positions))

    rng = np.random.default_rng(seed)
    initial_guess = np.clip(
        true_theta + rng.normal(scale=0.15, size=true_theta.shape), bounds_rad[:, 0], bounds_rad[:, 1]
    )

    result = solve_frame_ik(skeleton, dof_names, bounds_rad, rest_rad, targets, initial_guess)
    return result, true_theta


def test_fk_then_ik_round_trip_quadruped():
    result, true_theta = _fk_then_ik_round_trip(_QUADRUPED, _QUADRUPED_DOF_NAMES, _QUADRUPED_FEET, seed=0)
    assert result.converged
    for residual in result.residuals_m.values():
        assert residual <= 1e-4
    # Position match (above) is the solver's actual contract (Doc 4: "place each
    # target-bearing end-effector at its target position"). Exact angle recovery is
    # NOT guaranteed even for this nominally-fully-determined 3-DOF-per-leg chain -
    # a 3-DOF chain reaching a 3D point can have more than one solution branch, and
    # the null-space bias only "prefers" the rest-pose-closest one, it doesn't
    # guarantee convergence to it within a position-only stopping criterion. This is
    # a coarse sanity bound to catch gross failures (wrong axis convention, broken
    # Jacobian), not a precise-recovery assertion.
    assert np.allclose(result.theta_rad, true_theta, atol=0.1)


def test_fk_then_ik_round_trip_biped():
    result, true_theta = _fk_then_ik_round_trip(_BIPED, _BIPED_DOF_NAMES, _BIPED_END_EFFECTORS, seed=1)
    assert result.converged
    for residual in result.residuals_m.values():
        assert residual <= 1e-4
    # See the quadruped test's comment: position match is the real contract. The
    # biped's per-leg chain (3-DOF hip + 1-DOF knee + 2-DOF ankle = 6 DOF for a 3D
    # foot target) is genuinely redundant, so null-space-direction angle drift is
    # expected and not itself a bug - this is a coarse sanity bound only.
    assert np.allclose(result.theta_rad, true_theta, atol=0.1)


def test_null_space_rest_pose_bias_pulls_toward_rest():
    # Underdetermined case: only L_foot targeted (3-DOF hip + 1-DOF knee driven by a
    # single 3D position target) - many joint-angle solutions reach the same target.
    bounds_rad = dof_bounds_rad(_BIPED, _BIPED_DOF_NAMES)
    rest_rad = dof_rest_angles_rad(_BIPED, _BIPED_DOF_NAMES)
    target = effector_positions(_BIPED, _BIPED_DOF_NAMES, rest_rad, ["L_foot"])[0]
    targets = {"L_foot": target}

    cold_start = np.zeros_like(rest_rad)
    result_bias_on = solve_frame_ik(
        _BIPED, _BIPED_DOF_NAMES, bounds_rad, rest_rad, targets, cold_start, rest_pose_gain=0.3
    )
    result_bias_off = solve_frame_ik(
        _BIPED, _BIPED_DOF_NAMES, bounds_rad, rest_rad, targets, cold_start, rest_pose_gain=0.0
    )

    assert result_bias_on.converged
    assert result_bias_off.converged
    dist_on = np.linalg.norm(result_bias_on.theta_rad - rest_rad)
    dist_off = np.linalg.norm(result_bias_off.theta_rad - rest_rad)
    assert dist_on < dist_off


def test_unreachable_target_flags_without_crashing():
    bounds_rad = dof_bounds_rad(_BIPED, _BIPED_DOF_NAMES)
    rest_rad = dof_rest_angles_rad(_BIPED, _BIPED_DOF_NAMES)
    far_target = {"L_foot": np.array([10.0, 10.0, 10.0])}

    result = solve_frame_ik(_BIPED, _BIPED_DOF_NAMES, bounds_rad, rest_rad, far_target, rest_rad.copy(), max_iterations=50)

    assert result.converged is False
    assert result.unreachable_bones == ["L_foot"]
    assert np.all(result.theta_rad >= bounds_rad[:, 0])
    assert np.all(result.theta_rad <= bounds_rad[:, 1])


def test_solution_never_exceeds_joint_limits_even_when_unreachable():
    bounds_rad = dof_bounds_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    rest_rad = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    # A target requiring an out-of-limit configuration to reach exactly.
    far_target = {"FL_foot": np.array([5.0, 5.0, 5.0])}

    result = solve_frame_ik(
        _QUADRUPED, _QUADRUPED_DOF_NAMES, bounds_rad, rest_rad, far_target, rest_rad.copy(), max_iterations=50
    )

    assert result.converged is False
    assert np.all(result.theta_rad >= bounds_rad[:, 0] - 1e-9)
    assert np.all(result.theta_rad <= bounds_rad[:, 1] + 1e-9)


def test_zero_targets_decays_toward_rest_pose():
    bounds_rad = dof_bounds_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    rest_rad = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    cold_start = np.zeros_like(rest_rad)

    result = solve_frame_ik(_QUADRUPED, _QUADRUPED_DOF_NAMES, bounds_rad, rest_rad, {}, cold_start)

    assert result.converged
    assert result.unreachable_bones == []
    assert np.allclose(result.theta_rad, rest_rad, atol=1e-6)
