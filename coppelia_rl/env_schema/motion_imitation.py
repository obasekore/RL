"""Runtime state and DeepMimic-style (Peng et al. 2018) reward/termination logic for
the `motion_imitation` env type. Kept separate from generic_env.py, which stays
focused on the generic schema-driven observation/action/reward/termination dispatch
machinery - this module holds one env type's domain-specific state and math.

DOF/bone -> scene object convention (new, load-bearing for future scene-building
work - no such mapping format exists anywhere in the specs): every name in
`coppelia_rl.skeleton.parser.ordered_dof_names(skeleton)` must be aliased to a joint
object in the scene (a `revolute_2dof`/`revolute_3dof` joint is multiple scene joint
objects, one per axis - matching how the retargeting solver already treats DOFs);
`skeleton.root_bone` and every `skeleton.end_effectors` name must be aliased to a
plain scene object (dummy/shape). Reuses the exact ref="..." alias-resolution every
observation/action already uses.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from coppelia_rl.env_schema.spec import MotionImitationSpec
from coppelia_rl.motion.clip import Clip, open_clip
from coppelia_rl.motion.clip_yaml_parser import parse_clip_yaml
from coppelia_rl.retargeting.fk import end_effector_position, forward_kinematics, quat_to_matrix
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml
from coppelia_rl.skeleton.schema import SkeletonSpec
from coppelia_rl.sim_interface.client import SimClient

if TYPE_CHECKING:
    from coppelia_rl.env_schema.generic_env import XmlDefinedEnv

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# DeepMimic's own kernel-scale constants (k_p, k_v, k_e); not schema-configurable,
# matching both the paper (fixed shape constants) and this schema's existing
# pattern (only `weight` is per-env-tunable, kind-internal shape isn't).
_POSE_KERNEL_SCALE = 2.0
_VELOCITY_KERNEL_SCALE = 0.1
_END_EFFECTOR_KERNEL_SCALE = 40.0
# Not from DeepMimic (that's r_c = center-of-mass tracking, dropped - see
# build_motion_imitation_runtime's module docstring in the build plan). This
# project's own contact-matching term instead penalizes world-space foot slip
# during reference-declared contact, per skeleton-schema-and-frame-convention-SPEC.md's
# "Fixed contact points during stance" note. Chosen so ~0.5 m/s of slip -> ~0.37 reward.
_CONTACT_SLIP_KERNEL_SCALE = 4.0


def _ref_path(ref: str) -> str:
    return ref if ref.startswith("/") else f"/{ref}"


def _quat_angle_difference(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Shortest-arc angle (radians) between two quaternions, handling double-cover
    (q and -q represent the same rotation) via absolute value of the dot product."""
    dot = float(np.clip(abs(np.dot(q1_wxyz, q2_wxyz)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _reference_joint_velocities(clip: Clip) -> np.ndarray:
    if clip.tracks.joint_velocities is not None:
        return clip.tracks.joint_velocities
    return np.gradient(clip.tracks.joint_angles, 1.0 / clip.header.clip.frame_rate, axis=0)


class _MotionImitationRuntime:
    def __init__(self, client: SimClient, clip: Clip, skeleton: SkeletonSpec):
        self.clip = clip
        self.skeleton = skeleton
        self.dof_names = ordered_dof_names(skeleton)
        self.n_frames = clip.tracks.root_pose.shape[0]
        self.loop = clip.header.clip.loop
        self.up_axis_index = _AXIS_INDEX[skeleton.up_axis]
        self.up_axis_vec = np.zeros(3)
        self.up_axis_vec[self.up_axis_index] = 1.0

        self.dof_joints = [client.get_joint(_ref_path(name)) for name in self.dof_names]
        self.root_object = client.get_object(_ref_path(skeleton.root_bone))
        self.ee_objects = {name: client.get_object(_ref_path(name)) for name in skeleton.end_effectors}
        self.contact_by_bone = {c.name: c.contact for c in clip.tracks.contact}
        self.reference_joint_velocities = _reference_joint_velocities(clip)

        self.frame_index = 0
        self.direction = 1
        self.exhausted = False
        self.prev_ee_world: dict[str, np.ndarray] = {}

    def inject_reset_state(self, np_random, rsi: bool) -> None:
        self.frame_index = int(np_random.integers(0, self.n_frames)) if rsi else 0
        self.direction = 1
        self.exhausted = False
        self.prev_ee_world = {}

        angles = self.clip.tracks.joint_angles[self.frame_index]
        for joint, angle in zip(self.dof_joints, angles):
            joint.set_joint_position(float(angle))
        self.root_object.set_pose_wxyz(self.clip.tracks.root_pose[self.frame_index])

    def advance_frame(self) -> None:
        n = self.n_frames
        if self.loop == "cyclic":
            self.frame_index = (self.frame_index + 1) % n
        elif self.loop == "one_shot":
            if self.frame_index + 1 >= n:
                self.exhausted = True
            else:
                self.frame_index += 1
        elif self.loop == "ping_pong":
            next_index = self.frame_index + self.direction
            if next_index >= n or next_index < 0:
                self.direction = -self.direction
                next_index = self.frame_index + self.direction
            self.frame_index = next_index

    # -- live/reference state readers ------------------------------------------

    def live_dof_angles(self) -> np.ndarray:
        return np.array([joint.get_joint_position() for joint in self.dof_joints])

    def live_dof_velocities(self) -> np.ndarray:
        return np.array([joint.get_joint_velocity() for joint in self.dof_joints])

    def reference_dof_angles(self) -> np.ndarray:
        return self.clip.tracks.joint_angles[self.frame_index]

    def reference_dof_velocities(self) -> np.ndarray:
        return self.reference_joint_velocities[self.frame_index]

    def reference_end_effector_positions_root_relative(self) -> dict[str, np.ndarray]:
        angles = dict(zip(self.dof_names, self.reference_dof_angles()))
        transforms = forward_kinematics(self.skeleton, angles)
        return {name: end_effector_position(self.skeleton, transforms, name) for name in self.skeleton.end_effectors}

    def live_end_effector_positions_root_relative(self) -> dict[str, np.ndarray]:
        return {name: obj.get_position(relative_to=self.root_object.handle) for name, obj in self.ee_objects.items()}

    def live_end_effector_positions_world(self) -> dict[str, np.ndarray]:
        return {name: obj.get_position() for name, obj in self.ee_objects.items()}


def build_motion_imitation_runtime(client: SimClient, mi_spec: MotionImitationSpec) -> _MotionImitationRuntime:
    clip_header = parse_clip_yaml(mi_spec.clip_dir / "clip.yaml")
    if clip_header.validation.status == "error":
        raise ValueError(
            f"Clip at {mi_spec.clip_dir} has validation.status='error' and must not be used for "
            "training (motion-representation-format-SPEC.md's cross-mode consistency rule #3)"
        )
    skeleton = parse_skeleton_yaml(clip_header.skeleton.target)
    clip = open_clip(mi_spec.clip_dir, skeleton=skeleton)
    return _MotionImitationRuntime(client, clip, skeleton)


# -- reward terms -----------------------------------------------------------------


def pose_tracking_reward(mi: _MotionImitationRuntime) -> float:
    """DOF angle error + root height error + root orientation error (root x/y
    translation excluded, matching DeepMimic's own convention of not penalizing
    forward-progress drift). Pose error is scalar-per-DOF-axis squared error, not
    DeepMimic's per-body quaternion error - a direct adaptation to this schema's
    explicit-revolute-DOF representation rather than DeepMimic's ball-joint one."""
    theta_err = mi.live_dof_angles() - mi.reference_dof_angles()

    h_live = mi.root_object.get_position()[mi.up_axis_index]
    h_ref = mi.clip.tracks.root_pose[mi.frame_index, mi.up_axis_index]

    q_live = mi.root_object.get_pose_wxyz()[3:]
    q_ref = mi.clip.tracks.root_pose[mi.frame_index, 3:]
    phi_err = _quat_angle_difference(q_live, q_ref)

    pose_error = float(np.sum(theta_err**2)) + (h_live - h_ref) ** 2 + phi_err**2
    return math.exp(-_POSE_KERNEL_SCALE * pose_error)


def velocity_tracking_reward(mi: _MotionImitationRuntime) -> float:
    """DOF angular velocities only - root linear/angular velocity is explicitly
    deferred (needs real quaternion-derivative machinery for comparatively little
    wiring-verification value at this step; root motion is already exercised by
    pose_tracking's height/orientation terms)."""
    omega_err = mi.live_dof_velocities() - mi.reference_dof_velocities()
    return math.exp(-_VELOCITY_KERNEL_SCALE * float(np.sum(omega_err**2)))


def end_effector_tracking_reward(mi: _MotionImitationRuntime) -> float:
    """Reference positions via forward_kinematics from the reference frame's
    joint_angles (mandatory, always computable), not tracks.h5's optional-at-storage
    /end_effectors/*/target curves. Both live and reference are root-relative -
    CoppeliaSim's native get_position(relative_to=root) is exactly fk.py's
    root-relative convention (root transform is identity), no extra transform needed."""
    live = mi.live_end_effector_positions_root_relative()
    reference = mi.reference_end_effector_positions_root_relative()
    error = sum(float(np.sum((live[name] - reference[name]) ** 2)) for name in mi.skeleton.end_effectors)
    return math.exp(-_END_EFFECTOR_KERNEL_SCALE * error)


def contact_matching_reward(mi: _MotionImitationRuntime, step_dt: float) -> float:
    """Gated by the REFERENCE contact channel (never inferred from height, per the
    format spec's explicit warning), penalizing LIVE world-space foot velocity during
    reference-declared stance. Neutral (1.0) during reference swing and on the first
    evaluated step for a bone (no velocity estimate yet)."""
    tracked_bones = [name for name in mi.skeleton.end_effectors if name in mi.contact_by_bone]
    if not tracked_bones:
        return 1.0

    live_world = mi.live_end_effector_positions_world()
    scores = []
    for bone in tracked_bones:
        in_contact = bool(mi.contact_by_bone[bone][mi.frame_index])
        prev = mi.prev_ee_world.get(bone)
        if not in_contact or prev is None:
            scores.append(1.0)
        else:
            velocity = (live_world[bone] - prev) / step_dt
            scores.append(math.exp(-_CONTACT_SLIP_KERNEL_SCALE * float(np.dot(velocity, velocity))))
    mi.prev_ee_world = live_world
    return float(np.mean(scores))


# -- termination --------------------------------------------------------------------


def make_fall_detection_check(mi: _MotionImitationRuntime) -> Callable[["XmlDefinedEnv"], bool]:
    fall_detection = mi.skeleton.fall_detection
    if fall_detection is None:
        raise ValueError(
            f'termination type="fall_detection" requires skeleton {mi.skeleton.name!r} to declare a '
            "fall_detection block"
        )

    def check(env: "XmlDefinedEnv") -> bool:
        height = mi.root_object.get_position()[mi.up_axis_index]
        if height < fall_detection.min_root_height:
            return True
        rotation = quat_to_matrix(mi.root_object.get_pose_wxyz()[3:])
        tilt_cos = float(np.clip(float(np.dot(rotation @ mi.up_axis_vec, mi.up_axis_vec)), -1.0, 1.0))
        tilt_deg = math.degrees(math.acos(tilt_cos))
        return tilt_deg > fall_detection.max_tilt_deg

    return check
