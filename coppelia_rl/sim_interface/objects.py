"""Stable object model wrapping CoppeliaSim's `sim.*` regular API.

This is the only module (besides client.py) allowed to call `sim.*` directly.
Everything above the Communication Layer talks to these classes instead.
"""

from __future__ import annotations

import numpy as np


class SceneObject:
    """A handle to any CoppeliaSim scene object, plus generic pose accessors."""

    def __init__(self, sim, handle: int):
        self._sim = sim
        self.handle = handle

    @classmethod
    def from_path(cls, sim, path: str) -> "SceneObject":
        return cls(sim, sim.getObject(path))

    def get_position(self, relative_to: int = -1) -> np.ndarray:
        return np.array(self._sim.getObjectPosition(self.handle, relative_to), dtype=np.float64)

    def set_position(self, position, relative_to: int = -1) -> None:
        self._sim.setObjectPosition(self.handle, relative_to, list(position))

    def get_orientation(self, relative_to: int = -1) -> np.ndarray:
        return np.array(self._sim.getObjectOrientation(self.handle, relative_to), dtype=np.float64)

    def set_orientation(self, orientation, relative_to: int = -1) -> None:
        self._sim.setObjectOrientation(self.handle, relative_to, list(orientation))

    def get_pose(self, relative_to: int = -1) -> np.ndarray:
        return np.array(self._sim.getObjectPose(self.handle, relative_to), dtype=np.float64)

    def set_pose(self, pose, relative_to: int = -1) -> None:
        self._sim.setObjectPose(self.handle, relative_to, list(pose))

    def set_name(self, alias: str) -> None:
        self._sim.setObjectAlias(self.handle, alias)

    def set_parent(self, parent: "SceneObject | None", keep_in_place: bool = True) -> None:
        parent_handle = parent.handle if parent is not None else -1
        self._sim.setObjectParent(self.handle, parent_handle, keep_in_place)


_JOINT_DYNCTRL_MODES = ("free", "force", "velocity", "position", "spring", "callback")


class Joint(SceneObject):
    """A revolute or prismatic joint.

    Joint-space accessors are named ``*_joint_*`` to avoid colliding with the
    inherited Cartesian ``get_position``/``set_position`` from SceneObject.
    """

    def get_joint_position(self) -> float:
        return self._sim.getJointPosition(self.handle)

    def get_joint_velocity(self) -> float:
        return self._sim.getJointVelocity(self.handle)

    def set_target_velocity(self, velocity: float) -> None:
        self._sim.setJointTargetVelocity(self.handle, velocity)

    def set_target_position(self, position: float) -> None:
        self._sim.setJointTargetPosition(self.handle, position)

    def set_target_force(self, force: float) -> None:
        self._sim.setJointTargetForce(self.handle, force)

    def set_control_mode(self, mode: str) -> None:
        """Sets the joint's dynamic control mode (`sim.jointdynctrl_*`).

        Confirmed live and load-bearing: `set_target_velocity()` is silently
        near-inert unless the joint is actually in "velocity" mode - the
        stock UR5.ttm's joints default to "position" mode, so every
        joint_velocity action in this project was barely moving the arm
        until this is set. Called once per joint from generic_env.py,
        matched to whatever action kind (`joint_velocity`/`joint_position`)
        the schema declares for it.
        """
        if mode not in _JOINT_DYNCTRL_MODES:
            raise ValueError(f"Unknown joint control mode {mode!r}, expected one of {_JOINT_DYNCTRL_MODES}")
        param_id = getattr(self._sim, f"jointdynctrl_{mode}")
        self._sim.setObjectInt32Param(self.handle, self._sim.jointintparam_dynctrlmode, param_id)

    def get_interval(self) -> tuple[bool, float, float]:
        """Returns (cyclic, min, range). Max allowed value is min + range."""
        cyclic, interval = self._sim.getJointInterval(self.handle)
        return bool(cyclic), float(interval[0]), float(interval[1])


class Sensor(SceneObject):
    """A proximity or force sensor."""

    def read_proximity(self):
        """Returns (detected: bool, distance: float, point: np.ndarray, detected_handle: int, normal: np.ndarray)."""
        result, dist, point, obj, normal = self._sim.readProximitySensor(self.handle)
        return bool(result), float(dist), np.array(point, dtype=np.float64), obj, np.array(normal, dtype=np.float64)

    def read_force(self):
        """Returns (available: bool, force: np.ndarray, torque: np.ndarray)."""
        result, force, torque = self._sim.readForceSensor(self.handle)
        available = bool(result & 1)
        return available, np.array(force, dtype=np.float64), np.array(torque, dtype=np.float64)


class Signal:
    """A named signal, used for cross-script/episode event communication."""

    def __init__(self, sim, name: str):
        self._sim = sim
        self.name = name

    def get_int32(self):
        return self._sim.getInt32Signal(self.name)

    def set_int32(self, value: int) -> None:
        self._sim.setInt32Signal(self.name, value)

    def get_float(self):
        return self._sim.getFloatSignal(self.name)

    def set_float(self, value: float) -> None:
        self._sim.setFloatSignal(self.name, value)

    def get_string(self):
        return self._sim.getStringSignal(self.name)

    def set_string(self, value: str) -> None:
        self._sim.setStringSignal(self.name, value)

    def clear(self) -> None:
        self._sim.clearInt32Signal(self.name)
        self._sim.clearFloatSignal(self.name)
        self._sim.clearStringSignal(self.name)
