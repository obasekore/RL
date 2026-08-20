"""A minimal in-memory stand-in for CoppeliaSim's `sim` remote object.

Implements just the subset of the `sim.*` surface that sim_interface/ and
envs/ call, so the object model and ReachEnv can be exercised without a
running CoppeliaSim instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class FakeSim:
    simulation_stopped = 0
    simulation_running = 1
    sceneobject_joint = "joint"
    sceneobject_dummy = "dummy"
    sceneobject_script = "script"
    handle_scene = 0
    primitiveshape_cuboid = "cuboid"
    primitiveshape_spheroid = "sphere"
    primitiveshape_cylinder = "cylinder"
    primitiveshape_cone = "cone"

    # -- domain randomization: physics engine identification + friction params
    intparam_dynamic_engine = "dynamic_engine"
    physics_bullet = "bullet"
    physics_ode = "ode"
    physics_newton = "newton"
    physics_vortex = "vortex"
    physics_mujoco = "mujoco"
    bullet_body_friction = "bullet_body_friction"
    ode_body_friction = "ode_body_friction"
    newton_body_kineticfriction = "newton_body_kineticfriction"
    newton_body_staticfriction = "newton_body_staticfriction"
    vortex_body_primlinearaxisfriction = "vortex_body_primlinearaxisfriction"
    mujoco_body_friction1 = "mujoco_body_friction1"
    texturemap_plane = "plane"

    # -- joint dynamic control mode
    jointintparam_dynctrlmode = "dynctrlmode"
    jointdynctrl_free = "free"
    jointdynctrl_force = "force"
    jointdynctrl_velocity = "velocity"
    jointdynctrl_position = "position"
    jointdynctrl_spring = "spring"
    jointdynctrl_callback = "callback"

    def __init__(self):
        self._next_handle = 1
        self._next_texture_id = 0
        self.objects: dict[int, dict] = {}
        self.aliases: dict[str, int] = {}
        self.state = self.simulation_stopped
        self.stepping = False
        self.step_count = 0
        self.calls: list[tuple] = []
        self._int32_signals: dict[str, int] = {}
        self._float_signals: dict[str, float] = {}
        self._string_signals: dict[str, str] = {}
        self.collisions: dict[frozenset, bool] = {}
        self.dynamic_engine = self.physics_bullet

    def _record(self, name, *args):
        self.calls.append((name, args))

    def _new_handle(self, kind: str, **fields) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self.objects[handle] = {
            "kind": kind,
            "position": np.zeros(3),
            "orientation": np.zeros(3),
            "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "children": [],
            **fields,
        }
        return handle

    # -- object pose --------------------------------------------------

    def getObjectPosition(self, handle, relativeTo):
        self._record("getObjectPosition", handle, relativeTo)
        return list(self.objects[handle]["position"])

    def setObjectPosition(self, handle, relativeTo, position):
        self._record("setObjectPosition", handle, relativeTo, position)
        self.objects[handle]["position"] = np.array(position, dtype=float)

    def getObjectOrientation(self, handle, relativeTo):
        return list(self.objects[handle]["orientation"])

    def setObjectOrientation(self, handle, relativeTo, orientation):
        self.objects[handle]["orientation"] = np.array(orientation, dtype=float)

    def getObjectPose(self, handle, relativeTo):
        return list(self.objects[handle]["pose"])

    def setObjectPose(self, handle, relativeTo, pose):
        self.objects[handle]["pose"] = np.array(pose, dtype=float)

    def setObjectAlias(self, handle, alias):
        self.objects[handle]["alias"] = alias
        self.aliases[f"/{alias}"] = handle

    def getObject(self, path):
        return self.aliases[path]

    # -- joints ---------------------------------------------------------

    def getJointPosition(self, handle):
        return self.objects[handle].get("joint_position", 0.0)

    def getJointVelocity(self, handle):
        return self.objects[handle].get("joint_velocity", 0.0)

    def setJointTargetVelocity(self, handle, velocity):
        self._record("setJointTargetVelocity", handle, velocity)
        self.objects[handle]["joint_target_velocity"] = velocity

    def setJointTargetPosition(self, handle, position):
        self.objects[handle]["joint_target_position"] = position

    def setJointTargetForce(self, handle, force):
        self.objects[handle]["joint_target_force"] = force

    def getJointInterval(self, handle):
        return False, [-3.14159, 6.28318]

    def setObjectInt32Param(self, handle, param_id, value):
        self._record("setObjectInt32Param", handle, param_id, value)
        self.objects[handle][param_id] = value

    def getObjectInt32Param(self, handle, param_id):
        return self.objects[handle].get(param_id, 0)

    # -- sensors ----------------------------------------------------------

    def readProximitySensor(self, handle):
        return 1, 0.5, [0.0, 0.0, 0.5], -1, [0.0, 0.0, 1.0]

    def readForceSensor(self, handle):
        return 1, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]

    def setExplicitHandling(self, handle, flags):
        self._record("setExplicitHandling", handle, flags)
        self.objects[handle]["explicit_handling"] = flags

    def getExplicitHandling(self, handle):
        return self.objects[handle].get("explicit_handling", 0)

    def handleVisionSensor(self, handle):
        if not self.objects[handle].get("explicit_handling", 0) & 1:
            raise RuntimeError(f"{handle}: object not tagged for explicit handling")
        self._record("handleVisionSensor", handle)
        self.objects[handle]["vision_handled"] = True

    def getVisionSensorImg(self, handle, *args, **kwargs):
        if not self.objects[handle].pop("vision_handled", False):
            raise RuntimeError("getVisionSensorImg called without handleVisionSensor first")
        width, height = 4, 4
        return bytes(width * height * 3), [width, height]

    def getVisionSensorDepth(self, handle, *args, **kwargs):
        if not self.objects[handle].pop("vision_handled", False):
            raise RuntimeError("getVisionSensorDepth called without handleVisionSensor first")
        width, height = 4, 4
        return np.zeros(width * height, dtype=np.float32).tobytes(), [width, height]

    # -- signals ----------------------------------------------------------

    def getInt32Signal(self, name):
        return self._int32_signals.get(name)

    def setInt32Signal(self, name, value):
        self._int32_signals[name] = value

    def getFloatSignal(self, name):
        return self._float_signals.get(name)

    def setFloatSignal(self, name, value):
        self._float_signals[name] = value

    def getStringSignal(self, name):
        return self._string_signals.get(name)

    def setStringSignal(self, name, value):
        self._string_signals[name] = value

    def clearInt32Signal(self, name):
        self._int32_signals.pop(name, None)

    def clearFloatSignal(self, name):
        self._float_signals.pop(name, None)

    def clearStringSignal(self, name):
        self._string_signals.pop(name, None)

    # -- simulation lifecycle ----------------------------------------------

    def getSimulationState(self):
        return self.state

    def setStepping(self, enabled):
        self.stepping = enabled

    def startSimulation(self):
        self.state = self.simulation_running

    def stopSimulation(self):
        self.state = self.simulation_stopped

    def step(self):
        self._record("step")
        self.step_count += 1
        for obj in self.objects.values():
            if obj["kind"] == "joint":
                velocity = obj.get("joint_target_velocity", 0.0)
                obj["joint_velocity"] = velocity
                obj["joint_position"] = obj.get("joint_position", 0.0) + velocity * 0.05

    # -- scene / model management -------------------------------------------

    def loadScene(self, path):
        self._record("loadScene", path)

    def saveScene(self, path):
        Path(path).touch()

    def closeScene(self):
        self._record("closeScene")
        self.objects.clear()
        self.aliases.clear()
        return 0

    def loadModel(self, path):
        base = self._new_handle("model")
        joint_handles = []
        for _ in range(6):
            joint = self._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
            self.objects[base]["children"].append(joint)
            joint_handles.append(joint)
        tip = self._new_handle("dummy")
        self.objects[joint_handles[-1]]["children"].append(tip)
        return base

    def createDummy(self, size):
        return self._new_handle("dummy")

    def createPrimitiveShape(self, primitive_type, sizes, options=0):
        return self._new_handle("shape", primitive_type=primitive_type, sizes=list(sizes))

    def createVisionSensor(self, options, int_params, float_params):
        width, height = int_params[0], int_params[1]
        return self._new_handle("vision", resolution=(width, height))

    def removeObjects(self, handles, delayedRemoval=False):
        self._record("removeObjects", handles)
        for handle in handles:
            obj = self.objects.pop(handle, None)
            if obj is not None:
                alias = obj.get("alias")
                if alias is not None:
                    self.aliases.pop(f"/{alias}", None)
        for obj in self.objects.values():
            obj["children"] = [h for h in obj["children"] if h not in handles]

    def setObjectParent(self, handle, parent_handle, keep_in_place):
        for obj in self.objects.values():
            if handle in obj["children"]:
                obj["children"].remove(handle)
        if parent_handle != -1:
            self.objects[parent_handle]["children"].append(handle)

    def checkCollision(self, entity1_handle, entity2_handle):
        return int(self.collisions.get(frozenset({entity1_handle, entity2_handle}), False)), []

    # -- domain randomization: mass / friction / texture ---------------------

    def setShapeMass(self, handle, mass):
        self._record("setShapeMass", handle, mass)
        self.objects[handle]["mass"] = mass

    def getInt32Param(self, param_id):
        if param_id == self.intparam_dynamic_engine:
            return self.dynamic_engine
        return 0

    def setEngineFloatParam(self, param_id, handle, value):
        self._record("setEngineFloatParam", param_id, handle, value)
        self.objects[handle].setdefault("engine_params", {})[param_id] = value

    def createTexture(self, fileName, options, planeSizes=None, scalingUV=None, xy_g=None, fixedResolution=0, resolution=None):
        self._record("createTexture", fileName, options)
        throwaway_shape = self._new_handle("shape")
        self._next_texture_id += 1
        return throwaway_shape, self._next_texture_id, [512, 512]

    def setShapeTexture(self, handle, textureId, mappingMode, options, uvScaling, position=None, orientation=None):
        self._record("setShapeTexture", handle, textureId)
        self.objects[handle]["texture_id"] = textureId

    def getObjectsInTree(self, base_handle, object_type=None, options=0):
        only_first_children = bool(options & 2)
        exclude_base = bool(options & 1)

        if base_handle == self.handle_scene:
            all_children = {child for obj in self.objects.values() for child in obj["children"]}
            roots = [h for h in self.objects if h not in all_children]
            results = []
            for root in roots:
                results.extend(self.getObjectsInTree(root, object_type, options & ~1))
            return results

        if only_first_children:
            return [
                h
                for h in self.objects[base_handle]["children"]
                if object_type is None or self.objects[h]["kind"] == object_type
            ]

        results = []

        def _walk(handle, include_self):
            obj = self.objects[handle]
            if include_self and (object_type is None or obj["kind"] == object_type):
                results.append(handle)
            for child in obj["children"]:
                _walk(child, True)

        _walk(base_handle, not exclude_base)
        return results


@pytest.fixture
def fake_sim():
    return FakeSim()


@pytest.fixture
def fake_sim_factory():
    """For tests needing several independent simulated CoppeliaSim instances
    (e.g. a vectorized env's N sub-envs), each with its own object state."""
    return FakeSim
