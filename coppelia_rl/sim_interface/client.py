"""Connection + synchronous stepping wrapper around CoppeliaSim's ZMQ remote API.

`SimClient` is the only place, besides objects.py, allowed to hold a reference
to the raw `sim` remote object. Everything above the Communication Layer must
go through the factory methods here instead of importing the ZMQ client itself.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from coppelia_rl.sim_interface.objects import Joint, Sensor, SceneObject, Signal, VisionSensor

_PRIMITIVE_SHAPE_KINDS = {
    "cuboid": "primitiveshape_cuboid",
    "sphere": "primitiveshape_spheroid",
    "cylinder": "primitiveshape_cylinder",
    "cone": "primitiveshape_cone",
}

# Friction has no engine-agnostic API in CoppeliaSim - each physics engine
# exposes its own float param(s) via the (deprecated-tagged, but still the
# only working call - see sim_interface/client.py's set_friction docstring)
# sim.setEngineFloatParam(paramId, objectHandle, value). Support matrix:
# engine constant name -> list of per-engine friction param constant names to
# set to the same value. Vortex exposes ~5 axis-specific friction params;
# only the primary linear-axis one is mapped here (partial support).
_FRICTION_PARAMS_BY_ENGINE = {
    "physics_bullet": ["bullet_body_friction"],
    "physics_ode": ["ode_body_friction"],
    "physics_newton": ["newton_body_kineticfriction", "newton_body_staticfriction"],
    "physics_vortex": ["vortex_body_primlinearaxisfriction"],
    "physics_mujoco": ["mujoco_body_friction1"],
}


class UnsupportedPhysicsEngineError(RuntimeError):
    pass


class SimClient:
    def __init__(self, remote_client: RemoteAPIClient, sim, *, scene_load_settle_time: float = 0.5):
        self._client = remote_client
        self._sim = sim
        # sim.loadScene() empirically returns before the engine has actually
        # finished swapping the scene in (it applies on a later internal
        # tick) - a short settle delay avoids resolving refs against a scene
        # that's still mid-load. Not needed against a FakeSim test double
        # (which applies loadScene synchronously/no-op), hence configurable.
        self._scene_load_settle_time = scene_load_settle_time

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 23000) -> "SimClient":
        remote_client = RemoteAPIClient(host=host, port=port)
        sim = remote_client.require("sim")
        return cls(remote_client, sim)

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._sim.getSimulationState() != self._sim.simulation_stopped:
            self.stop_simulation()

    # -- stepping / simulation lifecycle 

    def set_stepping(self, enabled: bool = True) -> None:
        self._sim.setStepping(enabled)

    def start_simulation(self) -> None:
        self._sim.startSimulation()

    def stop_simulation(self) -> None:
        self._sim.stopSimulation()
        while self._sim.getSimulationState() != self._sim.simulation_stopped:
            pass

    def step(self) -> None:
        self._sim.step()

    # -- scene / model management 

    def load_scene(self, path: str | Path) -> None:
        self._sim.loadScene(str(path))
        if self._scene_load_settle_time > 0:
            time.sleep(self._scene_load_settle_time)
        self._strip_embedded_scripts(self._sim.handle_scene)

    def save_scene(self, path: str | Path) -> None:
        self._sim.saveScene(str(path))

    def close_scene(self) -> None:
        """Closes the current scene and switches/creates a fresh, empty one.

        Must be called before procedurally building a new scene from scratch -
        sim.loadModel()/createDummy() etc. add to whatever scene is currently
        open, they don't start a new one.
        """
        self._sim.closeScene()

    def load_model(self, path: str | Path) -> SceneObject:
        handle = self._sim.loadModel(str(path))
        self._strip_embedded_scripts(handle)
        return SceneObject(self._sim, handle)

    def _strip_embedded_scripts(self, base_handle: int) -> None:
        """Removes any embedded child scripts under base_handle.

        Stock CoppeliaSim models (UR5.ttm, Pioneer p3dx.ttm, ...) ship with
        their own demo child scripts. This project drives every object
        externally over this same API, so a leftover embedded script is
        always dead weight - and throws errors once code elsewhere renames
        the objects it expects to find by hardcoded name. Idempotent: a
        no-op on scenes/models already stripped.
        """
        handles = self._sim.getObjectsInTree(base_handle, self._sim.sceneobject_script)
        if handles:
            self._sim.removeObjects(handles)

    def create_dummy(self, size: float = 0.01) -> SceneObject:
        return SceneObject(self._sim, self._sim.createDummy(size))

    def get_joints_in_tree(self, base: SceneObject) -> list[Joint]:
        handles = self._sim.getObjectsInTree(base.handle, self._sim.sceneobject_joint)
        return [Joint(self._sim, h) for h in handles]

    def get_child_dummies(self, parent: SceneObject) -> list[SceneObject]:
        handles = self._sim.getObjectsInTree(parent.handle, self._sim.sceneobject_dummy, 1 | 2)
        return [SceneObject(self._sim, h) for h in handles]

    def create_primitive_shape(self, kind: str, sizes: list[float]) -> SceneObject:
        if kind not in _PRIMITIVE_SHAPE_KINDS:
            raise ValueError(f"Unknown primitive shape kind {kind!r}, expected one of {sorted(_PRIMITIVE_SHAPE_KINDS)}")
        primitive_type = getattr(self._sim, _PRIMITIVE_SHAPE_KINDS[kind])
        return SceneObject(self._sim, self._sim.createPrimitiveShape(primitive_type, list(sizes)))

    def create_vision_sensor(
        self,
        resolution: tuple[int, int],
        perspective_angle_deg: float = 60.0,
        near_clip: float = 0.01,
        far_clip: float = 3.0,
    ) -> VisionSensor:
        width, height = resolution
        int_params = [width, height, 0, 0]
        float_params = [near_clip, far_clip, math.radians(perspective_angle_deg), 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        options = 2  # perspective mode
        handle = self._sim.createVisionSensor(options, int_params, float_params)
        return VisionSensor(self._sim, handle)

    def check_collision(self, entity1: SceneObject, entity2: SceneObject) -> bool:
        result, _ = self._sim.checkCollision(entity1.handle, entity2.handle)
        return bool(result)

    # -- dynamics / appearance (for coppelia_rl.randomization)

    def set_mass(self, shape: SceneObject, mass: float) -> None:
        self._sim.setShapeMass(shape.handle, mass)

    def set_friction(self, shape: SceneObject, value: float) -> None:
        """Sets friction on `shape` for whichever physics engine is active.

        Friction has no engine-agnostic setter in CoppeliaSim - see the
        support matrix in `_FRICTION_PARAMS_BY_ENGINE` above. Raises
        UnsupportedPhysicsEngineError (never silently no-ops) if the active
        engine isn't in that matrix.
        """
        engine = self._sim.getInt32Param(self._sim.intparam_dynamic_engine)
        for engine_name, param_names in _FRICTION_PARAMS_BY_ENGINE.items():
            if engine == getattr(self._sim, engine_name):
                for param_name in param_names:
                    self._sim.setEngineFloatParam(getattr(self._sim, param_name), shape.handle, value)
                return
        raise UnsupportedPhysicsEngineError(
            f"No friction support mapped for the active physics engine (id={engine}). "
            f"Supported: {sorted(_FRICTION_PARAMS_BY_ENGINE)}"
        )

    def set_texture(self, shape: SceneObject, file_path: str | Path) -> None:
        """Applies the image at `file_path` as `shape`'s texture.

        sim.createTexture() always creates a throwaway planar shape as a side
        effect of registering the texture - it's discarded immediately, only
        the texture id is kept, then applied onto the real target shape.
        """
        throwaway_shape, texture_id, _res = self._sim.createTexture(str(file_path), 0)
        self._sim.removeObjects([throwaway_shape])
        self._sim.setShapeTexture(shape.handle, texture_id, self._sim.texturemap_plane, 0, [1.0, 1.0])

    # -- object-model factories

    def get_object(self, path: str) -> SceneObject:
        return SceneObject.from_path(self._sim, path)

    def get_joint(self, path: str) -> Joint:
        return Joint(self._sim, self._sim.getObject(path))

    def get_sensor(self, path: str) -> Sensor:
        return Sensor(self._sim, self._sim.getObject(path))

    def get_vision_sensor(self, path: str) -> VisionSensor:
        return VisionSensor(self._sim, self._sim.getObject(path))

    def get_signal(self, name: str) -> Signal:
        return Signal(self._sim, name)
