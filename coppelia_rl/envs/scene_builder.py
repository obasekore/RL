"""Procedurally builds (or reloads) the reach-task scene.

No hand-authored `.ttt` scene is checked in: the scene is assembled once
through the object model (load the stock UR5 model, add a target dummy) and
then saved, so later runs and the future GUI editor can just load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coppelia_rl.envs.ur5_arm import load_ur5_arm, resolve_ur5_arm
from coppelia_rl.sim_interface.client import SimClient
from coppelia_rl.sim_interface.objects import Joint, SceneObject


@dataclass
class ReachScene:
    base: SceneObject
    joints: list[Joint]
    tip: SceneObject
    target: SceneObject


def ensure_reach_scene(
    client: SimClient,
    scene_path: str | Path,
    ur5_model_path: str | Path | None = None,
) -> ReachScene:
    scene_path = Path(scene_path)

    if scene_path.exists():
        client.load_scene(scene_path)
        arm = resolve_ur5_arm(client)
        target = client.get_object("/ReachTarget")
        return ReachScene(base=arm.base, joints=arm.joints, tip=arm.tip, target=target)

    client.close_scene()
    arm = load_ur5_arm(client, ur5_model_path)

    target = client.create_dummy(size=0.03)
    target.set_name("ReachTarget")
    target.set_position(sample_target_position(np.random.default_rng()))

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    client.save_scene(scene_path)

    return ReachScene(base=arm.base, joints=arm.joints, tip=arm.tip, target=target)


def sample_target_position(rng: np.random.Generator) -> np.ndarray:
    """Samples a target position within a reachable envelope in front of the UR5 base."""
    x = rng.uniform(0.3, 0.6)
    y = rng.uniform(-0.3, 0.3)
    z = rng.uniform(0.2, 0.6)
    return np.array([x, y, z], dtype=np.float64)
