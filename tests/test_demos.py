from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

import h5py  # noqa: E402

from coppelia_rl.demos.recorder import DemoRecorder  # noqa: E402
from coppelia_rl.env_schema.generic_env import XmlDefinedEnv  # noqa: E402
from coppelia_rl.env_schema.spec import (  # noqa: E402
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    ObservationSpec,
    TerminationConditionSpec,
)
from coppelia_rl.sim_interface.client import SimClient  # noqa: E402

_SCENE_PATH = Path("unused.ttt")


def _client(fake_sim):
    return SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)


def _make_env(fake_sim, *, termination_conditions, max_steps_hint=100):
    joint = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    fake_sim.setObjectAlias(joint, "J1")
    spec = EnvSpec(
        name="t",
        step_dt=0.05,
        scene_path=_SCENE_PATH,
        observations=[ObservationSpec(kind="joint_position", key="j1_pos", ref="J1")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="J1", ref="J1", value_range=(-1.0, 1.0))],
        ),
        reward_terms=[],
        termination_conditions=termination_conditions,
    )
    return XmlDefinedEnv(spec, _client(fake_sim))


def _run_episode(recorder, steps=3):
    obs, info = recorder.reset()
    for _ in range(steps):
        action = np.array([0.1], dtype=np.float32)
        obs, reward, terminated, truncated, info = recorder.step(action)
        if terminated or truncated:
            break
    return terminated, truncated


def test_records_episode_structure(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=3)])
    out_path = tmp_path / "demos.h5"
    recorder = DemoRecorder(env, out_path, env_name="test_env")

    _run_episode(recorder, steps=5)
    recorder.close()

    with h5py.File(out_path, "r") as f:
        assert f.attrs["env_name"] == "test_env"
        assert f.attrs["num_episodes"] == 1
        ep = f["episode_0"]
        assert ep["actions"].shape == (3, 1)
        assert ep["rewards"].shape == (3,)
        assert ep["terminated"].shape == (3,)
        assert ep["truncated"].shape == (3,)
        # observations include the initial reset() obs too - one more than actions.
        assert ep["observations"]["j1_pos"].shape == (4, 1)
        assert "success" in ep.attrs


def test_default_success_labels(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="signal", name="task_done")])
    out_path = tmp_path / "demos.h5"
    recorder = DemoRecorder(env, out_path)

    recorder.reset()
    fake_sim.setInt32Signal("task_done", 1)
    recorder.step(np.array([0.1], dtype=np.float32))  # terminates (not truncated) -> success by default
    recorder.close()

    with h5py.File(out_path, "r") as f:
        assert bool(f["episode_0"].attrs["success"]) is True


def test_default_success_labels_truncation_as_failure(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    out_path = tmp_path / "demos.h5"
    recorder = DemoRecorder(env, out_path)

    recorder.reset()
    recorder.step(np.array([0.1], dtype=np.float32))  # hits max_steps -> truncated -> failure by default
    recorder.close()

    with h5py.File(out_path, "r") as f:
        assert bool(f["episode_0"].attrs["success"]) is False


def test_success_override_wins_over_default(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    out_path = tmp_path / "demos.h5"
    recorder = DemoRecorder(env, out_path)

    recorder.reset()
    recorder.set_success_override(True)
    recorder.step(np.array([0.1], dtype=np.float32))  # would default to failure, override says success
    recorder.close()

    with h5py.File(out_path, "r") as f:
        assert bool(f["episode_0"].attrs["success"]) is True


def test_custom_success_fn_receives_obs(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    out_path = tmp_path / "demos.h5"

    def success_fn(obs, terminated, truncated, info):
        return bool(obs["j1_pos"][0] > 0.0)

    recorder = DemoRecorder(env, out_path, success_fn=success_fn)
    recorder.reset()
    recorder.step(np.array([0.1], dtype=np.float32))
    recorder.close()

    with h5py.File(out_path, "r") as f:
        assert bool(f["episode_0"].attrs["success"]) is True


def test_append_mode_accumulates_across_recorder_instances(fake_sim, tmp_path):
    out_path = tmp_path / "demos.h5"

    env1 = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    recorder1 = DemoRecorder(env1, out_path)
    recorder1.reset()
    recorder1.step(np.array([0.1], dtype=np.float32))
    recorder1.close()

    env2 = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    recorder2 = DemoRecorder(env2, out_path)
    recorder2.reset()
    recorder2.step(np.array([0.1], dtype=np.float32))
    recorder2.close()

    with h5py.File(out_path, "r") as f:
        assert f.attrs["num_episodes"] == 2
        assert "episode_0" in f
        assert "episode_1" in f


def test_action_and_observation_spaces_delegate_to_wrapped_env(fake_sim, tmp_path):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    recorder = DemoRecorder(env, tmp_path / "demos.h5")

    assert recorder.action_space == env.action_space
    assert recorder.observation_space == env.observation_space
    recorder.close()


def test_wrapped_spec_access_needs_unwrapped(fake_sim, tmp_path):
    """Documents a real gotcha (crashed a live teleop_record.py run):
    XmlDefinedEnv.spec collides in *name* (not type) with gymnasium.Env's
    own reserved `.spec` - gym.Wrapper.__init__ tries to copy env.spec
    assuming the gymnasium type, fails, and silently sets the wrapper's own
    `.spec` to None (only a UserWarning). `.unwrapped.spec` is the only
    reliable way to reach our EnvSpec through a wrapper."""
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=1)])
    recorder = DemoRecorder(env, tmp_path / "demos.h5")

    assert recorder.spec is None
    assert recorder.unwrapped.spec is env.spec
    recorder.close()
