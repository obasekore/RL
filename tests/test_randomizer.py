import numpy as np
import pytest

from coppelia_rl.env_schema.spec import (
    CameraRandomizationSpec,
    DomainRandomizationSpec,
    RangeRandomizationSpec,
    TextureRandomizationSpec,
)
from coppelia_rl.randomization.randomizer import Randomizer
from coppelia_rl.sim_interface.client import SimClient, UnsupportedPhysicsEngineError


def _client(fake_sim):
    return SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)


def _alias(fake_sim, kind, name):
    handle = fake_sim._new_handle(kind)
    fake_sim.setObjectAlias(handle, name)
    return handle


# -- mass / friction --------------------------------------------------------------


def test_mass_resample_draws_within_range(fake_sim):
    handle = _alias(fake_sim, "shape", "cube")
    spec = DomainRandomizationSpec(masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))])
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))

    r.resample()

    assert 0.1 <= fake_sim.objects[handle]["mass"] <= 0.2


@pytest.mark.parametrize(
    "engine_attr, param_attrs",
    [
        ("physics_bullet", ["bullet_body_friction"]),
        ("physics_ode", ["ode_body_friction"]),
        ("physics_newton", ["newton_body_kineticfriction", "newton_body_staticfriction"]),
        ("physics_vortex", ["vortex_body_primlinearaxisfriction"]),
        ("physics_mujoco", ["mujoco_body_friction1"]),
    ],
)
def test_friction_resample_dispatches_per_engine(fake_sim, engine_attr, param_attrs):
    handle = _alias(fake_sim, "shape", "table")
    fake_sim.dynamic_engine = getattr(fake_sim, engine_attr)
    spec = DomainRandomizationSpec(frictions=[RangeRandomizationSpec(ref="table", value_range=(0.4, 1.0))])
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))

    r.resample()

    engine_params = fake_sim.objects[handle]["engine_params"]
    for param_attr in param_attrs:
        param_id = getattr(fake_sim, param_attr)
        assert 0.4 <= engine_params[param_id] <= 1.0


def test_friction_unsupported_engine_raises(fake_sim):
    _alias(fake_sim, "shape", "table")
    fake_sim.dynamic_engine = "some_future_engine"
    spec = DomainRandomizationSpec(frictions=[RangeRandomizationSpec(ref="table", value_range=(0.4, 1.0))])
    r = Randomizer(_client(fake_sim), spec)

    with pytest.raises(UnsupportedPhysicsEngineError):
        r.resample()


# -- texture ------------------------------------------------------------------------


def test_texture_resample_picks_from_pool(fake_sim, tmp_path):
    handle = _alias(fake_sim, "shape", "table")
    (tmp_path / "wood_1.png").touch()
    (tmp_path / "wood_2.png").touch()
    spec = DomainRandomizationSpec(textures=[TextureRandomizationSpec(ref="table", pool=str(tmp_path / "wood_*.png"))])
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))

    r.resample()

    assert fake_sim.objects[handle]["texture_id"] is not None


def test_texture_resample_raises_on_empty_pool(fake_sim, tmp_path):
    _alias(fake_sim, "shape", "table")
    spec = DomainRandomizationSpec(textures=[TextureRandomizationSpec(ref="table", pool=str(tmp_path / "nope_*.png"))])
    r = Randomizer(_client(fake_sim), spec)

    with pytest.raises(FileNotFoundError):
        r.resample()


# -- camera jitter --------------------------------------------------------------------


def test_camera_jitter_stays_within_bounds_of_fixed_base_pose(fake_sim):
    handle = _alias(fake_sim, "vision", "cam1")
    spec = DomainRandomizationSpec(cameras=[CameraRandomizationSpec(ref="cam1", jitter_pos=0.1, jitter_rot_deg=10.0)])
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))
    base_pos = fake_sim.objects[handle]["position"].copy()
    base_orient = fake_sim.objects[handle]["orientation"].copy()

    for _ in range(20):
        r.resample()
        pos = fake_sim.objects[handle]["position"]
        orient = fake_sim.objects[handle]["orientation"]
        assert np.all(np.abs(pos - base_pos) <= 0.1 + 1e-9)
        assert np.all(np.abs(orient - base_orient) <= np.radians(10.0) + 1e-9)


# -- resample_on policies ---------------------------------------------------------------


def _resample_count(fake_sim):
    return sum(1 for name, _ in fake_sim.calls if name == "setShapeMass")


def test_resample_on_episode_start_resamples_every_episode_start(fake_sim):
    _alias(fake_sim, "shape", "cube")
    spec = DomainRandomizationSpec(masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))])
    r = Randomizer(_client(fake_sim), spec)

    r.maybe_resample(episode_start=True)
    r.maybe_resample(episode_start=False)
    r.maybe_resample(episode_start=True)

    assert _resample_count(fake_sim) == 2


def test_resample_on_once_resamples_only_first_time(fake_sim):
    _alias(fake_sim, "shape", "cube")
    spec = DomainRandomizationSpec(
        masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))], resample_on="once"
    )
    r = Randomizer(_client(fake_sim), spec)

    r.maybe_resample(episode_start=True)
    r.notify_step()
    r.maybe_resample(episode_start=True)
    r.maybe_resample(episode_start=False)

    assert _resample_count(fake_sim) == 1


def test_resample_on_n_steps_resamples_after_interval(fake_sim):
    _alias(fake_sim, "shape", "cube")
    spec = DomainRandomizationSpec(
        masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))],
        resample_on="n_steps",
        resample_every_n_steps=3,
    )
    r = Randomizer(_client(fake_sim), spec)

    r.maybe_resample(episode_start=True)  # first call always resamples
    assert _resample_count(fake_sim) == 1

    r.notify_step()
    r.maybe_resample(episode_start=False)
    r.notify_step()
    r.maybe_resample(episode_start=False)
    assert _resample_count(fake_sim) == 1  # only 2 steps elapsed, interval is 3

    r.notify_step()
    r.maybe_resample(episode_start=False)
    assert _resample_count(fake_sim) == 2  # 3rd step reached


def test_n_steps_without_interval_raises_at_construction(fake_sim):
    spec = DomainRandomizationSpec(resample_on="n_steps")

    with pytest.raises(ValueError):
        Randomizer(_client(fake_sim), spec)


# -- latency: action delay + observation noise -----------------------------------------


def test_delay_action_holds_then_delays_by_fixed_amount(fake_sim):
    spec = DomainRandomizationSpec(action_delay_steps=(2, 2))
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))
    r.resample()  # draws action_delay_steps=2 from the fixed [2,2] range

    outputs = [r.delay_action(f"a{i}") for i in range(5)]

    assert outputs == ["a0", "a0", "a0", "a1", "a2"]


def test_delay_action_is_passthrough_when_not_configured(fake_sim):
    r = Randomizer(_client(fake_sim), DomainRandomizationSpec())

    assert r.delay_action("a0") == "a0"
    assert r.delay_action("a1") == "a1"


def test_observation_noise_perturbs_float_arrays_only(fake_sim):
    spec = DomainRandomizationSpec(observation_noise_std=0.1)
    r = Randomizer(_client(fake_sim), spec, rng=np.random.default_rng(0))
    obs = {
        "pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "img": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    noisy = r.add_observation_noise(obs)

    assert noisy["pos"].dtype == np.float32
    assert not np.array_equal(noisy["pos"], obs["pos"])
    assert np.array_equal(noisy["img"], obs["img"])


def test_observation_noise_is_noop_when_std_not_set(fake_sim):
    r = Randomizer(_client(fake_sim), DomainRandomizationSpec())
    obs = {"pos": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    assert r.add_observation_noise(obs) is obs
