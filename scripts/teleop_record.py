"""Records human-teleoperated demonstration episodes via keyboard, using the
same DemoRecorder-wrapped env training would use.

Connect to an already-running, **non-headless** CoppeliaSim instance (so you
can actually see what you're driving) before running this script.

Controls:
    Enter       start a new episode (also ends/saves the current one)
    Y / N       mark the current episode a success/failure once it ends
                (overrides the default auto-label)
    Esc         quit, closing the recording cleanly

Continuous action spaces: each action dimension gets a +/- key pair, in
order, from this fixed list (supports up to 10 dimensions):
    dim 0: 1 / q     dim 1: 2 / w     dim 2: 3 / e     dim 3: 4 / r
    dim 4: 5 / t      dim 5: 6 / y     dim 6: 7 / u     dim 7: 8 / i
    dim 8: 9 / o      dim 9: 0 / p
Holding a key jogs that dimension at a constant rate; release to stop.

Discrete action spaces: number keys 0..N-1 select the action index; no key
held defaults to index 0.

Usage:
    .venv/Scripts/python.exe scripts/teleop_record.py tasks/reach.xml --out demos/reach_teleop.h5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from gymnasium import spaces
from pynput import keyboard

from coppelia_rl.demos.recorder import DemoRecorder
from coppelia_rl.env_schema import load_env

_CONTINUOUS_KEY_PAIRS = [
    ("1", "q"),
    ("2", "w"),
    ("3", "e"),
    ("4", "r"),
    ("5", "t"),
    ("6", "y"),
    ("7", "u"),
    ("8", "i"),
    ("9", "o"),
    ("0", "p"),
]


class KeyState:
    def __init__(self):
        self._held: set[str] = set()
        self.start_episode = False
        self.mark_success: bool | None = None
        self.quit = False

    def _char(self, key) -> str | None:
        return getattr(key, "char", None)

    def on_press(self, key):
        char = self._char(key)
        if char is not None:
            self._held.add(char)
            if char in ("y", "Y"):
                self.mark_success = True
            elif char in ("n", "N"):
                self.mark_success = False
        elif key == keyboard.Key.enter:
            self.start_episode = True
        elif key == keyboard.Key.esc:
            self.quit = True

    def on_release(self, key):
        char = self._char(key)
        if char is not None:
            self._held.discard(char)

    def held(self, char: str) -> bool:
        return char in self._held


def _build_action(action_space, keys: KeyState, magnitude: float):
    if isinstance(action_space, spaces.Box):
        values = np.zeros(action_space.shape, dtype=action_space.dtype)
        for i in range(action_space.shape[0]):
            if i >= len(_CONTINUOUS_KEY_PAIRS):
                break
            plus_key, minus_key = _CONTINUOUS_KEY_PAIRS[i]
            if keys.held(plus_key):
                values[i] = magnitude
            elif keys.held(minus_key):
                values[i] = -magnitude
        return np.clip(values, action_space.low, action_space.high)

    if isinstance(action_space, spaces.Discrete):
        for index in range(action_space.n):
            if keys.held(str(index)):
                return index
        return 0

    raise TypeError(f"teleop_record.py doesn't support action space type {type(action_space)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml_path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--out", default="demos/teleop.h5")
    parser.add_argument("--magnitude", type=float, default=1.0, help="Continuous action jog magnitude.")
    args = parser.parse_args()

    env = load_env(args.xml_path, host=args.host, port=args.port)
    recorder = DemoRecorder(env, args.out, env_name=args.xml_path)
    # recorder.spec (not .unwrapped.spec) would silently be None here -
    # gym.Wrapper reserves .spec for its own registry bookkeeping and
    # doesn't understand XmlDefinedEnv's same-named EnvSpec (see the NOTE
    # in generic_env.py's XmlDefinedEnv.__init__).
    step_dt = recorder.unwrapped.spec.step_dt

    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    print(__doc__)
    print("Waiting for Enter to start the first episode...")

    try:
        episode_running = False
        while not keys.quit:
            if keys.start_episode:
                keys.start_episode = False
                if episode_running:
                    print("(episode already ended when max_steps/termination was hit - starting a fresh one)")
                obs, info = recorder.reset()
                episode_running = True
                print("Episode started. Drive the robot; Enter to end early and start a new one.")

            if keys.mark_success is not None:
                recorder.set_success_override(keys.mark_success)
                print(f"Marked current episode success={keys.mark_success}")
                keys.mark_success = None

            if episode_running:
                action = _build_action(recorder.action_space, keys, args.magnitude)
                obs, reward, terminated, truncated, info = recorder.step(action)
                if terminated or truncated:
                    print(f"Episode ended (terminated={terminated} truncated={truncated}) and was saved.")
                    episode_running = False

            time.sleep(step_dt)
    finally:
        listener.stop()
        recorder.close()
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
