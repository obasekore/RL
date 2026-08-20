"""One-time setup: ensures scenes/reach.ttt, pick_and_place.ttt, and
mobile_nav.ttt all exist, building whichever are missing.

Usage (with a CoppeliaSim instance + ZMQ remote API add-on running):
    .venv/Scripts/python.exe scripts/build_task_scenes.py --port 23000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coppelia_rl.envs.mobile_nav_scene import ensure_mobile_nav_scene
from coppelia_rl.envs.pick_place_scene import ensure_pick_and_place_scene
from coppelia_rl.envs.scene_builder import ensure_reach_scene
from coppelia_rl.sim_interface.client import SimClient

_SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args()

    client = SimClient.connect(host=args.host, port=args.port)
    try:
        ensure_reach_scene(client, _SCENES_DIR / "reach.ttt")
        print("reach.ttt ready")
        ensure_pick_and_place_scene(client, _SCENES_DIR / "pick_and_place.ttt")
        print("pick_and_place.ttt ready")
        ensure_mobile_nav_scene(client, _SCENES_DIR / "mobile_nav.ttt")
        print("mobile_nav.ttt ready")
    finally:
        client.close()


if __name__ == "__main__":
    main()
