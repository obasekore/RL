"""One-time setup: ensures scenes/biped_nao.ttt exists, building it (from the stock
NAO.ttm model, joints renamed to match skeletons/biped_nao.yaml) if missing.

Usage (with a CoppeliaSim instance + ZMQ remote API add-on running):
    .venv/Scripts/python.exe scripts/build_biped_nao_scene.py --port 23100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coppelia_rl.envs.biped_nao_scene import ensure_biped_nao_scene
from coppelia_rl.sim_interface.client import SimClient

_SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args()

    client = SimClient.connect(host=args.host, port=args.port)
    try:
        ensure_biped_nao_scene(client, _SCENES_DIR / "biped_nao.ttt")
        print("biped_nao.ttt ready")
    finally:
        client.close()


if __name__ == "__main__":
    main()
