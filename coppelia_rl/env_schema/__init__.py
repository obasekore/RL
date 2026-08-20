from __future__ import annotations

from pathlib import Path

from coppelia_rl.env_schema.generic_env import XmlDefinedEnv
from coppelia_rl.env_schema.parser import parse_env_xml
from coppelia_rl.sim_interface.client import SimClient


def load_env(
    xml_path: str | Path,
    host: str = "localhost",
    port: int = 23000,
    client: SimClient | None = None,
) -> XmlDefinedEnv:
    spec = parse_env_xml(xml_path)
    sim_client = client if client is not None else SimClient.connect(host=host, port=port)
    return XmlDefinedEnv(spec, sim_client)


__all__ = ["load_env", "XmlDefinedEnv", "parse_env_xml"]
