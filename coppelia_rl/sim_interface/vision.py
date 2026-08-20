"""Vision sensor wrapping: RGB/depth image reads for camera observations.

Split out from objects.py (per the explicit `sim_interface/vision.py`
naming) because image observations need real care that the rest of the
object model doesn't: every read requires the sensor to be explicitly
handled first, confirmed against a live instance this milestone:

- `sim.getVisionSensorImg`/`sim.getVisionSensorDepth`'s own manual pages
  state the returned data "doesn't make sense" unless `sim.handleVisionSensor`
  was called first.
- `sim.handleVisionSensor` itself errors ("object not tagged for explicit
  handling") unless the sensor has explicit handling enabled - confirmed
  live: calling it on a sensor created the way `SimClient.create_vision_sensor`
  used to (`options` without bit0 set) raises immediately.
- Explicit handling is a *runtime* flag, not something persisted into a
  saved .ttt scene - also confirmed live the hard way: setting it once at
  construction time works for the first read, but `XmlDefinedEnv.reset()`
  calls `client.load_scene()` again on every episode, which reloads the
  scene fresh from disk and silently reverts the flag, so a second episode's
  first camera read fails the same "not tagged for explicit handling" error.
  Fixed by re-asserting it on every read instead of once at construction -
  cheap (one extra call) and correct regardless of how many times the
  underlying scene gets reloaded. This also self-heals vision sensors baked
  into already-saved scenes (e.g. `wrist_cam` in `pick_and_place.ttt`, built
  in before this fix existed) without needing to rebuild them.
- `sim.getVisionSensorDepthBuffer` is deprecated in favor of
  `sim.getVisionSensorDepth` - same (bytes, resolution) return shape,
  confirmed live.
"""

from __future__ import annotations

import numpy as np

from coppelia_rl.sim_interface.objects import SceneObject


class VisionSensor(SceneObject):
    """An RGB/depth vision sensor."""

    def _handle_for_read(self) -> None:
        self._sim.setExplicitHandling(self.handle, 1)
        self._sim.handleVisionSensor(self.handle)

    def get_rgb(self) -> np.ndarray:
        self._handle_for_read()
        img, resolution = self._sim.getVisionSensorImg(self.handle)
        width, height = resolution
        arr = np.frombuffer(img, dtype=np.uint8)
        return arr.reshape(height, width, 3)[::-1]

    def get_depth(self) -> np.ndarray:
        self._handle_for_read()
        buf, resolution = self._sim.getVisionSensorDepth(self.handle)
        width, height = resolution
        arr = np.frombuffer(buf, dtype=np.float32)
        return arr.reshape(height, width)[::-1]
