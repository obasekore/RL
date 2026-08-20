"""RL Domain Randomizer GUI - implementation module.

The standalone counterpart to rl_env_editor_app.py: applies domain
randomization to whatever scene is currently open in this CoppeliaSim
instance, with no <rl_env> XML or gymnasium.Env involved at all - "function
independently for any non-RL problem". It shares its scene tree and Domain
Randomization tab widgets with the RL Env Editor (scene_tree_panel.py,
domain_randomization_panel.py) and drives the exact same
coppelia_rl.randomization.Randomizer engine the RL training path uses, so
behavior can't drift between the two.

It can also "enable by parsing an rl-env definition": Load DR from RL env
XML pulls just the <domain_randomization> block out of an existing <rl_env>
XML file (via DomainRandomizerState.load_from_env_xml), instead of requiring
every field to be authored by hand here.

`sim`/`simUI`/`self` are CoppeliaSim add-on globals - see
rl_env_editor_app.py's module docstring for why they're injected as module
attributes rather than available automatically.
"""

from __future__ import annotations

import threading

from coppelia_rl.gui import domain_randomization_panel, scene_tree_panel
from coppelia_rl.gui.domain_randomization_panel import DR_TAB_XML
from coppelia_rl.gui.scene_tree_panel import TREE_GROUP_XML

sim = None
simUI = None
self = None

ID_TABS = 20
ID_LOAD_XML_BTN = 21
ID_APPLY_BTN = 22
ID_STATUS_EDIT = 23

_UI_XML = """
<ui title="RL Domain Randomizer" closeable="true" resizable="true" on-close="onClose" size="750,600">
  <group flat="true" layout="hbox" stretch="1">
    {TREE_GROUP_XML}
    <group flat="true" layout="vbox" stretch="2">
      <tabs id="{ID_TABS}" stretch="1">
        {DR_TAB_XML}
      </tabs>
      <group flat="true" layout="hbox">
        <button id="{ID_LOAD_XML_BTN}" text="Load DR from RL env XML..." on-click="onLoadFromEnvXml"/>
        <button id="{ID_APPLY_BTN}" text="Apply Randomization Now" on-click="onApplyNow"/>
      </group>
      <group flat="true" layout="hbox">
        <edit id="{ID_STATUS_EDIT}" value="" enabled="false"/>
      </group>
    </group>
  </group>
</ui>
"""

__all__ = [
    "init_state",
    "create_ui",
    "check_background_tasks",
    "onClose",
    "onLoadFromEnvXml",
    "onApplyNow",
]


def init_state():
    from coppelia_rl.randomization.randomizer_state import DomainRandomizerState

    self.state = DomainRandomizerState()
    self.selected_dr_row = None
    self.leaveNow = False
    self.apply_result = None


def create_ui():
    """See rl_env_editor_app.create_ui()'s docstring for why sysCall_init +
    sysCall_nonSimulation (not sysCall_thread) is the pattern used here."""
    scene_tree_panel.sim = sim
    scene_tree_panel.simUI = simUI
    domain_randomization_panel.sim = sim
    domain_randomization_panel.simUI = simUI
    domain_randomization_panel.self = self
    # No larger preview to refresh here (unlike the RL Env Editor's XML tab) -
    # the DR table itself is the whole picture, already re-rendered by
    # render_domain_randomization_tab() before this hook runs.
    domain_randomization_panel.on_change_hook = lambda ui: None

    self.ui = simUI.create(_UI_XML.format(**globals()))
    scene_tree_panel.refresh_tree(self.ui)
    domain_randomization_panel.render_domain_randomization_tab(self.ui)


def _local_zmq_port():
    """The ZMQ remote API port for *this* running instance - see
    rl_env_editor_app._local_zmq_port() for why this can't be hardcoded."""
    try:
        port = sim.getNamedInt32Param("zmqRemoteApi.rpcPort")
        if port:
            return port
    except Exception:
        pass
    return 23000 + sim.getInt32Param(sim.intparam_processid)


def _set_status(ui, text):
    simUI.setEditValue(ui, ID_STATUS_EDIT, text, True)


def check_background_tasks():
    """Polled every idle tick from sysCall_nonSimulation - Apply Randomization
    Now needs a second ZMQ connection back into this same running instance,
    which must not happen synchronously inside a callback (see
    rl_env_editor_app.check_background_tasks() for the deadlock this avoids)."""
    result = self.apply_result
    if result is not None:
        self.apply_result = None
        kind, msg = result
        box_type = simUI.msgbox_type.info if kind == "ok" else simUI.msgbox_type.warning
        simUI.msgBox(box_type, simUI.msgbox_buttons.ok, "RL Domain Randomizer", msg)
        _set_status(self.ui, msg)


def onClose(ui, *args):
    self.leaveNow = True


def onLoadFromEnvXml(ui, *args):
    results = simUI.fileDialog(
        simUI.filedialog_type.load, "Select an RL env XML file", "", "", "RL env XML", "xml", False
    )
    if not results:
        return
    try:
        self.state.load_from_env_xml(results[0])
    except Exception as e:
        simUI.msgBox(
            simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Domain Randomizer", "Failed to load: " + str(e)
        )
        return
    domain_randomization_panel.render_domain_randomization_tab(ui)
    _set_status(ui, "Loaded domain_randomization from " + results[0])


def onApplyNow(ui, *args):
    if not self.state.has_domain_randomization():
        simUI.msgBox(
            simUI.msgbox_type.warning,
            simUI.msgbox_buttons.ok,
            "RL Domain Randomizer",
            "Add at least one randomization entry first.",
        )
        return

    if getattr(self, "apply_thread", None) is not None and self.apply_thread.is_alive():
        simUI.msgBox(
            simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Domain Randomizer", "Already running."
        )
        return

    port = _local_zmq_port()
    _set_status(ui, "Applying randomization to the currently open scene...")
    self.apply_thread = threading.Thread(target=_apply_worker, args=(self.state.domain_randomization, port), daemon=True)
    self.apply_thread.start()


def _apply_worker(dr_spec, port):
    from coppelia_rl.randomization.randomizer import Randomizer
    from coppelia_rl.sim_interface.client import SimClient

    try:
        # No client.close() here - unlike XmlDefinedEnv, this never starts a
        # simulation, so there's nothing to stop. Calling close() would risk
        # stopping a simulation the user already has running for unrelated
        # reasons, just because Apply Randomization Now happened to connect.
        client = SimClient.connect(host="localhost", port=port)
        Randomizer(client, dr_spec).resample()
        self.apply_result = ("ok", "Randomization applied to the current scene.")
    except Exception as e:
        self.apply_result = ("error", "Apply failed: " + str(e))
