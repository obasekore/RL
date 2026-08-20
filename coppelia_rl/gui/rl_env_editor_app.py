"""RL Env Editor GUI - implementation module.

Kept out of addOns/RL Env Editor.py so that file can stay a thin loader;
this is a normal importable module with the real widget layout and callback
logic, editable/testable without touching the add-on script itself.

`sim`/`simUI`/`self` are CoppeliaSim add-on globals. They're only injected
automatically into the add-on script's own execution namespace, not into a
regularly-imported module like this one - so the thin add-on script sets
them here explicitly (as module attributes) before calling anything.
"""

from __future__ import annotations

import threading

from coppelia_rl.gui import domain_randomization_panel, scene_tree_panel
from coppelia_rl.gui.domain_randomization_panel import DR_TAB_XML
from coppelia_rl.gui.scene_tree_panel import TREE_GROUP_XML
from coppelia_rl.gui.scene_tree_panel import selected_object_alias as _selected_object_alias

sim = None
simUI = None
self = None

# Widget ids (must be unique per simUI.create call). ID_TREE/DR tab widget
# ids live in scene_tree_panel.py/domain_randomization_panel.py instead -
# those tabs are shared with the standalone RL Domain Randomizer add-on.
ID_NAME_EDIT = 1
ID_STEP_DT_EDIT = 2
ID_SCENE_EDIT = 3

ID_TABS = 20

ID_OBS_TABLE = 30
ID_OBS_KIND_COMBO = 31
ID_OBS_REF_EDIT = 32
ID_OBS_NAME_EDIT = 33
ID_OBS_FRAME_EDIT = 34
ID_OBS_RESOLUTION_EDIT = 35
ID_OBS_CHANNELS_COMBO = 36
ID_OBS_CALLABLE_EDIT = 37
ID_OBS_USE_SELECTED_BTN = 38
ID_OBS_ADD_BTN = 39
ID_OBS_REMOVE_BTN = 40

ID_ACT_TYPE_COMBO = 41
ID_ACT_TABLE = 42
ID_ACT_KIND_COMBO = 43
ID_ACT_REF_EDIT = 44
ID_ACT_NAME_EDIT = 45
ID_ACT_RANGE_EDIT = 46
ID_ACT_SIGNAL_EDIT = 47
ID_ACT_CALLABLE_EDIT = 48
ID_ACT_USE_SELECTED_BTN = 49
ID_ACT_ADD_BTN = 50
ID_ACT_REMOVE_BTN = 51

ID_REWARD_TABLE = 52
ID_REWARD_KIND_COMBO = 53
ID_REWARD_WEIGHT_EDIT = 54
ID_REWARD_FROM_EDIT = 55
ID_REWARD_TO_EDIT = 56
ID_REWARD_OBJ1_EDIT = 57
ID_REWARD_OBJ2_EDIT = 58
ID_REWARD_SIGNAL_EDIT = 59
ID_REWARD_CALLABLE_EDIT = 60
ID_REWARD_ADD_BTN = 61
ID_REWARD_REMOVE_BTN = 62

ID_TERM_TABLE = 63
ID_TERM_KIND_COMBO = 64
ID_TERM_NAME_EDIT = 65
ID_TERM_VALUE_EDIT = 66
ID_TERM_CALLABLE_EDIT = 67
ID_TERM_ADD_BTN = 68
ID_TERM_REMOVE_BTN = 69

ID_XML_BROWSER = 80
ID_SAVE_BTN = 81
ID_LOAD_BTN = 82
ID_TEST_ENV_BTN = 83
ID_STATUS_BROWSER = 84
ID_ACT_APPLY_TYPE_BTN = 85
ID_BROWSE_SCENE_BTN = 86
ID_REWARD_FROM_USE_BTN = 87
ID_REWARD_TO_USE_BTN = 88
ID_REWARD_OBJ1_USE_BTN = 89
ID_REWARD_OBJ2_USE_BTN = 90

OBS_KINDS = ["joint_position", "joint_velocity", "object_pose", "object_position", "camera", "custom"]
OBS_CHANNELS = ["rgb", "depth"]
ACT_TYPES = ["continuous", "discrete"]
REWARD_KINDS = ["distance", "collision", "signal_event", "custom"]
TERM_KINDS = ["signal", "max_steps", "custom"]

_UI_XML = '''
<ui title="RL Env Editor" closeable="true" resizable="true" on-close="onClose" size="1150,750">
  <group flat="true" layout="hbox">
    <label text="Name:"/>
    <edit id="{ID_NAME_EDIT}" value="new_env" on-change="onMetaChange"/>
    <label text="step_dt:"/>
    <edit id="{ID_STEP_DT_EDIT}" value="0.05" on-change="onMetaChange"/>
    <label text="Scene (.ttt path):"/>
    <edit id="{ID_SCENE_EDIT}" value="" on-change="onMetaChange" stretch="1"/>
    <button id="{ID_BROWSE_SCENE_BTN}" text="Browse..." on-click="onBrowseScene"/>
  </group>
  <group flat="true" layout="hbox" stretch="1">
    {TREE_GROUP_XML}
    <tabs id="{ID_TABS}" stretch="2">
      <tab title="Observations">
        <table id="{ID_OBS_TABLE}" show-horizontal-header="true" selection-mode="row" editable="false" on-selection-change="onObsTableSelect"/>
        <group flat="true" layout="form">
          <label text="kind:"/>
          <combobox id="{ID_OBS_KIND_COMBO}"><item>joint_position</item><item>joint_velocity</item><item>object_pose</item><item>object_position</item><item>camera</item><item>custom</item></combobox>
          <label text="ref:"/>
          <edit id="{ID_OBS_REF_EDIT}" value=""/>
          <label text="name (optional, required for custom):"/>
          <edit id="{ID_OBS_NAME_EDIT}" value=""/>
          <label text="frame (object_pose/position, default world):"/>
          <edit id="{ID_OBS_FRAME_EDIT}" value=""/>
          <label text="resolution WxH (camera):"/>
          <edit id="{ID_OBS_RESOLUTION_EDIT}" value=""/>
          <label text="channels (camera):"/>
          <combobox id="{ID_OBS_CHANNELS_COMBO}"><item>rgb</item><item>depth</item></combobox>
          <label text="callable (custom):"/>
          <edit id="{ID_OBS_CALLABLE_EDIT}" value=""/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_OBS_USE_SELECTED_BTN}" text="Sel -> ref" on-click="onObsUseSelected"/>
          <button id="{ID_OBS_ADD_BTN}" text="Add" on-click="onObsAdd"/>
          <button id="{ID_OBS_REMOVE_BTN}" text="Remove selected" on-click="onObsRemove"/>
        </group>
      </tab>
      <tab title="Actions">
        <group flat="true" layout="hbox">
          <label text="type:"/>
          <combobox id="{ID_ACT_TYPE_COMBO}"><item>continuous</item><item>discrete</item></combobox>
          <button id="{ID_ACT_APPLY_TYPE_BTN}" text="Apply type (clears existing action rows)" on-click="onActApplyType"/>
        </group>
        <table id="{ID_ACT_TABLE}" show-horizontal-header="true" selection-mode="row" editable="false" on-selection-change="onActTableSelect"/>
        <group flat="true" layout="form">
          <label text="kind (joint_velocity/joint_position/custom if continuous, event/custom if discrete):"/>
          <edit id="{ID_ACT_KIND_COMBO}" value="joint_velocity"/>
          <label text="ref (joint_velocity/joint_position):"/>
          <edit id="{ID_ACT_REF_EDIT}" value=""/>
          <label text="name (event/custom):"/>
          <edit id="{ID_ACT_NAME_EDIT}" value=""/>
          <label text="range [lo,hi] (joint_velocity/joint_position):"/>
          <edit id="{ID_ACT_RANGE_EDIT}" value=""/>
          <label text="signal (event):"/>
          <edit id="{ID_ACT_SIGNAL_EDIT}" value=""/>
          <label text="callable (custom):"/>
          <edit id="{ID_ACT_CALLABLE_EDIT}" value=""/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_ACT_USE_SELECTED_BTN}" text="Sel -> ref" on-click="onActUseSelected"/>
          <button id="{ID_ACT_ADD_BTN}" text="Add" on-click="onActAdd"/>
          <button id="{ID_ACT_REMOVE_BTN}" text="Remove selected" on-click="onActRemove"/>
        </group>
      </tab>
      <tab title="Reward">
        <table id="{ID_REWARD_TABLE}" show-horizontal-header="true" selection-mode="row" editable="false" on-selection-change="onRewardTableSelect"/>
        <group flat="true" layout="form">
          <label text="kind:"/>
          <combobox id="{ID_REWARD_KIND_COMBO}"><item>distance</item><item>collision</item><item>signal_event</item><item>custom</item></combobox>
          <label text="weight:"/>
          <edit id="{ID_REWARD_WEIGHT_EDIT}" value="-1.0"/>
          <label text="from (distance):"/>
          <edit id="{ID_REWARD_FROM_EDIT}" value=""/>
          <label text="to (distance):"/>
          <edit id="{ID_REWARD_TO_EDIT}" value=""/>
          <label text="object1 (collision):"/>
          <edit id="{ID_REWARD_OBJ1_EDIT}" value=""/>
          <label text="object2 (collision):"/>
          <edit id="{ID_REWARD_OBJ2_EDIT}" value=""/>
          <label text="signal (signal_event):"/>
          <edit id="{ID_REWARD_SIGNAL_EDIT}" value=""/>
          <label text="callable (custom):"/>
          <edit id="{ID_REWARD_CALLABLE_EDIT}" value=""/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_REWARD_FROM_USE_BTN}" text="Sel -> from" on-click="onRewardUseSelectedFrom"/>
          <button id="{ID_REWARD_TO_USE_BTN}" text="Sel -> to" on-click="onRewardUseSelectedTo"/>
          <button id="{ID_REWARD_OBJ1_USE_BTN}" text="Sel -> object1" on-click="onRewardUseSelectedObj1"/>
          <button id="{ID_REWARD_OBJ2_USE_BTN}" text="Sel -> object2" on-click="onRewardUseSelectedObj2"/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_REWARD_ADD_BTN}" text="Add" on-click="onRewardAdd"/>
          <button id="{ID_REWARD_REMOVE_BTN}" text="Remove selected" on-click="onRewardRemove"/>
        </group>
      </tab>
      <tab title="Termination">
        <table id="{ID_TERM_TABLE}" show-horizontal-header="true" selection-mode="row" editable="false" on-selection-change="onTermTableSelect"/>
        <group flat="true" layout="form">
          <label text="kind:"/>
          <combobox id="{ID_TERM_KIND_COMBO}"><item>signal</item><item>max_steps</item><item>custom</item></combobox>
          <label text="name (signal):"/>
          <edit id="{ID_TERM_NAME_EDIT}" value=""/>
          <label text="value (max_steps):"/>
          <edit id="{ID_TERM_VALUE_EDIT}" value=""/>
          <label text="callable (custom):"/>
          <edit id="{ID_TERM_CALLABLE_EDIT}" value=""/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_TERM_ADD_BTN}" text="Add" on-click="onTermAdd"/>
          <button id="{ID_TERM_REMOVE_BTN}" text="Remove selected" on-click="onTermRemove"/>
        </group>
      </tab>
      {DR_TAB_XML}
      <tab title="XML">
        <text-browser id="{ID_XML_BROWSER}" type="plain" read-only="true"/>
        <group flat="true" layout="hbox">
          <edit id="{ID_STATUS_BROWSER}" value="" enabled="false"/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_LOAD_BTN}" text="Load from scene path's .xml" on-click="onLoad"/>
          <button id="{ID_SAVE_BTN}" text="Save to scene path's .xml" on-click="onSave"/>
          <button id="{ID_TEST_ENV_BTN}" text="Test env (random rollout)" on-click="onTestEnv"/>
        </group>
      </tab>
    </tabs>
  </group>
</ui>
'''

__all__ = [
    "init_state",
    "create_ui",
    "check_background_tasks",
    "onClose",
    "onMetaChange",
    "onBrowseScene",
    "onObsTableSelect",
    "onActTableSelect",
    "onRewardTableSelect",
    "onTermTableSelect",
    "onObsUseSelected",
    "onObsAdd",
    "onObsRemove",
    "onActApplyType",
    "onActUseSelected",
    "onActAdd",
    "onActRemove",
    "onRewardUseSelectedFrom",
    "onRewardUseSelectedTo",
    "onRewardUseSelectedObj1",
    "onRewardUseSelectedObj2",
    "onRewardAdd",
    "onRewardRemove",
    "onTermAdd",
    "onTermRemove",
    "onSave",
    "onLoad",
    "onTestEnv",
]


def init_state():
    from coppelia_rl.env_schema.editor_state import EnvBuilderState

    self.state = EnvBuilderState()
    self.selected_obs_row = None
    self.selected_act_row = None
    self.selected_reward_row = None
    self.selected_term_row = None
    self.selected_dr_row = None
    self.leaveNow = False
    self.test_env_result = None
    self.load_validate_result = None


def create_ui():
    """Creates and shows the panel, then returns immediately - the window is
    driven by CoppeliaSim's own idle-tick loop (sysCall_nonSimulation) from
    here on, not a manually-pumped thread. Matches the pattern used by the
    stock 'Convex decomposition coacd.py' add-on, which reliably shows a
    visible window; the sysCall_thread + sim.handleExtCalls() loop this
    replaced ran without error but never actually rendered anything.
    """
    scene_tree_panel.sim = sim
    scene_tree_panel.simUI = simUI
    domain_randomization_panel.sim = sim
    domain_randomization_panel.simUI = simUI
    domain_randomization_panel.self = self
    domain_randomization_panel.on_change_hook = _refresh_xml_preview

    self.ui = simUI.create(_UI_XML.format(**globals()))
    scene_tree_panel.refresh_tree(self.ui)
    _render_all(self.ui)


def _local_zmq_port():
    """The ZMQ remote API port for *this* running instance - reads the same
    named param the "ZMQ remote API server" add-on itself sets, so a second
    client connecting back into this instance (Load's ref validation, Test
    env) hits the right port instead of a hardcoded guess."""
    try:
        port = sim.getNamedInt32Param("zmqRemoteApi.rpcPort")
        if port:
            return port
    except Exception:
        pass
    return 23000 + sim.getInt32Param(sim.intparam_processid)


def check_background_tasks():
    """Polled every idle tick from sysCall_nonSimulation. Test env and Load's
    ref validation both need a second ZMQ connection back into this same
    running instance - doing that synchronously inside a callback deadlocks
    (this script and the ZMQ server add-on share CoppeliaSim's single main
    loop, so blocking here means the server never gets a turn to reply), so
    that work runs on a background thread and just leaves its result here for
    the main thread to pick up and turn into UI updates.
    """
    ui = self.ui

    test_result = self.test_env_result
    if test_result is not None:
        self.test_env_result = None
        kind, msg = test_result
        box_type = simUI.msgbox_type.info if kind == "ok" else simUI.msgbox_type.warning
        simUI.msgBox(box_type, simUI.msgbox_buttons.ok, "RL Env Editor - Test env", msg)
        _set_status(ui, "Test env: " + msg)

    load_result = self.load_validate_result
    if load_result is not None:
        self.load_validate_result = None
        if isinstance(load_result, str):
            simUI.msgBox(
                simUI.msgbox_type.warning,
                simUI.msgbox_buttons.ok,
                "RL Env Editor",
                "Could not validate refs against the running scene "
                "(is the ZMQ remote API add-on running?): " + load_result,
            )
        elif load_result:
            simUI.msgBox(
                simUI.msgbox_type.warning,
                simUI.msgbox_buttons.ok,
                "RL Env Editor",
                "These refs do not exist in the scene: " + ", ".join(sorted(set(load_result))),
            )
        else:
            resolved_handles = []
            for ref, _ in self.state.collect_scene_refs():
                try:
                    resolved_handles.append(sim.getObject("/" + ref))
                except Exception:
                    pass
            if resolved_handles:
                sim.setObjectSel(resolved_handles)


def _xml_path_for_scene(scene_path_text):
    if scene_path_text.endswith(".ttt"):
        return scene_path_text[:-4] + ".xml"
    return scene_path_text + ".xml"


def _set_table(ui, table_id, headers, rows):
    simUI.setColumnCount(ui, table_id, len(headers))
    for i, h in enumerate(headers):
        simUI.setColumnHeaderText(ui, table_id, i, h)
    simUI.setRowCount(ui, table_id, len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            simUI.setItem(ui, table_id, r, c, str(value) if value is not None else "")


def _render_observations(ui):
    rows = []
    for obs in self.state.observations:
        detail = ""
        if obs.kind in ("object_pose", "object_position"):
            detail = "frame=" + str(obs.frame)
        elif obs.kind == "camera":
            w, h = obs.resolution
            detail = str(w) + "x" + str(h) + " " + str(obs.channels)
        elif obs.kind == "custom":
            detail = str(obs.callable)
        rows.append([obs.kind, obs.ref or "", obs.key, detail])
    _set_table(ui, ID_OBS_TABLE, ["Kind", "Ref", "Key", "Detail"], rows)


def _render_actions(ui):
    rows = []
    for entry in self.state.actions.entries:
        detail = ""
        if entry.value_range is not None:
            detail = "[" + str(entry.value_range[0]) + "," + str(entry.value_range[1]) + "]"
        elif entry.signal is not None:
            detail = "signal=" + entry.signal
        elif entry.callable is not None:
            detail = entry.callable
        rows.append([entry.kind, entry.ref or entry.key, detail])
    _set_table(ui, ID_ACT_TABLE, ["Kind", "Ref/Name", "Detail"], rows)


def _render_reward(ui):
    rows = []
    for term in self.state.reward_terms:
        if term.kind == "distance":
            detail = str(term.from_ref) + " -> " + str(term.to_ref)
        elif term.kind == "collision":
            detail = str(term.object1) + " <-> " + str(term.object2)
        elif term.kind == "signal_event":
            detail = "signal=" + str(term.signal)
        else:
            detail = str(term.callable)
        rows.append([term.kind, term.weight, detail])
    _set_table(ui, ID_REWARD_TABLE, ["Kind", "Weight", "Detail"], rows)


def _render_termination(ui):
    rows = []
    for cond in self.state.termination_conditions:
        if cond.kind == "signal":
            detail = "name=" + str(cond.name)
        elif cond.kind == "max_steps":
            detail = "value=" + str(cond.value)
        else:
            detail = str(cond.callable)
        rows.append([cond.kind, detail])
    _set_table(ui, ID_TERM_TABLE, ["Kind", "Detail"], rows)


def _refresh_xml_preview(ui):
    from coppelia_rl.env_schema.serializer import spec_to_xml

    try:
        spec = self.state.build_spec()
        simUI.setText(ui, ID_XML_BROWSER, spec_to_xml(spec))
        _set_status(ui, "OK")
    except Exception as e:
        simUI.setText(ui, ID_XML_BROWSER, "")
        _set_status(ui, "Incomplete: " + str(e))


def _set_status(ui, text):
    simUI.setEditValue(ui, ID_STATUS_BROWSER, text, True)


def _render_all(ui):
    simUI.setEditValue(ui, ID_NAME_EDIT, self.state.name, True)
    simUI.setEditValue(ui, ID_STEP_DT_EDIT, str(self.state.step_dt), True)
    simUI.setEditValue(ui, ID_SCENE_EDIT, str(self.state.scene_path) if self.state.scene_path else "", True)
    simUI.setComboboxSelectedIndex(ui, ID_ACT_TYPE_COMBO, ACT_TYPES.index(self.state.actions.action_type))
    _render_observations(ui)
    _render_actions(ui)
    _render_reward(ui)
    _render_termination(ui)
    domain_randomization_panel.render_domain_randomization_tab(ui)
    _refresh_xml_preview(ui)


def onClose(ui, *args):
    self.leaveNow = True


def onMetaChange(ui, *args):
    self.state.name = simUI.getEditValue(ui, ID_NAME_EDIT)
    try:
        self.state.step_dt = float(simUI.getEditValue(ui, ID_STEP_DT_EDIT))
    except ValueError:
        pass
    scene_text = simUI.getEditValue(ui, ID_SCENE_EDIT)
    self.state.scene_path = scene_text or None
    _refresh_xml_preview(ui)


def onBrowseScene(ui, *args):
    results = simUI.fileDialog(
        simUI.filedialog_type.load, "Select CoppeliaSim scene", "", "", "CoppeliaSim Scene", "ttt", False
    )
    if results:
        simUI.setEditValue(ui, ID_SCENE_EDIT, results[0], True)
        self.state.scene_path = results[0]
        _refresh_xml_preview(ui)


def onObsTableSelect(ui, table_id, row, col):
    self.selected_obs_row = row


def onActTableSelect(ui, table_id, row, col):
    self.selected_act_row = row


def onRewardTableSelect(ui, table_id, row, col):
    self.selected_reward_row = row


def onTermTableSelect(ui, table_id, row, col):
    self.selected_term_row = row


def _parse_range(text):
    text = text.strip().lstrip("[").rstrip("]")
    lo, hi = text.split(",")
    return (float(lo), float(hi))


def _parse_resolution(text):
    w, h = text.lower().split("x")
    return (int(w), int(h))


def onObsUseSelected(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_OBS_REF_EDIT, alias, True)


def onObsAdd(ui, *args):
    kind = OBS_KINDS[simUI.getComboboxSelectedIndex(ui, ID_OBS_KIND_COMBO)]
    ref = simUI.getEditValue(ui, ID_OBS_REF_EDIT) or None
    name = simUI.getEditValue(ui, ID_OBS_NAME_EDIT) or None
    frame = simUI.getEditValue(ui, ID_OBS_FRAME_EDIT) or None
    resolution_text = simUI.getEditValue(ui, ID_OBS_RESOLUTION_EDIT)
    resolution = _parse_resolution(resolution_text) if resolution_text else None
    channels = OBS_CHANNELS[simUI.getComboboxSelectedIndex(ui, ID_OBS_CHANNELS_COMBO)]
    callable_ = simUI.getEditValue(ui, ID_OBS_CALLABLE_EDIT) or None
    try:
        self.state.add_observation(
            kind, ref=ref, name=name, frame=frame, resolution=resolution, channels=channels, callable=callable_
        )
        _render_observations(ui)
        _refresh_xml_preview(ui)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))


def onObsRemove(ui, *args):
    if self.selected_obs_row is not None and 0 <= self.selected_obs_row < len(self.state.observations):
        self.state.remove_observation(self.selected_obs_row)
        self.selected_obs_row = None
        _render_observations(ui)
        _refresh_xml_preview(ui)


def onActApplyType(ui, *args):
    self.state.set_action_type(ACT_TYPES[simUI.getComboboxSelectedIndex(ui, ID_ACT_TYPE_COMBO)])
    _render_actions(ui)
    _refresh_xml_preview(ui)


def onActUseSelected(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_ACT_REF_EDIT, alias, True)


def onActAdd(ui, *args):
    kind = simUI.getEditValue(ui, ID_ACT_KIND_COMBO)
    ref = simUI.getEditValue(ui, ID_ACT_REF_EDIT) or None
    name = simUI.getEditValue(ui, ID_ACT_NAME_EDIT) or None
    range_text = simUI.getEditValue(ui, ID_ACT_RANGE_EDIT)
    value_range = _parse_range(range_text) if range_text else None
    signal = simUI.getEditValue(ui, ID_ACT_SIGNAL_EDIT) or None
    callable_ = simUI.getEditValue(ui, ID_ACT_CALLABLE_EDIT) or None
    try:
        self.state.add_action_entry(
            kind, ref=ref, name=name, value_range=value_range, signal=signal, callable=callable_
        )
        _render_actions(ui)
        _refresh_xml_preview(ui)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))


def onActRemove(ui, *args):
    if self.selected_act_row is not None and 0 <= self.selected_act_row < len(self.state.actions.entries):
        self.state.remove_action_entry(self.selected_act_row)
        self.selected_act_row = None
        _render_actions(ui)
        _refresh_xml_preview(ui)


def onRewardUseSelectedFrom(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_REWARD_FROM_EDIT, alias, True)


def onRewardUseSelectedTo(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_REWARD_TO_EDIT, alias, True)


def onRewardUseSelectedObj1(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_REWARD_OBJ1_EDIT, alias, True)


def onRewardUseSelectedObj2(ui, *args):
    alias = _selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_REWARD_OBJ2_EDIT, alias, True)


def onRewardAdd(ui, *args):
    kind = REWARD_KINDS[simUI.getComboboxSelectedIndex(ui, ID_REWARD_KIND_COMBO)]
    try:
        weight = float(simUI.getEditValue(ui, ID_REWARD_WEIGHT_EDIT))
        self.state.add_reward_term(
            kind,
            weight=weight,
            from_ref=simUI.getEditValue(ui, ID_REWARD_FROM_EDIT) or None,
            to_ref=simUI.getEditValue(ui, ID_REWARD_TO_EDIT) or None,
            object1=simUI.getEditValue(ui, ID_REWARD_OBJ1_EDIT) or None,
            object2=simUI.getEditValue(ui, ID_REWARD_OBJ2_EDIT) or None,
            signal=simUI.getEditValue(ui, ID_REWARD_SIGNAL_EDIT) or None,
            callable=simUI.getEditValue(ui, ID_REWARD_CALLABLE_EDIT) or None,
        )
        _render_reward(ui)
        _refresh_xml_preview(ui)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))


def onRewardRemove(ui, *args):
    if self.selected_reward_row is not None and 0 <= self.selected_reward_row < len(self.state.reward_terms):
        self.state.remove_reward_term(self.selected_reward_row)
        self.selected_reward_row = None
        _render_reward(ui)
        _refresh_xml_preview(ui)


def onTermAdd(ui, *args):
    kind = TERM_KINDS[simUI.getComboboxSelectedIndex(ui, ID_TERM_KIND_COMBO)]
    value_text = simUI.getEditValue(ui, ID_TERM_VALUE_EDIT)
    try:
        self.state.add_termination_condition(
            kind,
            name=simUI.getEditValue(ui, ID_TERM_NAME_EDIT) or None,
            value=float(value_text) if value_text else None,
            callable=simUI.getEditValue(ui, ID_TERM_CALLABLE_EDIT) or None,
        )
        _render_termination(ui)
        _refresh_xml_preview(ui)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))


def onTermRemove(ui, *args):
    if self.selected_term_row is not None and 0 <= self.selected_term_row < len(self.state.termination_conditions):
        self.state.remove_termination_condition(self.selected_term_row)
        self.selected_term_row = None
        _render_termination(ui)
        _refresh_xml_preview(ui)


def onSave(ui, *args):
    from coppelia_rl.env_schema.serializer import save_spec_xml

    scene_text = simUI.getEditValue(ui, ID_SCENE_EDIT)
    if not scene_text:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", "Set the scene path first.")
        return
    try:
        spec = self.state.build_spec()
        xml_path = _xml_path_for_scene(scene_text)
        save_spec_xml(spec, xml_path)
        _set_status(ui, "Saved to " + xml_path)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))


def onLoad(ui, *args):
    from coppelia_rl.env_schema.parser import parse_env_xml

    scene_text = simUI.getEditValue(ui, ID_SCENE_EDIT)
    xml_path = _xml_path_for_scene(scene_text)
    try:
        spec = parse_env_xml(xml_path)
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", "Failed to load: " + str(e))
        return

    self.state.load(spec)
    _render_all(ui)

    port = _local_zmq_port()
    _set_status(ui, "Validating refs against the running scene...")
    threading.Thread(target=_validate_refs_worker, args=(port,), daemon=True).start()


def _validate_refs_worker(port):
    from coppelia_rl.sim_interface.client import SimClient

    try:
        client = SimClient.connect(host="localhost", port=port)
        missing = self.state.validate_refs(client)
        client.close()
        self.load_validate_result = missing
    except Exception as e:
        self.load_validate_result = str(e)


def onTestEnv(ui, *args):
    try:
        spec = self.state.build_spec()
    except Exception as e:
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", str(e))
        return

    if getattr(self, "test_env_thread", None) is not None and self.test_env_thread.is_alive():
        simUI.msgBox(simUI.msgbox_type.warning, simUI.msgbox_buttons.ok, "RL Env Editor", "A test run is already in progress.")
        return

    port = _local_zmq_port()
    _set_status(ui, "Test env running in the background...")
    self.test_env_thread = threading.Thread(target=_test_env_worker, args=(spec, port), daemon=True)
    self.test_env_thread.start()


def _test_env_worker(spec, port):
    import os
    import tempfile

    from coppelia_rl.env_schema import load_env
    from coppelia_rl.env_schema.serializer import save_spec_xml

    fd, tmp_path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        save_spec_xml(spec, tmp_path)
        env = load_env(tmp_path, host="localhost", port=port)
        try:
            obs, info = env.reset()
            episode_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < 50:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            msg = (
                "steps="
                + str(steps)
                + " return="
                + str(round(episode_reward, 3))
                + " terminated="
                + str(terminated)
                + " truncated="
                + str(truncated)
            )
            self.test_env_result = ("ok", msg)
        finally:
            env.close()
    except Exception as e:
        self.test_env_result = ("error", "Test env failed: " + str(e))
    finally:
        os.remove(tmp_path)
