"""Shared "Domain Randomization" tab widget.

Used by both the RL Env Editor add-on (where it's one tab of a larger
EnvSpec) and the standalone RL Domain Randomizer add-on (where it's the
whole panel) - factored out here so the two add-ons can't drift into
separate implementations. Works against `self.state` polymorphically: both
`EnvBuilderState` and `DomainRandomizerState` expose the same
add_texture_randomization/add_mass_randomization/domain_randomization
surface (see coppelia_rl/randomization/randomizer_state.py), so this module
never needs to know which one it's talking to.

`sim`/`simUI`/`self` are CoppeliaSim add-on globals - not injected
automatically into a regularly-imported module like this one, so whichever
app's `create_ui()` uses this module must set them here first (mirrors the
`sim`/`simUI`/`self` injection pattern in rl_env_editor_app.py).

Camera jitter, friction, and latency (action_delay_steps/observation_noise)
are parsed/serialized by the schema but intentionally have no editing
widgets here yet - texture/mass/resample_on only, matching the scope of the
GUI editor built. Hand-editing XML remains the escape hatch
for those, same as every other schema-coverage gap in this project.
"""

from __future__ import annotations

from coppelia_rl.gui.scene_tree_panel import selected_object_alias

sim = None
simUI = None
self = None

# Called after every DR state mutation, so each app can refresh whatever
# larger preview it owns (e.g. rl_env_editor_app's full XML preview). Each
# app's create_ui() overrides this before rendering; defaults to a no-op so
# the module is usable stand-alone without an override.
on_change_hook = lambda ui: None  # noqa: E731

ID_DR_TABLE = 210
ID_DR_TEXTURE_REF_EDIT = 211
ID_DR_TEXTURE_POOL_EDIT = 212
ID_DR_TEXTURE_ADD_BTN = 213
ID_DR_MASS_REF_EDIT = 214
ID_DR_MASS_RANGE_EDIT = 215
ID_DR_MASS_ADD_BTN = 216
ID_DR_RESAMPLE_COMBO = 217
ID_DR_TEXTURE_USE_BTN = 218
ID_DR_MASS_USE_BTN = 219
ID_DR_REMOVE_BTN = 220

RESAMPLE_ON_VALUES = ["episode_start", "n_steps", "once"]

DR_TAB_XML = """
    <tab title="Domain Randomization">
        <table id="{ID_DR_TABLE}" show-horizontal-header="true" selection-mode="row" editable="false" on-selection-change="onDrTableSelect"/>
        <group flat="true" layout="form">
          <label text="texture ref:"/>
          <edit id="{ID_DR_TEXTURE_REF_EDIT}" value=""/>
          <label text="texture pool (e.g. textures/wood_*):"/>
          <edit id="{ID_DR_TEXTURE_POOL_EDIT}" value=""/>
          <label text="mass ref:"/>
          <edit id="{ID_DR_MASS_REF_EDIT}" value=""/>
          <label text="mass range [lo,hi]:"/>
          <edit id="{ID_DR_MASS_RANGE_EDIT}" value=""/>
          <label text="resample_on:"/>
          <combobox id="{ID_DR_RESAMPLE_COMBO}"><item>episode_start</item><item>n_steps</item><item>once</item></combobox>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_DR_TEXTURE_USE_BTN}" text="Sel -> texture ref" on-click="onDrUseSelectedTexture"/>
          <button id="{ID_DR_MASS_USE_BTN}" text="Sel -> mass ref" on-click="onDrUseSelectedMass"/>
        </group>
        <group flat="true" layout="hbox">
          <button id="{ID_DR_TEXTURE_ADD_BTN}" text="Add texture randomization" on-click="onDrAddTexture"/>
          <button id="{ID_DR_MASS_ADD_BTN}" text="Add mass randomization" on-click="onDrAddMass"/>
          <button id="{ID_DR_REMOVE_BTN}" text="Remove selected" on-click="onDrRemove"/>
        </group>
    </tab>
""".format(
    ID_DR_TABLE=ID_DR_TABLE,
    ID_DR_TEXTURE_REF_EDIT=ID_DR_TEXTURE_REF_EDIT,
    ID_DR_TEXTURE_POOL_EDIT=ID_DR_TEXTURE_POOL_EDIT,
    ID_DR_TEXTURE_ADD_BTN=ID_DR_TEXTURE_ADD_BTN,
    ID_DR_MASS_REF_EDIT=ID_DR_MASS_REF_EDIT,
    ID_DR_MASS_RANGE_EDIT=ID_DR_MASS_RANGE_EDIT,
    ID_DR_MASS_ADD_BTN=ID_DR_MASS_ADD_BTN,
    ID_DR_RESAMPLE_COMBO=ID_DR_RESAMPLE_COMBO,
    ID_DR_TEXTURE_USE_BTN=ID_DR_TEXTURE_USE_BTN,
    ID_DR_MASS_USE_BTN=ID_DR_MASS_USE_BTN,
    ID_DR_REMOVE_BTN=ID_DR_REMOVE_BTN,
)

__all__ = [
    "ID_DR_TABLE",
    "ID_DR_TEXTURE_REF_EDIT",
    "ID_DR_TEXTURE_POOL_EDIT",
    "ID_DR_TEXTURE_ADD_BTN",
    "ID_DR_MASS_REF_EDIT",
    "ID_DR_MASS_RANGE_EDIT",
    "ID_DR_MASS_ADD_BTN",
    "ID_DR_RESAMPLE_COMBO",
    "ID_DR_TEXTURE_USE_BTN",
    "ID_DR_MASS_USE_BTN",
    "ID_DR_REMOVE_BTN",
    "DR_TAB_XML",
    "render_domain_randomization_tab",
    "onDrTableSelect",
    "onDrUseSelectedTexture",
    "onDrUseSelectedMass",
    "onDrAddTexture",
    "onDrAddMass",
    "onDrRemove",
]


def _parse_range(text):
    text = text.strip().lstrip("[").rstrip("]")
    lo, hi = text.split(",")
    return (float(lo), float(hi))


def render_domain_randomization_tab(ui):
    dr = self.state.domain_randomization
    rows = []
    for t in dr.textures:
        rows.append(["texture", t.ref, "pool=" + t.pool])
    for m in dr.masses:
        rows.append(["mass", m.ref, str(m.value_range)])
    resample_detail = dr.resample_on
    if dr.resample_every_n_steps:
        resample_detail += f" every_n_steps={dr.resample_every_n_steps}"
    rows.append(["resample_on", "", resample_detail])

    simUI.setColumnCount(ui, ID_DR_TABLE, 3)
    for i, h in enumerate(["Category", "Ref", "Detail"]):
        simUI.setColumnHeaderText(ui, ID_DR_TABLE, i, h)
    simUI.setRowCount(ui, ID_DR_TABLE, len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            simUI.setItem(ui, ID_DR_TABLE, r, c, str(value) if value is not None else "")


def onDrUseSelectedTexture(ui, *args):
    alias = selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_DR_TEXTURE_REF_EDIT, alias, True)


def onDrUseSelectedMass(ui, *args):
    alias = selected_object_alias()
    if alias:
        simUI.setEditValue(ui, ID_DR_MASS_REF_EDIT, alias, True)


def onDrAddTexture(ui, *args):
    ref = simUI.getEditValue(ui, ID_DR_TEXTURE_REF_EDIT)
    pool = simUI.getEditValue(ui, ID_DR_TEXTURE_POOL_EDIT)
    if ref and pool:
        self.state.add_texture_randomization(ref=ref, pool=pool)
        render_domain_randomization_tab(ui)
        on_change_hook(ui)


def onDrAddMass(ui, *args):
    ref = simUI.getEditValue(ui, ID_DR_MASS_REF_EDIT)
    range_text = simUI.getEditValue(ui, ID_DR_MASS_RANGE_EDIT)
    if ref and range_text:
        self.state.add_mass_randomization(ref=ref, value_range=_parse_range(range_text))
        render_domain_randomization_tab(ui)
        on_change_hook(ui)


def onDrTableSelect(ui, table_id, row, col):
    self.selected_dr_row = row


def onDrRemove(ui, *args):
    """Maps the selected table row back to its underlying list + index - the
    table mixes textures/masses/the resample_on summary into one flat view
    (see render_domain_randomization_tab), so removal has to un-mix them."""
    row = getattr(self, "selected_dr_row", None)
    if row is None:
        return

    dr = self.state.domain_randomization
    n_textures = len(dr.textures)
    n_masses = len(dr.masses)
    if row < n_textures:
        self.state.remove_texture_randomization(row)
    elif row < n_textures + n_masses:
        self.state.remove_mass_randomization(row - n_textures)
    else:
        return  # the resample_on summary row isn't removable

    self.selected_dr_row = None
    render_domain_randomization_tab(ui)
    on_change_hook(ui)
