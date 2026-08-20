"""Shared scene-tree mirror widget.

Used by both the RL Env Editor and the standalone RL Domain Randomizer
add-ons, factored out so tree behavior can't fork between them.

`sim`/`simUI` are CoppeliaSim add-on globals - not injected automatically
into a regularly-imported module like this one, so whichever app's
`create_ui()` uses this module must set them here first (mirrors the
`sim`/`simUI`/`self` injection pattern in rl_env_editor_app.py).
"""

from __future__ import annotations

sim = None
simUI = None

ID_TREE = 200
ID_REFRESH_TREE_BTN = 201

TREE_GROUP_XML = """
    <group flat="true" layout="vbox" stretch="1">
      <label text="Scene tree (click an object to select it)"/>
      <tree id="{ID_TREE}" show-header="false" on-selection-change="onTreeSelect"/>
      <button id="{ID_REFRESH_TREE_BTN}" text="Refresh tree" on-click="onRefreshTree"/>
    </group>
""".format(ID_TREE=ID_TREE, ID_REFRESH_TREE_BTN=ID_REFRESH_TREE_BTN)

__all__ = [
    "ID_TREE",
    "ID_REFRESH_TREE_BTN",
    "TREE_GROUP_XML",
    "refresh_tree",
    "selected_object_alias",
    "onTreeSelect",
    "onRefreshTree",
]


def refresh_tree(ui):
    simUI.clearTree(ui, ID_TREE)
    simUI.setColumnCount(ui, ID_TREE, 1)
    handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    parents = {}
    for h in handles:
        p = sim.getObjectParent(h)
        parents[h] = p if p != -1 else 0
    added = set()
    remaining = list(handles)
    while remaining:
        progressed = False
        still_remaining = []
        for h in remaining:
            p = parents[h]
            if p == 0 or p in added:
                alias = sim.getObjectAlias(h, -1)
                simUI.addTreeItem(ui, ID_TREE, h, [alias], p, True, True)
                added.add(h)
                progressed = True
            else:
                still_remaining.append(h)
        remaining = still_remaining
        if not progressed:
            break


def selected_object_alias():
    sel = sim.getObjectSel()
    if not sel:
        return None
    return sim.getObjectAlias(sel[0], -1)


def onTreeSelect(ui, tree_id, item_id):
    if item_id:
        sim.setObjectSel([item_id])


def onRefreshTree(ui, *args):
    refresh_tree(ui)
