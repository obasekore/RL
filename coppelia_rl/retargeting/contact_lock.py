"""Contact-locking preprocessing (Doc 4's "Contact-aware refinement" pass).

Runs BEFORE Stage B: holds each contact-capable end-effector's target fixed at its
first-contact-frame position for the full duration of each contiguous contact window,
preventing frame-to-frame IK residual noise from producing foot slip during nominal
stance.
"""

from __future__ import annotations

import numpy as np


def contiguous_true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(N,) bool -> inclusive [start, end] index pairs, one per maximal True run."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(flags):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def lock_contacts(
    end_effector_targets: dict[str, np.ndarray],  # bone -> (N,3), root-relative, pre-lock
    contact: dict[str, np.ndarray],  # bone -> (N,) bool
    contact_capable_bones: set[str],
) -> dict[str, np.ndarray]:
    """Returns a NEW dict (never mutates the input). For each bone present in both
    `end_effector_targets` and `contact_capable_bones` with a contact channel, holds
    `curve[start:end+1] = curve[start]` (the first-contact-frame position) for every
    contiguous True run in `contact[bone]`. Bones without a contact channel, or not
    contact_capable, are passed through unchanged.
    """
    result: dict[str, np.ndarray] = {}
    for bone, curve in end_effector_targets.items():
        if bone not in contact_capable_bones or bone not in contact:
            result[bone] = curve
            continue
        locked = curve.copy()
        for start, end in contiguous_true_runs(contact[bone]):
            locked[start : end + 1] = locked[start]
        result[bone] = locked
    return result
