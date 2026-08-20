"""Resolves `callable="module.path:attr"` escape-hatch references from XML.

Per the design constraint, every schema block accepts a `type="custom"`
entry pointing at user code instead of a built-in type, so power users are
never blocked by schema coverage gaps. The contract differs by block (see
generic_env.py's registries), but resolution is the same everywhere.
"""

from __future__ import annotations

import importlib


class InvalidCallableReferenceError(ValueError):
    pass


def resolve_callable(reference: str):
    """Resolves "module.path:attr" to the attr object (function or class instance)."""
    if ":" not in reference:
        raise InvalidCallableReferenceError(
            f"Expected 'module.path:attr', got {reference!r} (missing ':')"
        )
    module_name, _, attr_path = reference.partition(":")
    try:
        obj = importlib.import_module(module_name)
    except ImportError as exc:
        raise InvalidCallableReferenceError(f"Cannot import module {module_name!r}: {exc}") from exc

    for attr in attr_path.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            raise InvalidCallableReferenceError(
                f"{module_name!r} has no attribute path {attr_path!r}"
            ) from exc
    return obj
