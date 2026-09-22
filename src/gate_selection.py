"""Geometry-biology consistency gate metadata helpers."""

from __future__ import annotations


ABCDE_METHODS = {
    "A": "Rigid only",
    "B": "Rigid + GDEBA",
    "C": "Rigid + GDEBA + Gate",
    "D": "Rigid + GDEBA + Gate + NR",
    "E": "Original Spateo NR",
}


def describe_gate_status(gate_status: str, fallback_to_rigid: bool) -> str:
    """Return a human-readable gate decision string for summary reports."""
    if gate_status in (None, "", "nan"):
        return "not recorded"
    if fallback_to_rigid:
        return f"{gate_status}; fallback to rigid"
    return str(gate_status)
