"""Reverse class-conditional control helpers."""

from __future__ import annotations


def reverse_exposure_effect(normal_mvg: float, reverse_mvg: float) -> float:
    """DeltaMVG_reverse = MVG_normal - MVG_reverse."""
    return normal_mvg - reverse_mvg


def reversal_happened(normal_mvg: float, reverse_mvg: float, drop: float = 0.05) -> bool:
    """True when reversing exposure materially reduces the vulnerability gap."""
    return reverse_exposure_effect(normal_mvg, reverse_mvg) >= drop
