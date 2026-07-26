"""Pytest for day40-r2 bugfixes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "homeworks" / "artifacts" / "day_40_r2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_average_empty_and_values() -> None:
    mod = _load("day40_r2_avg", ART / "broken_avg.py")
    assert mod.average([]) is None
    assert mod.average([1.0, 3.0]) == 2.0


def test_last_n() -> None:
    mod = _load("day40_r2_slice", ART / "broken_slice.py")
    assert mod.last_n([1, 2, 3, 4, 5], 2) == [4, 5]
    assert mod.last_n([1, 2], 5) == [1, 2]


def test_append_item_no_shared_default() -> None:
    mod = _load("day40_r2_append", ART / "broken_append.py")
    assert mod.append_item("a") == ["a"]
    assert mod.append_item("b") == ["b"]
