"""Pytest for day40 FizzBuzz CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "homeworks" / "artifacts" / "day_40" / "fizzbuzz.py"


def _load():
    spec = importlib.util.spec_from_file_location("day40_fizzbuzz", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fizzbuzz_sequence() -> None:
    mod = _load()
    assert mod.fizzbuzz(15)[14] == "FizzBuzz"
    assert mod.fizzbuzz(3) == ["1", "2", "Fizz"]


def test_main_prints(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load()
    assert mod.main(["5"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["1", "2", "Fizz", "4", "Buzz"]


def test_main_usage_without_arg() -> None:
    mod = _load()
    assert mod.main([]) == 2
