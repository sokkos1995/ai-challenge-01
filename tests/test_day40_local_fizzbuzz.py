"""Pytest for day40-local FizzBuzz."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "homeworks" / "artifacts" / "day_40_local" / "fizzbuzz.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("day40_local_fizzbuzz", MODULE_PATH)
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
    # Local CLI treats argv like sys.argv (program name + N).
    assert mod.main(["fizzbuzz.py", "5"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["1", "2", "Fizz", "4", "Buzz"]


def test_main_usage_without_arg() -> None:
    assert _load().main(["fizzbuzz.py"]) == 2
