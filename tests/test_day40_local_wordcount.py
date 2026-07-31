"""Pytest for day40-local wordcount."""

from __future__ import annotations

import importlib.util
import sys
from io import StringIO
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "homeworks" / "artifacts" / "day_40_local" / "wordcount.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("day40_local_wordcount", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_count_text_basic() -> None:
    words, lines, chars = _load().count_text("hello world\nfoo")
    assert words == 3
    assert lines == 2
    assert chars == 15


def test_main_from_file(tmp_path: Path, capsys) -> None:
    mod = _load()
    path = tmp_path / "sample.txt"
    path.write_text("a b c", encoding="utf-8")
    assert mod.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "Words: 3" in out
    assert "Lines: 1" in out
    assert "Chars: 5" in out


def test_main_from_stdin(monkeypatch, capsys) -> None:
    mod = _load()
    monkeypatch.setattr(sys, "stdin", StringIO("one two"))
    assert mod.main([]) == 0
    assert "Words: 2" in capsys.readouterr().out
