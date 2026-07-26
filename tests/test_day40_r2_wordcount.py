"""Pytest for day40-r2 wordcount."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "homeworks" / "artifacts" / "day_40_r2" / "wordcount.py"


def _load():
    spec = importlib.util.spec_from_file_location("day40_r2_wordcount", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_count_text_basic() -> None:
    words, lines, chars = _load().count_text("hello world\nfoo")
    assert words == 3 and lines == 2 and chars == 15


def test_main_from_file(tmp_path: Path, capsys) -> None:
    mod = _load()
    path = tmp_path / "sample.txt"
    path.write_text("a b c", encoding="utf-8")
    assert mod.main([str(path)]) == 0
    assert "words=3" in capsys.readouterr().out


def test_main_from_stdin(monkeypatch, capsys) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys.stdin, "read", lambda: "one two")
    assert mod.main([]) == 0
    assert "words=2" in capsys.readouterr().out
