import argparse
import ast
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_TOP_LEVEL_SIDE_EFFECT_CALL_NAMES = {
    "run",
    "main",
    "server.run",
    "anyio.run",
}


def _run_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # Never raise on non-zero; CI should still produce a comment.
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _get_git_lines(args: list[str], cwd: Path) -> list[str]:
    res = _run_capture(["git", *args], cwd=cwd)
    if res.returncode != 0:
        return []
    return [line.rstrip("\n") for line in res.stdout.splitlines()]


def _read_all_py_files(repo_root: Path) -> list[Path]:
    excluded_dirnames = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".venv",
        "venv",
    }
    py_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirnames]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            if fn.endswith(".pyc"):
                continue
            full = Path(dirpath) / fn
            py_files.append(full)
    py_files.sort()
    return py_files


def _module_name_for_file(repo_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_root).as_posix()
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    else:
        rel = rel[: -len(".py")]
    return rel.replace("/", ".")


def _resolve_relative_import(current_module: str, level: int, module: Optional[str]) -> str:
    # current_module is something like "app.services.foo"
    # level=1 means "from .bar import X" -> current package
    # level=2 means "from ..bar import X" -> parent package
    parts = current_module.split(".")
    if len(parts) >= 2:
        package_parts = parts[:-1]  # treat current module as submodule
    else:
        package_parts = parts

    # level=1 -> keep package_parts; level=2 -> remove one; etc.
    cut = max(0, len(package_parts) - (level - 1))
    base_parts = package_parts[:cut]
    if module:
        return ".".join([*base_parts, *module.split(".")]) if base_parts else module
    return ".".join(base_parts)


def _collect_import_edges(repo_root: Path) -> tuple[dict[str, set[str]], list[str]]:
    py_files = _read_all_py_files(repo_root)
    module_for_path: dict[Path, str] = {p: _module_name_for_file(repo_root, p) for p in py_files}
    all_modules: set[str] = set(module_for_path.values())

    # Adjacency list: from_module -> set(to_module)
    edges: dict[str, set[str]] = defaultdict(set)
    side_effect_files: list[str] = []

    for file_path in py_files:
        current_module = module_for_path[file_path]
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            # compileall should have caught it, but keep analysis robust
            continue
        except OSError:
            continue

        # Heuristic for side effects: top-level Expr/Call that is not within if __name__ == "__main__".
        # This is not perfect, but good enough for a PR review heuristic.
        has_main_guard = False
        for node in tree.body:
            if isinstance(node, ast.If):
                # Detect if __name__ == "__main__"
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    has_main_guard = True
                    continue

        if not has_main_guard:
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    side_effect_files.append(current_module)
                    break
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    side_effect_files.append(current_module)
                    break

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in all_modules:
                        edges[current_module].add(name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if node.level and node.level > 0:
                    resolved = _resolve_relative_import(current_module, node.level, module)
                else:
                    resolved = module or ""

                if not resolved:
                    continue

                # If it's "from X import Y", X could be a package/module.
                # We'll try to connect both X and X.Y when those are known modules.
                if resolved in all_modules:
                    edges[current_module].add(resolved)
                if node.names:
                    first = node.names[0].name
                    candidate = f"{resolved}.{first}"
                    if candidate in all_modules:
                        edges[current_module].add(candidate)

    # Remove duplicates while preserving sort stability
    side_effect_files = sorted(set(side_effect_files))
    return edges, side_effect_files


def _find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    # Detect cycles with DFS. Return a list of cycles (may not be minimal).
    visited: set[str] = set()
    in_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        visited.add(node)
        in_stack.add(node)
        stack.append(node)
        for nxt in sorted(edges.get(node, set())):
            if nxt not in visited:
                dfs(nxt)
            elif nxt in in_stack:
                # extract cycle segment
                if nxt in stack:
                    idx = stack.index(nxt)
                    cycle = stack[idx:] + [nxt]
                    # normalize for duplicate avoidance
                    # rotate so the smallest string comes first
                    if len(cycle) >= 2:
                        core = cycle[:-1]
                        min_idx = min(range(len(core)), key=lambda i: core[i])
                        rotated = core[min_idx:] + core[:min_idx]
                        canonical = tuple(rotated)
                        if not any(tuple(c[:-1]) == canonical for c in cycles):
                            cycles.append(cycle)
        stack.pop()
        in_stack.remove(node)

    for node in sorted(edges.keys()):
        if node not in visited:
            dfs(node)

    cycles.sort(key=len)
    return cycles


def _classify_commit_message(msg: str) -> str:
    text = (msg or "").strip().lower()
    if not text:
        return "unknown"
    # Keep heuristics simple: keywords drive expectations for review focus.
    if re.search(r"\b(fix|bug|hotfix)\b", text):
        return "fix"
    if re.search(r"\b(refactor|rename|cleanup)\b", text):
        return "refactor"
    if re.search(r"\b(feat|feature|add)\b", text):
        return "feature"
    if re.search(r"\b(test|tests|qa)\b", text):
        return "tests"
    if re.search(r"\b(docs|readme)\b", text):
        return "docs"
    return "other"


def _analyze_commit_messages(repo_root: Path, base_sha: str, head_sha: str) -> tuple[list[dict], Counter[str]]:
    # Use a sentinel delimiter to safely parse multiline commit messages.
    sentinel = "\x1e"
    sep = "\x1f"
    fmt = f"%H{sep}%s{sep}%B{sentinel}"
    res = _run_capture(
        [
            "git",
            "log",
            f"--format={fmt}",
            f"{base_sha}..{head_sha}",
        ],
        cwd=repo_root,
    )
    text = (res.stdout or "").strip()
    commits: list[dict] = []
    if not text:
        # Fallback: at least one commit
        lines = _get_git_lines(["log", "-1", "--format=%H%x1f%s%x1f%B"], cwd=repo_root)
        if not lines:
            return [], Counter()
        # parse single line
        parts = lines[0].split("\x1f")
        sha, subject, body = parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""
        commits.append({"sha": sha, "subject": subject, "body": body})
    else:
        chunks = [c for c in text.split(sentinel) if c.strip()]
        for chunk in chunks:
            parts = chunk.split(sep)
            sha = parts[0]
            subject = parts[1] if len(parts) > 1 else ""
            body = parts[2] if len(parts) > 2 else ""
            commits.append({"sha": sha, "subject": subject, "body": body})

    types = Counter(_classify_commit_message(c.get("subject", "") + "\n" + c.get("body", "")) for c in commits)
    return commits, types


@dataclass
class RunResult:
    ok: bool
    returncode: int
    output_tail: str


def _tail_text(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_compile(repo_root: Path) -> RunResult:
    res = _run_capture([sys.executable, "-m", "compileall", "."], cwd=repo_root)
    tail = (res.stdout + "\n" + res.stderr).strip()
    return RunResult(ok=res.returncode == 0, returncode=res.returncode, output_tail=_tail_text(tail))


def _run_pytest(repo_root: Path) -> RunResult:
    res = _run_capture([sys.executable, "-m", "pytest", "-q"], cwd=repo_root)
    tail = (res.stdout + "\n" + res.stderr).strip()
    return RunResult(ok=res.returncode == 0, returncode=res.returncode, output_tail=_tail_text(tail))


def _get_changed_files(repo_root: Path, base_sha: str, head_sha: str) -> list[str]:
    files = _get_git_lines(["diff", "--name-only", f"{base_sha}..{head_sha}"], cwd=repo_root)
    return files


def _format_commit_overview(commit_types: Counter[str]) -> str:
    if not commit_types:
        return "не удалось определить"
    parts = [f"{k}: {v}" for k, v in commit_types.most_common()]
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repo", required=False, default="")
    parser.add_argument("--pr-number", required=False, default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    changed_files = _get_changed_files(repo_root, args.base_sha, args.head_sha)
    py_changed = [p for p in changed_files if p.endswith(".py")]

    commits, commit_types = _analyze_commit_messages(repo_root, args.base_sha, args.head_sha)
    commit_overview = _format_commit_overview(commit_types)

    compile_res = _run_compile(repo_root)
    pytest_res = _run_pytest(repo_root)

    edges, side_effect_files = _collect_import_edges(repo_root)
    cycles = _find_cycles(edges) if edges else []

    # ---- Build comment ----
    title = "AI Code Review (Day 32)"
    pr_hint = ""
    if args.repo and args.pr_number:
        pr_hint = f"\nRepo: {args.repo}\nPR: #{args.pr_number}"

    body_parts: list[str] = [f"{title}{pr_hint}"]

    body_parts.append("\n## Потенциальные баги")
    if pytest_res.ok:
        body_parts.append("- замечаний нет")
    else:
        body_parts.append(f"- `pytest` завершился с ошибкой (exit code: {pytest_res.returncode}).")
        body_parts.append("")
        body_parts.append("Фрагменты вывода:")
        body_parts.append("```")
        body_parts.append(pytest_res.output_tail or "(empty output)")
        body_parts.append("```")
        if py_changed:
            body_parts.append("")
            body_parts.append(f"- Изменены Python-файлы: {', '.join(py_changed[:15])}{'...' if len(py_changed) > 15 else ''}")

    body_parts.append("\n## Архитектурные проблемы")
    arch_issues: list[str] = []
    if cycles:
        # Keep comment short: show up to 3 cycles
        for i, cycle in enumerate(cycles[:3], start=1):
            arch_issues.append(f"{i}) цикл импортов: {' -> '.join(cycle[: min(len(cycle), 7)])}")
    if side_effect_files:
        top = side_effect_files[:10]
        arch_issues.append(
            "в некоторых модулях есть возможные сайд-эффекты при импорте (топ-левел вызовы): "
            + ", ".join(top)
            + ("..." if len(side_effect_files) > 10 else "")
        )
    if not arch_issues:
        body_parts.append("- замечаний нет")
    else:
        body_parts.extend(arch_issues)

    body_parts.append("\n## Синтаксис")
    if compile_res.ok:
        body_parts.append("- замечаний нет")
    else:
        body_parts.append(f"- `compileall` нашел ошибки (exit code: {compile_res.returncode}).")
        body_parts.append("")
        body_parts.append("Фрагменты вывода:")
        body_parts.append("```")
        body_parts.append(compile_res.output_tail or "(empty output)")
        body_parts.append("```")

    body_parts.append("\n## Рекомендации")
    recs: list[str] = []
    recs.append(f"- Типы commit-сообщений (эвристика): {commit_overview}.")
    if not compile_res.ok:
        recs.append("- Сначала исправьте синтаксические ошибки (compileall), затем повторите проверки.")
    if not pytest_res.ok:
        recs.append("- Устраните причины падения pytest; целесообразно добавлять/уточнять тесты вокруг измененных модулей.")
    else:
        recs.append("- Все проверки прошли. Дополнительно проверьте крайние случаи и регрессию для измененных модулей (и добавьте тесты при отсутствии покрытия).")

    if py_changed:
        recs.append("- Рекомендуемый фокус: изменения в " + ", ".join(py_changed[:10]) + ("..." if len(py_changed) > 10 else ""))
    if cycles:
        recs.append("- Для циклов импортов рассмотрите разрыв зависимости (интерфейсы/слои, перенос функций в более подходящие модули).")
    body_parts.extend(recs)

    output_path = Path(args.output)
    output_path.write_text("\n".join(body_parts) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

