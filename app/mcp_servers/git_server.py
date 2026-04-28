import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    name="git-mcp-server",
    log_level="ERROR",
    instructions=(
        "MCP server for local git repository. "
        "Exposes minimal tools to inspect the current branch and short status."
    ),
)


def _run_git(repo_path: str, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        error_message = process.stderr.strip() or process.stdout.strip() or "Unknown git error"
        raise RuntimeError(error_message)
    return process.stdout.strip()


@server.tool()
def get_current_branch(repo_path: str = ".") -> dict[str, Any]:
    """Return current branch for the given local git repository."""
    resolved_repo_path = str(Path(repo_path).resolve())
    branch = _run_git(resolved_repo_path, "branch", "--show-current")
    return {"repo_path": resolved_repo_path, "branch": branch}


@server.tool()
def get_short_status(repo_path: str = ".") -> dict[str, Any]:
    """Return `git status --short` output for the local repository."""
    resolved_repo_path = str(Path(repo_path).resolve())
    status = _run_git(resolved_repo_path, "status", "--short")
    lines = [line for line in status.splitlines() if line.strip()]
    return {"repo_path": resolved_repo_path, "count": len(lines), "lines": lines}


if __name__ == "__main__":
    server.run(transport="stdio")
