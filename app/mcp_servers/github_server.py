import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import load_env_file
from app.mcp_servers._http import add_query, request_json

GITHUB_API_BASE = "https://api.github.com"

server = FastMCP(
    name="github-mcp-server",
    log_level="ERROR",
    instructions=(
        "MCP server for GitHub REST API. "
        "Exposes minimal tools to list issues and create issue comments."
    ),
)


def _github_token() -> str:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN in environment or .env.")
    return token


def _github_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


@server.tool()
def list_repo_issues(owner: str, repo: str, state: str = "open", limit: int = 20) -> dict[str, Any]:
    """List issues in a GitHub repository."""
    token = _github_token()
    safe_limit = max(1, min(limit, 100))
    url = add_query(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
        {"state": state, "per_page": str(safe_limit)},
    )
    issues = request_json(
        method="GET",
        url=url,
        token=token,
        extra_headers=_github_headers(),
    )
    issues_list = issues if isinstance(issues, list) else []
    return {
        "count": len(issues_list),
        "issues": issues_list,
    }


@server.tool()
def create_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
    """Create a comment in a GitHub issue."""
    token = _github_token()
    result = request_json(
        method="POST",
        url=f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments",
        token=token,
        payload={"body": body},
        extra_headers=_github_headers(),
    )
    return result if isinstance(result, dict) else {"raw": result}


if __name__ == "__main__":
    load_env_file()
    server.run(transport="stdio")
