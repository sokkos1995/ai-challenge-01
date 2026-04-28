# Developer Assistant Knowledge Base

This file is used as project documentation for RAG experiments.

## Project structure (short)

- `app/agent.py` — core LLM agent entrypoint and request pipeline.
- `app/cli.py` — CLI chat loop and command routing.
- `app/services/` — RAG, memory, personalization, and command services.
- `app/mcp_servers/` — local MCP servers (`todoist`, `github`, `git`).
- `homeworks/` — daily reports and artifacts for assignments.

## Day 31 command

- `/help` in chat mode prints available command groups.
- `/help` is handled before regular LLM generation.
- Command purpose: quickly explain project capabilities without guessing.
