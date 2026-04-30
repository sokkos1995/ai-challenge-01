# MCP integration notes

Use this folder to add MCP client/provider for context sources.

Suggested next steps:
- implement `McpContextProvider` with methods:
  - `get_user(user_id: str)`
  - `get_ticket(ticket_id: str)`
- wire it into `app/main.py` instead of `LocalJsonContextProvider`
- keep a local JSON fallback for offline demo
