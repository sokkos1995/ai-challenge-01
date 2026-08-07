"""FastAPI landing page with hidden indirect-injection payloads."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

ART = Path(__file__).resolve().parents[2] / "artifacts" / "day_47"
LANDING_PATH = ART / "payloads" / "landing.html"

app = FastAPI(title="Day 47 Aurora Tea Co landing", version="0.1.0")


def load_landing_html() -> str:
    return LANDING_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return HTMLResponse(content=load_landing_html())


@app.get("/raw", response_class=PlainTextResponse)
def landing_raw() -> PlainTextResponse:
    return PlainTextResponse(content=load_landing_html(), media_type="text/plain; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "homeworks.src.day_47_indirect_injection.landing_app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
