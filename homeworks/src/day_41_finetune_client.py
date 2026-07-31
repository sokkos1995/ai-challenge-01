#!/usr/bin/env python3
"""OpenAI fine-tuning client: upload file → create job → poll status.

By default runs in --dry-run mode (no network). Pass --execute to actually start a job.
Homework day_41: do NOT execute — only prepare the client.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import build_ssl_context, load_env_file  # noqa: E402

API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TRAIN = ROOT / "homeworks" / "artifacts" / "day_41" / "train.jsonl"


def _api_key() -> str:
    load_env_file()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key == "...":
        raise RuntimeError("OPENAI_API_KEY is required for --execute")
    return key


def _request(
    method: str,
    path: str,
    *,
    api_key: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    hdrs = {
        "Authorization": f"Bearer {api_key}",
        **(headers or {}),
    }
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers=hdrs,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=120, context=build_ssl_context()) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc


def upload_training_file(path: Path, api_key: str) -> str:
    """Multipart upload with purpose=fine-tune. Returns file id."""
    boundary = "----day41finetune"
    filename = path.name
    file_bytes = path.read_bytes()
    content_type = mimetypes.guess_type(filename)[0] or "application/json"

    parts: list[bytes] = []
    # purpose field
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"fine-tune\r\n"
        ).encode("utf-8")
    )
    # file field
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    result = _request(
        "POST",
        "/files",
        api_key=api_key,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    file_id = str(result.get("id") or "")
    if not file_id:
        raise RuntimeError(f"upload missing id: {result}")
    return file_id


def create_fine_tuning_job(
    *,
    api_key: str,
    training_file_id: str,
    model: str,
    suffix: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "training_file": training_file_id,
        "model": model,
    }
    if suffix:
        payload["suffix"] = suffix
    return _request(
        "POST",
        "/fine_tuning/jobs",
        api_key=api_key,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def get_job(api_key: str, job_id: str) -> dict[str, Any]:
    return _request("GET", f"/fine_tuning/jobs/{job_id}", api_key=api_key)


def poll_job(
    api_key: str,
    job_id: str,
    *,
    interval_sec: float = 15.0,
    max_wait_sec: float = 3600.0,
) -> dict[str, Any]:
    deadline = time.time() + max_wait_sec
    terminal = {"succeeded", "failed", "cancelled"}
    while True:
        job = get_job(api_key, job_id)
        status = str(job.get("status") or "")
        print(f"job {job_id} status={status}")
        if status in terminal:
            return job
        if time.time() >= deadline:
            raise RuntimeError(f"timeout waiting for job {job_id}, last status={status}")
        time.sleep(interval_sec)


def dry_run_plan(train_path: Path, model: str) -> dict[str, Any]:
    size = train_path.stat().st_size if train_path.is_file() else 0
    n_lines = 0
    if train_path.is_file():
        with train_path.open("r", encoding="utf-8") as fh:
            n_lines = sum(1 for line in fh if line.strip())
    return {
        "mode": "dry-run",
        "steps": [
            {
                "step": 1,
                "action": "upload",
                "method": "POST",
                "path": "/v1/files",
                "purpose": "fine-tune",
                "file": str(train_path),
                "bytes": size,
                "lines": n_lines,
            },
            {
                "step": 2,
                "action": "create_job",
                "method": "POST",
                "path": "/v1/fine_tuning/jobs",
                "model": model,
                "training_file": "<file_id from step 1>",
            },
            {
                "step": 3,
                "action": "poll",
                "method": "GET",
                "path": "/v1/fine_tuning/jobs/{job_id}",
                "until": ["succeeded", "failed", "cancelled"],
            },
        ],
        "note": "Pass --execute to run for real. day_41 homework: keep dry-run.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI fine-tuning client (upload→job→poll)")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suffix", default="aviation-day41")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call OpenAI API (default is dry-run)",
    )
    parser.add_argument("--poll-interval", type=float, default=15.0)
    args = parser.parse_args()

    if not args.execute:
        plan = dry_run_plan(args.train, args.model)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry-run only. Re-run with --execute to start a fine-tuning job.")
        return 0

    if not args.train.is_file():
        print(f"training file not found: {args.train}", file=sys.stderr)
        return 1

    api_key = _api_key()
    print(f"Uploading {args.train}…")
    file_id = upload_training_file(args.train, api_key)
    print(f"file_id={file_id}")

    print(f"Creating fine-tuning job model={args.model}…")
    job = create_fine_tuning_job(
        api_key=api_key,
        training_file_id=file_id,
        model=args.model,
        suffix=args.suffix,
    )
    job_id = str(job.get("id") or "")
    if not job_id:
        raise RuntimeError(f"create job missing id: {job}")
    print(f"job_id={job_id}")

    final = poll_job(api_key, job_id, interval_sec=args.poll_interval)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if str(final.get("status")) == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
