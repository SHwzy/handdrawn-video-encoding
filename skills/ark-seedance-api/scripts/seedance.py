"""Minimal Seedance client for the Volcano Ark Agent Plan API."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

BASE = "https://ark.cn-beijing.volces.com/api/plan/v3/contents/generations/tasks"
MODEL = "doubao-seedance-2.5"


def request(method, url, body=None):
    key = os.environ.get("ARK_API_KEY")
    if not key:
        local = Path(__file__).resolve().parents[3] / "password.json"
        if local.is_file():
            key = json.loads(local.read_text(encoding="utf-8-sig")).get("password")
    if not key:
        raise RuntimeError("Agent Plan API Key is missing; set ARK_API_KEY or create project password.json")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else {"ok": True}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from None


def get(task_id):
    if not task_id.startswith("cgt-") or "/" in task_id or "?" in task_id:
        raise ValueError("Expected a cgt- task ID")
    return request("GET", f"{BASE}/{task_id}")


def main():
    parser = argparse.ArgumentParser(description="Seedance 2.5 via the standard Volcano Ark API")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Submit one billable generation task")
    inputs = create.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="Text-to-video prompt")
    inputs.add_argument("--payload", type=Path, help="JSON body for official multimodal API")
    create.add_argument("--model", default=MODEL)
    create.add_argument("--duration", type=int)
    create.add_argument("--ratio", choices=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"])
    create.add_argument("--resolution", choices=["480p", "720p", "1080p"])
    create.add_argument("--no-audio", action="store_true")
    create.add_argument("--dry-run", action="store_true", help="Print request body without submitting")
    for name in ("get", "wait", "download"):
        p = sub.add_parser(name)
        p.add_argument("task_id")
        if name == "wait":
            p.add_argument("--timeout", type=int, default=600)
            p.add_argument("--interval", type=int, default=10)
        if name == "download":
            p.add_argument("--output", type=Path, required=True)
    listing = sub.add_parser("list", help="List recent video tasks")
    listing.add_argument("--page-num", type=int, default=1)
    listing.add_argument("--page-size", type=int, default=10)
    listing.add_argument("--status", choices=["queued", "running", "succeeded", "failed", "cancelled"])
    listing.add_argument("--model")
    deleting = sub.add_parser("delete", help="Cancel or delete a video task")
    deleting.add_argument("task_id")
    args = parser.parse_args()
    if args.command == "create":
        if args.payload:
            body = json.loads(args.payload.read_text(encoding="utf-8"))
            body.setdefault("model", args.model)
        else:
            body = {"model": args.model, "content": [{"type": "text", "text": args.prompt}]}
            if args.duration is not None:
                if args.duration != -1 and not 4 <= args.duration <= 30:
                    parser.error("Seedance 2.5 duration must be -1 or 4..30")
                body["duration"] = args.duration
            for field in ("ratio", "resolution"):
                value = getattr(args, field)
                if value:
                    body[field] = value
            if args.no_audio:
                body["generate_audio"] = False
        print(json.dumps(body if args.dry_run else request("POST", BASE, body), ensure_ascii=False, indent=2))
    elif args.command == "get":
        print(json.dumps(get(args.task_id), ensure_ascii=False, indent=2))
    elif args.command == "list":
        params = {"page_num": args.page_num, "page_size": args.page_size}
        if args.status:
            params["filter.status"] = args.status
        if args.model:
            params["filter.model"] = args.model
        print(json.dumps(request("GET", BASE + "?" + urllib.parse.urlencode(params)), ensure_ascii=False, indent=2))
    elif args.command == "delete":
        get(args.task_id)  # Validate task ID before sending DELETE.
        print(json.dumps(request("DELETE", f"{BASE}/{args.task_id}"), ensure_ascii=False, indent=2))
    elif args.command == "wait":
        deadline = time.monotonic() + args.timeout
        while True:
            result = get(args.task_id)
            status = result.get("status")
            print(status, file=sys.stderr)
            if status in ("succeeded", "failed", "cancelled"):
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if status == "succeeded" else 1
            if time.monotonic() >= deadline:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 2
            time.sleep(max(1, args.interval))
    else:
        result = get(args.task_id)
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Task status is {result.get('status')}, not succeeded")
        url = result.get("content", {}).get("video_url")
        if not url:
            raise RuntimeError("Succeeded task did not contain content.video_url")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as source, args.output.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
