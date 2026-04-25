"""
Launch the ARC AI server.

Usage:
    python -m scripts.serve
    python -m scripts.serve --port 8080
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = parser.parse_args()

    banner = f"""
  ╭──────────────────────────────────────────╮
  │   ARC AI — Maryland Housing & Rental Law │
  │                                          │
  │   Open → http://{args.host}:{args.port:<5}                │
  ╰──────────────────────────────────────────╯
"""
    print(banner)

    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()