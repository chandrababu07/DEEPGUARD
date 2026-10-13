"""
CLI server runner for DeepGuard FastAPI application.

Usage:
    python scripts/run_server.py --host 127.0.0.1 --port 8000 --reload
"""

import argparse
import os
import sys
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard API and Web Application Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print(" DEEPGUARD — DEEPFAKE DETECTION API & WEB SERVER")
    print("=" * 72)
    print(f"[*] Host        : http://{args.host}:{args.port}")
    print(f"[*] Web UI      : http://{args.host}:{args.port}/")
    print(f"[*] API Docs    : http://{args.host}:{args.port}/docs")
    print(f"[*] Reload      : {args.reload}")
    print("-" * 72)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
