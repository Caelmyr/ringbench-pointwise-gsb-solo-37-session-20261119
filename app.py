"""RingBench local web entry point."""
import os
from pathlib import Path

from waitress import serve

from ringbench.web import create_app

ROOT = Path(__file__).resolve().parent


def main():
    state = Path(os.environ.get("RINGBENCH_DATA_DIR", ROOT / "data")).resolve()
    host = os.environ.get("RINGBENCH_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "18940"))
    app = create_app(state)
    serve(app, host=host, port=port, threads=5, max_request_body_size=148 * 1024 * 1024)


if __name__ == "__main__":
    main()
