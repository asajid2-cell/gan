from __future__ import annotations

import argparse

import gradio as gr

from gui.app import APP_CSS, build_app
from gui.backend import OUTPUT_ROOT, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the DGGR local inference GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for the local server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share links.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app()
    app.queue(default_concurrency_limit=2)
    app.launch(
        server_name=args.host,
        server_port=int(args.port),
        inbrowser=not bool(args.no_browser),
        share=bool(args.share),
        debug=bool(args.debug),
        show_error=True,
        theme=gr.themes.Soft(),
        css=APP_CSS,
        allowed_paths=[str(OUTPUT_ROOT), str(REPO_ROOT / "examples")],
    )


if __name__ == "__main__":
    main()
