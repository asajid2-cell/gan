from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = REPO_ROOT / "saves2" / "real_music_transfer" / "manual_review_envelope075_strength3"
PASS_FIELDS = [
    "realism_pass",
    "source_identity_pass",
    "target_recognizable_pass",
    "artifact_free_pass",
    "novelty_pass",
]
BASELINE_OPTIONS = ["new", "baseline", "tie", "unclear"]
COMPLETE_VALUES = {"1", "true", "True", "yes", "Yes", "y", "Y"}


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    tmp = csv_path.with_suffix(".tmp")
    fieldnames = list(rows[0].keys())
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def _load_review_dir(review_dir: Path) -> Tuple[Path, List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    csv_path = Path(review_dir) / "manual_notes_template.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manual notes CSV: {csv_path}")
    rows = _read_rows(csv_path)
    by_case = {str(row.get("case_id", "")): row for row in rows}
    return csv_path, rows, by_case


def _load_priority_case_ids(review_dir: Path) -> List[str]:
    path = Path(review_dir) / "priority_cases.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        out: List[str] = []
        for row in csv.DictReader(f):
            case_id = str(row.get("case_id", "")).strip()
            if case_id and case_id not in out:
                out.append(case_id)
    return out


def _is_reviewed(row: Dict[str, str]) -> bool:
    return not _missing_review_fields(row)


def _missing_review_fields(row: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    if str(row.get("review_complete", "")).strip() not in COMPLETE_VALUES:
        missing.append("review_complete")
    for field in PASS_FIELDS:
        if str(row.get(field, "")).strip() not in {"0", "1"}:
            missing.append(field)
    if str(row.get("baseline_preference", "")).strip() not in set(BASELINE_OPTIONS):
        missing.append("baseline_preference")
    return missing


def _resolve_media(row: Dict[str, str], key: str) -> Path:
    raw = str(row.get(key, "")).strip()
    if not raw:
        raise FileNotFoundError(f"Missing media field {key}")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #1d1d1f; background: #f7f7f8; }}
    header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: 12px 18px; z-index: 2; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
    .bar {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .pill {{ background: #eceff3; border-radius: 999px; padding: 4px 10px; font-size: 13px; }}
    .case {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 12px; }}
    .audio {{ background: #f2f4f7; border: 1px solid #dde1e6; border-radius: 6px; padding: 10px; }}
    audio {{ width: 100%; }}
    label {{ display: block; font-weight: 600; margin-top: 10px; }}
    select, input[type=text], textarea {{ width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #c8cdd4; border-radius: 5px; }}
    textarea {{ min-height: 90px; }}
    button, a.button {{ display: inline-block; border: 0; background: #165dba; color: #fff; padding: 9px 12px; border-radius: 5px; text-decoration: none; cursor: pointer; }}
    button.secondary {{ background: #4b5563; }}
    button.warn {{ background: #9a3412; }}
    a.link {{ color: #165dba; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 14px; flex-wrap: wrap; }}
    .quick {{ display: flex; gap: 8px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
    .muted {{ color: #606873; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e1e4e8; padding: 8px; font-size: 14px; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>{body}</body>
</html>
""".encode("utf-8")


class ReviewState:
    def __init__(self, review_dir: Path):
        self.review_dir = Path(review_dir).resolve()
        self.csv_path, self.rows, self.by_case = _load_review_dir(self.review_dir)
        self.priority_case_ids = _load_priority_case_ids(self.review_dir)
        self.lock = threading.RLock()

    def reload(self) -> None:
        self.csv_path, self.rows, self.by_case = _load_review_dir(self.review_dir)
        self.priority_case_ids = _load_priority_case_ids(self.review_dir)

    def ordered_rows(self, priority: bool = False) -> List[Dict[str, str]]:
        if not priority:
            return list(self.rows)
        priority_set = set(self.priority_case_ids)
        priority_rows = [self.by_case[case_id] for case_id in self.priority_case_ids if case_id in self.by_case]
        remainder = [row for row in self.rows if str(row.get("case_id", "")) not in priority_set]
        return priority_rows + remainder

    def next_unreviewed(self, *, priority: bool = False, after_case_id: str = "") -> Optional[Dict[str, str]]:
        rows = self.ordered_rows(priority=priority)
        if not rows:
            return None
        start = 0
        if after_case_id:
            for idx, row in enumerate(rows):
                if str(row.get("case_id", "")) == str(after_case_id):
                    start = idx + 1
                    break
        for row in rows[start:] + rows[:start]:
            if not _is_reviewed(row):
                return row
        return rows[0]

    def status(self) -> Dict[str, Any]:
        with self.lock:
            reviewed = sum(1 for row in self.rows if _is_reviewed(row))
            priority_rows = [self.by_case[case_id] for case_id in self.priority_case_ids if case_id in self.by_case]
            priority_reviewed = sum(1 for row in priority_rows if _is_reviewed(row))
            invalid = [
                {
                    "case_id": str(row.get("case_id", "")),
                    "missing": _missing_review_fields(row),
                    "priority": str(row.get("case_id", "")) in set(self.priority_case_ids),
                }
                for row in self.rows
                if not _is_reviewed(row)
            ]
            return {
                "review_dir": str(self.review_dir),
                "manual_notes_csv": str(self.csv_path),
                "n_cases": len(self.rows),
                "reviewed": reviewed,
                "remaining": len(self.rows) - reviewed,
                "priority_cases": len(priority_rows),
                "priority_reviewed": priority_reviewed,
                "priority_remaining": len(priority_rows) - priority_reviewed,
                "gate_ready": reviewed == len(self.rows) and len(self.rows) > 0,
                "invalid_cases": len(invalid),
                "first_invalid_cases": invalid[:12],
            }

    def update_case(self, case_id: str, values: Dict[str, str]) -> None:
        with self.lock:
            self.reload()
            if case_id not in self.by_case:
                raise KeyError(case_id)
            row = self.by_case[case_id]
            row["reviewer"] = values.get("reviewer", "").strip()
            row["review_complete"] = "1" if values.get("review_complete") == "1" else "0"
            for field in PASS_FIELDS:
                row[field] = values.get(field, "").strip()
            row["baseline_preference"] = values.get("baseline_preference", "").strip()
            row["notes"] = values.get("notes", "").replace("\r\n", " ").replace("\n", " ").strip()
            _write_rows(self.csv_path, self.rows)
            self.reload()


def _select(name: str, value: str, options: List[Tuple[str, str]]) -> str:
    opts = []
    for opt_value, label in options:
        selected = " selected" if str(value) == opt_value else ""
        opts.append(f'<option value="{html.escape(opt_value)}"{selected}>{html.escape(label)}</option>')
    return f'<select name="{html.escape(name)}">{"".join(opts)}</select>'


def _case_form(state: ReviewState, row: Dict[str, str], message: str = "") -> str:
    case_id = str(row.get("case_id", ""))
    missing = _missing_review_fields(row)
    pass_options = [("", "unreviewed"), ("1", "pass"), ("0", "fail")]
    baseline_options = [("", "unreviewed"), *[(x, x) for x in BASELINE_OPTIONS]]
    metrics = [
        ("new_style_margin", row.get("new_style_margin", "")),
        ("baseline_style_margin", row.get("baseline_style_margin", "")),
        ("new_content_chroma", row.get("new_content_chroma", "")),
        ("baseline_content_chroma", row.get("baseline_content_chroma", "")),
        ("new_warble", row.get("new_warble", "")),
        ("baseline_warble", row.get("baseline_warble", "")),
    ]
    metric_html = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in metrics)
    complete_checked = " checked" if str(row.get("review_complete", "")).strip() in {"1", "true", "True"} else ""
    body = f"""
<header>
  <div class="bar">
    <strong>Manual Listening Review</strong>
    <span class="pill">{html.escape(case_id)}</span>
    <span class="pill">missing: {html.escape(", ".join(missing) if missing else "none")}</span>
    <span class="pill">source: {html.escape(row.get("source_genre", ""))}</span>
    <span class="pill">target: {html.escape(row.get("target_genre", ""))}</span>
    <a class="link" href="/">case list</a>
    <a class="link" href="/status">status</a>
  </div>
</header>
<main>
  {f'<p class="pill">{html.escape(message)}</p>' if message else ''}
  <section class="case">
    <div class="grid">
      <div class="audio"><strong>Source</strong><audio controls preload="none" src="/media/{html.escape(case_id)}/source_audio"></audio></div>
      <div class="audio"><strong>New Model</strong><audio controls preload="none" src="/media/{html.escape(case_id)}/new_wav"></audio></div>
      <div class="audio"><strong>Codec Baseline</strong><audio controls preload="none" src="/media/{html.escape(case_id)}/baseline_wav"></audio></div>
    </div>
    <table><tbody>{metric_html}</tbody></table>
    <form method="post" action="/save/{html.escape(case_id)}">
      <label>Reviewer</label><input type="text" name="reviewer" value="{html.escape(row.get("reviewer", ""))}">
      <label>Realism</label>{_select("realism_pass", row.get("realism_pass", ""), pass_options)}
      <label>Source Identity</label>{_select("source_identity_pass", row.get("source_identity_pass", ""), pass_options)}
      <label>Target Recognizable</label>{_select("target_recognizable_pass", row.get("target_recognizable_pass", ""), pass_options)}
      <label>Artifact Free</label>{_select("artifact_free_pass", row.get("artifact_free_pass", ""), pass_options)}
      <label>Novelty</label>{_select("novelty_pass", row.get("novelty_pass", ""), pass_options)}
      <label>Baseline Preference</label>{_select("baseline_preference", row.get("baseline_preference", ""), baseline_options)}
      <label>Notes</label><textarea name="notes">{html.escape(row.get("notes", ""))}</textarea>
      <label><input type="checkbox" name="review_complete" value="1"{complete_checked}> Review complete</label>
      <div class="quick">
        <button class="secondary" type="button" data-quick="new">Quick fill: new passes</button>
        <button class="secondary" type="button" data-quick="baseline">Quick fill: baseline preferred</button>
        <button class="secondary" type="button" data-quick="tie">Quick fill: tie</button>
        <button class="warn" type="button" data-quick="fail">Quick fill: fail review</button>
      </div>
      <div class="actions">
        <button type="submit" name="redirect" value="case">Save Review</button>
        <button type="submit" name="redirect" value="next">Save & Next</button>
        <button type="submit" name="redirect" value="next-priority">Save & Next Priority</button>
        <a class="button" href="/next">Next Unreviewed</a>
        <a class="button" href="/next-priority">Next Priority</a>
      </div>
    </form>
  </section>
  <script>
    const form = document.querySelector("form");
    const reviewer = form.elements["reviewer"];
    const savedReviewer = localStorage.getItem("realMusicReviewer");
    if (savedReviewer && !reviewer.value) reviewer.value = savedReviewer;
    reviewer.addEventListener("input", () => localStorage.setItem("realMusicReviewer", reviewer.value));

    function setPassFields(value) {{
      for (const field of {json.dumps(PASS_FIELDS)}) {{
        form.elements[field].value = value;
      }}
    }}
    document.querySelectorAll("[data-quick]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const mode = button.getAttribute("data-quick");
        form.elements["review_complete"].checked = true;
        if (mode === "fail") {{
          setPassFields("0");
          form.elements["baseline_preference"].value = "unclear";
          return;
        }}
        setPassFields("1");
        form.elements["baseline_preference"].value = mode;
      }});
    }});
  </script>
</main>
"""
    return body


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        state = self.server.state
        if path == "/status":
            body = json.dumps(state.status(), indent=2).encode("utf-8")
            self._send(body, content_type="application/json")
            return
        if path == "/invalid":
            with state.lock:
                state.reload()
                invalid_rows = [
                    row
                    for row in state.ordered_rows(priority=True)
                    if not _is_reviewed(row)
                ]
            table = []
            for row in invalid_rows:
                case_id = str(row.get("case_id", ""))
                missing = ", ".join(_missing_review_fields(row))
                table.append(
                    "<tr>"
                    f'<td><a href="/case/{urllib.parse.quote(case_id)}">{html.escape(case_id)}</a></td>'
                    f"<td>{html.escape(missing)}</td>"
                    f"<td>{html.escape(row.get('source_genre', ''))}</td>"
                    f"<td>{html.escape(row.get('target_genre', ''))}</td>"
                    "</tr>"
                )
            body = f"""
<header><div class="bar"><strong>Gate-Invalid Review Rows</strong><span class="pill">{len(invalid_rows)} invalid</span><a class="link" href="/next-priority">next priority</a><a class="link" href="/">case list</a><a class="link" href="/status">status json</a></div></header>
<main><table><thead><tr><th>Case</th><th>Missing / Invalid Fields</th><th>Source</th><th>Target</th></tr></thead><tbody>{''.join(table)}</tbody></table></main>
"""
            self._send(_page("Gate-Invalid Review Rows", body))
            return
        if path in {"/next", "/next-priority"}:
            with state.lock:
                state.reload()
                row = state.next_unreviewed(priority=path == "/next-priority")
            if row is None:
                self._send(_page("Manual Review", "<main>No cases found.</main>"), status=404)
                return
            self.send_response(302)
            self.send_header("Location", f"/case/{urllib.parse.quote(str(row.get('case_id', '')))}")
            self.end_headers()
            return
        if path.startswith("/case/"):
            case_id = urllib.parse.unquote(path.split("/", 2)[2])
            with state.lock:
                state.reload()
                row = state.by_case.get(case_id)
            if row is None:
                self._send(_page("Missing Case", f"<main>Unknown case {html.escape(case_id)}</main>"), status=404)
                return
            self._send(_page(f"Review {case_id}", _case_form(state, row)))
            return
        if path.startswith("/media/"):
            parts = path.split("/")
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            case_id = urllib.parse.unquote(parts[2])
            key = urllib.parse.unquote(parts[3])
            with state.lock:
                state.reload()
                row = state.by_case.get(case_id)
            if row is None or key not in {"source_audio", "new_wav", "baseline_wav"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                media = _resolve_media(row, key)
                data = media.read_bytes()
            except Exception as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self._send(data, content_type=mimetypes.guess_type(str(media))[0] or "application/octet-stream")
            return
        priority_mode = path == "/priority" or urllib.parse.parse_qs(parsed.query).get("priority", ["0"])[-1] == "1"
        with state.lock:
            state.reload()
            status = state.status()
            rows = state.ordered_rows(priority=priority_mode)
        table = []
        for row in rows:
            case_id = str(row.get("case_id", ""))
            done = "yes" if _is_reviewed(row) else "no"
            table.append(
                "<tr>"
                f'<td><a href="/case/{urllib.parse.quote(case_id)}">{html.escape(case_id)}</a></td>'
                f"<td>{done}</td>"
                f"<td>{html.escape(row.get('source_genre', ''))}</td>"
                f"<td>{html.escape(row.get('target_genre', ''))}</td>"
                f"<td>{html.escape(row.get('baseline_preference', ''))}</td>"
                "</tr>"
            )
        body = f"""
<header><div class="bar"><strong>Manual Listening Review</strong><span class="pill">{status['reviewed']} / {status['n_cases']} reviewed</span><span class="pill">priority {status['priority_reviewed']} / {status['priority_cases']}</span><span class="pill">gate ready: {str(status['gate_ready']).lower()}</span><a class="link" href="/next">next unreviewed</a><a class="link" href="/next-priority">next priority</a><a class="link" href="/">all cases</a><a class="link" href="/priority">priority order</a><a class="link" href="/invalid">invalid rows</a><a class="link" href="/status">status json</a></div></header>
<main>
  <p class="muted">Saving a case updates {html.escape(str(state.csv_path))}. After all rows are complete, rerun the listening audit and completion gate.</p>
  <table><thead><tr><th>Case</th><th>Reviewed</th><th>Source</th><th>Target</th><th>Preference</th></tr></thead><tbody>{''.join(table)}</tbody></table>
</main>
"""
        self._send(_page("Manual Listening Review", body))

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/save/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        case_id = urllib.parse.unquote(parsed.path.split("/", 2)[2])
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        values = {k: v[-1] for k, v in urllib.parse.parse_qs(payload, keep_blank_values=True).items()}
        try:
            self.server.state.update_case(case_id, values)
        except Exception as exc:
            self._send(_page("Save Failed", f"<main>{html.escape(str(exc))}</main>"), status=400)
            return
        redirect = values.get("redirect", "case")
        location = f"/case/{urllib.parse.quote(case_id)}?saved=1"
        if redirect in {"next", "next-priority"}:
            with self.server.state.lock:
                self.server.state.reload()
                row = self.server.state.next_unreviewed(
                    priority=redirect == "next-priority",
                    after_case_id=case_id,
                )
            if row is not None:
                location = f"/case/{urllib.parse.quote(str(row.get('case_id', '')))}"
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, addr: Tuple[str, int], state: ReviewState):
        super().__init__(addr, ReviewHandler)
        self.state = state


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local manual listening review server for real-music completion gate notes.")
    ap.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    state = ReviewState(Path(args.review_dir))
    server = ReviewServer((str(args.host), int(args.port)), state)
    url = f"http://{args.host}:{args.port}"
    print(json.dumps({"event": "manual_review_server_started", "url": url, **state.status()}, indent=2), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
