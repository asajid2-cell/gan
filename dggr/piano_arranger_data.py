from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "data" / "real_music_manifests" / "spotify_discovered_genres.csv"
DEFAULT_DISCOVERY_REPORT = REPO_ROOT / "data" / "real_music_manifests" / "spotify_discovered_genres_report.json"
DEFAULT_PIANO_MANIFEST = REPO_ROOT / "data" / "piano_arranger_manifests" / "piano_candidates.csv"
DEFAULT_MIDI_MANIFEST = REPO_ROOT / "data" / "piano_arranger_manifests" / "midi_piano_targets.csv"
DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST = REPO_ROOT / "data" / "piano_arranger_manifests" / "paired_audio_midi_targets.csv"
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif"}
MIDI_SUFFIXES = {".mid", ".midi", ".kar"}


POSITIVE_TERMS: Dict[str, float] = {
    "piano": 10.0,
    "pianoforte": 10.0,
    "keyboard": 4.0,
    "keys": 4.0,
    "sonata": 5.0,
    "nocturne": 5.0,
    "prelude": 3.5,
    "etude": 4.0,
    "waltz": 2.0,
    "rag": 2.0,
    "ragtime": 3.0,
    "classical": 2.0,
    "acoustic": 2.0,
    "solo": 1.0,
    "instrumental": 1.0,
}

NEGATIVE_TERMS: Dict[str, float] = {
    "noise": -8.0,
    "white": -6.0,
    "airplane": -6.0,
    "sleep": -3.0,
    "meditation": -2.0,
    "recitation": -5.0,
    "podcast": -5.0,
    "speech": -5.0,
    "remix": -1.0,
}


@dataclass(frozen=True)
class PianoManifestSummary:
    source_manifest: str
    out_csv: str
    report_path: str
    input_rows: int
    selected_rows: int
    min_score: float
    top_genres: Dict[str, int]
    top_examples: List[Dict[str, Any]]


@dataclass(frozen=True)
class MidiManifestSummary:
    out_csv: str
    report_path: str
    roots: List[str]
    discovered_files: int
    selected_rows: int
    min_notes: int
    errors: int
    top_examples: List[Dict[str, Any]]


@dataclass(frozen=True)
class PairedManifestSummary:
    out_csv: str
    report_path: str
    audio_roots: List[str]
    midi_roots: List[str]
    discovered_audio_files: int
    discovered_midi_files: int
    selected_rows: int
    min_notes: int
    errors: int
    top_examples: List[Dict[str, Any]]


@dataclass(frozen=True)
class PairedAuditSummary:
    manifest: str
    report_path: str
    rows_csv: str
    passed_manifest_csv: str
    input_rows: int
    passed_rows: int
    warning_counts: Dict[str, int]
    top_examples: List[Dict[str, Any]]


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _term_score(text: str) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    for tok in _tokens(text):
        if tok in POSITIVE_TERMS:
            score += POSITIVE_TERMS[tok]
            reasons.append(f"+{tok}")
        if tok in NEGATIVE_TERMS:
            score += NEGATIVE_TERMS[tok]
            reasons.append(f"-{tok}")
    return float(score), reasons


def _cluster_boosts(discovery_report: Path) -> Dict[str, float]:
    path = Path(discovery_report)
    if not path.exists():
        return {}
    report = _read_json(path)
    boosts: Dict[str, float] = {}
    for cluster in report.get("clusters", []):
        genre = str(cluster.get("genre", ""))
        if not genre:
            continue
        text_parts: List[str] = [genre, " ".join(str(t) for t in cluster.get("top_terms", []))]
        for ex in cluster.get("examples", [])[:16]:
            text_parts.extend([str(ex.get("title", "")), str(ex.get("album", "")), str(ex.get("artist", ""))])
        score, _ = _term_score(" ".join(text_parts))
        profile = cluster.get("audio_profile", {})
        flatness = float(profile.get("flatness_mean", 0.0) or 0.0)
        centroid = float(profile.get("centroid_mean", 0.0) or 0.0)
        if flatness < 0.004:
            score += 1.0
        if 500.0 <= centroid <= 2600.0:
            score += 1.0
        boosts[genre] = max(0.0, float(score) * 0.15)
    return boosts


def score_manifest_row(row: pd.Series, genre_boosts: Dict[str, float] | None = None) -> Tuple[float, str]:
    fields = [
        str(row.get("title", "")),
        str(row.get("artist", "")),
        str(row.get("album", "")),
        str(row.get("tag_genre", "")),
        str(row.get("genre", "")),
    ]
    text_score, reasons = _term_score(" ".join(fields))
    genre = str(row.get("genre", ""))
    boost = float((genre_boosts or {}).get(genre, 0.0))
    score = float(text_score + boost)
    if boost > 0.0:
        reasons.append(f"+cluster_boost:{boost:.2f}")
    return score, ";".join(reasons)


def build_piano_candidate_manifest(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    discovery_report: Path = DEFAULT_DISCOVERY_REPORT,
    out_csv: Path = DEFAULT_PIANO_MANIFEST,
    report_path: Path | None = None,
    min_score: float = 8.0,
    max_rows: int = 0,
) -> PianoManifestSummary:
    source_manifest = Path(source_manifest)
    out_csv = Path(out_csv)
    report_path = Path(report_path) if report_path is not None else out_csv.with_suffix(".summary.json")
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest}")

    df = pd.read_csv(source_manifest)
    boosts = _cluster_boosts(Path(discovery_report))
    scores: List[float] = []
    reasons: List[str] = []
    for _, row in df.iterrows():
        score, reason = score_manifest_row(row, boosts)
        scores.append(score)
        reasons.append(reason)

    out = df.copy()
    out["piano_score"] = scores
    out["piano_reason"] = reasons
    out = out[out["piano_score"] >= float(min_score)].copy()
    out = out.sort_values(["piano_score", "duration_sec"], ascending=[False, False])
    if int(max_rows) > 0:
        out = out.head(int(max_rows)).copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    top_genres = {str(k): int(v) for k, v in out["genre"].value_counts().head(12).to_dict().items()} if len(out) else {}
    example_cols = [c for c in ["title", "artist", "album", "genre", "path", "piano_score", "piano_reason"] if c in out.columns]
    top_examples = out[example_cols].head(20).to_dict(orient="records") if len(out) else []
    summary = PianoManifestSummary(
        source_manifest=str(source_manifest),
        out_csv=str(out_csv),
        report_path=str(report_path),
        input_rows=int(len(df)),
        selected_rows=int(len(out)),
        min_score=float(min_score),
        top_genres=top_genres,
        top_examples=top_examples,
    )
    _write_json(report_path, summary.__dict__)
    return summary


def iter_manifest_paths(manifest: Path) -> Iterable[Path]:
    df = pd.read_csv(manifest)
    if "path" not in df.columns:
        raise ValueError(f"Manifest missing path column: {manifest}")
    for raw in df["path"].astype(str).tolist():
        yield Path(raw)


def _is_excluded_path(path: Path, exclude_dir_names: set[str]) -> bool:
    names = {part.lower() for part in Path(path).parts}
    return bool(names.intersection(exclude_dir_names))


def _iter_files_from_roots(roots: Iterable[Path], suffixes: set[str], exclude_dir_names: set[str]) -> Tuple[List[Path], List[Dict[str, Any]]]:
    files_out: List[Path] = []
    errors: List[Dict[str, Any]] = []
    for root in [Path(r) for r in roots]:
        if not root.exists():
            errors.append({"root": str(root), "path": "", "error": "root_missing"})
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            path = Path(path)
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if _is_excluded_path(path, exclude_dir_names):
                continue
            files_out.append(path)
    return files_out, errors


def _pair_key(path: Path) -> str:
    return "".join(_tokens(Path(path).stem))


def _pair_match_rank(audio_key: str, midi_key: str) -> int:
    if not audio_key or not midi_key:
        return 0
    if audio_key == midi_key:
        return 3
    if len(audio_key) >= 4 and len(midi_key) >= 4 and (audio_key.startswith(midi_key) or midi_key.startswith(audio_key)):
        return 2
    return 0


def _resolve_manifest_path(raw: Any, manifest: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(manifest).parent / path


def _audio_quick_stats(path: Path) -> Dict[str, Any]:
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            sr = int(f.samplerate)
            frames = int(len(f))
            channels = int(f.channels)
        return {
            "audio_duration_sec": float(frames / float(max(1, sr))),
            "audio_sample_rate": int(sr),
            "audio_channels": int(channels),
        }
    except Exception:
        import librosa

        duration = float(librosa.get_duration(path=str(path)))
        return {
            "audio_duration_sec": duration,
            "audio_sample_rate": 0,
            "audio_channels": 0,
        }


def _midi_quick_stats(path: Path) -> Dict[str, Any]:
    import mido

    midi = mido.MidiFile(str(path))
    tempo = 500000
    seconds = 0.0
    note_count = 0
    piano_program_events = 0
    active_pitches: set[int] = set()
    channels: set[int] = set()
    for msg in mido.merge_tracks(midi.tracks):
        seconds += float(mido.tick2second(int(msg.time), int(midi.ticks_per_beat), int(tempo)))
        if msg.type == "set_tempo":
            tempo = int(msg.tempo)
        elif msg.type == "program_change":
            if int(getattr(msg, "program", -1)) in range(0, 8):
                piano_program_events += 1
        elif msg.type == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
            channel = int(getattr(msg, "channel", 0))
            if channel == 9:
                continue
            pitch = int(getattr(msg, "note", 0))
            if 21 <= pitch <= 108:
                note_count += 1
                active_pitches.add(pitch)
                channels.add(channel)
    return {
        "duration_sec": float(seconds),
        "note_count": int(note_count),
        "unique_pitches": int(len(active_pitches)),
        "channels": int(len(channels)),
        "piano_program_events": int(piano_program_events),
        "ticks_per_beat": int(midi.ticks_per_beat),
    }


def build_midi_piano_target_manifest(
    *,
    roots: Iterable[Path],
    out_csv: Path = DEFAULT_MIDI_MANIFEST,
    report_path: Path | None = None,
    max_rows: int = 0,
    min_notes: int = 8,
    include_package_examples: bool = False,
) -> MidiManifestSummary:
    root_list = [Path(r) for r in roots]
    out_csv = Path(out_csv)
    report_path = Path(report_path) if report_path is not None else out_csv.with_suffix(".summary.json")
    if not root_list:
        raise ValueError("At least one --midi-root is required")

    exclude_names = set()
    if not bool(include_package_examples):
        exclude_names.update({".git", ".hg", ".svn", ".venv", ".venv_lab1", "venv", "env", "site-packages", "__pycache__"})

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    discovered = 0
    for root in root_list:
        if not root.exists():
            errors.append({"root": str(root), "path": "", "error": "root_missing"})
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            path = Path(path)
            if not path.is_file() or path.suffix.lower() not in {".mid", ".midi", ".kar"}:
                continue
            if _is_excluded_path(path, exclude_names):
                continue
            discovered += 1
            try:
                stats = _midi_quick_stats(path)
                if int(stats["note_count"]) < int(min_notes):
                    continue
                rows.append(
                    {
                        "path": str(path),
                        "title": path.stem,
                        "source": "midi_discovery",
                        "size_bytes": int(path.stat().st_size),
                        **stats,
                    }
                )
            except Exception as exc:
                errors.append({"root": str(root), "path": str(path), "error": str(exc)})

    midi_columns = [
        "path",
        "title",
        "source",
        "size_bytes",
        "duration_sec",
        "note_count",
        "unique_pitches",
        "channels",
        "piano_program_events",
        "ticks_per_beat",
    ]
    out = pd.DataFrame(rows, columns=midi_columns)
    if len(out):
        out = out.sort_values(["note_count", "unique_pitches", "duration_sec"], ascending=[False, False, False])
        if int(max_rows) > 0:
            out = out.head(int(max_rows)).copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_csv.with_suffix(".errors.csv"), index=False)

    examples = out.head(20).to_dict(orient="records") if len(out) else []
    summary = MidiManifestSummary(
        out_csv=str(out_csv),
        report_path=str(report_path),
        roots=[str(r) for r in root_list],
        discovered_files=int(discovered),
        selected_rows=int(len(out)),
        min_notes=int(min_notes),
        errors=int(len(errors)),
        top_examples=examples,
    )
    _write_json(report_path, summary.__dict__)
    return summary


def build_paired_audio_midi_manifest(
    *,
    audio_roots: Iterable[Path],
    midi_roots: Iterable[Path],
    out_csv: Path = DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST,
    report_path: Path | None = None,
    max_rows: int = 0,
    min_notes: int = 8,
    include_package_examples: bool = False,
) -> PairedManifestSummary:
    audio_root_list = [Path(r) for r in audio_roots]
    midi_root_list = [Path(r) for r in midi_roots]
    out_csv = Path(out_csv)
    report_path = Path(report_path) if report_path is not None else out_csv.with_suffix(".summary.json")
    if not audio_root_list:
        raise ValueError("At least one --audio-root is required for discover-paired")
    if not midi_root_list:
        raise ValueError("At least one --midi-root is required for discover-paired")

    exclude_names = set()
    if not bool(include_package_examples):
        exclude_names.update({".git", ".hg", ".svn", ".venv", ".venv_lab1", "venv", "env", "site-packages", "__pycache__"})

    audio_files, errors = _iter_files_from_roots(audio_root_list, AUDIO_SUFFIXES, exclude_names)
    midi_files, midi_root_errors = _iter_files_from_roots(midi_root_list, MIDI_SUFFIXES, exclude_names)
    errors.extend(midi_root_errors)

    midi_candidates: List[Dict[str, Any]] = []
    for midi_path in midi_files:
        try:
            stats = _midi_quick_stats(midi_path)
            if int(stats["note_count"]) < int(min_notes):
                continue
            midi_candidates.append({"path": midi_path, "key": _pair_key(midi_path), **stats})
        except Exception as exc:
            errors.append({"root": "", "path": str(midi_path), "error": str(exc)})

    rows: List[Dict[str, Any]] = []
    used_midi: set[str] = set()
    for audio_path in sorted(audio_files, key=lambda p: str(p).lower()):
        audio_key = _pair_key(audio_path)
        scored: List[Tuple[int, int, int, float, Dict[str, Any]]] = []
        for cand in midi_candidates:
            midi_path = Path(cand["path"])
            if str(midi_path) in used_midi:
                continue
            rank = _pair_match_rank(audio_key, str(cand["key"]))
            if rank <= 0:
                continue
            scored.append(
                (
                    int(rank),
                    int(cand.get("note_count", 0)),
                    int(cand.get("unique_pitches", 0)),
                    float(cand.get("duration_sec", 0.0)),
                    cand,
                )
            )
        if not scored:
            continue
        scored.sort(reverse=True, key=lambda item: item[:4])
        rank, _notes, _unique, _duration, best = scored[0]
        midi_path = Path(best["path"])
        used_midi.add(str(midi_path))
        rows.append(
            {
                "source_audio": str(audio_path),
                "target_midi": str(midi_path),
                "title": audio_path.stem,
                "artist": "",
                "source": "paired_discovery",
                "pair_match": "exact_stem" if int(rank) >= 3 else "prefix_stem",
                "audio_key": audio_key,
                "midi_key": str(best["key"]),
                "audio_size_bytes": int(audio_path.stat().st_size),
                "midi_size_bytes": int(midi_path.stat().st_size),
                "target_duration_sec": float(best.get("duration_sec", 0.0)),
                "target_note_count": int(best.get("note_count", 0)),
                "target_unique_pitches": int(best.get("unique_pitches", 0)),
                "target_channels": int(best.get("channels", 0)),
                "target_piano_program_events": int(best.get("piano_program_events", 0)),
            }
        )
        if int(max_rows) > 0 and len(rows) >= int(max_rows):
            break

    columns = [
        "source_audio",
        "target_midi",
        "title",
        "artist",
        "source",
        "pair_match",
        "audio_key",
        "midi_key",
        "audio_size_bytes",
        "midi_size_bytes",
        "target_duration_sec",
        "target_note_count",
        "target_unique_pitches",
        "target_channels",
        "target_piano_program_events",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_csv.with_suffix(".errors.csv"), index=False)

    examples = out.head(20).to_dict(orient="records") if len(out) else []
    summary = PairedManifestSummary(
        out_csv=str(out_csv),
        report_path=str(report_path),
        audio_roots=[str(r) for r in audio_root_list],
        midi_roots=[str(r) for r in midi_root_list],
        discovered_audio_files=int(len(audio_files)),
        discovered_midi_files=int(len(midi_files)),
        selected_rows=int(len(out)),
        min_notes=int(min_notes),
        errors=int(len(errors)),
        top_examples=examples,
    )
    _write_json(report_path, summary.__dict__)
    return summary


def audit_paired_audio_midi_manifest(
    *,
    manifest: Path = DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST,
    report_path: Path | None = None,
    min_notes: int = 8,
    max_duration_delta: float = 5.0,
) -> PairedAuditSummary:
    manifest = Path(manifest)
    report_path = Path(report_path) if report_path is not None else manifest.with_suffix(".audit.json")
    rows_csv = report_path.with_suffix(".rows.csv")
    passed_manifest_csv = report_path.with_suffix(".passed.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing paired manifest: {manifest}")
    df = pd.read_csv(manifest)
    required = {"source_audio", "target_midi"}
    missing = sorted(required.difference(set(df.columns)))
    if missing:
        raise ValueError(f"Paired manifest missing columns {missing}; expected source_audio,target_midi: {manifest}")

    rows: List[Dict[str, Any]] = []
    warning_counts: Dict[str, int] = {}
    for i, rec in df.reset_index(drop=True).iterrows():
        source_audio = _resolve_manifest_path(rec["source_audio"], manifest)
        target_midi = _resolve_manifest_path(rec["target_midi"], manifest)
        warnings: List[str] = []
        row: Dict[str, Any] = {
            "row_idx": int(i),
            "source_audio": str(source_audio),
            "target_midi": str(target_midi),
            "title": str(rec.get("title", source_audio.stem)),
            "pair_match": str(rec.get("pair_match", "")),
        }
        if str(row["pair_match"]) == "prefix_stem":
            warnings.append("prefix_match_needs_review")

        if not source_audio.exists():
            warnings.append("missing_source_audio")
        else:
            try:
                row.update(_audio_quick_stats(source_audio))
                if float(row.get("audio_duration_sec", 0.0)) <= 0.0:
                    warnings.append("audio_duration_empty")
            except Exception as exc:
                row["audio_error"] = str(exc)
                warnings.append("audio_probe_error")

        if not target_midi.exists():
            warnings.append("missing_target_midi")
        else:
            try:
                stats = _midi_quick_stats(target_midi)
                row.update(
                    {
                        "target_duration_sec": float(stats.get("duration_sec", 0.0)),
                        "target_note_count": int(stats.get("note_count", 0)),
                        "target_unique_pitches": int(stats.get("unique_pitches", 0)),
                        "target_channels": int(stats.get("channels", 0)),
                        "target_piano_program_events": int(stats.get("piano_program_events", 0)),
                    }
                )
                if int(stats.get("note_count", 0)) < int(min_notes):
                    warnings.append("target_too_few_notes")
            except Exception as exc:
                row["midi_error"] = str(exc)
                warnings.append("midi_probe_error")

        audio_duration = float(row.get("audio_duration_sec", 0.0) or 0.0)
        target_duration = float(row.get("target_duration_sec", 0.0) or 0.0)
        if audio_duration > 0.0 and target_duration > 0.0:
            delta = abs(audio_duration - target_duration)
            row["duration_delta_sec"] = float(delta)
            row["duration_ratio"] = float(max(audio_duration, target_duration) / max(1e-6, min(audio_duration, target_duration)))
            if delta > float(max_duration_delta):
                warnings.append("duration_mismatch")
        else:
            row["duration_delta_sec"] = None
            row["duration_ratio"] = None

        row["warnings"] = ";".join(warnings)
        row["passed"] = bool(not warnings)
        for warning in warnings:
            warning_counts[warning] = int(warning_counts.get(warning, 0)) + 1
        rows.append(row)

    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rows_csv, index=False)
    passed_manifest_rows = [
        {
            "source_audio": str(row["source_audio"]),
            "target_midi": str(row["target_midi"]),
            "title": str(row.get("title", Path(str(row["source_audio"])).stem)),
            "artist": "",
            "source": "paired_audit_passed",
            "pair_match": str(row.get("pair_match", "")),
        }
        for row in rows
        if bool(row.get("passed"))
    ]
    pd.DataFrame(
        passed_manifest_rows,
        columns=["source_audio", "target_midi", "title", "artist", "source", "pair_match"],
    ).to_csv(passed_manifest_csv, index=False)
    passed_rows = int(sum(1 for row in rows if bool(row.get("passed"))))
    report = {
        "manifest": str(manifest),
        "report_path": str(report_path),
        "rows_csv": str(rows_csv),
        "passed_manifest_csv": str(passed_manifest_csv),
        "input_rows": int(len(rows)),
        "passed_rows": passed_rows,
        "warning_counts": warning_counts,
        "top_examples": rows[:20],
        "thresholds": {
            "min_notes": int(min_notes),
            "max_duration_delta": float(max_duration_delta),
        },
    }
    _write_json(report_path, report)
    return PairedAuditSummary(
        manifest=str(manifest),
        report_path=str(report_path),
        rows_csv=str(rows_csv),
        passed_manifest_csv=str(passed_manifest_csv),
        input_rows=int(len(rows)),
        passed_rows=passed_rows,
        warning_counts=warning_counts,
        top_examples=rows[:20],
    )


__all__ = [
    "DEFAULT_DISCOVERY_REPORT",
    "DEFAULT_MIDI_MANIFEST",
    "DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST",
    "DEFAULT_PIANO_MANIFEST",
    "DEFAULT_SOURCE_MANIFEST",
    "MidiManifestSummary",
    "PairedAuditSummary",
    "PairedManifestSummary",
    "PianoManifestSummary",
    "audit_paired_audio_midi_manifest",
    "build_paired_audio_midi_manifest",
    "build_piano_candidate_manifest",
    "build_midi_piano_target_manifest",
    "iter_manifest_paths",
    "score_manifest_row",
]
