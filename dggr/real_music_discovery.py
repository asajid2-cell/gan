from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from mutagen import File as MutagenFile
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler

from .lab3_diffusion_data import DIFFUSION_SR
from .real_music_manifest import DEFAULT_REAL_MUSIC_ROOT, RealMusicSource, _iter_audio_files


DEFAULT_STOPWORDS = {
    "the",
    "and",
    "feat",
    "ft",
    "with",
    "from",
    "vol",
    "volume",
    "remaster",
    "remastered",
    "version",
    "explicit",
    "original",
    "single",
    "album",
    "live",
}

AUDIO_FEATURE_NAMES = (
    ["tempo", "centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean", "flatness_mean", "zcr_mean", "rms_mean", "rms_std", "onset_mean", "onset_std"]
    + [f"chroma_mean_{i:02d}" for i in range(12)]
    + [f"chroma_std_{i:02d}" for i in range(12)]
    + [f"mfcc_mean_{i:02d}" for i in range(13)]
    + [f"mfcc_std_{i:02d}" for i in range(13)]
)


def _slug(raw: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_")
    text = re.sub(r"_+", "_", text)
    return text[:48] or fallback


def _tag_value(tags: Any, names: Sequence[str]) -> str:
    if not tags:
        return ""
    for name in names:
        try:
            value = tags.get(name)
        except Exception:
            value = None
        if value:
            if isinstance(value, (list, tuple)):
                return "; ".join(str(v) for v in value if str(v).strip())
            return str(value)
    return ""


def _scan_one(path_str: str, root_str: str, source_name: str, min_bytes: int) -> Optional[Dict[str, Any]]:
    path = Path(path_str)
    try:
        size = int(path.stat().st_size)
    except OSError:
        return None
    if size < int(min_bytes):
        return None
    title = artist = album = tag_genre = date = ""
    duration = 0.0
    bitrate = 0
    try:
        audio = MutagenFile(str(path), easy=True)
        if audio is not None:
            tags = getattr(audio, "tags", None)
            title = _tag_value(tags, ["title"])
            artist = _tag_value(tags, ["artist", "albumartist", "performer"])
            album = _tag_value(tags, ["album"])
            tag_genre = _tag_value(tags, ["genre"])
            date = _tag_value(tags, ["date", "year"])
            info = getattr(audio, "info", None)
            duration = float(getattr(info, "length", 0.0) or 0.0)
            bitrate = int(getattr(info, "bitrate", 0) or 0)
    except Exception:
        pass
    return {
        "source": source_name,
        "path": str(path),
        "ext": path.suffix.lower().lstrip("."),
        "size_bytes": size,
        "is_music": 1,
        "genre": "undiscovered",
        "title": title,
        "artist": artist,
        "album": album,
        "tag_genre": tag_genre,
        "date": date,
        "duration_sec": duration,
        "bitrate": bitrate,
        "aacid": path.stem,
        "source_root": root_str,
        "metadata_text": " ".join(x for x in [title, artist, album, tag_genre] if x).strip(),
    }


def scan_audio_metadata(
    root: Path = DEFAULT_REAL_MUSIC_ROOT,
    *,
    min_bytes: int = 64_000,
    max_files: int = 0,
    seed: int = 328,
    source_name: str = "spotify_discovered",
) -> pd.DataFrame:
    root = Path(root)
    if int(max_files) > 0:
        files = [p for p in _iter_audio_files(root, prefer_ogg_export=True) if p.suffix.lower()]
        rng = np.random.default_rng(int(seed))
        if len(files) > int(max_files):
            take = rng.choice(len(files), size=int(max_files), replace=False)
            files = [files[int(i)] for i in sorted(take.tolist())]
        file_iter: Any = files
        total = len(files)
    else:
        file_iter = (p for p in _iter_audio_files(root, prefer_ogg_export=True) if p.suffix.lower())
        total = None
    rows: List[Dict[str, Any]] = []
    print(f"[discovery] scanning metadata root={root} max_files={max_files}", flush=True)
    for i, path in enumerate(file_iter):
        row = _scan_one(str(path), str(root), source_name, min_bytes)
        if row is not None:
            rows.append(row)
        if (i + 1) % 500 == 0:
            suffix = f"/{total}" if total is not None else ""
            print(f"[discovery] scanned metadata {i + 1}{suffix} files; kept={len(rows)}", flush=True)
    if not rows:
        raise RuntimeError(f"No usable files found under {root}")
    return pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)


def _audio_features_one(task: Tuple[int, str, float, float]) -> Tuple[int, Optional[np.ndarray], str]:
    idx, path_str, seconds, offset = task
    try:
        y, sr = librosa.load(
            path_str,
            sr=DIFFUSION_SR,
            mono=True,
            offset=max(0.0, float(offset)),
            duration=float(seconds),
            dtype=np.float32,
            res_type="soxr_hq",
        )
        if len(y) < int(0.5 * DIFFUSION_SR):
            return idx, None, "too_short"
        y = librosa.util.normalize(y)
        tempo_arr = librosa.feature.tempo(y=y, sr=sr)
        tempo = float(np.asarray(tempo_arr).reshape(-1)[0]) if np.asarray(tempo_arr).size else 0.0
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        feats = [
            tempo,
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(bandwidth)),
            float(np.mean(rolloff)),
            float(np.mean(flatness)),
            float(np.mean(zcr)),
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.mean(onset)),
            float(np.std(onset)),
        ]
        feats.extend(np.mean(chroma, axis=1).astype(float).tolist())
        feats.extend(np.std(chroma, axis=1).astype(float).tolist())
        feats.extend(np.mean(mfcc, axis=1).astype(float).tolist())
        feats.extend(np.std(mfcc, axis=1).astype(float).tolist())
        arr = np.asarray(feats, dtype=np.float32)
        arr[~np.isfinite(arr)] = 0.0
        return idx, arr, "ok"
    except Exception as exc:
        return idx, None, f"error:{type(exc).__name__}"


def extract_audio_feature_matrix(
    df: pd.DataFrame,
    *,
    seconds: float = 8.0,
    max_files: int = 0,
    seed: int = 328,
    workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    n = int(len(df))
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=bool), {}
    indices = np.arange(n, dtype=np.int64)
    if int(max_files) > 0 and int(max_files) < n:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_files), replace=False))
    tasks: List[Tuple[int, str, float, float]] = []
    for idx in indices.tolist():
        dur = float(df.iloc[int(idx)].get("duration_sec", 0.0) or 0.0)
        offset = 0.0
        if dur > float(seconds) + 8.0:
            offset = min(max(0.0, 0.25 * dur), max(0.0, dur - float(seconds)))
        tasks.append((int(idx), str(df.iloc[int(idx)]["path"]), float(seconds), float(offset)))

    print(f"[discovery] extracting audio features for {len(tasks)} files with workers={workers}", flush=True)
    feat_by_idx: Dict[int, np.ndarray] = {}
    status_counts: Counter[str] = Counter()
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(_audio_features_one, task) for task in tasks]
            for done_i, fut in enumerate(as_completed(futures), start=1):
                idx, feat, status = fut.result()
                status_counts[str(status)] += 1
                if feat is not None:
                    feat_by_idx[int(idx)] = feat
                if done_i % 100 == 0:
                    print(f"[discovery] extracted audio features {done_i}/{len(tasks)}", flush=True)
    else:
        for done_i, task in enumerate(tasks, start=1):
            idx, feat, status = _audio_features_one(task)
            status_counts[str(status)] += 1
            if feat is not None:
                feat_by_idx[int(idx)] = feat
            if done_i % 100 == 0:
                print(f"[discovery] extracted audio features {done_i}/{len(tasks)}", flush=True)

    feat_dim = len(next(iter(feat_by_idx.values()))) if feat_by_idx else 1
    X = np.zeros((n, feat_dim), dtype=np.float32)
    mask = np.zeros((n,), dtype=bool)
    for idx, feat in feat_by_idx.items():
        X[int(idx), :] = feat.astype(np.float32)
        mask[int(idx)] = True
    if mask.any():
        mean = np.mean(X[mask], axis=0)
        X[~mask] = mean
    return X, mask, {str(k): int(v) for k, v in status_counts.items()}


def _top_terms(texts: Sequence[str], n: int = 5) -> List[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", str(text).lower())
        for word in words:
            if word in DEFAULT_STOPWORDS:
                continue
            counts[word] += 1
    return [w for w, _ in counts.most_common(int(n))]


def _audio_profile_name(mean_feat: np.ndarray) -> Tuple[str, Dict[str, float]]:
    vals = {name: float(mean_feat[i]) if i < len(mean_feat) else 0.0 for i, name in enumerate(AUDIO_FEATURE_NAMES)}
    tempo = vals.get("tempo", 0.0)
    centroid = vals.get("centroid_mean", 0.0)
    flatness = vals.get("flatness_mean", 0.0)
    onset = vals.get("onset_mean", 0.0)
    zcr = vals.get("zcr_mean", 0.0)
    rms = vals.get("rms_mean", 0.0)
    parts: List[str] = []
    if tempo >= 130:
        parts.append("fast")
    elif tempo >= 90:
        parts.append("midtempo")
    elif tempo > 0:
        parts.append("slow")
    if centroid >= 3200 or zcr >= 0.10:
        parts.append("bright")
    elif centroid <= 1600 and centroid > 0:
        parts.append("warm")
    if flatness >= 0.09:
        parts.append("noisy")
    elif flatness <= 0.025:
        parts.append("tonal")
    if onset >= 1.4:
        parts.append("percussive")
    elif onset <= 0.45:
        parts.append("smooth")
    if rms >= 0.12:
        parts.append("loud")
    elif rms > 0 and rms <= 0.035:
        parts.append("quiet")
    if not parts:
        parts = ["mixed"]
    keep = {
        "tempo": tempo,
        "centroid_mean": centroid,
        "flatness_mean": flatness,
        "zcr_mean": zcr,
        "onset_mean": onset,
        "rms_mean": rms,
    }
    return "_".join(parts[:4]), keep


def discover_genre_manifest(
    *,
    root: Path = DEFAULT_REAL_MUSIC_ROOT,
    out_csv: Path,
    report_path: Optional[Path] = None,
    n_clusters: int = 12,
    min_bytes: int = 64_000,
    max_files: int = 0,
    audio_feature_limit: int = 2000,
    audio_feature_seconds: float = 8.0,
    audio_workers: int = 1,
    seed: int = 328,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = scan_audio_metadata(
        Path(root),
        min_bytes=int(min_bytes),
        max_files=int(max_files),
        seed=int(seed),
        source_name="spotify_discovered",
    )
    texts = df["metadata_text"].fillna("").astype(str).tolist()
    fallback_text = df["aacid"].fillna("").astype(str).tolist()
    texts = [t if t.strip() else f for t, f in zip(texts, fallback_text)]
    vectorizer = HashingVectorizer(
        n_features=2048,
        analyzer="char_wb",
        ngram_range=(3, 5),
        alternate_sign=False,
        norm="l2",
    )
    X_text = vectorizer.transform(texts)

    X_audio, audio_mask, audio_status = extract_audio_feature_matrix(
        df,
        seconds=float(audio_feature_seconds),
        max_files=int(audio_feature_limit),
        seed=int(seed),
        workers=int(audio_workers),
    )
    numeric = df[["size_bytes", "duration_sec", "bitrate"]].fillna(0).to_numpy(dtype=np.float32)
    if X_audio.shape[1] > 0:
        dense = np.concatenate([numeric, X_audio, audio_mask.astype(np.float32)[:, None]], axis=1)
    else:
        dense = np.concatenate([numeric, audio_mask.astype(np.float32)[:, None]], axis=1)
    dense = StandardScaler().fit_transform(dense).astype(np.float32)
    X = sparse.hstack([X_text, sparse.csr_matrix(dense * 0.35)], format="csr")

    k = max(3, min(int(n_clusters), int(len(df))))
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=min(4096, max(256, len(df))),
        n_init=5,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(X)
    df["discovered_cluster"] = labels.astype(int)

    cluster_names: Dict[int, str] = {}
    cluster_rows: List[Dict[str, Any]] = []
    for cluster_id, gdf in df.groupby("discovered_cluster", sort=True):
        terms = _top_terms(gdf["metadata_text"].fillna("").astype(str).tolist(), n=4)
        idxs = gdf.index.to_numpy(dtype=np.int64)
        profile_slug, audio_profile = _audio_profile_name(np.mean(X_audio[idxs], axis=0) if X_audio.shape[1] else np.zeros((1,), dtype=np.float32))
        name = _slug("_".join([profile_slug] + terms[:3]), fallback=f"style_{int(cluster_id):02d}")
        cluster_name = f"style_{int(cluster_id):02d}_{name}"
        cluster_names[int(cluster_id)] = cluster_name
        examples = []
        for _, row in gdf.head(8).iterrows():
            examples.append(
                {
                    "title": str(row.get("title", "")),
                    "artist": str(row.get("artist", "")),
                    "album": str(row.get("album", "")),
                    "path": str(row.get("path", "")),
                }
            )
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "genre": cluster_name,
                "count": int(len(gdf)),
                "top_terms": terms,
                "audio_profile": audio_profile,
                "examples": examples,
                "audio_feature_coverage": float(np.mean(audio_mask[gdf.index.to_numpy(dtype=np.int64)]))
                if len(gdf)
                else 0.0,
            }
        )
    df["genre"] = df["discovered_cluster"].map(lambda x: cluster_names[int(x)])
    df["source"] = "spotify_discovered"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["metadata_text"]).to_csv(out_csv, index=False)

    report = {
        "manifest_path": str(out_csv),
        "rows": int(len(df)),
        "n_clusters": int(k),
        "root": str(root),
        "settings": {
            "min_bytes": int(min_bytes),
            "max_files": int(max_files),
            "audio_feature_limit": int(audio_feature_limit),
            "audio_feature_seconds": float(audio_feature_seconds),
            "audio_workers": int(audio_workers),
            "seed": int(seed),
        },
        "metadata_coverage": {
            "title": float((df["title"].fillna("").astype(str).str.len() > 0).mean()),
            "artist": float((df["artist"].fillna("").astype(str).str.len() > 0).mean()),
            "album": float((df["album"].fillna("").astype(str).str.len() > 0).mean()),
            "tag_genre": float((df["tag_genre"].fillna("").astype(str).str.len() > 0).mean()),
        },
        "audio_feature_coverage": float(np.mean(audio_mask)) if len(audio_mask) else 0.0,
        "audio_feature_status": audio_status,
        "clusters": sorted(cluster_rows, key=lambda r: int(r["cluster_id"])),
        "genre_counts": {str(k): int(v) for k, v in df["genre"].value_counts().sort_index().to_dict().items()},
    }
    report_path = Path(report_path) if report_path is not None else out_csv.with_suffix(".discovery_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"manifest_path": str(out_csv), "rows": len(df), "n_clusters": k}, indent=2), flush=True)
    return df, report
