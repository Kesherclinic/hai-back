"""
Affinity Calculator — HAI Protocol v1.0
Computes Z-scores, detects peaks, scores life domains.
Formula: Affinity = Valence × Arousal (Phase 1)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional


# ── Life domains defined in HAI Protocol ──
LIFE_DOMAINS = [
    "art",
    "sport_challenge",
    "nature",
    "science_learning",
    "innovation_technology",
    "culture_leisure",
    "care_education",
    "personal_growth",
]

DOMAIN_LABELS_HE = {
    "art":                   "אומנות",
    "sport_challenge":       "ספורט ואתגר",
    "nature":                "טבע",
    "science_learning":      "מדע, למידה ומחקר",
    "innovation_technology": "חדשנות וטכנולוגיה",
    "culture_leisure":       "תרבות ופנאי",
    "care_education":        "טיפול, חינוך ותקשורת",
    "personal_growth":       "התפתחות אישית",
}

# Scoring weights per HAI Protocol Section 3
WEIGHTS = {
    "peak_z":       0.30,
    "mean_z":       0.25,
    "duration":     0.25,
    "consistency":  0.20,
}

# Thresholds per HAI Protocol Section 4
Z_STRONG    = 2.0   # Strong connection
Z_NOTABLE   = 1.5   # Notable connection
Z_MILD      = 0.5   # Mild interest
Z_REJECTION = -0.5  # Rejection / conflict
PEAK_MIN_SEC = 1.5  # Minimum duration for a "deep" peak
SIGNATURE_SEC = 5.0 # Signature peak threshold


@dataclass
class PeakMoment:
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_z: float
    is_signature: bool  # Z > 2.0 AND duration > 5s


@dataclass
class DomainScore:
    domain_id: str
    domain_label_he: str
    hai_score: float        # 0–100
    peak_z: float
    mean_z: float
    peak_duration_sec: float
    consistency: float       # 1 - normalized_std
    classification: str      # strong / notable / mild / neutral / rejection
    peaks: list


def calculate_affinity(
    video_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    stimulus_map: Optional[dict] = None,
) -> dict:
    """
    Main entry point.
    stimulus_map: {timestamp_sec: domain_id} — maps moments to life domains.
    If not provided, returns session-level stats only.
    """

    video_stats = _compute_session_stats(video_df, channel="video")
    audio_stats = _compute_session_stats(audio_df, channel="audio")

    peaks_video = _detect_peaks(video_df, channel="video")
    peaks_audio = _detect_peaks(audio_df, channel="audio")

    domain_scores = []
    if stimulus_map:
        domain_scores = _score_domains(video_df, audio_df, stimulus_map)

    return {
        "video": video_stats,
        "audio": audio_stats,
        "peaks": {
            "video": [asdict(p) for p in peaks_video],
            "audio": [asdict(p) for p in peaks_audio],
        },
        "domain_scores": [asdict(d) for d in domain_scores],
        "top_domains": _get_top_domains(domain_scores),
        "signature_peaks": _get_signature_peaks(peaks_video + peaks_audio),
        "rejection_flags": _get_rejections(video_df, audio_df),
    }


def _compute_session_stats(df: pd.DataFrame, channel: str) -> dict:
    """Compute session-level descriptive stats."""
    if df.empty or "affinity_z" not in df.columns:
        return {"channel": channel, "available": False}

    return {
        "channel":        channel,
        "available":      True,
        "n_samples":      len(df),
        "mean_valence":   round(float(df["valence"].mean()), 4),
        "mean_arousal":   round(float(df["arousal"].mean()), 4),
        "mean_affinity":  round(float(df["affinity_raw"].mean()), 4),
        "std_affinity":   round(float(df["affinity_raw"].std()), 4),
        "max_z":          round(float(df["affinity_z"].max()), 4),
        "min_z":          round(float(df["affinity_z"].min()), 4),
        "pct_above_1_5":  round(float((df["affinity_z"] > Z_NOTABLE).mean() * 100), 2),
        "pct_below_neg":  round(float((df["affinity_z"] < Z_REJECTION).mean() * 100), 2),
        "dominant_emotion": df["dominant"].mode().iloc[0] if not df.empty else "unknown",
    }


def _detect_peaks(df: pd.DataFrame, channel: str) -> list[PeakMoment]:
    """
    Identify contiguous windows where affinity_z > Z_NOTABLE (1.5).
    Returns list of PeakMoment objects.
    """
    if df.empty or "affinity_z" not in df.columns:
        return []

    ts_col = "timestamp" if channel == "video" else "timestamp"
    te_col = "timestamp"

    peaks = []
    in_peak = False
    peak_start = 0.0
    peak_max_z = 0.0

    for _, row in df.iterrows():
        z = row["affinity_z"]
        t = row[ts_col]

        if z >= Z_NOTABLE:
            if not in_peak:
                in_peak = True
                peak_start = float(t)
                peak_max_z = float(z)
            else:
                peak_max_z = max(peak_max_z, float(z))
        else:
            if in_peak:
                peak_end = float(row.get(te_col, t))
                duration = peak_end - peak_start
                if duration >= PEAK_MIN_SEC or peak_max_z >= Z_STRONG:
                    peaks.append(PeakMoment(
                        start_sec=round(peak_start, 3),
                        end_sec=round(peak_end, 3),
                        duration_sec=round(duration, 3),
                        peak_z=round(peak_max_z, 4),
                        is_signature=(peak_max_z >= Z_STRONG and duration >= SIGNATURE_SEC),
                    ))
                in_peak = False
                peak_max_z = 0.0

    return peaks


def _score_domains(
    video_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    stimulus_map: dict,
) -> list[DomainScore]:
    """
    Score each life domain using HAI Protocol weights.
    stimulus_map: {domain_id: [(start_sec, end_sec), ...]}
    """
    scores = []

    for domain_id in LIFE_DOMAINS:
        intervals = stimulus_map.get(domain_id, [])
        if not intervals:
            continue

        v_rows = _filter_by_intervals(video_df, intervals, "timestamp")
        a_rows = _filter_by_intervals(audio_df, intervals, "timestamp")

        if v_rows.empty and a_rows.empty:
            continue

        # Dynamic weighting by Signal Strength per channel
        v_z = v_rows["affinity_z"].dropna().clip(upper=3.5) if not v_rows.empty else pd.Series(dtype=float)
        a_z = a_rows["affinity_z"].dropna().clip(upper=3.5) if not a_rows.empty else pd.Series(dtype=float)

        signal_face  = float(v_z.std()) if len(v_z) > 1 else 0.0
        signal_audio = float(a_z.std()) if len(a_z) > 1 else 0.0
        total_signal = signal_face + signal_audio

        if total_signal > 0:
            w_face  = signal_face  / total_signal
            w_audio = signal_audio / total_signal
        else:
            w_face, w_audio = 0.5, 0.5

        mean_z_face  = float(v_z.mean()) if not v_z.empty else 0.0
        mean_z_audio = float(a_z.mean()) if not a_z.empty else 0.0
        mean_z = w_face * mean_z_face + w_audio * mean_z_audio

        combined = pd.concat([v_rows, a_rows], ignore_index=True)
        z_vals = combined["affinity_z"].dropna().clip(upper=3.5)
        if z_vals.empty:
            continue

        peak_z      = float(z_vals.max())
        duration    = float((z_vals > Z_NOTABLE).sum())
        consistency = max(0.0, 1.0 - float(z_vals.std() / (abs(mean_z) + 1e-6)))

        # Weighted HAI score 0–100
        raw_score = (
            WEIGHTS["peak_z"]      * _normalize_z(peak_z) +
            WEIGHTS["mean_z"]      * _normalize_z(mean_z) +
            WEIGHTS["duration"]    * min(duration / 30, 1.0) +
            WEIGHTS["consistency"] * consistency
        )
        hai_score = round(raw_score * 100, 1)

        classification = _classify(peak_z)

        domain_peaks = _detect_peaks(combined, channel="video")

        scores.append(DomainScore(
            domain_id=domain_id,
            domain_label_he=DOMAIN_LABELS_HE[domain_id],
            hai_score=hai_score,
            peak_z=round(peak_z, 4),
            mean_z=round(mean_z, 4),
            peak_duration_sec=round(max((p.duration_sec for p in domain_peaks), default=0), 3),
            consistency=round(consistency, 4),
            classification=classification,
            peaks=[asdict(p) for p in domain_peaks],
        ))

    return sorted(scores, key=lambda d: d.hai_score, reverse=True)


def _filter_by_intervals(df: pd.DataFrame, intervals: list, ts_col: str) -> pd.DataFrame:
    """Return rows whose timestamp falls within any of the given intervals."""
    if df.empty or ts_col not in df.columns:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for start, end in intervals:
        mask |= (df[ts_col] >= start) & (df[ts_col] <= end)
    return df[mask]


def _normalize_z(z: float) -> float:
    """Map Z-score to 0–1 range (sigmoid-like, clamped)."""
    return float(np.clip((z + 3) / 6, 0, 1))


def _classify(peak_z: float) -> str:
    if peak_z >= Z_STRONG:    return "strong"
    if peak_z >= Z_NOTABLE:   return "notable"
    if peak_z >= Z_MILD:      return "mild"
    if peak_z >= Z_REJECTION: return "neutral"
    return "rejection"


def _get_top_domains(domain_scores: list[DomainScore]) -> list[dict]:
    notable = [d for d in domain_scores if d.classification in ("strong", "notable")]
    return [
        {"rank": i+1, "domain": d.domain_id, "label_he": d.domain_label_he, "peak_z": d.peak_z, "mean_z": d.mean_z,
         "hai_score": d.hai_score, "classification": d.classification}
        for i, d in enumerate(notable[:3])
    ]


def _get_signature_peaks(peaks: list[PeakMoment]) -> list[dict]:
    return [asdict(p) for p in peaks if p.is_signature]


def _get_rejections(video_df: pd.DataFrame, audio_df: pd.DataFrame) -> list[dict]:
    """Find time windows with strong negative affinity."""
    rejections = []
    for df, channel in [(video_df, "video"), (audio_df, "audio")]:
        if df.empty or "affinity_z" not in df.columns:
            continue
        ts_col = "timestamp" if channel == "video" else "timestamp"
        neg = df[df["affinity_z"] < Z_REJECTION]
        for _, row in neg.iterrows():
            if float(row["arousal"]) > 0.4:  # High arousal + negative = strong rejection
                rejections.append({
                    "channel":   channel,
                    "timestamp": float(row[ts_col]),
                    "z_score":   round(float(row["affinity_z"]), 4),
                    "dominant":  row["dominant"],
                })
    return rejections
