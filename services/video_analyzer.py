"""
Video Analyzer — DeepFace facial emotion extraction
Processes video frames → Valence, Arousal, Intensity + 8 emotions per frame
"""

import cv2
import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# DeepFace import with graceful fallback for install check
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not installed. Run: pip install deepface")


# ── Emotion → Valence/Arousal mapping (Russell's circumplex model) ──
# Each emotion maps to approximate valence (-1 to +1) and arousal (0 to 1)
EMOTION_VA = {
    "happy":     {"valence":  0.85, "arousal": 0.65},
    "surprise":  {"valence":  0.10, "arousal": 0.85},
    "neutral":   {"valence":  0.00, "arousal": 0.10},
    "sad":       {"valence": -0.70, "arousal": 0.25},
    "fear":      {"valence": -0.75, "arousal": 0.90},
    "angry":     {"valence": -0.80, "arousal": 0.85},
    "disgust":   {"valence": -0.75, "arousal": 0.55},
    "contempt":  {"valence": -0.60, "arousal": 0.30},
}

SAMPLE_RATE_FPS = 5          # Analyze every Nth frame (5fps from 30fps video)
EXECUTOR = ThreadPoolExecutor(max_workers=2)


async def analyze_video_frames(video_path: Path, output_csv: Path) -> pd.DataFrame:
    """
    Extract facial emotions from video at SAMPLE_RATE_FPS.
    Returns DataFrame + saves CSV.
    """
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        EXECUTOR,
        _analyze_sync,
        video_path,
        output_csv
    )
    return df


def _analyze_sync(video_path: Path, output_csv: Path) -> pd.DataFrame:
    """Synchronous frame analysis (runs in thread pool)."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / SAMPLE_RATE_FPS))

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            timestamp_sec = frame_idx / fps
            row = _analyze_frame(frame, frame_idx, timestamp_sec)
            if row:
                rows.append(row)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    if df.empty:
        # Return empty DataFrame with correct columns
        df = _empty_video_df()
    else:
        df = _enrich_dataframe(df)

    df.to_csv(output_csv, index=False)
    print(f"✅ Video analysis complete: {len(df)} frames → {output_csv}")
    return df


def _analyze_frame(frame: np.ndarray, frame_idx: int, timestamp: float) -> dict | None:
    """
    Run DeepFace on a single frame.
    Returns dict with all emotion scores + derived Valence/Arousal.
    """
    if not DEEPFACE_AVAILABLE:
        # Mock data for development/testing
        return _mock_frame(frame_idx, timestamp)

    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        dominant = result.get("dominant_emotion", "neutral")

        # Normalize emotion scores to 0-1
        total = sum(emotions.values()) or 1
        norm = {k: v / total for k, v in emotions.items()}

        # Compute weighted Valence and Arousal
        valence = sum(
            norm.get(em, 0) * va["valence"]
            for em, va in EMOTION_VA.items()
        )
        arousal = sum(
            norm.get(em, 0) * va["arousal"]
            for em, va in EMOTION_VA.items()
        )
        intensity = abs(valence) * arousal

        return {
            "frame":      frame_idx,
            "timestamp":  round(timestamp, 3),
            "dominant":   dominant,
            "valence":    round(valence, 4),
            "arousal":    round(arousal, 4),
            "intensity":  round(intensity, 4),
            "angry":      round(norm.get("angry", 0), 4),
            "disgust":    round(norm.get("disgust", 0), 4),
            "fear":       round(norm.get("fear", 0), 4),
            "happy":      round(norm.get("happy", 0), 4),
            "sad":        round(norm.get("sad", 0), 4),
            "surprise":   round(norm.get("surprise", 0), 4),
            "neutral":    round(norm.get("neutral", 0), 4),
            "contempt":   round(norm.get("contempt", 0), 4),
        }

    except Exception as e:
        # Face not detected in frame — skip silently
        return None


def _enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add Affinity (Phase 1) and Z-score columns."""
    df = df.copy()

    # Phase 1 Affinity
    df["affinity_raw"] = df["valence"] * df["arousal"]

    # Z-score normalisation (personal profile)
    mean = df["affinity_raw"].mean()
    std  = df["affinity_raw"].std()
    if std > 0:
        df["affinity_z"] = (df["affinity_raw"] - mean) / std
    else:
        df["affinity_z"] = 0.0

    df["affinity_z"] = df["affinity_z"].round(4)
    df["affinity_z"] = df["affinity_z"].clip(upper=3.5)
    return df


def _empty_video_df() -> pd.DataFrame:
    cols = [
        "frame","timestamp","dominant","valence","arousal","intensity",
        "angry","disgust","fear","happy","sad","surprise","neutral","contempt",
        "affinity_raw","affinity_z"
    ]
    return pd.DataFrame(columns=cols)


def _mock_frame(frame_idx: int, timestamp: float) -> dict:
    """
    Deterministic mock data for testing without DeepFace installed.
    Simulates a realistic emotional arc over a 15-min session.
    """
    import math
    t = timestamp / 952  # normalise to 0-1

    # Simulate gradual shift: surprise → neutral with occasional peaks
    base_neutral  = 0.5 + 0.3 * t
    base_surprise = 0.3 * (1 - t)
    base_happy    = 0.1 * math.sin(t * 6) ** 2

    valence  = round( base_happy * 0.85 + base_surprise * 0.1 - (1 - base_neutral) * 0.2, 4)
    arousal  = round( 0.6 * (1 - t) + 0.1, 4)
    dominant = "neutral" if base_neutral > 0.6 else ("surprise" if base_surprise > 0.3 else "happy")

    return {
        "frame":     frame_idx,
        "timestamp": round(timestamp, 3),
        "dominant":  dominant,
        "valence":   valence,
        "arousal":   arousal,
        "intensity": round(abs(valence) * arousal, 4),
        "angry":     0.02, "disgust": 0.01, "fear": round(0.05 * (1-t), 4),
        "happy":     round(base_happy, 4),
        "sad":       0.02,
        "surprise":  round(base_surprise, 4),
        "neutral":   round(base_neutral, 4),
        "contempt":  0.005,
    }
