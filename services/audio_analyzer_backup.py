"""
Audio Analyzer — SpeechBrain emotion extraction
Pipeline: video → FFmpeg (extract audio) → SpeechBrain → Valence/Arousal per segment
"""

import subprocess
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import torchaudio
    import torch
    from speechbrain.inference.interfaces import foreign_class
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("⚠️  SpeechBrain not installed. Run: pip install speechbrain")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Emotion → Valence/Arousal/Intensity mapping (audio domain)
AUDIO_EMOTION_VA = {
    "happy":    {"valence":  0.80, "arousal":  0.70},
    "angry":    {"valence": -0.75, "arousal":  0.90},
    "sad":      {"valence": -0.65, "arousal":  0.20},
    "neutral":  {"valence":  0.00, "arousal":  0.05},
    "fearful":  {"valence": -0.70, "arousal":  0.85},
    "disgust":  {"valence": -0.65, "arousal":  0.55},
    "surprise": {"valence":  0.15, "arousal":  0.80},
    "calm":     {"valence":  0.30, "arousal": -0.10},
}

SEGMENT_DURATION_SEC = 5   # Analyse audio in N-second windows
MIN_SEGMENT_ENERGY   = 0.001  # Skip near-silent segments


async def analyze_audio(video_path: Path, output_csv: Path) -> pd.DataFrame:
    """
    Extract audio from video → analyse emotion per segment.
    Returns DataFrame + saves CSV.
    """
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        EXECUTOR,
        _analyze_audio_sync,
        video_path,
        output_csv
    )
    return df


def _analyze_audio_sync(video_path: Path, output_csv: Path) -> pd.DataFrame:
    """Synchronous audio pipeline."""
    audio_path = video_path.with_suffix(".wav")

    # Step 1: Extract audio track with FFmpeg
    _extract_audio(video_path, audio_path)

    if not audio_path.exists():
        print("⚠️  Audio extraction failed — returning empty DataFrame")
        df = _empty_audio_df()
        df.to_csv(output_csv, index=False)
        return df

    # Step 2: Load and segment audio
    segments = _segment_audio(audio_path)

    if not segments:
        df = _empty_audio_df()
        df.to_csv(output_csv, index=False)
        return df

    # Step 3: Analyse each segment
    rows = []
    for seg_idx, (start_sec, end_sec, audio_chunk, sr) in enumerate(segments):
        row = _analyze_segment(seg_idx, start_sec, end_sec, audio_chunk, sr)
        if row:
            rows.append(row)

    # Cleanup temp audio file
    audio_path.unlink(missing_ok=True)

    df = pd.DataFrame(rows) if rows else _empty_audio_df()

    if not df.empty:
        df = _enrich_audio_df(df)

    df.to_csv(output_csv, index=False)
    print(f"✅ Audio analysis complete: {len(df)} segments → {output_csv}")
    return df


def _extract_audio(video_path: Path, audio_path: Path):
    """Use FFmpeg to extract 16kHz mono WAV from video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ar", "16000",     # 16kHz — SpeechBrain standard
        "-ac", "1",          # mono
        "-vn",               # no video
        "-f", "wav",
        str(audio_path)
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=True
        )
        print(f"✅ Audio extracted: {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  FFmpeg error: {e.stderr.decode()[:200]}")
    except FileNotFoundError:
        print("⚠️  FFmpeg not found. Install: https://ffmpeg.org/download.html")


def _segment_audio(audio_path: Path) -> list:
    """
    Load WAV and split into SEGMENT_DURATION_SEC chunks.
    Returns list of (start_sec, end_sec, audio_array, sample_rate).
    """
    if not LIBROSA_AVAILABLE:
        print("⚠️  librosa not installed — using mock audio segments")
        return _mock_segments()

    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        total_sec = len(audio) / sr
        step = int(SEGMENT_DURATION_SEC * sr)
        segments = []

        for i in range(0, len(audio), step):
            chunk = audio[i : i + step]
            if len(chunk) < sr * 0.5:   # Skip chunks < 0.5s
                continue

            energy = float(np.sqrt(np.mean(chunk ** 2)))
            if energy < MIN_SEGMENT_ENERGY:
                continue  # Near-silent, skip

            start_sec = i / sr
            end_sec   = min((i + step) / sr, total_sec)
            segments.append((start_sec, end_sec, chunk, sr))

        return segments

    except Exception as e:
        print(f"⚠️  Audio segmentation error: {e}")
        return []


def _analyze_segment(
    seg_idx: int,
    start_sec: float,
    end_sec: float,
    audio_chunk: np.ndarray,
    sr: int
) -> dict | None:
    """Run SpeechBrain (or mock) on one audio segment."""
    if not SPEECHBRAIN_AVAILABLE:
        return _mock_segment(seg_idx, start_sec, end_sec)

    try:
        import torch
        waveform = torch.tensor(audio_chunk).unsqueeze(0)  # [1, T]

        # SpeechBrain IEMOCAP model — 4 classes: happy, angry, sad, neutral
        # For production: use wav2vec2-based model for richer output
        classifier = _get_speechbrain_model()
        out_prob, score, index, text_lab = classifier.classify_batch(waveform)

        dominant = text_lab[0]
        conf = float(score[0])

        va = AUDIO_EMOTION_VA.get(dominant.lower(), {"valence": 0.0, "arousal": 0.1})
        valence   = round(va["valence"] * conf, 4)
        arousal   = round(va["arousal"] * conf, 4)
        intensity = round(abs(valence) * abs(arousal) ** 0.5, 4)

        return {
            "segment":    seg_idx,
            "start_sec":  round(start_sec, 3),
            "end_sec":    round(end_sec, 3),
            "dominant":   dominant,
            "confidence": round(conf, 4),
            "valence":    valence,
            "arousal":    arousal,
            "intensity":  intensity,
        }

    except Exception as e:
        print(f"⚠️  SpeechBrain error on segment {seg_idx}: {e}")
        return None


# Singleton model loader — loads once, reuses
_speechbrain_model = None

def _get_speechbrain_model():
    global _speechbrain_model
    if _speechbrain_model is None:
        print("📦 Loading SpeechBrain emotion model (first run: downloads ~200MB)...")
        _speechbrain_model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
        )
        print("✅ SpeechBrain model loaded")
    return _speechbrain_model


def _enrich_audio_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add Affinity and Z-score columns to audio DataFrame."""
    df = df.copy()
    df["affinity_raw"] = df["valence"] * df["arousal"]

    mean = df["affinity_raw"].mean()
    std  = df["affinity_raw"].std()
    if std > 0:
        df["affinity_z"] = ((df["affinity_raw"] - mean) / std).round(4)
    else:
        df["affinity_z"] = 0.0

    return df


def _empty_audio_df() -> pd.DataFrame:
    cols = [
        "segment","start_sec","end_sec","dominant","confidence",
        "valence","arousal","intensity","affinity_raw","affinity_z"
    ]
    return pd.DataFrame(columns=cols)


# ── Mock data (used when SpeechBrain/librosa not installed) ──
def _mock_segments():
    """Return mock segment boundaries for a 15-min session."""
    return [
        (26, 98,  np.zeros(16000*5), 16000),
        (119, 235, np.zeros(16000*5), 16000),
        (239, 245, np.zeros(16000*5), 16000),
        (330, 342, np.zeros(16000*5), 16000),
        (412, 419, np.zeros(16000*5), 16000),
        (492, 505, np.zeros(16000*5), 16000),
        (579, 592, np.zeros(16000*5), 16000),
        (665, 675, np.zeros(16000*5), 16000),
        (748, 751, np.zeros(16000*5), 16000),
        (832, 843, np.zeros(16000*5), 16000),
        (915, 940, np.zeros(16000*5), 16000),
    ]


def _mock_segment(seg_idx: int, start: float, end: float) -> dict:
    """Realistic mock audio emotion output matching original CSV pattern."""
    mock_emotions = [
        ("happy",   0.72, 0.39, 0.24),
        ("disgust", 0.68, -0.46, 0.20),
        ("calm",    0.99, -0.01, 0.00),
        ("happy",   0.70, 0.42, 0.19),
        ("disgust", 0.75, -0.56, 0.27),
        ("disgust", 0.65, -0.40, 0.23),
        ("fearful", 0.58, -0.37, -0.04),
        ("disgust", 0.70, -0.46, 0.05),
        ("disgust", 0.78, -0.53, 0.20),
        ("fearful", 0.82, 0.00,  0.00),
        ("disgust", 0.55, -0.18, 0.16),
    ]
    em, conf, val, aro = mock_emotions[seg_idx % len(mock_emotions)]
    return {
        "segment":    seg_idx,
        "start_sec":  round(start, 3),
        "end_sec":    round(end, 3),
        "dominant":   em,
        "confidence": conf,
        "valence":    val,
        "arousal":    aro,
        "intensity":  round(abs(val) * abs(aro) ** 0.5, 4),
    }
