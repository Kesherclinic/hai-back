"""
Audio Analyzer — HuggingFace transformers emotion extraction
Pipeline: video → FFmpeg (extract audio) → wav2vec2 emotion model → Valence/Arousal per segment
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
    from transformers import pipeline
    AUDIO_MODEL_AVAILABLE = True
except ImportError:
    AUDIO_MODEL_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install transformers")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

EXECUTOR = ThreadPoolExecutor(max_workers=2)

AUDIO_EMOTION_VA = {
    "hap":      {"valence":  0.80, "arousal":  0.70},
    "ang":      {"valence": -0.75, "arousal":  0.90},
    "sad":      {"valence": -0.65, "arousal":  0.20},
    "neu":      {"valence":  0.00, "arousal":  0.05},
    "happy":    {"valence":  0.80, "arousal":  0.70},
    "angry":    {"valence": -0.75, "arousal":  0.90},
    "neutral":  {"valence":  0.00, "arousal":  0.05},
    "fearful":  {"valence": -0.70, "arousal":  0.85},
    "disgust":  {"valence": -0.65, "arousal":  0.55},
    "surprise": {"valence":  0.15, "arousal":  0.80},
    "calm":     {"valence":  0.30, "arousal": -0.10},
}

SEGMENT_DURATION_SEC = 5
MIN_SEGMENT_ENERGY   = 0.001

_emotion_pipeline = None

def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print("📦 Loading emotion model (first run: downloads ~200MB)...")
        _emotion_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=-1
        )
        print("✅ Emotion model loaded")
    return _emotion_pipeline

def _extract_audio(video_path: Path, wav_path: Path) -> bool:
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000",
            "-vn", str(wav_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and wav_path.exists()
    except Exception as e:
        print(f"⚠️  Audio extraction error: {e}")
        return False

def _analyze_segment(waveform_np: np.ndarray, sr: int, seg_idx: int) -> dict | None:
    try:
        classifier = _get_emotion_pipeline()
        import soundfile as sf
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, waveform_np, sr)
        results = classifier(tmp_path, top_k=1)
        os.unlink(tmp_path)
        if not results:
            return None
        top = results[0]
        label = top["label"].lower()
        score = top["score"]
        va = AUDIO_EMOTION_VA.get(label, {"valence": 0.0, "arousal": 0.0})
        return {
            "emotion": label,
            "confidence": round(score, 4),
            "valence": va["valence"],
            "arousal": va["arousal"],
        }
    except Exception as e:
        print(f"⚠️  Audio error on segment {seg_idx}: {e}")
        return None

async def analyze_audio(video_path: Path, output_csv: Path) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=[
        "segment","timestamp","dominant","confidence",
        "valence","arousal","intensity","affinity_raw","affinity_z"
    ])

    if not AUDIO_MODEL_AVAILABLE:
        print("⚠️  Audio model not available")
        empty_df.to_csv(output_csv, index=False)
        return empty_df

    wav_path = output_csv.parent / f"{video_path.stem}_audio.wav"
    print(f"✅ Audio extracted: {wav_path}")

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(EXECUTOR, _extract_audio, video_path, wav_path)

    if not success:
        print("⚠️  Audio extraction failed — returning empty DataFrame")
        empty_df.to_csv(output_csv, index=False)
        return empty_df

    try:
        if LIBROSA_AVAILABLE:
            import librosa
            waveform, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        else:
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.numpy()[0]
            sr = 16000
    except Exception as e:
        print(f"⚠️  Audio load error: {e}")
        empty_df.to_csv(output_csv, index=False)
        return empty_df

    seg_samples = SEGMENT_DURATION_SEC * sr
    total_samples = len(waveform)
    records = []

    for seg_idx in range(0, total_samples // seg_samples):
        start = seg_idx * seg_samples
        end = start + seg_samples
        segment = waveform[start:end]
        energy = float(np.sqrt(np.mean(segment ** 2)))
        if energy < MIN_SEGMENT_ENERGY:
            continue
        timestamp = seg_idx * SEGMENT_DURATION_SEC
        result = await loop.run_in_executor(EXECUTOR, _analyze_segment, segment, sr, seg_idx)
        if result is None:
            continue
        valence = result["valence"]
        arousal = result["arousal"]
        intensity = energy
        affinity_raw = (valence * 0.6 + arousal * 0.4) * intensity
        records.append({
            "segment":     seg_idx,
            "timestamp":   timestamp,
            "dominant":    result["emotion"],
            "confidence":  result["confidence"],
            "valence":     valence,
            "arousal":     arousal,
            "intensity":   round(intensity, 4),
            "affinity_raw": round(affinity_raw, 4),
            "affinity_z":  0.0,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        mean = df["affinity_raw"].mean()
        std = df["affinity_raw"].std()
        if std > 0:
            df["affinity_z"] = ((df["affinity_raw"] - mean) / std).round(4)
        else:
            df["affinity_z"] = 0.0

    df.to_csv(output_csv, index=False)
    print(f"✅ Audio analysis complete: {len(df)} segments → {output_csv}")
    return df
