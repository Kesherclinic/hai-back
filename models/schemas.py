"""
Pydantic schemas — request/response models for HAI API
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class SessionStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETE   = "complete"
    ERROR      = "error"


class SubjectInfo(BaseModel):
    name: Optional[str] = None
    age:  Optional[int] = None
    notes: Optional[str] = None


class AnalysisRequest(BaseModel):
    subject: Optional[SubjectInfo] = None
    protocol_version: str = "1.0"


class EmotionFrame(BaseModel):
    frame:     int
    timestamp: float
    dominant:  str
    valence:   float
    arousal:   float
    intensity: float
    affinity_z: Optional[float] = None


class AudioSegment(BaseModel):
    segment:    int
    start_sec:  float
    end_sec:    float
    dominant:   str
    confidence: float
    valence:    float
    arousal:    float
    intensity:  float
    affinity_z: Optional[float] = None


class PeakResult(BaseModel):
    start_sec:    float
    end_sec:      float
    duration_sec: float
    peak_z:       float
    is_signature: bool


class DomainResult(BaseModel):
    domain_id:         str
    domain_label_he:   str
    hai_score:         float
    peak_z:            float
    mean_z:            float
    peak_duration_sec: float
    consistency:       float
    classification:    str


class AnalysisResult(BaseModel):
    session_id:   str
    status:       SessionStatus
    video_rows:   Optional[int] = None
    audio_segments: Optional[int] = None
    video_stats:  Optional[dict] = None
    audio_stats:  Optional[dict] = None
    top_domains:  Optional[List[dict]] = None
    signature_peaks: Optional[List[dict]] = None
    rejection_flags: Optional[List[dict]] = None
    video_csv:    Optional[str] = None
    audio_csv:    Optional[str] = None
