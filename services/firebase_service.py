"""
Firebase Service — HAI Backend
Connects Python backend to Firebase Firestore
Handles: subjects, sessions, emotion_results, conclusions
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from pathlib import Path
import os

# ── Initialize Firebase ──
_db = None

def get_db():
    global _db
    if _db is None:
        key_path = Path(__file__).parent.parent / "firebase_key.json"
        if not key_path.exists():
            raise FileNotFoundError(
                f"firebase_key.json not found at {key_path}\n"
                "Download it from Firebase Console → Project Settings → Service Accounts"
            )
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(key_path))
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db


# ══════════════════════════════════════════
# SUBJECTS — נבדקים
# ══════════════════════════════════════════

def create_subject(name: str, age: int = None, notes: str = "") -> str:
    """
    Create a new subject record.
    Returns the new subject_id.

    Schema:
    subjects/{subject_id}
        name:        str
        age:         int | None
        notes:       str
        created_at:  timestamp
    """
    db = get_db()
    ref = db.collection("subjects").document()
    ref.set({
        "name":       name,
        "age":        age,
        "notes":      notes,
        "created_at": datetime.utcnow(),
    })
    print(f"✅ Subject created: {ref.id} ({name})")
    return ref.id


def get_subject(subject_id: str) -> dict | None:
    db = get_db()
    doc = db.collection("subjects").document(subject_id).get()
    return doc.to_dict() if doc.exists else None


def list_subjects() -> list[dict]:
    db = get_db()
    docs = db.collection("subjects").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    return [{"id": d.id, **d.to_dict()} for d in docs]


# ══════════════════════════════════════════
# SESSIONS — סשני בדיקה
# ══════════════════════════════════════════

def create_session(subject_id: str, protocol_version: str = "1.0") -> str:
    """
    Create a new test session for a subject.
    Returns the new session_id.

    Schema:
    sessions/{session_id}
        subject_id:        str
        protocol_version:  str
        status:            "pending" | "recording" | "processing" | "complete" | "error"
        video_path:        str | None      ← local path or Firebase Storage URL
        created_at:        timestamp
        completed_at:      timestamp | None
        error:             str | None
    """
    db = get_db()
    ref = db.collection("sessions").document()
    ref.set({
        "subject_id":       subject_id,
        "protocol_version": protocol_version,
        "status":           "pending",
        "video_path":       None,
        "created_at":       datetime.utcnow(),
        "completed_at":     None,
        "error":            None,
    })
    print(f"✅ Session created: {ref.id}")
    return ref.id


def update_session_status(session_id: str, status: str, **kwargs):
    """Update session status and any extra fields."""
    db = get_db()
    data = {"status": status, **kwargs}
    if status == "complete":
        data["completed_at"] = datetime.utcnow()
    db.collection("sessions").document(session_id).update(data)


def get_session(session_id: str) -> dict | None:
    db = get_db()
    doc = db.collection("sessions").document(session_id).get()
    return {"id": doc.id, **doc.to_dict()} if doc.exists else None


def list_sessions(subject_id: str = None) -> list[dict]:
    db = get_db()
    q = db.collection("sessions").order_by("created_at", direction=firestore.Query.DESCENDING)
    if subject_id:
        q = q.where("subject_id", "==", subject_id)
    return [{"id": d.id, **d.to_dict()} for d in q.stream()]


# ══════════════════════════════════════════
# EMOTION RESULTS — תוצאות רגשיות
# ══════════════════════════════════════════

def save_emotion_results(session_id: str, video_stats: dict, audio_stats: dict,
                          peaks: dict, domain_scores: list, rejection_flags: list):
    """
    Save full emotion analysis results for a session.

    Schema:
    emotion_results/{session_id}
        session_id:      str
        video_stats:     dict   ← mean valence/arousal, max Z, dominant emotion
        audio_stats:     dict   ← same for audio channel
        peaks:           dict   ← {video: [...], audio: [...]}
        domain_scores:   list   ← [{domain_id, hai_score, peak_z, ...}, ...]
        top_domains:     list   ← top 3 domains with Z > 1.5
        signature_peaks: list   ← peaks with Z > 2.0 and duration > 5s
        rejection_flags: list   ← moments of strong negative affinity
        saved_at:        timestamp
    """
    db = get_db()
    top_domains = sorted(
        [d for d in domain_scores if d.get("classification") in ("strong", "notable")],
        key=lambda x: x.get("hai_score", 0), reverse=True
    )[:3]

    signature_peaks = [
        p for p in (peaks.get("video", []) + peaks.get("audio", []))
        if p.get("is_signature")
    ]

    db.collection("emotion_results").document(session_id).set({
        "session_id":      session_id,
        "video_stats":     video_stats,
        "audio_stats":     audio_stats,
        "peaks":           peaks,
        "domain_scores":   domain_scores,
        "top_domains":     top_domains,
        "signature_peaks": signature_peaks,
        "rejection_flags": rejection_flags,
        "saved_at":        datetime.utcnow(),
    })
    print(f"✅ Emotion results saved for session: {session_id}")


def get_emotion_results(session_id: str) -> dict | None:
    db = get_db()
    doc = db.collection("emotion_results").document(session_id).get()
    return doc.to_dict() if doc.exists else None


# ══════════════════════════════════════════
# CONCLUSIONS — מסקנות
# ══════════════════════════════════════════

def save_conclusion(session_id: str, subject_id: str,
                    top_domains: list, hai_scores: dict,
                    narrative: str, protocol_version: str = "1.0"):
    """
    Save the final HAI conclusion for a session.

    Schema:
    conclusions/{session_id}
        session_id:        str
        subject_id:        str
        protocol_version:  str
        top_domains:       list   ← [{rank, domain, label_he, hai_score, classification}]
        hai_scores:        dict   ← {domain_id: score} for all 8 domains
        narrative:         str    ← Claude-generated Hebrew narrative
        generated_at:      timestamp
    """
    db = get_db()
    db.collection("conclusions").document(session_id).set({
        "session_id":       session_id,
        "subject_id":       subject_id,
        "protocol_version": protocol_version,
        "top_domains":      top_domains,
        "hai_scores":       hai_scores,
        "narrative":        narrative,
        "generated_at":     datetime.utcnow(),
    })
    print(f"✅ Conclusion saved for session: {session_id}")


def get_conclusion(session_id: str) -> dict | None:
    db = get_db()
    doc = db.collection("conclusions").document(session_id).get()
    return doc.to_dict() if doc.exists else None


def list_conclusions(subject_id: str = None) -> list[dict]:
    db = get_db()
    q = db.collection("conclusions").order_by("generated_at", direction=firestore.Query.DESCENDING)
    if subject_id:
        q = q.where("subject_id", "==", subject_id)
    return [{"id": d.id, **d.to_dict()} for d in q.stream()]


# ══════════════════════════════════════════
# FULL PIPELINE SAVE
# ══════════════════════════════════════════

def save_full_session_results(session_id: str, subject_id: str, affinity_results: dict, narrative: str):
    """
    Convenience function: saves emotion_results + conclusion + updates session status.
    Called at the end of the full analysis pipeline.
    """
    # 1. Save emotion results
    save_emotion_results(
        session_id=session_id,
        video_stats=affinity_results.get("video", {}),
        audio_stats=affinity_results.get("audio", {}),
        peaks=affinity_results.get("peaks", {}),
        domain_scores=affinity_results.get("domain_scores", []),
        rejection_flags=affinity_results.get("rejection_flags", []),
    )

    # 2. Save conclusion
    top_domains = affinity_results.get("top_domains", [])
    hai_scores  = {
        d["domain_id"]: d["hai_score"]
        for d in affinity_results.get("domain_scores", [])
    }
    save_conclusion(
        session_id=session_id,
        subject_id=subject_id,
        top_domains=top_domains,
        hai_scores=hai_scores,
        narrative=narrative,
    )

    # 3. Mark session complete
    update_session_status(session_id, "complete")
    print(f"✅ Full session saved: {session_id}")
