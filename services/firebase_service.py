"""
Firebase Service — HAI Backend
Connects Python backend to Firebase Firestore + Firebase Storage
Handles: subjects, sessions, emotion_results, conclusions, session video uploads
"""

import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import uuid
import os
import json
import csv
import io
import re

# ── Firebase Storage bucket for this project ──
FIREBASE_STORAGE_BUCKET = "hai-app-2ebc3.firebasestorage.app"

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
            firebase_admin.initialize_app(cred, {
                "storageBucket": FIREBASE_STORAGE_BUCKET,
            })
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
        video_path:        str | None      ← Firebase Storage download URL (or local path on failure)
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
# SESSION VIDEO — הקלטת הנבדק (Firebase Storage)
# ══════════════════════════════════════════

def _subject_folder_name(subject_id: str, subject_name: str = None) -> str:
    """
    Builds a readable-but-unique Firebase Storage folder name for a subject:
    the subject's name (sanitized) plus a short suffix taken from their
    subject_id — e.g. "דוד כהן_a3f91c" — so two subjects who happen to share
    the same name (e.g. two people both named "דוד") still get separate
    folders and their recordings never get mixed into one place. Falls back
    to the bare subject_id if the subject has no name on file.
    """
    short_id = (subject_id or "")[:6] or "000000"
    name = (subject_name or "").strip()
    # מסירים תווים שעלולים ליצור תת-תיקיות לא מכוונות (/) או לשבור את הנתיב
    name = re.sub(r"[\/\\\x00-\x1f]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name[:40]  # מגבילים אורך כדי לא ליצור נתיבים ארוכים מדי בסטוראג'
    if not name:
        return subject_id
    return f"{name}_{short_id}"


def get_subject_folder_name(subject_id: str) -> str:
    """Looks up the subject's name and returns their Storage folder name."""
    subject = get_subject(subject_id)
    name = subject.get("name") if subject else None
    return _subject_folder_name(subject_id, name)


def upload_session_video(subject_id: str, session_id: str, local_path: Path) -> str:
    """
    Uploads a subject's recorded session video (local .webm file, currently
    sitting on Render's ephemeral disk) to Firebase Storage, so it survives
    server restarts/redeploys and is reachable by Nico at any time.

    Stored at: recordings/{subject name + short id}/{session_id}.webm —
    grouped in a per-subject folder, alongside that same session's timeline
    CSV (see upload_session_timeline below), so both files for a given
    subject sit together in Storage.
    Returns a public Firebase Storage download URL (same format used for the
    baseline relaxation video).

    NOTE: sessions recorded before this change are still sitting at the old
    flat path (recordings/{session_id}.webm), or the earlier
    recordings/{subject_id}/... path — they are not moved.
    """
    get_db()  # ensures firebase_admin app (with storageBucket) is initialized

    folder = get_subject_folder_name(subject_id)
    storage_path = f"recordings/{folder}/{session_id}.webm"
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(str(local_path))

    # Attach a download token so the file is fetchable via a stable public URL
    # (same mechanism Firebase Console uses when you click "Get download URL").
    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.patch()

    download_url = (
        f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_STORAGE_BUCKET}"
        f"/o/{quote(storage_path, safe='')}?alt=media&token={token}"
    )
    print(f"✅ Session video uploaded to Firebase Storage: {storage_path}")
    return download_url


def _expand_timeline_to_csv(timeline_json: str) -> str:
    """
    Takes the raw segment log sent from the frontend (one row per phase/step
    the subject was in, each with start_sec/end_sec relative to the start of
    the recording) and expands it into a per-second table: one CSV row for
    every whole second of the recording, showing exactly what the subject was
    exposed to at that second. This makes it trivial to line this table up
    later against any other per-second signal (e.g. frame-by-frame emotion
    scores), instead of having to look up which segment a given second falls
    into by hand.
    """
    segments = json.loads(timeline_json) if timeline_json else []
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["second", "phase", "domain", "domain_label_he", "step", "step_label_he", "sub_phase", "label_he"])
    for seg in segments:
        start = seg.get("start_sec")
        if start is None:
            continue
        end = seg.get("end_sec")
        if end is None:
            end = start  # הקטע האחרון, אם משום מה לא נסגר כראוי — שורה אחת בלבד
        first_sec = int(start)
        last_sec = max(first_sec, int(end))
        for sec in range(first_sec, last_sec + 1):
            writer.writerow([
                sec,
                seg.get("phase", ""),
                seg.get("domain", "") or "",
                seg.get("domain_label_he", "") or "",
                seg.get("step", "") or "",
                seg.get("step_label_he", "") or "",
                seg.get("sub_phase", "") if seg.get("sub_phase") is not None else "",
                seg.get("label_he", "") or "",
            ])
    # BOM בתחילת הקובץ — כדי שגם תוכנות שלא מסתכלות על כותרת ה-Content-Type
    # (כמו Excel בפתיחת קובץ שהורד מקומית) יזהו נכון שהקובץ ב-UTF-8 ויציגו
    # את העברית נכון, במקום לנחש קידוד אחר ולהציג ג'יבריש.
    return "﻿" + output.getvalue()


def upload_session_timeline(subject_id: str, session_id: str, timeline_json: str) -> str:
    """
    Expands the frontend's timeline log (what the subject was exposed to,
    and when) into a per-second CSV table, and uploads it to Firebase
    Storage in the same per-subject folder as that session's video — so
    Nico can open both together and know exactly what was on screen/in the
    subject's ears at any given second of the recording.

    Stored at: recordings/{subject name + short id}/{session_id}_timeline.csv
    """
    get_db()
    csv_content = _expand_timeline_to_csv(timeline_json)

    folder = get_subject_folder_name(subject_id)
    storage_path = f"recordings/{folder}/{session_id}_timeline.csv"
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    # charset=utf-8 מפורש — בלעדיו, דפדפן שפותח את הלינק ישירות מנחש קידוד
    # (לרוב Latin-1/Windows-1252) ומציג את העברית כג'יבריש, למרות שהבתים
    # בקובץ עצמו תקינים ב-UTF-8.
    blob.upload_from_string(csv_content, content_type="text/csv; charset=utf-8")

    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    # Content-Disposition: attachment — בלעדיו, הדפדפן מציג את הקובץ כטקסט
    # בתוך הטאב כשנכנסים ללינק; עם זה, הוא יורד/נשמר ישירות כקובץ .csv בעל
    # שם קריא, מוכן לפתיחה ב-Excel/Numbers כטבלה מסודרת.
    blob.content_disposition = f'attachment; filename="{session_id}_timeline.csv"'
    blob.patch()

    download_url = (
        f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_STORAGE_BUCKET}"
        f"/o/{quote(storage_path, safe='')}?alt=media&token={token}"
    )
    print(f"✅ Session timeline uploaded to Firebase Storage: {storage_path}")
    return download_url


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
