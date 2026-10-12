"""
HAI Backend — Heart Affinity Index
FastAPI server — גרסת "הקלטה בלבד" (ללא ניתוח ביומטרי, זמנית)

שלב הניתוח הביומטרי (DeepFace / SpeechBrain וכו') מושבת כרגע כדי לאפשר פריסה
על תוכנית אירוח חינמית/קלה, בלי לדרוש RAM כבד. השרת הזה רק מקבל את הסרטון,
שומר אותו, ומסמן את ה-session כ"הוקלט" — בלי להריץ שום ניתוח.

כדי להחזיר את הניתוח המלא בעתיד:
1. הסר את ה-# משלוש שורות היבוא של video_analyzer / audio_analyzer / affinity_calculator למטה.
2. הסר את ה-# מהפונקציות run_full_analysis / _build_temp_narrative בתחתית הקובץ.
3. בתוך analyze(), החזר את שורת ה-background_tasks.add_task(...) (מסומנת למטה).
4. עדכן את requirements.txt חזרה לגרסה המלאה עם deepface/torch/speechbrain/librosa וכו'.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import shutil
from pathlib import Path

# ── מושבת זמנית (ML כבד) — ראה הערה למעלה כדי להחזיר ──
# from services.video_analyzer import analyze_video_frames
# from services.audio_analyzer import analyze_audio
# from services.affinity_calculator import calculate_affinity

from services.firebase_service import (
    create_subject, get_subject, list_subjects,
    create_session, update_session_status, get_session, list_sessions,
    get_emotion_results, get_conclusion, list_conclusions,
    save_full_session_results, upload_session_video
)

app = FastAPI(
    title="HAI Backend — Heart Affinity Index",
    description="Recording-only mode: captures and stores subject video; biometric analysis not yet enabled",
    version="1.1.0-recording-only"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    return {"status": "HAI backend running (recording only)", "version": "1.1.0-recording-only"}

# ── SUBJECTS ──
@app.post("/subjects")
def new_subject(name: str, age: int = None, notes: str = ""):
    subject_id = create_subject(name=name, age=age, notes=notes)
    return {"subject_id": subject_id, "name": name}

@app.get("/subjects")
def all_subjects():
    return list_subjects()

@app.get("/subjects/{subject_id}")
def one_subject(subject_id: str):
    s = get_subject(subject_id)
    if not s:
        raise HTTPException(404, "Subject not found")
    return s

# ── SESSIONS ──
@app.post("/sessions")
def new_session(subject_id: str, protocol_version: str = "1.0"):
    session_id = create_session(subject_id=subject_id, protocol_version=protocol_version)
    return {"session_id": session_id, "subject_id": subject_id}

@app.post("/analyze/{session_id}")
async def analyze(
    session_id: str,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    stimulus_map: str = Form(None),
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    video_path = UPLOAD_DIR / f"{session_id}.webm"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # מצב הקלטה בלבד: שומר את הסרטון ומסמן "recorded" — בלי להפעיל ניתוח ביומטרי.
    # כדי להחזיר את הניתוח המלא, מחק את השורה עם upload_session_video/update_session_status
    # למטה והחזר את שתי השורות המוערות שאחריה.
    #
    # הדיסק של Render הוא זמני (ephemeral) — קבצים שנשמרים רק שם נמחקים בכל
    # הפעלה מחדש/דיפלוי. לכן מעלים את ההקלטה ל-Firebase Storage מיד אחרי
    # השמירה המקומית, כדי שהיא תישמר לצמיתות ותהיה נגישה לך בכל זמן.
    try:
        storage_url = upload_session_video(session_id, video_path)
        update_session_status(session_id, "recorded", video_path=storage_url)
        # מחיקת העותק המקומי אחרי העלאה מוצלחת — חוסך מקום בדיסק הזמני של Render
        video_path.unlink(missing_ok=True)
    except Exception as e:
        # אם ההעלאה ל-Firebase נכשלה, לפחות שומרים את הנתיב המקומי כדי לא לאבד
        # את ההקלטה (אם Render לא התאתחל בינתיים), ומתעדים את השגיאה.
        print(f"⚠️ Failed to upload session video to Firebase Storage: {e}")
        update_session_status(session_id, "recorded", video_path=str(video_path), storage_error=str(e))
    # import json
    # stimulus_map_parsed = json.loads(stimulus_map) if stimulus_map else {}
    # background_tasks.add_task(run_full_analysis, session_id, session["subject_id"], video_path, stimulus_map_parsed)

    return {"session_id": session_id, "status": "recorded"}

@app.get("/sessions/{session_id}")
def one_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s

@app.get("/sessions")
def all_sessions(subject_id: str = None):
    return list_sessions(subject_id=subject_id)

# ── RESULTS (לא יחזירו כלום כל עוד הניתוח מושבת — זה תקין, לא באג) ──
@app.get("/results/{session_id}")
def session_results(session_id: str):
    results = get_emotion_results(session_id)
    if not results:
        raise HTTPException(404, "Results not found")
    return results

@app.get("/conclusions/{session_id}")
def session_conclusion(session_id: str):
    conclusion = get_conclusion(session_id)
    if not conclusion:
        raise HTTPException(404, "Conclusion not found")
    return conclusion

@app.get("/conclusions")
def all_conclusions(subject_id: str = None):
    return list_conclusions(subject_id=subject_id)

# ── PIPELINE — מושבת זמנית, ראה הערה בראש הקובץ כדי להחזיר ──
# async def run_full_analysis(session_id: str, subject_id: str, video_path: Path, stimulus_map: dict = {}):
#     try:
#         result_dir = RESULTS_DIR / session_id
#         result_dir.mkdir(exist_ok=True)
#         video_df = await analyze_video_frames(video_path, result_dir / "video_analysis.csv")
#         audio_df = await analyze_audio(video_path, result_dir / "audio_emotions.csv")
#         affinity = calculate_affinity(video_df, audio_df, stimulus_map)
#         narrative = _build_temp_narrative(affinity)
#         save_full_session_results(session_id=session_id, subject_id=subject_id,
#                                    affinity_results=affinity, narrative=narrative)
#     except Exception as e:
#         update_session_status(session_id, "error", error=str(e))
#         print(f"Analysis error for {session_id}: {e}")
#
# def _build_temp_narrative(affinity: dict) -> str:
#     top = affinity.get("top_domains", [])
#     if not top:
#         return "לא זוהו תחומי חיים מובהקים בסשן זה."
#     lines = ["תוצאות ניתוח נטיית הלב:"]
#     for d in top:
#         lines.append(f"• {d['label_he']} — ציון HAI: {d['hai_score']:.0f} (Z-peak: {d.get('peak_z',0):.2f})")
#     sig = affinity.get("signature_peaks", [])
#     if sig:
#         p = sig[0]
#         lines.append(f"\nSignature Peak ב-{p.get('start_sec', p.get('timestamp', 0)):.0f}s (משך: {p['duration_sec']:.1f}s, Z={p['peak_z']:.2f}).")
#     lines.append("\n⚠️ מסקנה מלאה תיוצר על ידי Claude AI בשלב 4.")
#     return "\n".join(lines)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
