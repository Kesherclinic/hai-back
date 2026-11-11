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

from fastapi import FastAPI, UploadFile, File, Form, Header, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import shutil
import os
from pathlib import Path

# ── מושבת זמנית (ML כבד) — ראה הערה למעלה כדי להחזיר ──
# from services.video_analyzer import analyze_video_frames
# from services.audio_analyzer import analyze_audio
# from services.affinity_calculator import calculate_affinity

from services.firebase_service import (
    create_subject, get_subject, list_subjects,
    create_session, update_session_status, get_session, list_sessions,
    get_emotion_results, get_conclusion, list_conclusions,
    save_full_session_results, upload_session_video, upload_session_timeline,
    generate_signed_video_url, generate_signed_timeline_url,
)

# ══════════════════════════════════════════
# שלב 0 (תוכנית העבודה — אבטחה ופרטיות): נעילת הבקאנד
# ══════════════════════════════════════════
# פתרון ביניים מהיר ודחוף: כל קריאה לבקאנד חייבת לשאת הדר X-API-Key עם סוד
# משותף אחד לכל האפליקציה (לא אימות אישי — זה יוחלף בשלב 1 בטוקן Firebase
# אמיתי לכל משתמש). הסוד נשמר אך ורק כמשתנה סביבה ב-Render, לעולם לא בקוד.
API_SHARED_SECRET = os.environ.get("API_SHARED_SECRET")
if not API_SHARED_SECRET:
    print("⚠️ API_SHARED_SECRET אינו מוגדר — כל קריאה לבקאנד תיחסם (401) עד שיוגדר ב-Render.")

async def verify_shared_secret(x_api_key: str = Header(None)):
    if not API_SHARED_SECRET or x_api_key != API_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

# דומיין(ים) מורשים לקרוא לבקאנד — מחליף את allow_origins=["*"] הפתוח לכולם.
# מוגדר דרך משתנה סביבה FRONTEND_ORIGIN ב-Render (אפשר כמה דומיינים מופרדים
# בפסיק, למשל דומיין production + דומייני preview של Vercel). חשוב: חייבים
# להגדיר כאן בדיוק את הדומיין האמיתי של haifront — דומיין שגוי יחסום את
# האפליקציה עצמה, לא רק תוקפים.
_frontend_origins_env = os.environ.get("FRONTEND_ORIGIN", "")
FRONTEND_ORIGINS = [o.strip() for o in _frontend_origins_env.split(",") if o.strip()]
if not FRONTEND_ORIGINS:
    print("⚠️ FRONTEND_ORIGIN אינו מוגדר ב-Render — נופלים זמנית לדומיין production הידוע; מומלץ בכל זאת להגדיר אותו ב-Render במקום להסתמך על הפולבאק הזה.")
    FRONTEND_ORIGINS = ["https://haifront-sigma.vercel.app"]

app = FastAPI(
    title="HAI Backend — Heart Affinity Index",
    description="Recording-only mode: captures and stores subject video; biometric analysis not yet enabled",
    version="1.1.0-recording-only",
)
# הערה: ה-dependency לא מוגדר ברמת האפליקציה כולה (שהיה חוסם גם את "/"), אלא
# בנפרד על כל endpoint רגיש למטה — כדי ש-"/" יישאר פתוח כ-health check
# שRender (או כל מנגנון ניטור) יכול לפנות אליו בלי הדר, ועדיין לא יחשוף מידע
# כלשהו (הוא מחזיר רק מחרוזת סטטוס קבועה).

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
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
@app.post("/subjects", dependencies=[Depends(verify_shared_secret)])
def new_subject(name: str, age: int = None, notes: str = ""):
    subject_id = create_subject(name=name, age=age, notes=notes)
    return {"subject_id": subject_id, "name": name}

@app.get("/subjects", dependencies=[Depends(verify_shared_secret)])
def all_subjects():
    return list_subjects()

@app.get("/subjects/{subject_id}", dependencies=[Depends(verify_shared_secret)])
def one_subject(subject_id: str):
    s = get_subject(subject_id)
    if not s:
        raise HTTPException(404, "Subject not found")
    return s

# ── SESSIONS ──
@app.post("/sessions", dependencies=[Depends(verify_shared_secret)])
def new_session(subject_id: str, protocol_version: str = "1.0"):
    session_id = create_session(subject_id=subject_id, protocol_version=protocol_version)
    return {"session_id": session_id, "subject_id": subject_id}

@app.post("/analyze/{session_id}", dependencies=[Depends(verify_shared_secret)])
async def analyze(
    session_id: str,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    stimulus_map: str = Form(None),
    timeline: str = Form(None),
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    video_path = UPLOAD_DIR / f"{session_id}.webm"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    subject_id = session["subject_id"]

    # מצב הקלטה בלבד: שומר את הסרטון ומסמן "recorded" — בלי להפעיל ניתוח ביומטרי.
    # כדי להחזיר את הניתוח המלא, מחק את השורה עם upload_session_video/update_session_status
    # למטה והחזר את שתי השורות המוערות שאחריה.
    #
    # הדיסק של Render הוא זמני (ephemeral) — קבצים שנשמרים רק שם נמחקים בכל
    # הפעלה מחדש/דיפלוי. לכן מעלים את ההקלטה ל-Firebase Storage מיד אחרי
    # השמירה המקומית, כדי שהיא תישמר לצמיתות ותהיה נגישה לך בכל זמן.
    try:
        storage_url = upload_session_video(subject_id, session_id, video_path)

        # טבלת הזמנים (מה הנבדק נחשף אליו בכל שנייה) — מועלית בנפרד, לתוך אותה
        # תיקיית נבדק. אם ההעלאה הזו נכשלת, זה לא אמור לפגוע בשמירת הסרטון עצמו
        # (הסרטון הוא הדבר הקריטי), לכן הכישלון שלה מטופל בנפרד ולא עוצר את הזרימה.
        timeline_url = None
        if timeline:
            try:
                timeline_url = upload_session_timeline(subject_id, session_id, timeline)
            except Exception as te:
                print(f"⚠️ Failed to upload session timeline to Firebase Storage: {te}")

        update_session_status(session_id, "recorded", video_path=storage_url, timeline_path=timeline_url)
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

@app.get("/sessions/{session_id}", dependencies=[Depends(verify_shared_secret)])
def one_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s

@app.get("/sessions", dependencies=[Depends(verify_shared_secret)])
def all_sessions(subject_id: str = None):
    return list_sessions(subject_id=subject_id)

# ── RESULTS (לא יחזירו כלום כל עוד הניתוח מושבת — זה תקין, לא באג) ──
@app.get("/results/{session_id}", dependencies=[Depends(verify_shared_secret)])
def session_results(session_id: str):
    results = get_emotion_results(session_id)
    if not results:
        raise HTTPException(404, "Results not found")
    return results

@app.get("/conclusions/{session_id}", dependencies=[Depends(verify_shared_secret)])
def session_conclusion(session_id: str):
    conclusion = get_conclusion(session_id)
    if not conclusion:
        raise HTTPException(404, "Conclusion not found")
    return conclusion

@app.get("/conclusions", dependencies=[Depends(verify_shared_secret)])
def all_conclusions(subject_id: str = None):
    return list_conclusions(subject_id=subject_id)

# ── RECORDINGS — קישורי הורדה זמניים (signed URLs), במקום הטוקן הציבורי הקבוע ──
# שלב 0 (המשך): מחליפים את firebaseStorageDownloadTokens (שתקף לנצח לכל מי
# שמחזיק את הלינק) בקישור חתום שפג תוקפו אחרי זמן קצר ונוצר רק דרך קריאה
# מאומתת לבקאנד. ה-URL עצמו עדיין לא דורש התחברות (זו מגבלה של signed URLs),
# אבל הוא נוצר רק למי שכבר עבר את verify_shared_secret וגם פג תוקף במהירות —
# שיפור משמעותי לעומת טוקן שחי לנצח.
@app.get("/recordings/{session_id}/video", dependencies=[Depends(verify_shared_secret)])
def recording_video_url(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    url = generate_signed_video_url(session["subject_id"], session_id)
    if not url:
        raise HTTPException(404, "Video not found")
    return {"url": url}

@app.get("/recordings/{session_id}/timeline", dependencies=[Depends(verify_shared_secret)])
def recording_timeline_url(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    url = generate_signed_timeline_url(session["subject_id"], session_id)
    if not url:
        raise HTTPException(404, "Timeline not found")
    return {"url": url}

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
