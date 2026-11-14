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
import re
from pathlib import Path

# ── מושבת זמנית (ML כבד) — ראה הערה למעלה כדי להחזיר ──
# from services.video_analyzer import analyze_video_frames
# from services.audio_analyzer import analyze_audio
# from services.affinity_calculator import calculate_affinity

from services.firebase_service import (
    get_db, create_subject, get_subject, list_subjects,
    create_session, update_session_status, get_session, list_sessions,
    get_emotion_results, get_conclusion, list_conclusions,
    save_full_session_results, upload_session_video, upload_session_timeline,
    generate_signed_video_url, generate_signed_timeline_url,
    update_login_credentials, complete_subject_profile, increment_session_count,
)

# שלב 1 (תוכנית העבודה — אבטחה ופרטיות): אימות נבדקים אמיתי דרך Firebase
# Authentication. ה-import הזה חייב לבוא אחרי ה-import מ-firebase_service
# למעלה, כי הוא מה שמאתחל את firebase_admin (דרך get_db בתוך אחת הפונקציות
# שיובאו) — בלעדי אתחול קודם, firebase_admin.auth לא יודע לאיזה פרויקט לפנות.
from firebase_admin import auth as firebase_auth

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

# ══════════════════════════════════════════
# שלב 1 (תוכנית העבודה — אבטחה ופרטיות): אימות אישי לנבדקים
# ══════════════════════════════════════════
# להבדיל מ-verify_shared_secret למעלה (סוד אחד משותף לכל האפליקציה), כל נבדק
# שולח כעת גם Authorization: Bearer <idToken> — טוקן אישי שה-SDK של Firebase
# בפרונט מנפיק לו לאחר התחברות (Email/Password עם פסבדו-אימייל, ראו
# WelcomePage/SignInPage/SignUpPage). הבקאנד מאמת את הטוקן מול Firebase עצמו
# (לא סומך על שום דבר שהלקוח טוען) ומוציא ממנו את ה-uid האמיתי + ה-custom
# claim role. ה-uid הזה הוא בדיוק מזהה המסמך ב-subjects/{uid} (ראו
# firebase_service.create_provisioned_subject) — כך שאין אפשרות לנבדק לפעול
# בשם נבדק אחר, אפילו אם הוא מנסה לשלוח subject_id שלא שלו בפרמטר.
#
# משמש רק את ה-endpoints החדשים של זרימת הנבדק האמיתית (/auth/*,
# /subjects/me/*) — שאר ה-endpoints הישנים ממשיכים כרגע עם verify_shared_secret
# בלבד (ניקוי/איחוד בין שני סוגי האימות מתוכנן לשלב 1.9, לא כרגע).
async def verify_subject_token(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    id_token = authorization[len("Bearer "):].strip()
    get_db()  # מבטיח ש-firebase_admin כבר מאותחל לפני שפונים ל-auth
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if decoded.get("role") != "subject":
        raise HTTPException(status_code=403, detail="Not a subject account")
    return decoded

# ══════════════════════════════════════════
# שלב 2.3 (תוכנית העבודה — אבטחה ופרטיות): הרשאת admin אמיתית ל-endpoints רגישים
# ══════════════════════════════════════════
# בדיוק כמו verify_subject_token למעלה, רק בודק role == "admin" במקום
# role == "subject". ה-admin היחיד שקיים כרגע הוא זה שנוצר ב-
# create_admin_account.py, ואיתו מתחברים גם ב-AccessGate בפרונט (App.js).
#
# משמש רק את שלושת ה-endpoints שמחזירים את *כל* הרשומות בלי שום סינון
# (כל הנבדקים / כל הסשנים / כל המסקנות) — אלה בדיוק ה-endpoints שהסקירה
# סימנה כ"קריטיים": מי שמחלץ את ה-API Key המשותף (שגלוי לכל מי שבודק את
# תעבורת האתר בדפדפן) יכול איתו לשלוף נתונים אישיים (שם/גיל/הערות) על כל
# נבדק במערכת, לא רק על אחד שהוא כבר יודע עליו. endpoints אחרים שמקבלים
# מזהה ספציפי (subject_id/session_id בודד שכבר ידוע לקורא) נשארים כרגע עם
# verify_shared_secret בלבד, כדי לא לשבור את זרימת הנבדק/הניתוח הקיימת
# בלי לבדוק קודם את כל העמודים שתלויים בהם.
async def verify_admin_token(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    id_token = authorization[len("Bearer "):].strip()
    get_db()  # מבטיח ש-firebase_admin כבר מאותחל לפני שפונים ל-auth
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if decoded.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return decoded

# ══════════════════════════════════════════
# שלב 2.4/2.5 (תוכנית העבודה — אבטחה ופרטיות): בדיקת "בעלות" על סשן ספציפי
# ══════════════════════════════════════════
# להבדיל מ-verify_subject_token/verify_admin_token למעלה (שבודקים role אחד
# ספציפי), כאן גם נבדק וגם admin מורשים להגיע ל-endpoint — אבל מי שנגשים אליו
# בפועל תלוי ברשומה הספציפית שמתבקשת (session_id נתון), ולא ניתן להכריע את זה
# מראש ב-dependency בלי לטעון את הרשומה. לכן verify_subject_or_admin_token
# רק מאמת את הטוקן ומחזיר אותו, וכל endpoint קורא בעצמו ל-_require_owner_or_admin
# אחרי שהוא כבר יודע למי שייכת הרשומה (session["subject_id"] וכו').
#
# משמש את ה-endpoints שמקבלים session_id ספציפי שכבר ידוע לקורא (לא רשימות
# מלאות — אלה כבר מוגנות ב-verify_admin_token למעלה): POST /analyze/{id},
# GET /sessions/{id}, GET /results/{id}, GET /conclusions/{id},
# GET /recordings/{id}/video|timeline. קודם כל חמשת ה-endpoints האלה בדקו רק
# את ה-API Key המשותף — מי שהחזיק אותו (גלוי בדפדפן לכל אחד) וגם הצליח לקבל
# איכשהו session_id ספציפי (למשל דליפה של קישור) יכל לגשת לנתונים/להעלות וידאו
# לכל סשן, לא רק לסשן שהוא עצמו פתח. מעכשיו גם מי שמחזיק ID ספציפי לא שלו
# יידחה, אלא אם הוא admin.
async def verify_subject_or_admin_token(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    id_token = authorization[len("Bearer "):].strip()
    get_db()
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if decoded.get("role") not in ("subject", "admin"):
        raise HTTPException(status_code=403, detail="Unrecognized account role")
    return decoded

def _require_owner_or_admin(decoded: dict, owner_subject_id: str) -> None:
    if decoded.get("role") == "admin":
        return
    if decoded.get("uid") != owner_subject_id:
        raise HTTPException(status_code=403, detail="הסשן הזה לא שלך")

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
# שלב 2.3: יצירת נבדק "אנונימי" (זרימת האינטייק הישנה של הסגל — IntakePage.js)
# אין כאן זהות נבדק אמיתית ליצור "בעלות" עליה, אז דורשים admin בלבד (לא
# verify_shared_secret הישן), בדיוק כמו שאר כלי הסגל.
@app.post("/subjects", dependencies=[Depends(verify_admin_token)])
def new_subject(name: str, age: int = None, notes: str = ""):
    subject_id = create_subject(name=name, age=age, notes=notes)
    return {"subject_id": subject_id, "name": name}

@app.get("/subjects", dependencies=[Depends(verify_admin_token)])
def all_subjects():
    return list_subjects()

@app.get("/subjects/{subject_id}", dependencies=[Depends(verify_shared_secret)])
def one_subject(subject_id: str):
    s = get_subject(subject_id)
    if not s:
        raise HTTPException(404, "Subject not found")
    return s

# ── AUTH — זרימת הנבדק האמיתית (שלב 1) ──
# כל ה-endpoints כאן מאומתים דרך verify_subject_token (טוקן Firebase אישי),
# לא verify_shared_secret. ה-uid נלקח מהטוקן המאומת בלבד — אף endpoint כאן
# לא מקבל subject_id כפרמטר מהלקוח, כדי שאף נבדק לא יוכל לפעול בשם נבדק אחר.

@app.get("/auth/me")
def auth_me(decoded=Depends(verify_subject_token)):
    """
    נקרא בכל כניסה לאתר אחרי ההתחברות, כדי שהפרונט ידע לאן לנתב את הנבדק:
    needs_reset → מסך איפוס שם משתמש+סיסמא; profile_completed=False → טופס
    פרטים אישיים; אחרת → מסך אישור "סשן שני/שלישי" לפי session_count.
    """
    uid = decoded["uid"]
    subject = get_subject(uid)
    if not subject:
        raise HTTPException(404, "Subject not found")
    return {
        "uid": uid,
        "login_username":    subject.get("login_username"),
        "needs_reset":       subject.get("needs_reset", False),
        "profile_completed": subject.get("profile_completed", False),
        "session_count":     subject.get("session_count", 0),
        "name":              subject.get("name"),
        "gender":            subject.get("gender"),
    }


@app.post("/auth/reset")
def auth_reset(new_username: str, new_password: str, decoded=Depends(verify_subject_token)):
    """
    איפוס חד-פעמי בכניסה הראשונה (needs_reset=True): הנבדק בוחר שם משתמש
    וסיסמא משלו במקום אלה שנשלחו לו מראש. משנה גם את חשבון Firebase
    Authentication עצמו (האימייל-הפסבדו שמשמש להתחברות) וגם את מסמך
    ה-Firestore (login_username, needs_reset).
    """
    uid = decoded["uid"]
    # סניטציה — חייבת להיות זהה בדיוק ל-usernameToPseudoEmail ב-firebase.js
    # בפרונט: Firebase, ב-Email/Password sign-in, לא מקבל רווחים או תווים
    # מיוחדים בחלק שלפני ה-@. מחליפים רווחים ב-"_" ומסננים כל תו שאינו
    # אות/ספרה/"_"/"."/"-", כדי שמשתמש שמקליד "dana cohen" עם רווח לא ייתקל
    # ב-INVALID_EMAIL — ושה-login_username שנשמר יתאים בדיוק למה שהפרונט
    # יבנה שוב בהתחברות הבאה מאותה הקלדה.
    raw_username = (new_username or "").strip().lower()
    new_username = re.sub(r"\s+", "_", raw_username)
    new_username = re.sub(r"[^a-z0-9_.-]", "", new_username)
    if len(new_username) < 3:
        raise HTTPException(400, "שם המשתמש חייב להיות באורך 3 תווים לפחות (אותיות באנגלית/ספרות בלבד)")
    if len(new_password) < 6:
        raise HTTPException(400, "הסיסמא חייבת להיות באורך 6 תווים לפחות")

    pseudo_email = f"{new_username}@hai-subjects.local"
    try:
        firebase_auth.update_user(uid, email=pseudo_email, password=new_password)
    except firebase_auth.EmailAlreadyExistsError:
        raise HTTPException(409, "שם המשתמש הזה תפוס — נסה שם אחר")
    except Exception as e:
        raise HTTPException(500, f"עדכון החשבון נכשל: {e}")

    update_login_credentials(uid, new_username)
    return {"ok": True, "login_username": new_username}


@app.post("/subjects/me/profile")
def subjects_me_profile(name: str, age: int = None, gender: str = None, notes: str = "",
                         decoded=Depends(verify_subject_token)):
    """
    טופס הפרטים האישיים — פעם אחת בלבד, בסשן הראשון (אחרי שה-flag
    profile_completed עובר ל-True, הטופס הזה לא יוצג יותר לנבדק הזה). ה-name
    כאן הוא ששם התיקייה בסטוראג' ממשיך להסתמך עליו — בלי קשר ל-login_username.
    """
    uid = decoded["uid"]
    if not name or not name.strip():
        raise HTTPException(400, "נא להזין שם מלא")
    complete_subject_profile(uid, name=name.strip(), age=age, gender=gender, notes=notes)
    return {"ok": True}


@app.post("/subjects/me/sessions")
def subjects_me_new_session(protocol_version: str = "1.0", decoded=Depends(verify_subject_token)):
    """
    מחליף את POST /sessions הישן (שמקבל subject_id מהלקוח) לזרימת הנבדק
    האמיתית — ה-subject_id נלקח מהטוקן המאומת (uid), לא מפרמטר, כך שנבדק לא
    יכול ליצור סשן על שם נבדק אחר.
    """
    uid = decoded["uid"]
    session_id = create_session(subject_id=uid, protocol_version=protocol_version)
    return {"session_id": session_id, "subject_id": uid}

# ── SESSIONS ──
# שלב 2.3: אותו היגיון כמו POST /subjects למעלה — זרימת האינטייק הישנה, admin בלבד.
@app.post("/sessions", dependencies=[Depends(verify_admin_token)])
def new_session(subject_id: str, protocol_version: str = "1.0"):
    session_id = create_session(subject_id=subject_id, protocol_version=protocol_version)
    return {"session_id": session_id, "subject_id": subject_id}

# שלב 2.4/2.5: במקום verify_shared_secret, דורש טוקן אמיתי (נבדק או admin) ובודק
# בעלות על הסשן הספציפי הזה — ראו verify_subject_or_admin_token/_require_owner_or_admin
# למעלה. כך רק הנבדק שבאמת פתח את הסשן (uid שלו == session["subject_id"]) או admin
# יכולים להעלות וידאו אליו; מי שרק מחזיק את ה-session_id (למשל דלף) לא יכול.
@app.post("/analyze/{session_id}")
async def analyze(
    session_id: str,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    stimulus_map: str = Form(None),
    timeline: str = Form(None),
    decoded: dict = Depends(verify_subject_or_admin_token),
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    _require_owner_or_admin(decoded, session["subject_id"])
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

        # שלב 1: מונה הסשנים שנשמרו בהצלחה לנבדק הזה — זה מה שמאפשר למסך
        # הכניסה הבא לדעת אוטומטית שזו הכניסה ל"סשן השני/שלישי" (ראו
        # GET /auth/me) ולהציג את מסך האישור במקום טופס פרטים אישיים. עולה
        # רק כאן, כלומר רק כשהווידאו נשמר בפועל — לא בכל קריאה ל-/analyze.
        try:
            increment_session_count(subject_id)
        except Exception as ce:
            print(f"⚠️ Failed to increment session_count for subject {subject_id}: {ce}")
    except Exception as e:
        # אם ההעלאה ל-Firebase נכשלה, לפחות שומרים את הנתיב המקומי כדי לא לאבד
        # את ההקלטה (אם Render לא התאתחל בינתיים), ומתעדים את השגיאה.
        print(f"⚠️ Failed to upload session video to Firebase Storage: {e}")
        update_session_status(session_id, "recorded", video_path=str(video_path), storage_error=str(e))
    # import json
    # stimulus_map_parsed = json.loads(stimulus_map) if stimulus_map else {}
    # background_tasks.add_task(run_full_analysis, session_id, session["subject_id"], video_path, stimulus_map_parsed)

    return {"session_id": session_id, "status": "recorded"}

# שלב 2.4/2.5: אותו היגיון — נדרש טוקן אמיתי + בעלות על הסשן הספציפי הזה.
@app.get("/sessions/{session_id}")
def one_session(session_id: str, decoded: dict = Depends(verify_subject_or_admin_token)):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    _require_owner_or_admin(decoded, s["subject_id"])
    return s

@app.get("/sessions", dependencies=[Depends(verify_admin_token)])
def all_sessions(subject_id: str = None):
    return list_sessions(subject_id=subject_id)

# ── RESULTS (לא יחזירו כלום כל עוד הניתוח מושבת — זה תקין, לא באג) ──
# שלב 2.4/2.5: emotion_results לא כולל subject_id משלו, אז בודקים בעלות דרך
# הסשן עצמו (get_session) לפני שמחזירים את התוצאות.
@app.get("/results/{session_id}")
def session_results(session_id: str, decoded: dict = Depends(verify_subject_or_admin_token)):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    _require_owner_or_admin(decoded, session["subject_id"])
    results = get_emotion_results(session_id)
    if not results:
        raise HTTPException(404, "Results not found")
    return results

# שלב 2.4/2.5: conclusions כבר כולל subject_id משלו (ראו save_conclusion),
# אז אין צורך לטעון גם את הסשן בנפרד.
@app.get("/conclusions/{session_id}")
def session_conclusion(session_id: str, decoded: dict = Depends(verify_subject_or_admin_token)):
    conclusion = get_conclusion(session_id)
    if not conclusion:
        raise HTTPException(404, "Conclusion not found")
    _require_owner_or_admin(decoded, conclusion.get("subject_id"))
    return conclusion

@app.get("/conclusions", dependencies=[Depends(verify_admin_token)])
def all_conclusions(subject_id: str = None):
    return list_conclusions(subject_id=subject_id)

# ── RECORDINGS — קישורי הורדה זמניים (signed URLs), במקום הטוקן הציבורי הקבוע ──
# שלב 0 (המשך): מחליפים את firebaseStorageDownloadTokens (שתקף לנצח לכל מי
# שמחזיק את הלינק) בקישור חתום שפג תוקפו אחרי זמן קצר ונוצר רק דרך קריאה
# מאומתת לבקאנד. ה-URL עצמו עדיין לא דורש התחברות (זו מגבלה של signed URLs),
# אבל הוא נוצר רק למי שכבר עבר את verify_shared_secret וגם פג תוקף במהירות —
# שיפור משמעותי לעומת טוקן שחי לנצח.
# שלב 2.4/2.5: אותו היגיון — קישור חתום להקלטה של סשן ספציפי דורש בעלות עליו.
@app.get("/recordings/{session_id}/video")
def recording_video_url(session_id: str, decoded: dict = Depends(verify_subject_or_admin_token)):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    _require_owner_or_admin(decoded, session["subject_id"])
    url = generate_signed_video_url(session["subject_id"], session_id)
    if not url:
        raise HTTPException(404, "Video not found")
    return {"url": url}

@app.get("/recordings/{session_id}/timeline")
def recording_timeline_url(session_id: str, decoded: dict = Depends(verify_subject_or_admin_token)):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    _require_owner_or_admin(decoded, session["subject_id"])
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
