# HAI Backend — Heart Affinity Index
# FastAPI + DeepFace + SpeechBrain

## מבנה הפרויקט

```
hai_backend/
├── main.py                        ← שרת FastAPI — נקודת כניסה
├── requirements.txt               ← כל החבילות
├── services/
│   ├── video_analyzer.py          ← DeepFace — ניתוח פנים
│   ├── audio_analyzer.py          ← SpeechBrain + FFmpeg — ניתוח שמע
│   └── affinity_calculator.py     ← נוסחת HAI — Z-scores + ציוני תחומים
├── models/
│   └── schemas.py                 ← מודלי Pydantic לAPI
├── uploads/                       ← וידאו שהועלו (נוצר אוטומטית)
└── results/                       ← CSV של תוצאות (נוצר אוטומטית)
```

## התקנה

### 1. דרישות מקדימות
```bash
# Python 3.11 מומלץ
python --version

# FFmpeg (חובה לניתוח שמע)
# Windows: https://ffmpeg.org/download.html
# Mac:     brew install ffmpeg
# Linux:   sudo apt install ffmpeg
```

### 2. סביבה וירטואלית
```bash
cd hai_backend
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. התקנת חבילות
```bash
pip install -r requirements.txt
```
> ⚠️ ההתקנה הראשונה לוקחת ~5-10 דקות (DeepFace + SpeechBrain כוללים מודלים כבדים)

### 4. הפעלת השרת
```bash
python main.py
# או:
uvicorn main:app --reload --port 8000
```

### 5. בדיקה
פתח דפדפן: http://localhost:8000
תיעוד API אינטראקטיבי: http://localhost:8000/docs

---

## שימוש בAPI

### העלאת וידאו לניתוח
```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@/path/to/video.webm"
```
מחזיר: `{"session_id": "abc-123", "status": "processing"}`

### בדיקת התקדמות
```bash
curl http://localhost:8000/status/abc-123
```

### קבלת תוצאות
```bash
curl http://localhost:8000/results/abc-123
```

---

## מה קורה מתחת למכסה

```
וידאו נכנס
    ↓
DeepFace (כל פריים 5):    angry/disgust/fear/happy/sad/surprise/neutral/contempt
                          → Valence + Arousal + Intensity
    ↓
FFmpeg:                   חילוץ שמע WAV 16kHz
    ↓
SpeechBrain:              סגמנטים של 5 שניות → סיווג רגשי
                          → Valence + Arousal + Intensity
    ↓
Affinity Calculator:      Affinity = Valence × Arousal
                          Z = (Affinity − mean) / std
                          ↓
                          זיהוי פיקים (Z > 1.5)
                          ציון לפי תחום חיים (אם יש stimulus_map)
    ↓
תוצאה:                    video_analysis.csv
                          audio_emotions.csv
                          affinity results JSON
```

---

## הערות פיתוח

- **Mock mode**: אם DeepFace/SpeechBrain לא מותקנים, הקוד חוזר לנתוני mock ריאליסטיים
- **Thread pool**: הניתוח רץ ב-background threads כדי לא לחסום את השרת
- **Step 2**: Firebase יחליף את מילון ה-`sessions` בזיכרון
- **Step 4**: Claude API יקבל את תוצאות ה-Z-score וייצר את המסקנה הנרטיבית
