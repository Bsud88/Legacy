import os
import sqlite3
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# 🔥 .env laden
load_dotenv()

# 🔥 OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔥 App init
app = FastAPI()

# 🔥 CORS (Frontend Zugriff)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 DB Pfad
DB_PATH = os.path.join("storage", "legacy.db")


# =========================
# DATABASE HELPERS
# =========================

def get_connection():
    os.makedirs("storage", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            created_at TEXT,
            transcript_raw TEXT,
            generated_text TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================
# PERSON
# =========================

def get_or_create_person(name: str) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM persons WHERE name = ?", (name,))
    row = cur.fetchone()

    if row:
        conn.close()
        return row["id"]

    cur.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    conn.commit()
    person_id = cur.lastrowid
    conn.close()

    return person_id


# =========================
# SAVE SESSION + BACKUP
# =========================

def save_session(person_id: int, transcript_raw: str, generated_text: str) -> int:
    conn = get_connection()
    cur = conn.cursor()

    now = datetime.utcnow().isoformat()

    cur.execute(
        """
        INSERT INTO sessions (person_id, created_at, transcript_raw, generated_text)
        VALUES (?, ?, ?, ?)
        """,
        (person_id, now, transcript_raw, generated_text),
    )

    conn.commit()
    session_id = cur.lastrowid
    conn.close()

    # 🔥 JSON Backup
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)

    backup_data = {
        "person_id": person_id,
        "session_id": session_id,
        "created_at": now,
        "transcript_raw": transcript_raw,
        "generated_text": generated_text,
    }

    backup_path = os.path.join(backup_dir, f"session_{session_id}.json")

    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, ensure_ascii=False, indent=2)

    return session_id


# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    person_name: str = Form(...)
):
    # 🔥 temp file speichern
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 🔥 Speech to Text
    with open(temp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )

    transcript_text = transcript.text

    # 🔥 Datei löschen
    os.remove(temp_path)

    # 🔥 Strukturierung
    structured = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Erstelle aus dem gesprochenen Text eine strukturierte Lebensgeschichte mit sinnvollen Abschnitten wie Kindheit, Jugend, Erwachsenenleben, aktuelle Situation."
            },
            {
                "role": "user",
                "content": transcript_text
            }
        ]
    )

    generated_text = structured.choices[0].message.content

    # 🔥 Person + Session speichern
    person_id = get_or_create_person(person_name)
    save_session(person_id, transcript_text, generated_text)

    return {
        "transcript": transcript_text,
        "generated": generated_text
    }