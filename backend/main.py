import os
import sqlite3
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://legacy-six-delta.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# SESSION HELPERS
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


def get_sessions_for_person(person_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, transcript_raw, generated_text
        FROM sessions
        WHERE person_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (person_id,),
    )

    rows = cur.fetchall()
    conn.close()
    return rows


def build_combined_transcript(person_id: int) -> str:
    sessions = get_sessions_for_person(person_id)

    if not sessions:
        return ""

    parts = []
    for i, session in enumerate(sessions, start=1):
        parts.append(
            f"SESSION {i}\n"
            f"Zeitpunkt: {session['created_at']}\n"
            f"Transkript:\n{session['transcript_raw']}\n"
        )

    return "\n\n".join(parts)


def generate_biography_from_all_sessions(person_name: str, combined_transcript: str) -> str:
    structured = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Du bist ein Assistent für die strukturierte Aufbereitung von Lebenserinnerungen.

Deine Aufgabe:
Du erhältst mehrere Sessions einer Person. Jede Session enthält Ausschnitte aus Erzählungen über das Leben dieser Person.
Du sollst daraus eine ehrliche, vorsichtige, zusammengeführte Lebensskizze in deutscher Sprache erstellen.

WICHTIGE REGELN:
1. Verwende ausschließlich Informationen, die in den Sessions tatsächlich genannt wurden.
2. Erfinde keine Fakten, Motive, Hintergründe, Beziehungen, Daten oder Lebensphasen.
3. Überinterpretiere keine einzelnen Aussagen.
4. Nutze Außenperspektive, nicht Ich-Perspektive.
5. Keine Psychologisierung.
6. Keine Aussagen wie „schon immer“, „prägte sein Leben“, „war zentral“, wenn das nicht ausdrücklich belegt ist.
7. Wenn Informationen lückenhaft sind, bleibe ehrlich und zurückhaltend.
8. Lieber unvollständig als erfunden.
9. Wenn mehrere Sessions vorhanden sind, führe sie zusammen, ohne doppelte Aussagen unnötig zu wiederholen.
10. Wenn Zeitabfolgen erkennbar sind, ordne den Text möglichst chronologisch.
11. Wenn die Zeitabfolge unklar ist, formuliere neutral und erfinde keine Reihenfolge.
12. Kein Listenformat, sondern lesbarer Fließtext mit sinnvollen Abschnitten.
13. Verwende nur Überschriften, wenn sie wirklich zum Material passen.

AUSGABELOGIK:
- Bei sehr wenig Material:
  Erstelle eine kurze, ehrliche Erstfassung mit 2 bis 5 Sätzen.
  Formuliere nur das, was wirklich gesagt wurde.
  Weise am Ende knapp darauf hin, dass für eine ausführlichere Lebensgeschichte mehr Erzählung nötig ist.

- Bei ausreichend Material:
  Erstelle eine zusammenhängende Fassung mit sinnvollen Abschnitten.
  Wenn möglich, ordne vorsichtig in zeitlicher Reihenfolge.
  Geeignete Abschnittsarten können sein:
  Frühe Jahre, Kindheit, Jugend, Ausbildung, Beruf, Familie, Wendepunkte, Gegenwart.
  Nutze aber nur Abschnitte, die im Material wirklich erkennbar sind.

STIL:
- ruhig
- menschlich
- nüchtern
- respektvoll
- nicht kitschig
- nicht wie ein Roman
- nicht übertrieben glatt
- nicht wie Werbung
- nicht wie eine erfundene Biografie

ZIEL:
Die Person soll sich in dem Text wiedererkennen, ohne dass etwas hinzugedichtet wurde.
"""
            },
            {
                "role": "user",
                "content": f"Person: {person_name}\n\nHier sind alle bisher vorhandenen Sessions in zeitlicher Reihenfolge:\n\n{combined_transcript}"
            }
        ]
    )

    return structured.choices[0].message.content


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
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    with open(temp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )

    transcript_text = transcript.text

    os.remove(temp_path)

    person_id = get_or_create_person(person_name)

    # Erst eine vorsichtige Session-Zusammenfassung für Backup/Sessionhistorie erzeugen
    session_summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Du fasst eine einzelne Sprachaufnahme kurz und ehrlich zusammen.

Regeln:
- Nur Informationen verwenden, die tatsächlich genannt wurden
- Nichts erfinden
- Keine Psychologisierung
- Kurz und nüchtern bleiben
- Wenn der Input sehr dünn ist, ehrlich knapp bleiben
- Deutsch
"""
            },
            {
                "role": "user",
                "content": f"Person: {person_name}\n\nTranskript der neuen Session:\n\n{transcript_text}"
            }
        ]
    )

    session_generated_text = session_summary_response.choices[0].message.content

    save_session(person_id, transcript_text, session_generated_text)

    combined_transcript = build_combined_transcript(person_id)
    generated_text = generate_biography_from_all_sessions(person_name, combined_transcript)

    return {
        "transcript": transcript_text,
        "generated": generated_text
    }