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


def get_person_id_by_name(name: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM persons WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()

    if row:
        return row["id"]
    return None


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


# =========================
# OPENAI HELPERS
# =========================

def generate_single_session_summary(person_name: str, transcript_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Du fasst eine einzelne Sprachaufnahme kurz und ehrlich zusammen.

REGELN:
1. Verwende nur Informationen, die tatsächlich genannt wurden.
2. Erfinde nichts.
3. Keine Psychologisierung.
4. Keine typischen Lebensstationen ergänzen.
5. Kein Romanstil.
6. Kurz, ruhig, nüchtern, klar.
7. Wenn sehr wenig Inhalt vorhanden ist, bleibe ehrlich knapp.
8. Keine Höflichkeitsform.
9. Nicht das Geschlecht raten.
10. Wenn das Geschlecht unklar ist, nutze lieber den Namen oder "die Person".

ZIEL:
Eine kurze, sachliche Zusammenfassung der einzelnen Session.
"""
            },
            {
                "role": "user",
                "content": f"Person: {person_name}\n\nTranskript der neuen Session:\n\n{transcript_text}"
            }
        ]
    )

    return response.choices[0].message.content.strip()


def generate_biography_and_questions(person_name: str, combined_transcript: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Du bist ein System zur strukturierten Rekonstruktion von Lebenserinnerungen.

Du erhältst mehrere Sessions einer Person.
Deine Aufgabe ist zweigeteilt:

1. Erstelle eine zusammengeführte, ehrliche Lebensskizze.
2. Erstelle 3 bis 5 gezielte Folgefragen, die helfen, Lücken sinnvoll zu füllen.

WICHTIGE REGELN FÜR DIE BIOGRAFIE:
1. Verwende ausschließlich Informationen, die in den Sessions tatsächlich genannt wurden.
2. Erfinde keine Fakten, Motive, Beziehungen, Daten, Berufe, Abschlüsse oder Lebensphasen.
3. Überinterpretiere keine einzelnen Aussagen.
4. Du darfst Informationen vorsichtig zeitlich ordnen, wenn die Reihenfolge aus dem Gesagten klar oder naheliegend ist.
5. Du darfst NICHT inhaltlich ausschmücken.
6. Keine Psychologisierung.
7. Keine Floskeln wie "prägte sein Leben", "war schon immer", "zentraler Lebensfaktor", wenn das nicht ausdrücklich gesagt wurde.
8. Keine Höflichkeitsform im Biografie-Text.
9. Geschlecht nur verwenden, wenn es im Material klar erkennbar ist.
10. Wenn das Geschlecht nicht klar ist, nutze den Namen oder neutrale Formulierungen wie "die Person".
11. Kein Listenformat in der Biografie, sondern lesbarer Fließtext mit sinnvollen Abschnitten.
12. Überschriften nur verwenden, wenn sie wirklich zum Material passen.
13. Wenn wenig Material vorhanden ist, schreibe keine künstlich vollständige Lebensgeschichte.

WICHTIGE REGELN FÜR DIE FRAGEN:
1. Stelle 3 bis 5 konkrete, hilfreiche Folgefragen.
2. Die Fragen sollen auf echten Lücken basieren.
3. Keine generischen Fragen wie "Erzähl mehr".
4. Keine Fragen zu Dingen, die bereits klar gesagt wurden.
5. Die Fragen sollen Erinnerungen auslösen und zu weiterem Erzählen einladen.
6. Die Fragen sollen nicht in Höflichkeitsform formuliert sein.
7. Keine Ja/Nein-Fragen, wenn es vermeidbar ist.
8. Fragen offen, konkret und ruhig formulieren.

AUSGABELOGIK:
- Bei sehr wenig Material:
  - Biografie kurz und ehrlich halten.
  - Nichts aufblasen.
  - Folgefragen trotzdem gezielt stellen, damit die nächste Session leichter fällt.

- Bei ausreichend Material:
  - Biografie zusammenhängend und möglichst chronologisch ordnen.
  - Doppelte Aussagen zusammenführen.
  - Offene Lücken bewusst offen lassen.
  - Folgefragen auf die wichtigsten fehlenden Bereiche richten.

STIL DER BIOGRAFIE:
- ruhig
- menschlich
- nüchtern
- respektvoll
- nicht kitschig
- nicht werblich
- nicht überdramatisch
- nicht wie eine erfundene Romanbiografie

WICHTIGES FORMAT:
Gib ausschließlich gültiges JSON zurück, ohne Markdown und ohne zusätzliche Einleitung.

Das JSON muss genau diese Struktur haben:
{
  "generated": "string",
  "follow_up_questions": ["string", "string", "string"]
}
"""
            },
            {
                "role": "user",
                "content": f"Person: {person_name}\n\nHier sind alle bisher vorhandenen Sessions in zeitlicher Reihenfolge:\n\n{combined_transcript}"
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
        generated = str(parsed.get("generated", "")).strip()
        follow_up_questions = parsed.get("follow_up_questions", [])

        if not isinstance(follow_up_questions, list):
            follow_up_questions = []

        follow_up_questions = [
            str(q).strip()
            for q in follow_up_questions
            if str(q).strip()
        ]

        return generated, follow_up_questions[:5]

    except Exception:
        fallback_generated = content
        fallback_questions = [
            "Welche Erinnerungen aus der Kindheit oder Jugend fehlen bisher noch?",
            "Welche Station nach Schule oder Ausbildung war für den weiteren Weg besonders wichtig?",
            "Welche Menschen, Orte oder Ereignisse sollten in der Geschichte auf keinen Fall fehlen?",
        ]
        return fallback_generated, fallback_questions


# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/person/{name}/latest")
def get_latest_biography(name: str):
    person_id = get_person_id_by_name(name)

    if not person_id:
        return {
            "generated": "",
            "follow_up_questions": []
        }

    combined_transcript = build_combined_transcript(person_id)

    if not combined_transcript.strip():
        return {
            "generated": "",
            "follow_up_questions": []
        }

    generated_text, follow_up_questions = generate_biography_and_questions(
        name,
        combined_transcript
    )

    return {
        "generated": generated_text,
        "follow_up_questions": follow_up_questions
    }


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

    transcript_text = transcript.text.strip()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    person_id = get_or_create_person(person_name)

    session_summary = generate_single_session_summary(person_name, transcript_text)
    save_session(person_id, transcript_text, session_summary)

    combined_transcript = build_combined_transcript(person_id)
    generated_text, follow_up_questions = generate_biography_and_questions(
        person_name,
        combined_transcript
    )

    return {
        "transcript": transcript_text,
        "generated": generated_text,
        "follow_up_questions": follow_up_questions
    }