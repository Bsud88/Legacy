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

# =========================
# PERSISTENT DATA PATHS
# =========================

BASE_DATA_DIR = "/app/data"
STORAGE_DIR = os.path.join(BASE_DATA_DIR, "storage")
BACKUP_DIR = os.path.join(BASE_DATA_DIR, "backups")
DB_PATH = os.path.join(STORAGE_DIR, "legacy.db")


# =========================
# DATABASE HELPERS
# =========================

def ensure_data_dirs():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def get_connection():
    ensure_data_dirs()
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
    name = name.strip()

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
    name = name.strip()

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

    backup_data = {
        "person_id": person_id,
        "session_id": session_id,
        "created_at": now,
        "transcript_raw": transcript_raw,
        "generated_text": generated_text,
    }

    backup_path = os.path.join(BACKUP_DIR, f"session_{session_id}.json")

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
Du fasst eine einzelne Sprachaufnahme kurz, ehrlich und zurückhaltend zusammen.

REGELN:
1. Verwende ausschließlich Informationen, die tatsächlich genannt wurden.
2. Erfinde nichts.
3. Ergänze keine typischen Lebensstationen oder Hintergründe.
4. Keine Psychologisierung.
5. Keine Ausschmückung und kein Romanstil.
6. Kurz, ruhig, nüchtern und klar formulieren.
7. Wenn wenig Inhalt vorhanden ist, ehrlich knapp bleiben.
8. Keine Höflichkeitsform verwenden.
9. Das Geschlecht nicht raten.
10. Wenn das Geschlecht unklar ist, lieber den Namen oder "die Person" verwenden.
11. Keine direkte Ansprache an die Person.
12. Kein Listenformat.

ZIEL:
Eine kurze, sachliche Zusammenfassung der einzelnen Session, die nur festhält, was wirklich gesagt wurde.
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

1. Erstelle eine zusammengeführte, ehrliche und strukturierte Lebensskizze.
2. Erstelle 3 bis 5 gezielte Folgefragen, die helfen, noch fehlende Bereiche sinnvoll zu ergänzen.

WICHTIGE GRUNDREGEL:
Verwende ausschließlich Informationen, die in den Sessions tatsächlich genannt wurden.
Wenn etwas nicht gesagt wurde, darfst du es nicht ergänzen.

REGELN FÜR DIE BIOGRAFIE:
1. Erfinde keine Fakten, Motive, Beziehungen, Daten, Berufe, Abschlüsse oder Lebensphasen.
2. Überinterpretiere keine einzelnen Aussagen.
3. Keine Psychologisierung.
4. Keine Ausschmückung und keine Dramatisierung.
5. Keine Floskeln wie "prägte sein Leben", "war schon immer", "zentraler Lebensfaktor", wenn das nicht ausdrücklich gesagt wurde.
6. Keine Höflichkeitsform im Biografie-Text.
7. Geschlecht nur verwenden, wenn es im Material klar erkennbar ist.
8. Wenn das Geschlecht nicht klar ist, nutze den Namen oder neutrale Formulierungen wie "die Person".
9. Schreibe nicht in Ich-Form.
10. Schreibe keinen Roman und keine werbliche Sprache.
11. Bleibe ruhig, menschlich, nüchtern und respektvoll.

REGELN FÜR DIE STRUKTUR:
1. Ordne die Informationen möglichst chronologisch, wenn die Reihenfolge aus dem Gesagten klar oder naheliegend ist.
2. Du darfst weiche zeitliche Ordnung herstellen, zum Beispiel frühe Jahre, Schulzeit, Ausbildung, Beruf, spätere Jahre oder Gegenwart.
3. Du darfst Abschnitte oder Zwischenüberschriften verwenden, aber nur wenn sie aus dem vorhandenen Material sinnvoll hervorgehen.
4. Erfinde keine Abschnitte, wenn dafür keine Grundlage vorhanden ist.
5. Wenn bestimmte Lebensphasen nicht vorkommen, lass sie einfach weg.
6. Führe doppelte Aussagen aus mehreren Sessions zusammen.
7. Lücken sollen sichtbar bleiben und nicht künstlich geschlossen werden.

REGELN FÜR WENIG ODER LÜCKENHAFTES MATERIAL:
1. Wenn wenig Material vorhanden ist, schreibe keine künstlich vollständige Lebensgeschichte.
2. Dann soll die Biografie kurz, ehrlich und zurückhaltend bleiben.
3. Lieber unvollständig als erfunden.

REGELN FÜR DIE FOLGEFRAGEN:
1. Stelle 3 bis 5 konkrete, hilfreiche Folgefragen.
2. Die Fragen sollen auf echten Lücken basieren.
3. Keine generischen Fragen wie "Erzähl mehr".
4. Keine Fragen zu Dingen, die bereits klar gesagt wurden.
5. Die Fragen sollen Erinnerungen auslösen und zum Weitererzählen einladen.
6. Keine Höflichkeitsform.
7. Wenn möglich offene Fragen statt Ja/Nein-Fragen.
8. Die Fragen sollen ruhig, natürlich und konkret klingen.

ZIEL DER BIOGRAFIE:
Die Person soll sich im Text wiedererkennen.
Der Text darf ordnen, aber nicht erfinden.
Der Text darf verbinden, aber nicht verfälschen.

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
        name.strip(),
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
    person_name = person_name.strip()

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