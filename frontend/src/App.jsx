import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [statusText, setStatusText] = useState('Bereit für deine erste Erzählung.')
  const [transcript, setTranscript] = useState('')
  const [personName, setPersonName] = useState('Vater')

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)

  const loadBiography = async (selectedPerson) => {
    try {
      const response = await fetch(
        `https://legacy-production-6e24.up.railway.app/person/${encodeURIComponent(selectedPerson)}/latest`
      )

      if (!response.ok) {
        throw new Error(`Fehler beim Laden: ${response.status}`)
      }

      const data = await response.json()

      if (data.generated) {
        setTranscript(data.generated)
        setStatusText(`Letzte Biografie geladen für: ${selectedPerson}`)
      } else {
        setTranscript('')
        setStatusText(`Noch keine gespeicherte Biografie für: ${selectedPerson}`)
      }
    } catch (error) {
      console.error(error)
      setTranscript('')
      setStatusText('Gespeicherte Biografie konnte nicht geladen werden.')
    }
  }

  useEffect(() => {
    loadBiography(personName)
  }, [])

  const handlePersonChange = async (e) => {
    const value = e.target.value
    setPersonName(value)
    await loadBiography(value)
  }

  const handleRecordingToggle = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        streamRef.current = stream

        const mediaRecorder = new MediaRecorder(stream)
        mediaRecorderRef.current = mediaRecorder
        audioChunksRef.current = []

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data)
          }
        }

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop())
            streamRef.current = null
          }

          setStatusText('Verarbeite Aufnahme…')

          try {
            const formData = new FormData()
            formData.append('file', audioBlob, 'recording.webm')
            formData.append('person_name', personName)

            const response = await fetch('https://legacy-production-6e24.up.railway.app/transcribe', {
              method: 'POST',
              body: formData,
            })

            if (!response.ok) {
              throw new Error(`Serverfehler: ${response.status}`)
            }

            const data = await response.json()

            setTranscript(data.generated)
            setStatusText(`Fertig! Session gespeichert für: ${personName}`)
          } catch (error) {
            console.error(error)
            setStatusText('Fehler bei der Verarbeitung.')
          }
        }

        mediaRecorder.start()
        setIsRecording(true)
        setStatusText('Aufnahme läuft… Sprich einfach frei.')
      } catch (error) {
        console.error(error)
        setStatusText('Mikrofon-Zugriff verweigert.')
      }
    } else {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      setIsRecording(false)
    }
  }

  return (
    <div className="app">
      <main className="hero">
        <div className="hero-card">
          <p className="eyebrow">Legacy System</p>
          <h1>Erzähl mir von deinem Leben</h1>

          <p className="subtitle">
            Sprich einfach frei. Du musst nichts vorbereiten.
            Aus deinen Worten entsteht eine erste, lesbare Lebensgeschichte.
          </p>

          <div style={{ marginTop: '24px', marginBottom: '10px' }}>
            <label htmlFor="personName" style={{ display: 'block', marginBottom: '8px', color: '#cfd8f6' }}>
              Testperson
            </label>
            <select
              id="personName"
              value={personName}
              onChange={handlePersonChange}
              disabled={isRecording}
              style={{
                width: '100%',
                maxWidth: '280px',
                padding: '12px 14px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.12)',
                background: 'rgba(255,255,255,0.06)',
                color: '#f5f7fb',
                fontSize: '1rem'
              }}
            >
              <option value="Vater">Vater</option>
              <option value="Freundin">Freundin</option>
              <option value="Ich">Ich</option>
            </select>
          </div>

          <div className="actions">
            <button
              className={`primary-button ${isRecording ? 'recording' : ''}`}
              onClick={handleRecordingToggle}
            >
              {isRecording ? 'Aufnahme stoppen' : 'Aufnahme starten'}
            </button>
          </div>

          <p className="status-text">{statusText}</p>

          <p className="hint">
            Für den MVP starten wir mit einem stabilen Aufnahme-Flow.
          </p>
        </div>

        <section className="result-card">
          <p className="result-label">Vorschau</p>
          <h2>Deine Lebensgeschichte erscheint hier</h2>

          <p>
            Sobald die Aufnahme verarbeitet wurde, siehst du hier eine erste
            strukturierte Fassung deiner Geschichte.
          </p>

          {transcript ? (
            <div className="result-preview">
              <p><strong>Deine Lebensgeschichte</strong></p>
              <p style={{ whiteSpace: 'pre-line' }}>{transcript}</p>
            </div>
          ) : (
            <div className="result-preview">
              <p><strong>Kindheit</strong></p>
              <p>
                Hier entsteht später ein erster Abschnitt, der wichtige frühe
                Erfahrungen und den Einstieg ins Leben beschreibt.
              </p>

              <p><strong>Beruf und Weg</strong></p>
              <p>
                Danach folgen weitere Abschnitte, die deinen Weg, Entscheidungen
                und Wendepunkte lesbar zusammenfassen.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App