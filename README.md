# Emma Studio

Studio für Avatar-Kommunikation: Audioanrufe, animierte Avatar-Videoanrufe
und fotorealistische Video-Messages. Neubau — ersetzt langfristig den alten Chatbot.

## Architektur

```
studio/
├── config.py          # eine Config für alles (LLM, Speech, Render, RAG)
├── core/
│   ├── store.py       # SQLite: personas / jobs / facts (FTS5 + bge-m3 hybrid)
│   └── pipeline.py    # Job-State-Machine: draft→scripted→voiced→rendering→done
├── agent/
│   ├── dialog.py      # Persona-Prompt + Faktenkarten → LLM-Turn
│   └── search.py      # SearXNG-Fallback bei unbeantwortbaren Fragen
├── speech/
│   ├── engine.py      # Realtime-Engine: VAD→STT→LLM→TTS Thread-Pool (Lazy-Build)
│   ├── vad_handler.py # Silero-VAD + smart-turn (semantisches Ende, Barge-in)
│   ├── stt.py         # faster-whisper (lazy, resident — geteilt mit Realtime-STT)
│   ├── tts.py         # OmniVoice (/v1/audio/speech, Profile-Voices)
│   └── router.py      # h3_native vs tts_pre Entscheidung
├── render/comfy.py    # ComfyUI-Client + Workflow-Injektion (fl2va BF16, RTX 2x)
└── deliver/
    ├── app.py         # FastAPI + Web-UI (:8890)
    ├── realtime/      # OpenAI-Realtime WebSocket (/v1/realtime) + Service/Router
    └── telegram.py    # Smart_Hermione_bot Transport (--telegram Flag)
```

## Ebenen

| Ebene | Optik | Status |
|---|---|---|
| Telefonat | Audio-only | **live** — OpenAI-Realtime WS (`/v1/realtime`): VAD→STT→LLM→TTS, Barge-in, Persona+RAG |
| Facetime | VRM-Avatar im Browser | **live** (three-vrm, Viseme-Lip-Sync via `emma.visemes`) |
| Video Message | fotoreal (MiniMax H3) | Modul fertig, Anbindung offen |

## Web-Oberfläche

- **Anrufe:** Kontakt auswählen, dann Audio oder Videoanruf starten. Mikrofon und
  Kamera werden erst auf Anfrage aktiviert; im Gespräch sind Stummschalten,
  Kamera ein/aus und Auflegen direkt erreichbar.
- **Transkript:** Gespräch mitlesen und während eines Anrufs Text senden.
- **Einstellungen:** Personas erstellen und Persönlichkeit, Kontaktportrait,
  Avatar (Emma oder Standard), Stimme und Sprache anpassen. Unter **Sprachmodell**
  wählt jede Persona ihren konfigurierten OpenAI-kompatiblen Endpunkt oder
  **OpenCode Go** und ein Modell. Die Modellliste wird beim Öffnen, beim
  Anbieterwechsel und alle 30 Sekunden in den sichtbaren Einstellungen vom
  Anbieter abgerufen; **Aktualisieren** lädt sie sofort neu. Während eines
  Anrufs bleiben Persona und Einstellungen gesperrt; die Navigation bleibt möglich.
- **Wissen:** Erinnerungen für den ausgewählten Kontakt verwalten und suchen.
- Technische Anzeigen liegen unter **Einstellungen → Studio-Diagnose**.
  `Alt+P` öffnet die Pipeline-Diagnose; Strg + Backtick öffnet die Debug-Konsole.
- Mikrofon und Kamera benötigen HTTPS oder einen lokalen `localhost`-Kontext.
  Portrait-Anfragen verwenden `/api/personas/portrait?persona_id=<id>`.

## RAG — Faktenkarten, keine Session-Dumps

`facts`-Tabelle mit FTS5 + bge-m3-Embeddings (Hybrid-Ranking). Einfügen per
Web-UI oder `/fact <text>` in Telegram. Persona-scoped Facts ranken vor globalen.

## Start

```bash
./studio.sh start            # ganzer Stack: Gemma-LLM (:8080) → TTS (:8880) → Studio (:8890)
./studio.sh status           # Units + Ports + geteilte Dienste
./studio.sh stop
./studio.sh start --no-llm   # ohne lokales LLM (Cloud-Modell, oder wenn der VRAM anderswo läuft)
```

Kein Autostart: `emma-studio.service` und `omnivoice.service` sind bewusst *disabled* — der
Stack belegt VRAM nur auf Zuruf. Das LLM steuert `~/models/gemma4_26b/launch-llama-server.sh`
(GPU0, Kontext-Autosizing nach freiem VRAM, MTP-Draft); die Unit `llama-llm.service` ist
veraltet und wird nicht benutzt. Geteilte Dienste (SearXNG :8081, bge-m3 :7777) gehören nicht
zum Stack und werden nie angefasst — `status` zeigt nur, ob sie erreichbar sind.

## Telegram: Textdialog mit vollständiger Sprachantwort

Telegram ist ein eigenständiger Übergangs-Transport für Textchat mit
Sprachantwort, kein Messenger-Videoanruf und keine Realtime-Audiobrücke.
Der normale Dialogpfad verwendet direkt Session-Store, Dialog-Agent und
vollständige TTS-Dateien; er startet keine Facetime-Pipeline und keine
satzweise geschnittenen `call_turn`-Jobs.

```bash
systemctl --user start omnivoice.service
python -m studio.deliver.telegram
# Alternativ gemeinsam mit der Web-App (nur eine Variante starten):
python -m studio --telegram
```

Die Dashboard-Karte **Emma Studio** steuert `emma-studio.service` direkt.
Die Unit startet `python -m studio --telegram` mit HTTPS auf Port 8890;
Dashboard-Start/Stop/Restart betrifft nur Studio, nicht OmniVoice oder das LLM.
Für direkte HTTPS-Starts sind `--ssl-certfile` und `--ssl-keyfile` gemeinsam
anzugeben; ohne beide Optionen bleibt der direkte CLI-Start HTTP.

Bei lokalen Sprachmodellen muss auch deren Server laufen. Cloud-Auswahl
benötigt kein lokales LLM. Telegram-only lädt weder VAD noch STT/VLM/TTS-
Streaming-Handler; Sprachnachrichten laden STT erst bei Bedarf.
`ffmpeg` mit `libopus` wird für echte OGG/Opus-Voice-Nachrichten benötigt.

### Konto und Zugriff

Standardmäßig werden Bot-Token und Name **in-place** aus
`~/projects/chatbot/telegram_secrets.json` (`bot_token`) und
`telegram_settings.json` (`bot_name`) gelesen. Der Vorgänger wird dabei
weder importiert noch gestartet; Zugangsdaten werden nicht kopiert.
Overrides: `STUDIO_TG_CHATBOT_DIR`, `STUDIO_TG_BOT_TOKEN`,
`STUDIO_TG_BOT_NAME` und `STUDIO_TG_ALLOWED_CHATS`.

Nur private Chats aus einer nichtleeren Allowlist sind erlaubt. Ohne explizites
`STUDIO_TG_ALLOWED_CHATS` werden bekannte positive Chat-IDs aus den
Vorgänger-Mappings `chat_sessions` und `chat_audio_modes` übernommen.
Eine explizit leere Allowlist bricht den Start ab, statt öffentlichen Zugriff
zu erlauben. Gruppen werden unabhängig von der Allowlist ignoriert.

**Nur ein Poller pro Bot-Token:** Vor dem Start den alten Chatbot-Poller
beenden; eigenständigen Telegram-Transport und `--telegram` nicht zugleich
mit demselben Konto verwenden.

### Befehle und Sessions

- `/start`: Hilfe und aktuelle Persona, Session, Anbieter und Modell.
- `/persona`: Personas auflisten; `/persona <Name oder ID>` wechselt die Persona.
  Die bisherige Session wird vorher zusammengefasst und geschlossen.
- `/provider`: Antippbare Anbieter-Auswahl mit tatsächlichem Endpunktnamen
  (z. B. **Local [127.0.0.1:8080]** oder **OpenCode Go**) statt interner IDs.
  **Use persona settings** übernimmt wieder die Einstellungen der Web-Persona.
  Nach der Anbieterwahl erscheint direkt die Modellauswahl.
- `/model`: Dynamisch abgerufene, antippbare Modellliste mit Seitennavigation
  und markierter aktueller Auswahl. Der Server-Standard wird mit seinem
  konkreten Modellnamen angezeigt. Es muss keine ID eingegeben werden;
  `/model <Modell-ID>` bleibt als optionale Experten-Eingabe möglich.
- `/new`: Aktuelle Session zusammenfassen, in den gemeinsamen Persona-
  Memory-Pool aufnehmen und eine neue Session mit derselben Auswahl beginnen.
  Bei einem Zusammenfassungs- oder Memory-Schreibfehler bleibt die bisherige
  Session aktiv, ohne eine neue Session anzulegen.
- `/fact <Text>`: Persona-Fakt speichern. Sprachnachrichten und
  `/video <Thema>` bleiben als optionale Kompatibilitätsfunktionen erhalten.
  `/video` nutzt weiterhin den separaten Video-Jobpfad mit der im Web
  gespeicherten Persona-Konfiguration, nicht die Telegram-Modellüberschreibung.

Laufende Telegram-Sessions überstehen Neustarts. Ohne ausdrückliche Auswahl
im Telegram-Chat werden Anbieter und Modell von der aktuellen Web-Persona
übernommen; Änderungen im Web wirken auch auf eine bestehende geerbte Session.
Eine explizite Telegram-Auswahl bleibt pro Bot/Chat gespeichert, verändert
keine Web-Persona-Konfiguration und bleibt über `/new` erhalten.
Statusmeldungen zeigen den tatsächlichen Anbieter/Endpunkt, das aufgelöste
Modell und die Auswahlquelle (**persona settings** oder **session choice**).
Alte leere lokale Standard-Auswahlen werden beim Datenbank-Upgrade als geerbt
behandelt; vorhandene nichtleere Modellauswahlen bleiben explizit erhalten.
`/provider` bleibt auch bei fehlenden Cloud-Zugangsdaten verfügbar, damit ein
anderer Endpunkt gewählt werden kann. Ändert sich die Persona während eines
Modellabrufs, wird die veraltete Menüauswahl ohne Session-Änderung abgewiesen;
`/provider` oder `/model` öffnet eine aktuelle Liste.
Neuer Telegram-Kontext enthält nur die Nachrichten
seiner eigenen Session plus den bestehenden Systemprompt mit früheren
Persona-Zusammenfassungen und passenden Fakten. Facetime-Historie und
Telegram-Live-Historie bleiben getrennt; Telegram blockiert oder beendet
keine Web-Anrufe. Der Persona-Memory-Pool ist absichtlich transportübergreifend.

Jede Antwort wird als vollständiger Text in der Session gespeichert, aber im
Telegram-Chat ausschließlich als Voice-Nachricht versendet — ohne zusätzliche
Textantwort. OmniVoice synthetisiert die komplette Antwort mit Stimme und
Sprache der Persona in eine WAV-Datei; der Transport konvertiert sie zu
OGG/Opus. Temporäre Dateien werden auch bei Fehlern entfernt. Scheitert TTS
oder Versand, bleiben Text und Session erhalten; der Chat erhält nur eine
kurze Fehlermeldung. Befehle, Auswahlmenüs und Statusmeldungen bleiben Text.
Modellfehler nennen die betroffene Anbieter-/Modellauswahl und unterscheiden
Verbindungsfehler, Zeitüberschreitungen und HTTP-Fehler des Anbieters.
Schlüssel, Telegram-Token und rohe Anbieter-Antworten werden nicht ausgegeben.

## Secrets (.env)

```
STUDIO_LLM_API_KEY=...        # OpenRouter
STUDIO_TG_BOT_TOKEN=...       # aus ~/projects/chatbot/telegram_secrets.json
STUDIO_TG_ALLOWED_CHATS=<chat-id>
```

### Cloud-Modelle: OpenCode Go

In den Persona-Einstellungen **OpenCode Go** auswählen, ein Modell aus der
Liste wählen und speichern. Die Auswahl gilt für Echtzeitgespräche,
Textantworten, Video-Skripte und Gesprächszusammenfassungen. Bestehende Personas
übernehmen weiterhin `STUDIO_LLM_BASE_URL`, `STUDIO_LLM_MODEL` und
`STUDIO_LLM_API_KEY`; ein leerer Modellwert beim konfigurierten Anbieter
übernimmt dessen Server-Standard.
Ein persona-spezifischer Endpunkt erhält den konfigurierten API-Schlüssel nur,
wenn Protokoll, Host und effektiver Port mit dem Server-Endpunkt übereinstimmen.
Andere Endpunkte werden ohne diesen Schlüssel angesprochen.
Ohne `STUDIO_LLM_API_KEY` wird für diesen vertrauenswürdigen Ursprung auch
`OPENAI_API_KEY` übernommen, wie bisher beim Echtzeit-Client.

Der Server liest OpenCode-Go-Zugangsdaten zuerst aus
`STUDIO_OPENCODE_GO_API_KEY`, sonst aus dem `opencode-go`-Credential-Pool in
`~/.hermes/auth.json`, danach aus `OPENCODE_GO_API_KEY` in Hermes' `.env`.
`HERMES_HOME` kann das Hermes-Verzeichnis ändern. Der Schlüssel bleibt auf dem
Server und wird weder in Persona-Daten noch an den Browser übertragen.
Optional setzt `STUDIO_OPENCODE_GO_BASE_URL` den serverseitigen Endpunkt;
standardmäßig ist es `https://opencode.ai/zen/go/v1`, beziehungsweise der im
Hermes-Pool hinterlegte Endpunkt. Go-Zugangsdaten werden niemals an eine
persona-spezifische URL gesendet.

Die Modellauswahl kommt authentifiziert von `/models` am API-Basisendpunkt
(für Go: `/zen/go/v1/models`); es gibt keine fest hinterlegte Modellliste.
Eine gespeicherte Auswahl bleibt bei Discovery-Fehlern oder verschwundenen
Modellen erhalten und wird entsprechend gekennzeichnet. Cloud-Fehler werden
nicht durch einen stillen Wechsel zum lokalen Modell kaschiert.

Cloud-Nutzung überträgt Gesprächsinhalte und den für die Anfrage verwendeten
Persona-/Erinnerungskontext an den Anbieter. [OpenCode Go](https://opencode.ai/docs/go/)
ist laut Anbieter für Coding-Agenten gedacht; Avatar-Chat kann daher
anbieterbedingten Einschränkungen unterliegen. Emma Studio identifiziert sich
mit eigenem User-Agent und sendet eine stabile Gesprächskennung.
`./studio.sh start --no-llm` lässt das lokale Sprachmodell aus; STT, TTS und
Kamera-Bildanalyse bleiben lokal und benötigen weiterhin ihre bisherigen Dienste.

## Wichtige Hinweise

- **Port-Konflikte:** `studio.sh start` prüft vorab, wem :8080/:8880/:8890 gehören.
  Hält ein Fremdprozess einen davon (typisch ein vergessener Dev-Server
  `python -m studio --port 8890`), bricht der Start sofort mit PID und Kommandozeile
  des Halters ab — statt in den 90-s-Health-Timeout zu laufen. `status` zeigt denselben
  Konflikt an.
- **Bot-Token-Konflikt:** Smart_Hermione_bot wird vom alten Chatbot gepollt.
  Beide gleichzeitig = Update-Kampf. Erst Chatbot stoppen, dann `--telegram`.
- **GPU 1 ist geteilt (STT + VLM):** Der VLM-Kaltstart (moondream, ~15,5 GB) belegt
  Gerät 1 in der Spitze mit ~23 GB (bge-m3 + OmniVoice + Whisper + VLM auf 24 GB).
  Er läuft deshalb unter einem gemeinsamen Schloss mit jeder Whisper-Inferenz
  (`studio/speech/core/device_lock.py`) und wird bei einem Videoanruf **beim
  Anrufstart** vorgeladen (`POST /api/calls/start {"video": true}`), nicht erst beim
  ersten Kamerabild. Ohne das reißt der parallele Kaltstart den ganzen Prozess per
  SIGABRT ab (CUDA-OOM in CTranslate2s Graph-Capture — Anruf tot). Passt das Modell
  nicht, läuft der Videoanruf ohne Bildkontext weiter, statt zu sterben.
  Nachprüfbar mit `scripts/repro_vlm_stt_race.py` (braucht die echten Modelle).
- **Sprachfelder sind zweierlei:** das Persona-Feld `language` (und der Silent-Turn-Pfad)
  enthält **Klarnamen** wie `"German"`, der Session-/STT-Pfad **ISO-Codes** wie `"de"`.
  Wer daraus einen Whisper-Code braucht, muss `speech.utils.whisper_language_code`
  benutzen — Whisper kennt nur Codes und wirft sonst `ValueError`, womit die
  Visem-Timeline eines Satzes komplett ausfällt (Lippen stehen still).
- **Hängende Call-Sessions:** Eine „aktive" Session ohne lebende Realtime-Verbindung
  (Prozess abgestürzt/neu gestartet, Tab weg ohne Auflegen) wird beim nächsten
  Anrufstart nach `STUDIO_STALE_CALL_GRACE` (90 s) automatisch abgeschlossen —
  sonst sperrt der Überrest die Persona dauerhaft aus jedem weiteren Anruf.
- OmniVoice: `~/services/omnivoice/toggle-omnivoice` (Port 8880, 3060 Ti)
- ComfyUI muss laufen für Video-Messages (`systemctl --user start comfyui`)
- Workflows in `workflows/` stammen aus video-gen-bot (fl2va BF16, beta57,
  Musubi-LoRA 0.4, RTX-VSR 2× ULTRA)
- **Logs enthalten standardmäßig keine Gesprächsinhalte** (nur Zeichenzahlen).
  Zum Debuggen: `STUDIO_LOG_TRANSCRIPTS=1` setzen bzw. `python -m studio --log_transcripts`
  — dann stehen die Transkripte im Journal und bleiben dort.

## Avatar memory

Full Telegram and Facetime transcripts and summaries are searchable in the existing
SQLite database. Each session has a six-character memory ID shown before its
summary in the persona prompt. The avatar can search across its history or within
one session, then read exact paginated conversation text. Memory calls and results
stay private; up to five calls are allowed before the final user-facing answer.

Retrieval uses Unicode-aware FTS5 keyword search without another service or GPU
model. Memory-enabled voice turns buffer model output before speech so tool JSON
and stray search preambles cannot be spoken. See
[the architecture and tool protocol](docs/memory-architecture.md) for migration,
limits, extension points and tests.

## Tests

```bash
python -m pytest tests/ -q
```
