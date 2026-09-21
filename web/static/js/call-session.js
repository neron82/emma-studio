/* Emma Studio call session engine — OpenAI-Realtime transport.

Owns: the /v1/realtime WebSocket (subprotocol 'realtime', same origin),
continuous mic capture via AudioWorklet (16 kHz PCM16 — the server-side VAD
segments turns and auto-responds, so the mic is NEVER paused while Emma
speaks; barge-in must work), ring-buffer playback of response.output_audio
deltas with barge-in flushing, backchannel filler scheduling, emma.visemes
lip-sync timelines for the Facetime engine, and captions.

Used by both Telefonat and Facetime views — identical pipeline, different
visuals.
*/
const CallSession = (() => {
  // ---- state ----
  let micOn = false;            // explicit toggle state
  let handsfree = false;        // auto-conversation mode flag (UI status text)
  let speaking = false;         // response audio playing
  let ws = null, wsReady = false, intentionalClose = false;
  let mediaStream = null, micSrc = null, micNode = null;
  let audioCtx = null, workletsLoaded = null;
  let playbackNode = null, pendingAudio = [];
  let onEvent = () => {};       // UI callbacks
  let personaId = null;
  let evtSeq = 0;
  const outbox = [];            // control events queued until session.updated

  const FILLER_DELAY_MS = 450;  // unchanged backchannel timing
  const FILLERS_ENABLED = false; // Zwischengeräusche bis auf Weiteres aus (siehe unten)
  const MIC_CHUNK_MS = 100;     // 100 ms = 1600 samples = 3200 bytes per append
  const FRAME_MS = 1000;        // Kamera: ein Frame pro Sekunde an den VLM
  const FRAME_MAX_EDGE = 512;   // Kantenlänge fürs Modell (mehr bringt nichts)
  let frameTimer = null, framesSent = 0;
  let pipEl = null, pipVideo = null, pipLabel = null;
  const videoEl = document.createElement('video');
  videoEl.playsInline = true; videoEl.muted = true; videoEl.autoplay = true;
  const frameCanvas = document.createElement('canvas');
  const WIRE_RATE = 16000;      // both audio directions on the wire

  // ---- response bookkeeping ----
  let currentResponseId = null;
  let responseText = '';
  let transcriptDeltaSeen = false;
  let responseAudioDone = false;
  let audioFed = false;         // ≥1 audio delta fed for the current response
  let flushed = false;          // drop incoming deltas until next response.created

  // ---- visemes (accumulated per response) ----
  let visemeTL = [];            // [{v,t0,t1}] with offsets folded in
  let visemeOffset = 0;         // sum of duration_s of earlier seqs
  let visemeRespId = null;

  // ---- playback clock (drives the facetime lip-sync timeline) ----
  let playedFrames = 0, clockBaseline = 0, clockActive = false;

  // ---- Pipeline-Statusanzeige (index.html setzt den Sink) ----
  let logSink = null;

  // ---- Debugging ----
  // Compact state snapshot appended to every [call] debug line, so each
  // console entry shows which gate is currently closed.
  function snap() {
    const w = ws ? (wsReady ? 'ready' : 'conn') : 'off';
    return `mic=${+micOn} hf=${+handsfree} spk=${+speaking} ws=${w} fl=${+flushed}`;
  }
  function dbg(msg) {
    window.DebugConsole && DebugConsole.log(`[call] ${msg} · ${snap()}`);
    if (logSink) { try { logSink(msg); } catch (e) { /* Anzeige darf nie stören */ } }
  }

  /** Ziel für die Pipeline-Statusanzeige (index.html hängt sich hier ein). */
  function setLogSink(fn) { logSink = fn; }

  function emit(ev) {
    try {
      onEvent(ev);
      if (window.DebugConsole && [
          'user','assistant','listening','playback-end','interjected',
          'mic','handsfree','speech-detected','transcribing','playback-start','flushed',
        ].includes(ev.type)) {
        const m = { user: () => '👤 „' + (ev.text||'').slice(0,50) + '“',
                    assistant: () => '🗣 „' + (ev.text||'').slice(0,60) + '…',
                    listening: () => '🎧 hört zu…',
                    'playback-end': () => '🔊 Playback fertig',
                    interjected: () => '⏸ unterbrochen — Buffer geflusht',
                    mic: () => ev.on ? '🎤 Mikro an' : '🎤 Mikro aus',
                    handsfree: () => ev.on ? '♾ Handsfree an' : '♾ Handsfree aus',
                    'speech-detected': () => '💬 Sprache erkannt (Server-VAD)',
                    transcribing: () => '⏳ transkribiere…',
                    'playback-start': () => '▶️ Playback startet',
                    flushed: () => '⏹ Playback verworfen' };
        DebugConsole.log(m[ev.type]());
      }
    } catch {}
  }

  function setPersona(id) {
    personaId = id;
    window._studioPersonaId = id;   // backchannel filler picker reads this
    if (wsReady) sendSessionUpdate('Persona-Wechsel');   // live re-apply
  }

  // ================= WEBSOCKET =================

  function nextEventId() { return `evt_${Date.now()}_${++evtSeq}`; }

  /** Send one client event. Every event carries a unique event_id — the
      server validates against strict OpenAI realtime models and rejects
      payloads without one. Control events queue in the outbox until the
      session is configured; audio appends are gated by wsReady instead. */
  function send(obj) {
    obj.event_id = nextEventId();
    const open = ws && ws.readyState === WebSocket.OPEN;
    if (obj.type === 'session.update') {
      if (open) ws.send(JSON.stringify(obj));
      dbg(`→ ${obj.type}`);
      return;
    }
    if (wsReady && open) {
      ws.send(JSON.stringify(obj));
      if (obj.type !== 'input_audio_buffer.append' && obj.type !== 'emma.video_frame') dbg(`→ ${obj.type}`);
    } else if (obj.type !== 'input_audio_buffer.append' && obj.type !== 'emma.video_frame') {
      outbox.push(obj);
      dbg(`${obj.type} vorgemerkt (Session noch nicht bereit)`);
    }
  }

  function sendSessionUpdate(origin) {
    // session.type is the pydantic union discriminator — mandatory.
    // persona_id is the custom extra: the server derives instructions,
    // history and voice from it — never send those ourselves.
    send({
      type: 'session.update',
      session: {
        type: 'realtime',
        persona_id: personaId,
        input_audio_format: 'pcm16',
        output_audio_format: 'pcm16',
        turn_detection: { type: 'server_vad', interrupt_response: true, create_response: true },
      },
    });
    dbg(`session.update gesendet (${origin || '握手'}): persona=${personaId}`);
  }

  async function connect() {
    if (ws) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/v1/realtime`;
    dbg(`→ WebSocket ${url} (Subprotokoll: realtime)`);
    ws = new WebSocket(url, ['realtime']);
    ws.onopen = () => dbg('WS offen — warte auf session.created');
    ws.onmessage = onWsMessage;
    ws.onerror = () => dbg('WS onerror (Details siehe onclose-Code)');
    ws.onclose = (e) => {
      dbg(`WS geschlossen (code=${e.code}${e.reason ? ', ' + e.reason : ''})`);
      const wasIntentional = intentionalClose;
      intentionalClose = false;
      teardownWs();
      if (!wasIntentional) resetUiAfterLoss();
    };
  }

  function teardownWs() {
    ws = null;
    wsReady = false;
    outbox.length = 0;
    pendingAudio.length = 0;
    stopFrameLoop();
    if (micNode) micNode.port.postMessage({ kind: 'enable', value: false });
    resetResponseState();
    flushPlaybackBuffer();
    clockActive = false;
    speaking = false;
  }

  /** Unexpected connection loss: reset mic/handsfree and surface it in the UI. */
  function resetUiAfterLoss() {
    const wasLive = micOn || handsfree || speaking;
    micOn = false; handsfree = false; speaking = false;
    stopFrameLoop();
    hidePip();
    if (pipVideo) pipVideo.srcObject = null;
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    micSrc = null;
    setVisemes(null);
    if (wasLive) emit({ type: 'handsfree', on: false });
    emit({ type: 'mic', on: false });
    emit({ type: 'error', error: 'Realtime-Verbindung getrennt' });
  }

  function closeWs() {
    if (!ws) return;
    intentionalClose = true;
    try { ws.close(); } catch {}
    teardownWs();
  }

  function resetResponseState() {
    currentResponseId = null;
    responseText = '';
    transcriptDeltaSeen = false;
    responseAudioDone = false;
    audioFed = false;
    flushed = false;
    visemeTL = []; visemeOffset = 0; visemeRespId = null;
  }

  function shortId(id) { return id ? String(id).slice(-6) : '—'; }

  function onWsMessage(e) {
    let ev;
    try { ev = JSON.parse(e.data); }
    catch { dbg('WS-Rahmen unparsebar'); return; }
    handleServerEvent(ev);
  }

  function handleServerEvent(ev) {
    switch (ev.type) {
      case 'session.created':
        dbg(`← session.created (${shortId(ev.session?.id)})`);
        sendSessionUpdate();
        break;

      case 'emma.status':
        // Server-Diagnose (VLM/STT/PROMPT/LM/TTS/RESP) → Pipeline-Anzeige
        dbg(`[${(ev.stage || '?').toUpperCase()}] ${ev.detail}${ev.ms != null ? ` (${Math.round(ev.ms)} ms)` : ''}`);
        break;

      case 'session.updated':
        dbg('← session.updated — Session konfiguriert');
        wsReady = true;
        while (outbox.length) send(outbox.shift());
        if (micOn && micNode) micNode.port.postMessage({ kind: 'enable', value: true });
        if (micOn) emit({ type: 'listening' });
        if (micOn) startFrameLoop();
        break;

      case 'input_audio_buffer.speech_started':
        // Barge-in: the server cancels its response internally — we must
        // flush OUR playback buffer immediately.
        Backchannel.cancel();
        if (speaking || audioFed) {
          flushed = true;
          flushPlaybackBuffer();
          if (speaking) { speaking = false; clockActive = false; emit({ type: 'interjected' }); }
        }
        emit({ type: 'speech-detected' });
        if (!speaking) emit({ type: 'listening' });
        break;

      case 'input_audio_buffer.speech_stopped':
        dbg(`← speech_stopped (${ev.audio_end_ms} ms)`);
        emit({ type: 'transcribing' });
        break;

      case 'conversation.item.input_audio_transcription.delta':
        // partial — only interesting for the debug trail
        if (ev.delta) dbg(`… Transkript: „${ev.delta.slice(0, 30)}“`);
        break;

      case 'conversation.item.input_audio_transcription.completed': {
        const text = (ev.transcript || '').trim();
        if (!text) { dbg('∅ leere Transkription'); return; }
        emit({ type: 'user', text });
        // backchannel: prepare now, schedule at 450 ms — canceled when the
        // answer's first audio delta arrives (race guard in backchannel.js)
        // TEMPORÄR AUS (Nervfaktor > Nutzen): nur der Aufruf ist deaktiviert,
        // Code und Filler-Clips bleiben liegen. Zum Reaktivieren FILLERS_ENABLED
        // auf true setzen.
        if (FILLERS_ENABLED) {
          Backchannel.prepare(text).then(() => Backchannel.play(FILLER_DELAY_MS));
        }
        break;
      }

      case 'response.created':
        currentResponseId = ev.response?.id || null;
        responseText = '';
        transcriptDeltaSeen = false;
        responseAudioDone = false;
        audioFed = false;
        flushed = false;                 // fresh response → accept audio again
        clockActive = false;
        visemeTL = []; visemeOffset = 0; visemeRespId = currentResponseId;
        setVisemes(null);
        dbg(`← response.created (${shortId(currentResponseId)})`);
        break;

      case 'response.output_audio.delta':
        if (flushed) { dbg('Audio-Delta verworfen (geflushht)'); return; }
        if (!audioFed) {
          audioFed = true;
          speaking = true;
          clockBaseline = playedFrames;  // lip-sync clock starts at this response
          clockActive = true;
          Backchannel.cancel();          // answer arrived → kill the filler
          emit({ type: 'playback-start' });
        }
        feedAudio(ev.delta);
        break;

      case 'response.output_audio.done':
        responseAudioDone = true;
        dbg('← response.output_audio.done');
        break;

      case 'response.output_audio_transcript.delta':
        if (!currentResponseId || !ev.response_id || ev.response_id === currentResponseId) {
          if (ev.delta) { responseText += ev.delta; transcriptDeltaSeen = true; }
        }
        break;

      case 'response.output_audio_transcript.done':
        if (ev.transcript) responseText = ev.transcript;   // authoritative full text
        break;

      case 'emma.visemes':
        onVisemes(ev);
        break;

      case 'response.done': {
        const status = ev.response?.status;
        const finalText = extractDoneText(ev.response) || responseText;
        dbg(`← response.done (${status || '?'}): „${finalText.slice(0, 40)}“`);
        if ((status === 'completed' || status === 'incomplete') && finalText) {
          emit({ type: 'assistant', text: finalText });
        }
        currentResponseId = null;
        if (!audioFed && !speaking) {
          // text-only or cancelled-without-audio: reset the status right away
          if (!flushed) emit({ type: 'playback-end' });
        }
        // with audio: playback-end fires when the ring buffer drains (underrun)
        break;
      }

      case 'error': {
        const msg = (ev.error && ev.error.message) || 'Unbekannter Realtime-Fehler';
        dbg(`⚠️ Server-Fehler: ${msg}`);
        emit({ type: 'error', error: msg });
        break;
      }

      default:
        dbg(`← ${ev.type}`);
    }
  }

  /** response.done carries the full reply in output[].content[].transcript|.text —
      source of truth when no transcript deltas arrived. */
  function extractDoneText(response) {
    const output = response?.output;
    if (!Array.isArray(output)) return '';
    const parts = [];
    for (const item of output) {
      for (const part of item?.content ?? []) {
        const t = part?.transcript ?? part?.text;
        if (typeof t === 'string' && t.trim()) parts.push(t.trim());
      }
    }
    return parts.join(' ').trim();
  }

  // ================= VISEMES (emma.visemes → Facetime engine) =================

  /** Timeline offsets are seconds relative to THIS sentence's audio; the
      engine expects one flat [{v,t0,t1}] array measured from the START of
      the response audio → shift each sentence by the summed duration_s of
      its predecessors and accumulate. */
  function onVisemes(ev) {
    if (ev.response_id && visemeRespId && ev.response_id !== visemeRespId) {
      dbg(`emma.visemes verworfen (resp ${shortId(ev.response_id)} ≠ ${shortId(visemeRespId)})`);
      return;
    }
    const tl = Array.isArray(ev.timeline) ? ev.timeline : [];
    for (const s of tl) {
      if (s && typeof s.v === 'string') {
        visemeTL.push({ v: s.v, t0: (s.t0 || 0) + visemeOffset, t1: (s.t1 || 0) + visemeOffset });
      }
    }
    visemeOffset += Number(ev.duration_s) || 0;
    if (!transcriptDeltaSeen && ev.text) responseText += (responseText ? ' ' : '') + ev.text;
    setVisemes(visemeTL.length ? visemeTL : null);
    dbg(`← emma.visemes seq=${ev.seq} · ${tl.length} Segmente · ${(ev.duration_s || 0).toFixed(2)}s (Offset ${visemeOffset.toFixed(2)}s)`);
  }

  function setVisemes(tl) {
    window._ftEngine?.setVisemeTimeline?.(tl);
  }

  /** Playback position in seconds since the current response's audio began.
      The Facetime engine evaluates the viseme timeline against this clock. */
  function playbackClock() {
    const rate = audioCtx ? audioCtx.sampleRate : 48000;
    return { t: Math.max(0, (playedFrames - clockBaseline) / rate), active: clockActive };
  }

  // ================= AUDIO GRAPH =================

  async function ensureAudioCtx() {
    audioCtx = audioCtx || new AudioContext({ latencyHint: 'interactive' });
    if (!workletsLoaded) {
      workletsLoaded = Promise.all([
        audioCtx.audioWorklet.addModule('/static/worklets/mic-capture.js'),
        audioCtx.audioWorklet.addModule('/static/worklets/audio-playback.js'),
      ]);
    }
    await workletsLoaded;
    if (audioCtx.state === 'suspended') await audioCtx.resume().catch(() => {});
  }

  async function ensurePlayback() {
    await ensureAudioCtx();
    if (playbackNode) return;
    playbackNode = new AudioWorkletNode(audioCtx, 'audio-playback', {
      numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
    });
    playbackNode.port.postMessage({ kind: 'config', inputRate: WIRE_RATE });
    playbackNode.port.onmessage = onPlaybackMsg;
    playbackNode.connect(audioCtx.destination);
    dbg(`Playback-Worklet aktiv (${audioCtx.sampleRate} Hz Kontext, ${WIRE_RATE} Hz Wire)`);
    while (pendingAudio.length) postSamples(pendingAudio.shift());
  }

  function postSamples(samples) {
    playbackNode.port.postMessage({ kind: 'audio', samples }, [samples.buffer]);
  }

  /** base64 PCM16 @16 kHz → Float32 for the worklet ring buffer. */
  function feedAudio(b64) {
    if (!b64) return;
    if (!playbackNode) { ensurePlayback().catch(e => dbg('Playback-Aufbau fehlgeschlagen: ' + (e && e.message || e))); }
    const samples = base64ToSamples(b64);
    if (!samples.length) return;
    if (playbackNode) postSamples(samples);
    else pendingAudio.push(samples);
  }

  function onPlaybackMsg(e) {
    const msg = e.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.kind === 'stats') {
      playedFrames = msg.playedFrames || 0;
    } else if (msg.kind === 'underrun') {
      if (!responseAudioDone) { dbg('Playback-Lücke (underrun) — Response noch nicht fertig'); return; }
      if (!speaking) return;
      speaking = false;
      clockActive = false;
      setVisemes(null);
      visemeTL = []; visemeOffset = 0;
      emit({ type: 'playback-end' });
    }
  }

  function flushPlaybackBuffer() {
    pendingAudio.length = 0;
    if (playbackNode) playbackNode.port.postMessage({ kind: 'clear' });
  }

  // ================= MIC =================

  async function acquireMic() {
    if (!window.isSecureContext) {
      const msg = 'Mikrofon braucht HTTPS — Seite muss über https:// geladen werden (nicht http://)';
      window.DebugConsole && DebugConsole.error(msg);
      emit({ type: 'error', error: msg });
      throw new Error('insecure context');
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = 'getUserMedia nicht verfügbar (Browser/Context blockiert Mikrofonzugriff)';
      window.DebugConsole && DebugConsole.error(msg);
      emit({ type: 'error', error: msg });
      throw new Error('no mediaDevices');
    }
    try {
      // echoCancellation keeps Emma's own TTS out of the mic — critical for
      // handsfree barge-in (proven chatbot config).
      // Video kommt dazu (VLM-Blickkontext). Verweigert der Nutzer die Kamera
      // oder fehlt sie, läuft der Call unverändert audio-only weiter.
      const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio, video: { width: { ideal: 640 }, height: { ideal: 480 } },
        });
        window.DebugConsole && DebugConsole.ok('Mikrofon + Kamera freigeschaltet');
      } catch (videoErr) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio });
        window.DebugConsole && DebugConsole.ok(
          `Mikrofon freigeschaltet (ohne Kamera: ${videoErr.name}) — Blickkontext aus`);
      }
    } catch (e) {
      window.DebugConsole && DebugConsole.error('getUserMedia fehlgeschlagen: ' + e.name);
      throw e;
    }
    if (hasVideo()) {
      videoEl.srcObject = mediaStream;
      videoEl.play().catch(() => {});
      showPip();                       // Vorschau: was Emma sieht
    }
    await ensureAudioCtx();
    if (micSrc) { try { micSrc.disconnect(); } catch {} }
    if (micNode) { try { micNode.disconnect(); } catch {} }
    micSrc = audioCtx.createMediaStreamSource(mediaStream);
    micNode = new AudioWorkletNode(audioCtx, 'mic-capture', {
      numberOfInputs: 1, numberOfOutputs: 0,
      processorOptions: { chunkMs: MIC_CHUNK_MS },
    });
    micNode.port.onmessage = (e) => { if (e.data instanceof ArrayBuffer) onMicChunk(e.data); };
    micSrc.connect(micNode);
    // stays DISABLED until session.updated — no audio before the persona is applied
    dbg(`Mic-Kanal gebaut (audioCtx.state=${audioCtx.state}, ${audioCtx.sampleRate} Hz → ${WIRE_RATE} Hz Wire)`);
  }

  /** Int16 PCM chunk from the worklet → base64 → input_audio_buffer.append. */
  function onMicChunk(buf) {
    if (!wsReady || !micOn) return;   // gated by the worklet enable flag
    send({ type: 'input_audio_buffer.append', audio: base64FromArrayBuffer(buf) });
  }

  // ================= KAMERA (VLM-Blickkontext) =================

  function hasVideo() {
    return !!(mediaStream && mediaStream.getVideoTracks().some(t => t.readyState === 'live'));
  }

  /** Kleine Vorschau: zeigt, was Emma gerade zu sehen bekommt.
      Live-Bild (glatt) + Zeile mit dem zuletzt gesendeten Frame. */
  function ensurePip() {
    if (pipEl) return;
    pipEl = document.createElement('div');
    pipEl.id = 'camPip';
    pipEl.innerHTML = '<video playsinline muted autoplay></video><div class="lbl">Kamera</div>';
    document.body.appendChild(pipEl);
    pipVideo = pipEl.querySelector('video');
    pipLabel = pipEl.querySelector('.lbl');
  }

  function showPip() {
    if (!hasVideo()) return;
    ensurePip();
    pipVideo.srcObject = mediaStream;
    pipVideo.play().catch(() => {});
    pipLabel.innerHTML = '<b>Emma sieht:</b> Kamera bereit';
    pipEl.classList.add('on');
  }

  function hidePip() {
    if (pipEl) pipEl.classList.remove('on');
  }

  function startFrameLoop() {
    if (frameTimer || !hasVideo()) return;
    frameTimer = setInterval(sendFrame, FRAME_MS);
    dbg(`Kamera-Kontext aktiv: 1 Frame/${FRAME_MS / 1000}s, max ${FRAME_MAX_EDGE} px`);
  }

  function stopFrameLoop() {
    if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
  }

  /** Ein herunterskalierter JPEG-Frame → emma.video_frame (fire-and-forget).
      Bei 512 px sind das ~15-20 KB/s — vernachlässigbar neben dem Audio. */
  function sendFrame() {
    if (!wsReady || !micOn || !hasVideo()) return;
    const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
    if (!vw || !vh) return;                      // Kamera liefert noch kein Bild
    const scale = Math.min(1, FRAME_MAX_EDGE / Math.max(vw, vh));
    frameCanvas.width = Math.max(1, Math.round(vw * scale));
    frameCanvas.height = Math.max(1, Math.round(vh * scale));
    frameCanvas.getContext('2d').drawImage(videoEl, 0, 0, frameCanvas.width, frameCanvas.height);
    const dataUrl = frameCanvas.toDataURL('image/jpeg', 0.8);
    send({ type: 'emma.video_frame', frame: dataUrl, at: Date.now() / 1000 });
    framesSent++;
    const kb = Math.round(dataUrl.length * 0.75 / 1024);
    if (pipLabel) {
      pipLabel.innerHTML = `<b>Emma sieht:</b> ${frameCanvas.width}×${frameCanvas.height} · `
        + `${kb} KB · ${new Date().toLocaleTimeString('de-DE')}`;
    }
    if (framesSent % 10 === 0) {
      dbg(`Kamera: ${framesSent} Frames gesendet (${kb} KB/Frame, ${frameCanvas.width}×${frameCanvas.height})`);
    }
  }

  async function setMic(on) {
    if (on === micOn && !!mediaStream === on) {
      dbg(`setMic(${on}) früh raus — bereits im Sollzustand`);
      return;
    }
    if (on) {
      await acquireMic();
      micOn = true;
      emit({ type: 'mic', on: true });
      await connect();
      if (wsReady) micNode.port.postMessage({ kind: 'enable', value: true });
      if (handsfree && wsReady) emit({ type: 'listening' });
    } else {
      if (micNode) micNode.port.postMessage({ kind: 'enable', value: false });
      stopFrameLoop();
      hidePip();
      if (pipVideo) pipVideo.srcObject = null;
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      videoEl.srcObject = null;
      micSrc = null;
      micOn = false;
      flushPlaybackBuffer();
      clockActive = false;
      emit({ type: 'mic', on: false });
      if (!handsfree) closeWs();   // nobody left talking → release the server slot
    }
  }
  function isMicOn() { return micOn; }
  function toggleMic() { return setMic(!micOn); }   // real toggle

  // ================= TURN CONTROL =================

  /** Interject: user interrupted manually — kill playback + ask the server
      to cancel its response. Silent = don't surface a status change. */
  function flushQueue(silent) {
    dbg(`Flush angefordert (silent=${!!silent})`);
    flushed = true;
    flushPlaybackBuffer();
    Backchannel.cancel();
    setVisemes(null);
    visemeTL = []; visemeOffset = 0;
    if (currentResponseId && wsReady) send({ type: 'response.cancel', response_id: currentResponseId });
    if (speaking || clockActive) {
      speaking = false; clockActive = false;
      emit({ type: 'interjected' });
    } else if (!silent) {
      emit({ type: 'flushed' });
    }
    // flushed stays latched until the next response.created drops it
  }

  // ================= PUBLIC =================

  async function pushText(text) {           // typed input path
    Backchannel.cancel();
    emit({ type: 'user', text });
    if (FILLERS_ENABLED) {                  // siehe FILLERS_ENABLED oben
      Backchannel.prepare(text).then(() => Backchannel.play(FILLER_DELAY_MS));
    }
    await ensurePlayback().catch(() => {});
    if (!ws) connect();
    send({
      type: 'conversation.item.create',
      item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] },
    });
    send({ type: 'response.create' });
  }

  function setHandsfree(on) {
    handsfree = !!on;
    emit({ type: 'handsfree', on: handsfree });
    if (handsfree && micOn && wsReady) emit({ type: 'listening' });
    if (!handsfree && !micOn) closeWs();    // hangup path: mic already off → disconnect
  }
  function isHandsfree() { return handsfree; }
  function isBusy() { return speaking; }

  return {
    setPersona, setMic, toggleMic, isMicOn,
    setHandsfree, isHandsfree, isBusy,
    pushText, flushQueue, playbackClock, setLogSink,
    set callback(fn) { onEvent = fn; },
    get state() { return { micOn, handsfree, speaking, connected: wsReady }; },
  };
})();

// ---- base64 helpers (wire format: base64 PCM16 mono @16 kHz) ----

/** @param {ArrayBuffer} buf */
function base64FromArrayBuffer(buf) {
  const bytes = new Uint8Array(buf);
  // Chunked encoding so we don't blow up the call stack on long buffers.
  let binary = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(binary);
}

/** @param {string} b64  @returns {Float32Array} mono samples */
function base64ToSamples(b64) {
  const binary = atob(b64);
  const n = binary.length >> 1;                      // PCM16 → whole samples
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lo = binary.charCodeAt(2 * i);
    const hi = binary.charCodeAt(2 * i + 1);
    let s = lo | (hi << 8);
    if (s >= 0x8000) s -= 0x10000;                   // little-endian signed
    out[i] = s / 32768;
  }
  return out;
}

window.CallSession = CallSession;