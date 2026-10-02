/* Backchannel player: plays a filler clip between user turn and answer.

Contract:
  backchannel.prepare(text)  — call right after STT/text submit; fetches a
                               matching clip in the background (~100 ms).
  backchannel.play(delayMs)  — starts playback after `delayMs` (default 450).
  backchannel.cancel()       — answer arrived before/at filler end: stop it.

Race rules:
  - If the real answer's first chunk is ready before play() is called, never
    start the filler.
  - If the filler is playing when the answer arrives, cancel immediately
    (hard cut, no crossfade — a fading filler sounds like an error).
*/

const Backchannel = (() => {
  let pending = null;   // {url, phrase, duration_s}
  let audio = null;     // dedicated Audio element (separate from answer audio)
  let timer = null;
  let playing = false;

  function el() {
    if (!audio) {
      audio = new Audio();
      audio.volume = 0.9;
      audio.addEventListener('ended', () => { playing = false; });
    }
    return audio;
  }

  async function prepare(text) {
    cancel();
    try {
      const r = await fetch('/api/fillers/pick', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, persona_id: window._studioPersonaId}),
      });
      const data = await r.json();
      if (data.url) {
        // preload so play() starts instantly
        el().src = data.url;
        el().load();
        pending = data;
      } else pending = null;
    } catch { pending = null; }
    return pending;
  }

  function play(delayMs = 450) {
    if (!pending || playing) return false;
    timer = setTimeout(() => {
      if (!pending) return;
      const a = el();
      if (a.src && !a.paused && !a.ended) return; // already going
      a.currentTime = 0;
      a.play().catch(() => {});
      playing = true;
      window._onFillerStart && window._onFillerStart(pending.phrase);
    }, delayMs);
    return true;
  }

  function cancel() {
    if (timer) { clearTimeout(timer); timer = null; }
    if (playing || (audio && !audio.paused)) {
      audio.pause(); audio.currentTime = 0;
      window._onFillerEnd && window._onFillerEnd();
    }
    playing = false;
    pending = null;
  }

  function isPlaying() { return playing; }

  return {prepare, play, cancel, isPlaying};
})();
window.Backchannel = Backchannel;
