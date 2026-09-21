/* Emma Studio mic capture worklet.

Resamples the AudioContext rate (typically 48 kHz) down to 16 kHz mono,
packs the result as little-endian Int16 PCM and posts it to the main thread
in fixed-size chunks (default 100 ms = 1600 samples = 3200 bytes — the
cadence the /v1/realtime input_audio_buffer.append events use).

Adapted from the speech-to-speech demo worklet; the noise gate was dropped
because Emma's server-side VAD does the gating (barge-in needs the raw,
continuous stream).

Messaging:
  main -> worklet: { kind: "enable", value: bool }   gate the OUTGOING stream
  worklet -> main: ArrayBuffer (Int16 PCM, transferable) per chunk

Disabled at construction: the main thread enables it once the realtime
session is configured (session.updated), so no audio is sent before the
persona is applied. While disabled the worklet still consumes input so the
graph never stalls or grows unbounded.
*/

const TARGET_RATE = 16000;
const DEFAULT_CHUNK_MS = 100;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = options?.processorOptions?.chunkMs ?? DEFAULT_CHUNK_MS;
    this._ratio = sampleRate / TARGET_RATE;
    this._chunkSamples = Math.round((TARGET_RATE * chunkMs) / 1000);
    this._scratch = new Float32Array(0);
    this._decimated = new Float32Array(this._chunkSamples);
    this._enabled = false;

    this.port.onmessage = (e) => {
      const data = e.data;
      if (data?.kind === "enable") this._enabled = !!data.value;
    };
  }

  /** Append `incoming` to the scratch buffer, then emit full chunks.
   *  @param {Float32Array} incoming */
  _ingest(incoming) {
    if (incoming.length === 0) return;
    const next = new Float32Array(this._scratch.length + incoming.length);
    next.set(this._scratch, 0);
    next.set(incoming, this._scratch.length);
    this._scratch = next;
    this._maybeEmit();
  }

  _maybeEmit() {
    const r = this._ratio;
    const n = this._chunkSamples;
    const needIn = Math.ceil(n * r);
    const dec = this._decimated;
    const k = Math.round(r);
    const integerRatio = k >= 2 && Math.abs(r - k) < 1e-6;
    while (this._scratch.length >= needIn) {
      // 1. Decimate to 16 kHz floats. Integer ratios use a k-tap boxcar
      //    average (cheap low-pass before decimation, e.g. 48 -> 16 with k=3);
      //    everything else falls back to linear interpolation (44.1 kHz etc.).
      if (integerRatio) {
        for (let i = 0; i < n; i++) {
          let acc = 0;
          const base = i * k;
          for (let j = 0; j < k; j++) acc += this._scratch[base + j];
          dec[i] = acc / k;
        }
      } else {
        for (let i = 0; i < n; i++) {
          const srcPos = i * r;
          const idx = Math.floor(srcPos);
          const frac = srcPos - idx;
          const a = this._scratch[idx];
          const b = this._scratch[idx + 1] ?? a;
          dec[i] = a + (b - a) * frac;
        }
      }

      // 2. Pack to Int16 (saturate to [-1, 1] before scaling).
      const out = new Int16Array(n);
      for (let i = 0; i < n; i++) {
        const clamped = dec[i] < -1 ? -1 : dec[i] > 1 ? 1 : dec[i];
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      }

      // Keep only the trailing unused samples.
      this._scratch = this._scratch.slice(Math.floor(n * r));

      if (this._enabled) this.port.postMessage(out.buffer, [out.buffer]);
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const mono = input[0];
    if (mono.length > 0) this._ingest(mono);
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);