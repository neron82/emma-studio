/* Emma Studio facetime engine: VRM avatar with amplitude-driven lip-sync,
   idle blink/sway, portrait fallback. ES module, loaded on demand. */

/* Blick- und Idle-Tuning. Alles prozedural (Bone-Rotationen pro Frame) —
   dafür braucht es keine Animations-Sequenzen: der Körper schwankt, der Kopf
   hält die Blickachse auf der Kamera, die Augen folgen über three-vrm. */
const GAZE = {
  headShare: 0.62,        // Anteil der Korrektur, den der Kopf übernimmt
  neckShare: 0.38,        // Rest geht an den Hals (menschliche Verteilung)
  maxAngleDeg: 32,        // darüber würde der Hals brechen — hart begrenzt
  smoothPerS: 10.0,       // 1/s: Annäherung an die Zielkorrektur (Latenz ↔ Ruhe)
  microAmp: 0.0075,       // rad Mikro-Rauschen um die Blickachse (~0.4°)
  glanceEveryS: [9, 20],  // Pause zwischen kurzen Seitenblicken
  glanceDurS: [0.8, 1.7], // Dauer eines Seitenblicks
  glanceAmp: 0.13,        // rad (~7.5°) — "denkt nach", bleibt aber zugewandt
  glanceEnabled: true,
};

export class FacetimeEngine {
  constructor() {
    this.vrm = null;
    this.renderer = null; this.scene = null; this.camera = null;
    this.canvas = null; this.fbEl = null; this.badgeEl = null;
    this.running = false;
  }

  attach(canvas, fallbackEl, badgeEl) {
    this.canvas = canvas; this.fbEl = fallbackEl; this.badgeEl = badgeEl;
    if (this.running) return;
    this.init().catch(e => {
      console.error('[facetime] init failed:', e);
      this.showFallback('WebGL/VRM: ' + (e && e.message ? e.message : e));
    });
  }

  showFallback(reason) {
    if (this.badgeEl) this.badgeEl.textContent = 'Standbild (' + reason + ')';
    if (this.fbEl) this.fbEl.style.display = 'flex';
    // gentle breathing on the still portrait
    const img = this.fbEl && this.fbEl.querySelector('img');
    if (img) { let t = 0;
      setInterval(() => { t += .05; img.style.transform = `scale(${1 + Math.sin(t) * .008})`; }, 50);
    }
  }

  async init() {
    const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js');
    const { GLTFLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/loaders/GLTFLoader.js');
    const { VRMLoaderPlugin } = await import('https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.1.3/lib/three-vrm.module.min.js');

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setSize(this.canvas.clientWidth || innerWidth, this.canvas.clientHeight || innerHeight);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(28, (this.canvas.clientWidth||innerWidth)/(this.canvas.clientHeight||innerHeight), .1, 20);
    this.camera.position.set(0, 1.34, 1.15);

    const key = new THREE.DirectionalLight(0xfff0f5, Math.PI * .9);
    key.position.set(.8, 2, 2); this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(0xffffff, .55));
    const rim = new THREE.DirectionalLight(0xe8a0bf, .5); rim.position.set(-2, 1, -1); this.scene.add(rim);

    const head = await fetch('/static/avatars/emma.vrm', { method: 'HEAD' });
    if (!head.ok) throw new Error('kein VRM');
    const loader = new GLTFLoader();
    loader.register(p => new VRMLoaderPlugin(p));
    const gltf = await loader.loadAsync('/static/avatars/emma.vrm');
    this.vrm = gltf.userData.vrm;
    this.scene.add(gltf.scene);
    gltf.scene.traverse(o => { o.frustumCulled = false; });

    // VRM models face +Z toward the default camera; we keep the model as-is
    // and place the camera on that side instead of rotating the model —
    // rotating the scene turns the head away (head-forward stays -Z).

    // Relax the T-pose into an A-pose (arms ~65° down) — VRM rest pose is
    // a T; without baked idle animation that looks broken on camera.
    // vrm.update() resets bone rotations every frame unless we disable
    // autoUpdateHumanBones first (verified empirically).
    if (this.vrm.humanoid) {
      this.vrm.humanoid.autoUpdateHumanBones = false;
      // restore the normalized rest pose once so our rotations start clean
      this.vrm.humanoid.resetNormalizedPose();
    }
    this._applyAPose(gltf.scene, THREE);

    // --- auto-framing: normalize model origin + aim camera at the head ---
    // VRM samples often have a non-ground-level origin; measure the real
    // bounds and head position, then place camera for a head-and-shoulders
    // shot regardless of the model's internal offsets.
    {
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const height = Math.max(0.1, box.max.y - box.min.y);
      const modelMinY = box.min.y;          // origin offset of this model
      gltf.scene.updateMatrixWorld(true);
      let headBone = null;
      gltf.scene.traverse(o => { if (o.isBone && /head/i.test(o.name)) headBone = o; });
      const v = new THREE.Vector3();
      const rawHeadY = headBone
        ? headBone.getWorldPosition(v).y
        : box.max.y - height * 0.07;
      // store everything relative to the model's own origin — we keep the
      // scene where the asset author put it and move only our viewpoint.
      this.modelOriginY = modelMinY;
      this.headOffsetY = rawHeadY - modelMinY;   // head height above feet
      this.height = height;
      const headWorldTarget = this.headOffsetY;  // after we ground the model
      gltf.scene.position.y -= modelMinY;        // stand on y=0
      gltf.scene.updateMatrixWorld(true);

      // Facing detection: which way does the FACE point? Eye bones sit in
      // front of the skull center on the face side (VRoid rigs: −Z, Mixamo:
      // +Z). Measure eye-vs-head depth and put the camera on that side.
      let eyeZ = null, headZ = null;
      if (headBone) {
        headZ = headBone.getWorldPosition(v).z;
        gltf.scene.traverse(o => {
          if (!o.isBone || !/eye/i.test(o.name) || eyeZ !== null) return;
          eyeZ = o.getWorldPosition(v).z;
        });
      }
      const facingPlusZ = this._facingPlusZ = (eyeZ === null || headZ === null)
        ? true                                  // keine Augen messbar: VRM-Spezial-Default (+Z)
        : (eyeZ > headZ);
      window.DebugConsole && DebugConsole.ok(
        `Blickrichtung: ${facingPlusZ ? '+Z' : '−Z'} (eye z=${eyeZ === null ? '?' : eyeZ.toFixed(2)}, head z=${headZ === null ? '?' : headZ.toFixed(2)})`
      );

      // head-and-shoulders: ~25% headroom (lookAt sits slightly below eyes),
      // camera on whichever side the face points to
      const dist = Math.max(0.55, height * 0.5);
      this.camera.position.set(0, headWorldTarget + height * 0.02,
                               (facingPlusZ ? +1 : -1) * dist);
      this.camera.lookAt(0, headWorldTarget - height * 0.055, 0);
      this._frame = { height, headWorldTarget, dist };
    }

    if (this.badgeEl) this.badgeEl.textContent =
      `VRM aktiv (${(this._frame.height).toFixed(2)}m)`;
    if (this.fbEl) this.fbEl.style.display = 'none';
    this.THREE = THREE;
    this._initGaze(this._facingPlusZ);   // Kopf/Augen auf die Kamera ausrichten
    this.running = true;
    this.loop();
  }

  /** Lower both arms from T-pose into a natural A-pose.
      VRM0 bone names use L_/R_ prefixes (J_Bip_L_UpperArm), not "left/right".
      Self-calibrating: tries both rotation directions per arm and keeps the
      one that actually lowers the hand — bind orientation differs between
      VRM rigs, so a fixed sign breaks on new models. */
  _applyAPose(sceneRoot, THREE) {
    let leftArm = null, rightArm = null;
    sceneRoot.traverse(o => {
      if (!o.isBone) return;
      const n = o.name.toLowerCase();
      if (!n.includes('upperarm')) return;
      if (n.includes('_l_') || n.startsWith('l_') || n.includes('left')) leftArm = o;
      else if (n.includes('_r_') || n.startsWith('r_') || n.includes('right')) rightArm = o;
    });
    window.DebugConsole && DebugConsole.ok(
      `A-Pose: L=${leftArm ? leftArm.name : '—'} R=${rightArm ? rightArm.name : '—'}`
    );
    const handOf = (arm) => {
      let found = null;
      arm.traverse(o => {
        if (found || !o.isBone || o === arm) return;
        if (/hand/i.test(o.name)) found = o;
      });
      return found;
    };
    const lower = (arm) => {
      if (!arm) return;
      const hand = handOf(arm);
      if (!hand || !THREE) { arm.rotation.z += 1.15; return; }   // fallback: guess
      sceneRoot.updateMatrixWorld(true);
      const v = new THREE.Vector3();
      const y0 = hand.getWorldPosition(v).y;
      // Bind-Orientierung ist rig-abhängig (VRoid: senkt um Z, Mixamo: oft X)
      // → alle drei Achsen × beide Richtungen probieren, beste Senkung gewinnt.
      let best = null, bestY = y0;
      for (const axis of ['x', 'y', 'z']) {
        for (const sign of [+1, -1]) {
          arm.rotation[axis] += sign * 1.15;
          sceneRoot.updateMatrixWorld(true);
          const y1 = hand.getWorldPosition(v).y;
          arm.rotation[axis] -= sign * 1.15;
          sceneRoot.updateMatrixWorld(true);
          if (y1 < bestY) { bestY = y1; best = [axis, sign]; }
        }
      }
      if (!best) { arm.rotation.z += 1.15; return; }   // keine Achse senkt: Rate-Fallback
      const [axis, sign] = best;
      arm.rotation[axis] += sign * 1.15;
      sceneRoot.updateMatrixWorld(true);
      window.DebugConsole && DebugConsole.log(
        `A-Pose: ${arm.name} ${axis}${sign > 0 ? '+' : '−'}1.15 · Hand ${y0.toFixed(2)}→${bestY.toFixed(2)}`
      );
    };
    lower(leftArm);
    lower(rightArm);
  }

  /** Blick-Anker: Kopf/Hals drehen so, dass die *gemessene* Blickachse auf
      der Kamera bleibt, während der Körper schwankt. Rein prozedural —
      keine Animations-Sequenzen, nur Bone-Deltas pro Frame.
      Rig-agnostisch: die Korrektur wird als Welt-Rotation berechnet und über
      den Parent-Raum auf die Bind-Pose gelegt (funktioniert für VRoid-VRM0
      wie für Mixamo-Rigs, ohne Achsen zu raten). */
  _initGaze(facingPlusZ) {
    const THREE = this.THREE;
    const h = this.vrm.humanoid;
    const raw = (n) => (h && h.getRawBoneNode ? h.getRawBoneNode(n) : null);
    const head = raw('head'), neck = raw('neck'), chest = raw('chest');
    if (!head) {
      window.DebugConsole && DebugConsole.log('Gaze: kein Kopf-Bone — Idle ohne Blickanker');
      return;
    }
    this._gaze = {
      head, neck, chest,
      rest: {
        head: head.quaternion.clone(),
        neck: neck ? neck.quaternion.clone() : null,
        chest: chest ? chest.quaternion.clone() : null,
      },
      forward: facingPlusZ ? 1 : -1,     // gemessene Blickachse des Modells
      corr: new THREE.Quaternion(),      // geglättete Welt-Korrektur
      t: 0, nextGlance: 7 + Math.random() * 6, glanceT: -1, glanceAmp: 0, glanceDur: 1,
      errDeg: null, corrDeg: 0, eyes: 'aus',
    };
    // Augen: three-vrm bringt den LookAt-Applier mit (VRM0 → Ausdrucks-Gewichte,
    // VRM1 → Eye-Bones). Läuft unabhängig von autoUpdateHumanBones.
    try {
      if (this.vrm.lookAt) {
        this.vrm.lookAt.target = this.camera;
        this.vrm.lookAt.autoUpdate = true;
        // Faktisch bleiben: welchen Applier three-vrm für dieses Asset wählt,
        // ist asset-abhängig (VRM0 wird intern konvertiert) — deshalb keine
        // Versions-Behauptung, nur "aktiv".
        this._gaze.eyes = this.vrm.lookAt.applier ? 'aktiv' : 'ohne Applier';
      }
    } catch (e) { /* LookAt ist optional — der Kopf trägt den Blick dann allein */ }
    window.DebugConsole && DebugConsole.ok(
      `Blick: Kopf→Kamera aktiv (Augen: ${this._gaze.eyes}, Achse: ${facingPlusZ ? '+Z' : '−Z'})`);
  }

  /** Welt-Rotation R auf einen Bone legen, ohne die Bind-Pose zu verändern:
      local' = parentWorld⁻¹ · R · parentWorld · rest. */
  _applyBoneWorldRotation(bone, rest, R) {
    const THREE = this.THREE;
    const qp = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    bone.quaternion.copy(rest).premultiply(qp.clone().invert().multiply(R).multiply(qp));
  }

  /** Pro Frame: Blickachse auf die Kamera nachführen (+ Mikro-Rauschen,
      + gelegentlicher Seitenblick) und auf Kopf/Hals verteilen. */
  _updateGaze(dt) {
    const THREE = this.THREE, g = this._gaze;
    if (!g) return;
    const v = new THREE.Vector3(), q = new THREE.Quaternion();

    // WICHTIG: erst die Bind-Pose wiederherstellen und dann messen. Würde die
    // Basisrichtung mit der noch aktiven Korrektur des Vorframes gemessen,
    // pendelt sich der Regler bei halbem Fehler ein (E/2 statt 0).
    g.head.quaternion.copy(g.rest.head);
    if (g.neck && g.rest.neck) g.neck.quaternion.copy(g.rest.neck);
    this.vrm.scene.updateMatrixWorld(true);
    const baseFwd = new THREE.Vector3(0, 0, g.forward)
      .applyQuaternion(g.head.getWorldQuaternion(q)).normalize();
    const headPos = g.head.getWorldPosition(v).clone();

    // Seitenblick-Zeitplan: kurz wegschauen, dann zurück
    g.t += dt;
    if (GAZE.glanceEnabled && g.glanceT < 0 && g.t > g.nextGlance) {
      g.glanceT = 0;
      g.nextGlance = g.t + GAZE.glanceEveryS[0]
        + Math.random() * (GAZE.glanceEveryS[1] - GAZE.glanceEveryS[0]);
      g.glanceAmp = (Math.random() < .5 ? -1 : 1) * GAZE.glanceAmp * (.7 + Math.random() * .6);
      g.glanceDur = GAZE.glanceDurS[0] + Math.random() * (GAZE.glanceDurS[1] - GAZE.glanceDurS[0]);
    }
    let glance = 0;
    if (g.glanceT >= 0) {
      g.glanceT += dt;
      const p = g.glanceT / g.glanceDur;
      if (p >= 1) { g.glanceT = -1; }
      else glance = (p < .3 ? p / .3 : p > .7 ? (1 - p) / .3 : 1) * g.glanceAmp;
    }

    // Zielrichtung = Kamera, plus Mikro-Rauschen (zwei ungleiche Frequenzen,
    // damit es nicht wie ein Sinus aussieht) plus Seitenblick
    const microY = (Math.sin(g.t * 0.31) + Math.sin(g.t * 0.53 + 1.7)) * GAZE.microAmp;
    const microP = (Math.sin(g.t * 0.23 + 0.9) + Math.sin(g.t * 0.41 + 2.3)) * GAZE.microAmp * .6;
    const desired = this.camera.getWorldPosition(new THREE.Vector3()).sub(headPos).normalize();
    desired.applyAxisAngle(new THREE.Vector3(0, 1, 0), microY + glance);
    desired.applyAxisAngle(new THREE.Vector3(1, 0, 0), microP);

    // minimale Drehung von der Basis- auf die Zielrichtung, Winkel begrenzt
    const target = new THREE.Quaternion().setFromUnitVectors(baseFwd, desired);
    const maxA = THREE.MathUtils.degToRad(GAZE.maxAngleDeg);
    const ang = 2 * Math.acos(Math.min(1, Math.abs(target.w)));
    if (ang > maxA) {
      // eigener Zwischenspeicher: slerpQuaternions mit identischem Ziel-Objekt
      // würde sich selbst überschreiben (this === qb) und die Drehung verlieren
      const clamped = new THREE.Quaternion().slerpQuaternions(new THREE.Quaternion(), target, maxA / ang);
      target.copy(clamped);
    }

    // weich nachführen (exponentielle Annäherung, framerate-unabhängig)
    g.corr.slerp(target, 1 - Math.exp(-GAZE.smoothPerS * dt));
    g.corrDeg = THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(g.corr.w))));

    const ident = new THREE.Quaternion();
    if (g.neck && g.rest.neck) {
      this._applyBoneWorldRotation(g.neck, g.rest.neck,
        new THREE.Quaternion().slerpQuaternions(ident, g.corr, GAZE.neckShare));
      this.vrm.scene.updateMatrixWorld(true);   // Kopf erbt den Hals-Anteil
    }
    this._applyBoneWorldRotation(g.head, g.rest.head,
      new THREE.Quaternion().slerpQuaternions(ident, g.corr, GAZE.headShare));

    // Restfehler messen (Debug + Verifikation: sollte nahe 0° liegen)
    this.vrm.scene.updateMatrixWorld(true);
    const nowFwd = new THREE.Vector3(0, 0, g.forward)
      .applyQuaternion(g.head.getWorldQuaternion(new THREE.Quaternion())).normalize();
    const toCam = this.camera.getWorldPosition(new THREE.Vector3())
      .sub(g.head.getWorldPosition(new THREE.Vector3())).normalize();
    g.errDeg = THREE.MathUtils.radToDeg(Math.acos(Math.max(-1, Math.min(1, nowFwd.dot(toCam)))));
    if (!g.logged && g.t > 1.5) {
      g.logged = true;
      window.DebugConsole && DebugConsole.ok(
        `Blick eingependelt: Restfehler ${g.errDeg.toFixed(2)}° · Korrektur ${g.corrDeg.toFixed(1)}° · Augen: ${g.eyes}`);
    }
  }

  /** Blick-Zustand für Debug/Verifikation (Restfehler in Grad etc.). */
  gazeState() {
    const g = this._gaze;
    if (!g) return null;
    return { errDeg: g.errDeg, corrDeg: g.corrDeg, eyes: g.eyes,
             glancing: g.glanceT >= 0, neck: !!g.neck };
  }

  audioLevel() {
    const el = document.getElementById('audioOut');
    if (!el || el.paused) return 0;
    if (!this.analyser) return 0;
    this.analyser.getByteFrequencyData(this.levelData);
    let s = 0; for (let i = 2; i < 24; i++) s += this.levelData[i];
    return Math.min(1, (s / 22) / 140);
  }

  /** Vokal-Näherung fürs Lip-Sync: Der spektrale Zentroid bestimmt die
      Mundform-Klasse (dunkel→o/u, mittel→a, hell→i/e), die Gesamtenergie den
      Öffnungsgrad. Kein echtes Phonem-Tracking — aber der Mund artikuliert
      statt nur mit der Lautstärke aufzugehen. */
  mouthShape() {
    const el = document.getElementById('audioOut');
    if (!el || el.paused || !this.analyser) return null;
    const sr = this.audioCtx ? this.audioCtx.sampleRate : 44100;
    const bins = this.levelData.length;
    const hzPerBin = sr / 2 / bins;
    this.analyser.getByteFrequencyData(this.levelData);
    const bandAvg = (f0, f1) => {
      let s = 0, n = 0;
      for (let i = Math.max(1, Math.floor(f0 / hzPerBin));
           i <= Math.min(bins - 1, Math.ceil(f1 / hzPerBin)); i++) {
        s += this.levelData[i]; n++;
      }
      return n ? s / n / 255 : 0;
    };
    const total = bandAvg(120, 4000);
    if (total < 0.045) return null;          // Stille → alle Kanäle schließen
    // Frequenz-Schwerpunkt über dem Sprachband
    let wsum = 0, esum = 0;
    const lo = Math.max(1, Math.floor(150 / hzPerBin));
    const hi = Math.min(bins - 1, Math.ceil(3800 / hzPerBin));
    for (let i = lo; i <= hi; i++) {
      const e = this.levelData[i] / 255;
      wsum += e * i; esum += e;
    }
    const centroidHz = esum > 0 ? (wsum / esum) * hzPerBin : 1000;
    const open = Math.min(1, total * 2.2);
    if (centroidHz < 850)  return { o: open * .85, u: open * .3, aa: open * .12 };
    if (centroidHz > 1900) return { i: open * .85, e: open * .35 };
    return { aa: open, e: open * .18 };
  }

  /** Aktive Visem-Timeline setzen (Luxus-Lip-Sync). null = Fallback auf
      Formant-Modus. Die Loop wertet sie anhand der Audio-Playback-Uhr aus. */
  setVisemeTimeline(tl) {
    this._visemeTL = Array.isArray(tl) ? tl : null;
    if (this._visemeTL) {
      window.DebugConsole && DebugConsole.log(
        `Visem-Timeline aktiv (${this._visemeTL.length} Segmente)`);
    }
  }

  /** Visem-Werte an Position t (Sekunden im aktuellen Chunk) — null wenn
      keine Timeline aktiv ist oder gerade Stille herrscht. */
  _visemeAt(t) {
    const tl = this._visemeTL;
    if (!tl || !tl.length) return null;
    // binäre Suche: letzter Eintrag mit t0 <= t
    let lo = 0, hi = tl.length - 1, idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (tl[mid].t0 <= t) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (idx < 0) return null;
    const seg = tl[idx];
    if (t > seg.t1) return null;             // Lücke zwischen Segmenten
    // kurzes Attack/Release am Segmentrand für organische Bewegung
    const dur = Math.max(0.02, seg.t1 - seg.t0);
    const phase = Math.min(1, (t - seg.t0) / dur);
    const env = phase < 0.25 ? phase / 0.25 : phase > 0.8 ? (1 - phase) / 0.2 : 1;
    return { v: seg.v, w: Math.max(0, env) };
  }

  bindAudio(el) {
    try {
      this.audioCtx = this.audioCtx || new AudioContext();
      const src = this.audioCtx.createMediaElementSource(el);
      this.analyser = this.analyser || this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;          // genug Auflösung für Vokal-Bänder
      this.levelData = new Uint8Array(this.analyser.frequencyBinCount);
      src.connect(this.analyser); this.analyser.connect(this.audioCtx.destination);
      window.DebugConsole && DebugConsole.ok(
        `Lip-Sync: Formant-Modus (${this.audioCtx.sampleRate} Hz, ${this.levelData.length} Bins)`);
    } catch {}
  }
}

FacetimeEngine.prototype.loop = function () {
  requestAnimationFrame(() => this.loop());
  const vrm = this.vrm;
  if (vrm) {
    // Mund-Priorität: (1) Visem-Timeline vom Server [Luxus],
    // (2) Formant-Näherung aus dem Spektrum [Fallback], (3) zu.
    const el = document.getElementById('audioOut');
    // Realtime-Playback (CallSession) liefert die Takt-Uhr; ohne sie zählt
    // die audioOut-Element-Uhr (alter Chunk-Playback-Pfad).
    const clk = window.CallSession?.playbackClock?.() || null;
    const tlActive = !!(this._visemeTL && this._visemeTL.length && (clk?.active || (el && !el.paused)));
    const target = { aa: 0, i: 0, e: 0, o: 0, u: 0 };
    let mode = 'zu';
    const vis = tlActive ? this._visemeAt(clk?.active ? clk.t : el.currentTime) : null;
    if (vis) {
      // Timeline: Zielwert = Visem-Gewicht × Attack/Release-Hüllkurve;
      // 'mbp' (Lippen b) wird als geschlossener Mund auf alle Kanäle abgebildet
      if (vis.v !== 'mbp') {
        target[vis.v] = Math.max(target[vis.v] ?? 0, vis.w);
        if ((vis.v === 'e' || vis.v === 'i') && vis.v in target) {
          target.aa = Math.max(target.aa, vis.w * 0.35);   // leichte Öffnung mit
        }
      }
      mode = `tl:${vis.v}`;
    } else if (!tlActive) {
      const shape = this.mouthShape();
      if (shape) { Object.assign(target, shape); mode = 'formant'; }
    }
    for (const k of ['aa', 'i', 'e', 'o', 'u']) {
      const cur = this._mouth?.[k] ?? 0;
      const next = cur + ((target[k] || 0) - cur) * .45;
      this._mouth = this._mouth || {};
      this._mouth[k] = next;
      vrm.expressionManager?.setValue(k, next);
    }
    this._lipMode = mode;
    this._t = (this._t || 0) + 1 / 60;
    if (this._t > (this._nextBlink || 2)) { this._t = 0; this._nextBlink = 2 + Math.random() * 3; }
    vrm.expressionManager?.setValue('blink', Math.max(0, 1 - Math.abs(this._t - .12) * 12));
    // idle sway: zwei ungleiche Frequenzen statt einer Sinuswelle (wirkt
    // weniger mechanisch) plus minimale Roll-/Seitwärtsbewegung. Der Kopf
    // wird direkt danach wieder auf die Kamera gedreht, das Schwanken bleibt
    // also im Körper — genau der Unterschied zwischen "Drehteller" und "lebt".
    const tS = performance.now() / 1000;
    vrm.scene.rotation.y = Math.sin(tS / 2.4) * .045 + Math.sin(tS / 1.13 + .8) * .018;
    vrm.scene.rotation.z = Math.sin(tS / 3.7 + 1.4) * .006;
    vrm.scene.position.x = Math.sin(tS / 3.1 + .4) * .006;
    vrm.scene.rotation.x = Math.sin(tS / 4.3 + .2) * .004;
    // idle breathing: tiny sway around the grounded position (y=0),
    // never an absolute height — that broke framing on offset models
    vrm.scene.position.y = Math.sin(Date.now() / 1900) * .008;
    // Blick nachführen, NACHDEM das Schwanken gesetzt ist: der Körper bewegt
    // sich, der Kopf bleibt auf der Kamera (sonst wandert der Blick mit).
    this._updateGaze(1 / 60);
    vrm.update(1 / 60);
    this.renderer.render(this.scene, this.camera);
  }
};

// auto-bind analyser to the shared audio element once it plays
document.addEventListener('play', e => {
  if (e.target.id === 'audioOut' && window._ftEngine) window._ftEngine.bindAudio(e.target);
}, true);
window._ftEngine = null;
const _origInit = FacetimeEngine.prototype.init;
FacetimeEngine.prototype.init = function () { window._ftEngine = this; return _origInit.call(this); };
