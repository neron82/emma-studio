// E2E lip-sync test: play a REAL generated audio chunk in the facetime page
// and verify the aa-expression tracks the audio amplitude.
const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome', headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 640, height: 640 });
  await page.goto('http://127.0.0.1:8890/#facetime', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await new Promise(r => setTimeout(r, 18000)); // VRM load

  // fetch a real filler clip URL from the API
  const clip = await page.evaluate(async () => {
    const r = await fetch('/api/fillers/pick', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Erzähl mir eine spannende Geschichte!', persona_id: 'bb8b992fc5b1' }),
    });
    return r.json();
  });
  if (!clip.url) { console.log('FAIL: no filler clip'); process.exit(1); }

  // create an audio element, bind it like the answer pipeline does, play, sample aa
  const samples = await page.evaluate(async (url) => {
    const eng = window._ftEngine;
    const em = eng.vrm.expressionManager;
    const el = document.getElementById('audioOut') || document.createElement('audio');
    el.id = 'audioOut'; el.style.display = 'none'; document.body.appendChild(el);
    eng.bindAudio(el);

    el.src = url;
    await el.play();
    // sample aa + raw level over 2s while audio plays
    const samples = [];
    const t0 = performance.now();
    await new Promise(res => {
      const iv = setInterval(() => {
        samples.push({ t: Math.round(performance.now() - t0), level: +eng.audioLevel().toFixed(3), aa: +em.getValue('aa').toFixed(3) });
        if (el.paused || performance.now() - t0 > 2500) { clearInterval(iv); res(); }
      }, 100);
    });
    el.pause();
    return { samples, durationPlayed: samples.length };
  }, 'http://127.0.0.1:8890' + clip.url);

  const levels = samples.samples.map(s => s.level);
  const aas = samples.samples.map(s => s.aa);
  const maxLevel = Math.max(...levels);
  const maxAa = Math.max(...aas);
  const moved = samples.samples.filter(s => s.aa > 0.05).length;

  console.log(`clip: ${clip.phrase} (${clip.duration_s}s) | samples: ${samples.durationPlayed}`);
  console.log(`max audio level: ${maxLevel} | max aa: ${maxAa} | frames with aa>0.05: ${moved}`);

  let fail = false;
  if (maxLevel <= 0.01) { console.log('FAIL: analyser gets no signal (bindAudio broken)'); fail = true; }
  if (maxAa <= 0.03) { console.log('FAIL: aa expression never moved — lip-sync dead'); fail = true; }
  if (moved < 3) { console.log('FAIL: lip-sync barely moved'); fail = true; }
  console.log(fail ? '\nLIP-SYNC E2E FAILED' : '\nLip-sync E2E: audio amplitude drives the avatar mouth. ✓');
  await browser.close();
  process.exit(fail ? 1 : 0);
})();
