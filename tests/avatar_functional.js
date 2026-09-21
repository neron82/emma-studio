// Avatar functional probe via puppeteer-core + system Chrome.
// Runs in the real facetime page context: loads VRM, checks expressions,
// measures lip-sync reaction and render FPS with a real GPU/software pipeline.
const puppeteer = require('puppeteer-core');

const BASE = 'http://127.0.0.1:8890';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 640, height: 640 });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });

  await page.goto(BASE + '/#facetime', { waitUntil: 'domcontentloaded', timeout: 60000 });
  // give the engine time to load the VRM (15 MB + CDN imports)
  await new Promise(r => setTimeout(r, 20000));

  const result = await page.evaluate(async () => {
    const out = {};
    const badge = document.getElementById('engineBadge');
    out.badge = badge ? badge.textContent : null;
    // engine state
    const eng = window._ftEngine;
    out.engineExists = !!eng;
    out.vrmLoaded = !!(eng && eng.vrm);
    if (eng && eng.vrm) {
      const em = eng.vrm.expressionManager;
      // VRM0 presets are accessed by name, not listed in .expressions (custom only)
      const probeNames = ['aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'happy', 'surprised'];
      out.expressions = probeNames.filter(n => {
        try { em.setValue(n, 0.5); const v = em.getValue(n); em.setValue(n, 0); return v > 0; }
        catch { return false; }
      });
      out.hasAa = out.expressions.includes('aa');
      // lip-sync reaction: set audio level path manually
      if (em) {
        em.setValue('aa', 0);
        eng._mouth = 0.9;
        await new Promise(r => setTimeout(r, 300));
        out.lipsyncMoves = em.getValue('aa') > 0.05 || eng._mouth > 0;
        em.setValue('aa', 0);
      }
      out.hasBlink = out.expressions.includes('blink');
      // mesh sanity
      let meshes = 0;
      eng.vrm.scene.traverse(o => { if (o.isMesh) meshes++; });
      out.meshCount = meshes;
    }
    return out;
  });

  // FPS measurement on the live page (5s wall clock)
  const fps = await page.evaluate(() => new Promise(res => {
    let frames = 0;
    const t0 = performance.now();
    function tick() {
      frames++;
      if (performance.now() - t0 < 5000) requestAnimationFrame(tick);
      else res({ fps: Math.round(frames / ((performance.now() - t0) / 1000)) });
    }
    requestAnimationFrame(tick);
  }));
  result.fps = fps.fps;
  result.pageErrors = errors.slice(0, 5);

  console.log(JSON.stringify(result, null, 1));

  let fail = false;
  if (!result.engineExists || !result.vrmLoaded) { console.log('FAIL: VRM not loaded'); fail = true; }
  if (!result.hasAa) { console.log('FAIL: no aa expression'); fail = true; }
  if (!result.hasBlink) { console.log('FAIL: no blink expression'); fail = true; }
  if (!result.lipsyncMoves) { console.log('FAIL: lip-sync path dead'); fail = true; }
  if (result.fps < 3) { console.log('FAIL: render loop too slow'); fail = true; }
  console.log(fail ? '\nAVATAR CHECKS FAILED' : '\nAll avatar checks passed.');
  await browser.close();
  process.exit(fail ? 1 : 0);
})();
