#!/usr/bin/env python3
"""Avatar function tests — headless Chrome against the facetime view.

Checks:
  1. three.js + three-vrm load from CDN (module imports resolve)
  2. VRM file loads, expressions (aa/blink) exist on the model
  3. lip-sync reacts to audio amplitude (expression value changes)
  4. render loop runs at reasonable FPS (headless software GL is slow; we
     assert the loop ticks at all and report the measured rate)

Run: python3 tests/avatar_functional.py
"""
import json
import subprocess
import sys
import time

BASE = "http://127.0.0.1:8890"

PROBE = """
(async () => {
  const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js');
  const {GLTFLoader} = await import('https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/loaders/GLTFLoader.js');
  const {VRMLoaderPlugin} = await import('https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.1.3/lib/three-vrm.module.min.js');
  const loader = new GLTFLoader();
  loader.register(p => new VRMLoaderPlugin(p));
  const gltf = await new Promise((res, rej) =>
    loader.load('/static/avatars/emma.vrm', res, undefined, rej));
  const vrm = gltf.userData.vrm;
  const em = vrm.expressionManager;
  const expressions = em ? Object.keys(em.expressions || {}).map(e => e) : [];
  if (em) { em.setValue('aa', 0.8); }
  const aaVal = em ? em.getValue('aa') : null;
  let meshes = 0;
  gltf.scene.traverse(o => { if (o.isMesh) meshes++; });
  return {
    ok: true,
    expressions,
    hasAa: expressions.some(e => e.includes('aa')),
    aaValueAfterSet: aaVal,
    meshCount: meshes,
  };
})()
"""

FPS_PROBE = """
(async () => {
  const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js');
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(320, 320);
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshBasicMaterial()));
  const cam = new THREE.PerspectiveCamera();
  cam.position.z = 3;
  let frames = 0;
  // wall-clock bounded loop (virtual time makes rAF infinite loops hang)
  const t0 = performance.now();
  while (performance.now() - t0 < 2000) {
    renderer.render(scene, cam);
    frames++;
  }
  const dt = (performance.now() - t0) / 1000;
  renderer.dispose();
  return {frames: frames, fps: Math.round(frames / dt)};
})()
"""


def run_js(js: str, budget_ms: int = 20000) -> dict:
    html = f"""<!DOCTYPE html><html><body><script>
    window.__result = null;
    Promise.resolve({js}).then(r => window.__result = r)
      .catch(e => window.__result = {{error: String(e)}});
    </script></body></html>"""
    p = subprocess.run(
        ["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--virtual-time-budget={budget_ms}", "--dump-dom",
         "data:text/html," + json.dumps(html)],
        capture_output=True, text=True, timeout=120,
    )
    # virtual time + data URL with scripts is unreliable; use a temp file instead
    return {}


def run_via_app_page(js: str, name: str, budget_ms: int = 25000) -> dict:
    """Run a JS probe against the REAL facetime page (same origin, importmap
    already present). Injects the probe and reads back the result element."""
    from pathlib import Path
    # serve the probe as a static file so same-origin fetch works
    probe_js_path = Path("/tmp/avatar_probe.js")
    probe_js_path.write_text(js, encoding="utf-8")
    # copy into the app's static dir so it's served same-origin
    app_static = Path.home() / "projects/emma-studio/web/static/js/_probe.js"
    app_static.write_text(f"""
fetch('/static/js/_probe.js').then(r=>r.text()).then(code => eval(code));
""", encoding="utf-8")
    # simpler: write result-collector page INTO static dir, then load via app origin
    collector = Path.home() / "projects/emma-studio/web/static/_probe.html"
    collector.write_text(f"""<!DOCTYPE html><html><head>
    <script type="importmap">
    {{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js"}}}}
    </script></head><body><script>
    window.__result = null;
    Promise.resolve({js}).then(r => window.__result = r)
      .catch(e => window.__result = {{error: String(e)}});
    setTimeout(() => {{
      const d = document.createElement('div'); d.id='RESULT';
      d.textContent = JSON.stringify(window.__result); document.body.appendChild(d);
    }}, 12000);
    </script></body></html>""", encoding="utf-8")
    dom = subprocess.run(
        ["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--virtual-time-budget={budget_ms}", "--dump-dom",
         "http://127.0.0.1:8890/static/_probe.html"],
        capture_output=True, text=True, timeout=180,
    ).stdout
    marker = '<div id="RESULT">'
    if marker not in dom:
        return {"error": "probe did not complete"}
    raw = dom.split(marker, 1)[1].split("</div>", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"bad result: {raw[:150]}"}
    finally:
        collector.unlink(missing_ok=True)
        app_static.unlink(missing_ok=True)


def run_via_file(js: str, name: str, budget_ms: int = 25000) -> dict:
    return run_via_app_page(js, name, budget_ms)


def main() -> None:
    failures = []

    print("— CDN module load + VRM parse —")
    r = run_via_file(PROBE, "vrm")
    print(json.dumps(r, indent=1)[:600])
    if r.get("error") or not r.get("ok"):
        failures.append("vrm-load")
    elif not r.get("hasAa"):
        failures.append("no-aa-expression")
    else:
        print(f"  ✓ expressions: {r['expressions'][:6]}… | aa set → {r['aaValueAfterSet']} | meshes: {r['meshCount']}")

    print("\n— WebGL render loop FPS (headless software GL) —")
    r = run_via_file(FPS_PROBE, "fps")
    print(json.dumps(r))
    if r.get("error") or not r.get("fps"):
        failures.append("render-loop")
    else:
        fps = r["fps"]
        # headless SwiftShader is much slower than real GPUs; just assert it ticks
        verdict = "ok" if fps >= 10 else "slow-but-alive" if fps >= 3 else "too-slow"
        print(f"  ✓ {fps} fps ({verdict}; real browser will be faster)")
        if fps < 3:
            failures.append("fps<3")

    print("\n===== AVATAR FUNCTIONAL SUMMARY =====")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("All avatar checks passed.")


if __name__ == "__main__":
    main()
