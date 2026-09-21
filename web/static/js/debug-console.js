/* Emma Studio debug console — collapsible overlay showing session events,
   errors (window.onerror + unhandled rejections), and service state.
   Toggle: the ⚙ button bottom-left, or Ctrl+` . */
(() => {
  const lines = [];
  let open = false, el = null, logEl = null, btn = null;

  function stamp() {
    const d = new Date();
    return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0');
  }

  function render() {
    if (!logEl) return;
    logEl.innerHTML = lines.slice(-200).map(l =>
      `<div class="dbg-line ${l.cls}"><span class="dbg-t">${l.t}</span>${l.msg}</div>`).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }

  function log(msg, cls = 'info') {
    lines.push({ t: stamp(), msg: String(msg).replace(/</g, '&lt;'), cls });
    if (lines.length > 400) lines.splice(0, lines.length - 400);
    render();
  }

  function build() {
    el = document.createElement('div');
    el.id = 'debugConsole';
    el.innerHTML = `
      <div class="dbg-head">
        <span>Debug</span>
        <button class="dbg-clear" title="leeren">clear</button>
        <button class="dbg-close" title="zu">×</button>
      </div>
      <div class="dbg-log"></div>`;
    document.body.appendChild(el);
    logEl = el.querySelector('.dbg-log');
    el.querySelector('.dbg-close').onclick = () => toggle(false);
    el.querySelector('.dbg-clear').onclick = () => { lines.length = 0; render(); };
  }

  function toggle(force) {
    if (!el) { build(); }
    open = force !== undefined ? force : !open;
    el.classList.toggle('open', open);
    if (open) render();
  }

  // global error capture
  window.addEventListener('error', e => log('❌ ' + e.message + (e.filename ? ` (${(e.filename||'').split('/').pop()}:${e.lineno})` : ''), 'err'));
  window.addEventListener('unhandledrejection', e => {
    const r = e.reason;
    log('⚠️ unhandled rejection: ' + (r && r.message ? r.message : String(r)), 'err');
  });

  // public API
  window.DebugConsole = {
    log,
    error: m => log(m, 'err'),
    ok: m => log(m, 'ok'),
    toggle,
  };

  // floating toggle button
  document.addEventListener('DOMContentLoaded', () => {
    btn = document.createElement('button');
    btn.id = 'dbgBtn';
    btn.title = 'Debug-Konsole (Strg+`)';
    btn.textContent = '⚙';
    btn.onclick = () => toggle();
    document.body.appendChild(btn);
    log('Studio geladen', 'ok');
  });
  document.addEventListener('keydown', e => {
    if (e.key === '`' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); toggle(); }
  });
})();
