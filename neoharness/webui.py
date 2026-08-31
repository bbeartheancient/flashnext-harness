"""Built-in test web UI for `neoharness serve` (option 1: zero dependencies).

A single self-contained page served at GET /ui/. It talks to the same
server's OpenAI-compatible endpoint (SSE streaming) and the small JSON
inspection APIs (/api/state, /api/skills). Plain HTML/JS, no build step,
no external assets — safe to expose on the LAN behind a --host 0.0.0.0
drop-in for testing.
"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeoHarness</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0b0e14; --panel:#141a26; --fg:#d5dbe7; --dim:#7d8698;
          --acc:#f2a33c; --ok:#4caf50; --bad:#e05252; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace, Menlo, Consolas, monospace; }
  header { padding:10px 16px; background:var(--panel); border-bottom:1px solid #232c3d;
           display:flex; gap:12px; align-items:baseline; }
  header h1 { font-size:15px; margin:0; color:var(--acc); letter-spacing:1px; }
  header span { color:var(--dim); font-size:12px; }
  main { display:grid; grid-template-columns: 2fr 1fr; gap:10px; padding:10px; }
  .panel { background:var(--panel); border:1px solid #232c3d; border-radius:6px;
           padding:10px; min-height:100px; }
  .panel h2 { margin:0 0 8px; font-size:12px; color:var(--dim);
              text-transform:uppercase; letter-spacing:1px; }
  #goal { width:100%; background:#0d121c; color:var(--fg); border:1px solid #2a3550;
          border-radius:4px; padding:8px; font:inherit; resize:vertical; }
  button { background:var(--acc); color:#111; border:0; border-radius:4px;
           padding:7px 18px; font:inherit; font-weight:bold; cursor:pointer; }
  button:disabled { opacity:.4; cursor:wait; }
  #stream { white-space:pre-wrap; word-break:break-word; max-height:55vh;
            overflow-y:auto; margin-top:8px; }
  #stream .done-line { color:var(--ok); }
  #stream .err { color:var(--bad); }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; font-size:12px; }
  #status { font-size:12px; color:var(--dim); margin-left:10px; }
  .skills div { padding:2px 0; border-bottom:1px dotted #232c3d; font-size:12px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>NEOHARNESS</h1>
  <span>BPE agentic endpoint &middot; duty LLM &middot; XYZ n-gram buffer</span>
  <span id="status"></span>
</header>
<main>
  <section class="panel">
    <h2>Goal</h2>
    <textarea id="goal" rows="3" placeholder="e.g. Read notes.txt and write a summary to summary.txt, then finish."></textarea>
    <div style="margin-top:8px">
      <button id="run">Run goal</button>
      <span id="status2"></span>
    </div>
    <div id="stream" class="panel" style="background:#0d121c"></div>
  </section>
  <section>
    <div class="panel"><h2>Harness state (BPE)</h2><pre id="bpe">loading…</pre></div>
    <div class="panel skills" style="margin-top:10px"><h2>Skills</h2><div id="skills">loading…</div></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);

async function refreshPanels() {
  try {
    const s = await (await fetch('/api/state')).json();
    $('bpe').textContent = s.render;
  } catch (e) { $('bpe').textContent = 'state unavailable: ' + e; }
  try {
    const k = await (await fetch('/api/skills')).json();
    $('skills').innerHTML = k.skills.length
      ? k.skills.map(x => `<div><b>${x.name}</b> — ${x.description}</div>`).join('')
      : '<div>(no skills yet)</div>';
  } catch (e) { $('skills').textContent = 'skills unavailable: ' + e; }
}

async function runGoal() {
  const goal = $('goal').value.trim();
  if (!goal) return;
  $('run').disabled = true;
  const st = $('stream');
  st.innerHTML = '';
  $('status2').textContent = 'running…';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'neoharness', stream: true,
                            messages: [{role: 'user', content: goal}]})
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {
        const evt = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (evt.trimStart().startsWith(':')) {
          $('status2').textContent = evt.replace(':', '').trim() + ' ' +
            new Date().toLocaleTimeString();
          continue;
        }
        const line = evt.split('\\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        try {
          const delta = JSON.parse(data).choices[0].delta.content || '';
          const div = document.createElement('div');
          if (delta.includes('solved=True')) div.className = 'done-line';
          if (delta.includes('error')) div.className = 'err';
          div.textContent = delta;
          st.appendChild(div);
          st.scrollTop = st.scrollHeight;
        } catch (e) { /* partial chunk */ }
      }
    }
    $('status2').textContent = 'done';
  } catch (e) {
    st.innerHTML += `<div class="err">request failed: ${e}</div>`;
    $('status2').textContent = 'failed';
  }
  $('run').disabled = false;
  refreshPanels();
}

$('run').onclick = runGoal;
$('goal').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runGoal();
});
fetch('/health').then(r => r.json()).then(h => {
  $('status').textContent = h.ok ? 'online' : 'degraded';
}).catch(() => { $('status').textContent = 'offline'; });
refreshPanels();
</script>
</body>
</html>
"""
