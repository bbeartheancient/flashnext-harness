/* Modular LCARS shell: one department, one function, only that I/O + viewport. */
(() => {
  const PALETTE = ["nav-1", "nav-2", "nav-3", "nav-4"];
  const STORE = "fabric-shell-v5";
  const THEME_STORE = "fabric-theme";
  const TILE_CAP = 16;
  const WALL = 4;
  const THEMES = [
    { id: "classic", label: "classic", swatch: "#baa4e5" },
    { id: "voyager", label: "voyager", swatch: "#55a7ff" },
    { id: "nemesis", label: "nemesis", swatch: "#2266ff" },
    { id: "lower-decks", label: "lower decks", swatch: "#ff7700" },
    { id: "picard", label: "picard", swatch: "#37a6d1" },
    { id: "lcars", label: "drexler", swatch: "#ec943a" },
    { id: "amber", label: "amber", swatch: "#ffb000" },
    { id: "green", label: "green", swatch: "#33ff66" },
    { id: "plasma", label: "plasma", swatch: "#ff4a1f" },
    { id: "mono", label: "mono", swatch: "#d9d9d9" },
    { id: "vga", label: "vga", swatch: "#5555ff" },
  ];

  const $ = (id) => document.getElementById(id);
  const termEl = $("term");
  const ioEl = $("io");
  const liveCol = $("liveCol");
  const ioRight = $("ioRight");
  // output parameters (auto-route figure emissions to viewports)
  let outRouting = {};
  try { outRouting = JSON.parse(localStorage.getItem("mfx-out-routing")
       || "{}") || {}; } catch (_) { outRouting = {}; }
  // matrix-wide display shader (App-Mode settings popup; applies to
  // the whole viewport matrix, not per tile)
  let shader = { scanlines: false, glow: false, bright: 1 };
  try {
    shader = { ...shader,
      ...JSON.parse(localStorage.getItem("mfx-shader") || "{}") };
  } catch (_) { /* private */ }
  let appEdit = false;          // build/edit switch (TL corner)
  let lastSelTile = null;       // tile picked in edit mode (library target)
  let wireDraft = null;         // live patch-cable drag {from, ghost}
  const ctxNav = $("contextNav");
  const tileDeck = $("tileDeck");
  const lblDept = $("lblDept");
  const lblTask = $("lblTask");
  const lblLive = $("lblLive");
  const cmdForm = $("cmd");
  const cmdIn = $("cmdIn");
  const maxBox = $("maxBox");
  const maxLeft = $("maxLeft");
  const maxRight = $("maxRight");
  const maxCenter = $("maxCenter");
  const termStack = $("termStack");
  const dutyLog = $("dutyLog");
  const settings = $("settings");
  const boxCol = $("boxCol");
  const page = $("column-3");
  let lastQuery = "";
  let kitCache = null;
  let clarifyState = null;
  let slashCmds = [];
  let slashIdx = 0;
  let settingsTab = "library";
  const wiz = { dept: "science", inputs: ["place"], outputs: ["heatmap"], viewports: 1,
  sigSource: "sensor", sigViz: "series" };

  let depts = [];
  let aliases = {};
  let catalogFns = [];
  let dept = null;
  let fn = null;
  let live = [];
  let mode = "split"; // split | term | viz
  let termPane = 0;
  let wallScale = 4;
  let soloId = null;
  let tileSeq = 0;
  const home = {};
  // per-side tile counts: 1 / 2x2 / 4x4 (16-slot viewport matrix)
  const WALL_SCALES = [1, 2, 4];
  // Rack grid — see Rack/include/app/common.hpp RACK_GRID_SIZE
  // Every module and param widget snaps to RACK_GRID (15×380). We
  // reuse the same grid inside each tile's control layer so controls
  // layout like Rack params do (box.pos / box.size, helpers
  // createParam/createWidget, ModuleWidget grid snapping).
  const RACK_GRID = 15;
  const RACK_KNOB_MIN_ANGLE = -0.83 * Math.PI;
  const RACK_KNOB_MAX_ANGLE = 0.83 * Math.PI;
  function rackSnap(v) { return Math.round(v / RACK_GRID) * RACK_GRID; }
  function mm2px(mm) { return mm * 96 / 25.4; }

  // framebuffer geometry standard (displayarea.png/svg): each tile is
  // a 632x396 framebuffer; the 4x4 matrix is 2528x1584 inside a
  // 3200x1800 fullscreen. The ratio is preserved at every screen
  // resolution — tiles scale, never stretch.
  const FB_TILE = { w: 632, h: 396 };
  const FB_MATRIX = { w: FB_TILE.w * 4, h: FB_TILE.h * 4 };

  function beep(id) {
    const a = $(id || "audio2");
    if (!a) return;
    try { a.currentTime = 0; a.play().catch(() => {}); } catch (_) { /* autoplay */ }
  }

  function clock() {
    const d = new Date();
    $("clock").textContent = d.toISOString().slice(11, 19);
  }
  setInterval(clock, 1000);
  clock();

  function persist() {
    try {
      localStorage.setItem(STORE, JSON.stringify({
        ext: dept && dept.ext, fn: fn && fn.id, mode,
      }));
    } catch (_) { /* private mode */ }
  }

  function restore() {
    try { return JSON.parse(localStorage.getItem(STORE) || "{}"); }
    catch (_) { return {}; }
  }

  // matrix display shader: whole-matrix effects (settings popup)
  function applyShader() {
    if (!tileDeck) return;
    tileDeck.classList.toggle("fx-scanlines", !!shader.scanlines);
    tileDeck.classList.toggle("fx-glow", !!shader.glow);
    tileDeck.style.setProperty("--fx-bright", String(shader.bright ?? 1));
    try {
      localStorage.setItem("mfx-shader", JSON.stringify(shader));
    } catch (_) { /* private */ }
  }

  // stream-i/o apps (MicroFX signal programs) use the bus rails —
  // no v1 parameter form in App Mode
  function isStreamApp(f) {
    if (!f) return false;
    const io = f.io || {};
    return ["ins", "outs", "controls", "tiles", "triggers", "wires"]
      .some((k) => Array.isArray(io[k]) && io[k].length)
      || f.kernel === "signal";
  }

  function shellLine(data, id) {
    if (data == null) return "";
    if (typeof data === "string") return data;
    if (data.error) return "err " + data.error;
    if (id === "hadamard") {
      return `ok  H(${data.n}) ${data.method}  det ${data.det_log10} / bound ${data.det_bound_log10}`;
    }
    if (data.reply) return data.reply;
    if (Array.isArray(data.goals)) {
      return data.goals.length
        ? data.goals.map((g) => (g.status || "") + " " + (g.text || "")).join(" · ")
        : "no goals";
    }
    if (data.result != null) return String(data.result);
    if (data.figure && data.figure.caption) return data.figure.caption;
    const skip = new Set(["figure", "grid", "dist_m", "elev_m", "azimuths_deg",
      "power_norm", "samples", "heightmap"]);
    const bits = [];
    for (const [k, v] of Object.entries(data)) {
      if (skip.has(k) || v == null || typeof v === "object") continue;
      bits.push(`${k}=${v}`);
    }
    return bits.slice(0, 8).join("  ") || "ok";
  }

  function log(who, payload, err) {
    if (termEl.firstChild && termEl.firstChild.nodeType === 3) {
      termEl.textContent = "";
    }
    const line = document.createElement("div");
    if (err) line.className = "err";
    if (who === "cmd" || who === ">") {
      const p = document.createElement("span");
      p.className = "prompt";
      p.textContent = "> ";
      line.appendChild(p);
      line.appendChild(document.createTextNode(String(payload)));
    } else {
      // Show a lane tag for non-standard / non-duty lanes
      const LANE_MAP = {
        bicameral: "biceph", swarm: "swarm", trace: "trace",
        improve: "improve", domain: "domain", shard: "shard",
        atomic: "atomic", teach: "teach", buffer: "buffer",
        roundtrip: "rtrip", optimize: "opt",
      };
      const badge = LANE_MAP[who];
      if (badge) {
        line.appendChild(laneTag(badge));
      }
      const t = document.createElement("span");
      t.className = "lane-text";
      t.textContent = shellLine(payload, who).slice(0, 1800);
      line.appendChild(t);
    }
    termEl.appendChild(line);
    while (termEl.childElementCount > 80) termEl.removeChild(termEl.firstChild);
    scrollBottom(termEl);
  }

  function hop(n) {
    const s = document.createElement("span");
    s.className = "hop";
    s.textContent = n;
    return s;
  }

  function rememberHomes() {
    if (!home.ok) {
      home.boxCol = boxCol && boxCol.parentNode;
      home.boxColNext = boxCol && boxCol.nextSibling;
      home.tiles = tileDeck && tileDeck.parentNode;
      home.tilesNext = tileDeck && tileDeck.nextSibling;
      home.right = ioRight && ioRight.parentNode;
      home.rightNext = ioRight && ioRight.nextSibling;
      home.term = termStack && termStack.parentNode;
      home.termNext = termStack && termStack.nextSibling;
      home.ok = true;
    }
  }

  function putBack(el, parent, next) {
    if (!el || !parent) return;
    if (next && next.parentNode === parent) parent.insertBefore(el, next);
    else parent.appendChild(el);
  }

  function leaveMax() {
    if (!home.ok) return;
    putBack(boxCol, home.boxCol, home.boxColNext);
    putBack(tileDeck, home.tiles, home.tilesNext);
    putBack(ioRight, home.right, home.rightNext);
    if (ioRight) ioRight.hidden = true;
    putBack(termStack, home.term, home.termNext);
    if (termStack) termStack.hidden = false;
    if (dutyLog) dutyLog.hidden = true;
    if (settings) settings.hidden = true;
    if (page) page.hidden = false;
    if (maxBox) maxBox.hidden = true;
    requestAnimationFrame(() => reflowWall());
  }

  function showTermPane() {
    if (settings) settings.hidden = true;
    if (termStack) termStack.hidden = termPane !== 0;
    if (dutyLog) dutyLog.hidden = termPane !== 1;
    const title = $("maxTitle");
    if (title) title.textContent = "";
  }

  function cycleTerm(dir) {
    termPane = (termPane + dir + 2) % 2;
    showTermPane();
  }

  function wallTiles() {
    if (!tileDeck) return [];
    return [...tileDeck.querySelectorAll(".viz-tile:not([data-placeholder])")];
  }

  // viewports: every visible tile is a display-driver slot on the
  // 4x4 matrix wall (multimodal: data, media, 3D, charts, scopes)
  function stampViewports() {
    let n = 0;
    for (const t of wallTiles()) {
      const visible = t.offsetWidth > 0 && t.offsetHeight > 0;
      t.dataset.vp = visible ? `vp${++n}` : "";
      t.title = visible ? `viewport ${n} · ${t.dataset.app
        || t.dataset.fn || "display"}` : "";
    }
  }

  function cycleLive(dir) {
    const all = wallTiles();
    if (!all.length) return;
    const ids = all.map((t) => t.dataset.tile);
    let i = ids.indexOf(soloId);
    if (i < 0) i = ids.indexOf((all.find((t) => t.dataset.fn === (fn && fn.id)) || {}).dataset.tile);
    if (i < 0) i = 0;
    soloTile(ids[(i + dir + ids.length) % ids.length]);
  }

  function enterMax(kind) {
    rememberHomes();
    if (page) page.hidden = true;
    if (!maxBox) return;
    maxBox.hidden = false;
    maxBox.classList.toggle("kind-term", kind === "term");
    maxBox.classList.toggle("kind-viz", kind === "viz");
    if (kind === "term") {
      if (termStack && maxCenter) maxCenter.insertBefore(termStack, dutyLog);
      showTermPane();
    } else {
      if (boxCol && maxLeft) maxLeft.appendChild(boxCol);
      if (tileDeck && maxCenter) maxCenter.appendChild(tileDeck);
      if (ioRight && maxRight) {
        maxRight.appendChild(ioRight);
        ioRight.hidden = false;
      }
      syncVizRails();
      paintMaxTitle();
      requestAnimationFrame(() => reflowWall());
    }
  }

  function setMode(next) {
    const dest = (next === mode) ? "split" : next;
    closePopup();
    if (appEdit && dest === "split") setEditMode(false);
    leaveMax();
    mode = dest;
    document.body.classList.toggle("mode-term", mode === "term");
    document.body.classList.toggle("mode-viz", mode === "viz");
    const t = $("ctlTerm"), v = $("ctlViz");
    if (t) t.classList.toggle("active", mode === "term");
    if (v) v.classList.toggle("active", mode === "viz");
    if (mode !== "split") enterMax(mode);
    persist();
  }

  const wallObs = new ResizeObserver((entries) => {
    for (const ent of entries) {
      const tile = ent.target;
      if (tile._fig && tile._fig.type !== "html") {
        const canvas = tile.querySelector("canvas");
        if (canvas) fitCanvas(canvas, tile._fig, tile);
      }
    }
  });

  function resetTiles() {
    for (const t of tileDeck.querySelectorAll(".viz-tile")) {
      try { wallObs.unobserve(t); } catch (_) { /* gone */ }
    }
    tileDeck.innerHTML = "";
    live = [];
    soloId = null;
    const tile = makeTile("standby");
    tile.dataset.placeholder = "1";
    tileDeck.appendChild(tile);
    watchTile(tile);
    reflowWall();
    renderLiveCol();
  }

  function shapeOf(fig) {
    if (!fig) return "square";
    if (fig.type === "matrix" || fig.type === "heatmap" ||
        fig.type === "polar" || fig.type === "space3d") {
      return "square";
    }
    return "wide";
  }

  function drawSeries(canvas, fig) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const pad = 30;
    const pts = (fig.series || []).map((p) => p);
    if (!pts.length) {
      ctx.fillStyle = "#888";
      ctx.font = "12px Antonio, sans-serif";
      ctx.fillText("no samples yet", pad, h / 2);
      return;
    }
    const t0 = pts[0][0], t1 = Math.max(pts[pts.length - 1][0], t0 + 1);
    let lo = Infinity, hi = -Infinity;
    for (const [, v] of pts) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) { lo -= 1; hi += 1; }
    const X = (t) => pad + ((t - t0) / (t1 - t0)) * (w - pad * 2);
    const Y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - pad * 2);
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    ctx.strokeRect(pad, pad, w - pad * 2, h - pad * 2);
    ctx.strokeStyle = fig.color || "#eb943a";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach(([t, v], i) => i ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)));
    ctx.stroke();
    const last = pts[pts.length - 1][1];
    ctx.fillStyle = "#fff";
    ctx.font = "bold 20px Antonio, sans-serif";
    ctx.fillText(String(last), pad + 4, pad + 18);
    ctx.fillStyle = "#baa4e5";
    ctx.font = "11px Antonio, sans-serif";
    ctx.fillText(fig.ylabel || "", pad + 4, h - 8);
    ctx.fillText(fig.caption || "", w - pad - ctx.measureText(fig.caption || "").width - 4, h - 8);
  }

  const liveTiles = new Set();
  function registerLive(tile, fig) {
    if (!fig || !fig.live) { liveTiles.delete(tile); return; }
    tile._live = { url: fig.live.url, key: fig.live.key,
                   interval: fig.live.interval || 5 };
    liveTiles.add(tile);
  }
  setInterval(() => {
    for (const tile of [...liveTiles]) {
      if (!tile.isConnected) { liveTiles.delete(tile); continue; }
      fetch(tile._live.url).then((r) => r.json()).then((data) => {
        const conv = data.series ? { type: "series", ...data } : figureFrom(data);
        if (!conv) return;
        tile._fig = conv;
        const canvas = tile.querySelector("canvas");
        if (canvas) fitCanvas(canvas, conv);
      }).catch(() => {});
    }
  }, 5000);

  function spanOfScale() {
    return WALL / wallScale; // wallScale 4 -> 16 slots at span 1
  }

  function bumpScale(dir) {
    const i = WALL_SCALES.indexOf(wallScale);
    const next = WALL_SCALES[Math.max(0, Math.min(WALL_SCALES.length - 1, i + dir))];
    if (next === wallScale) return;
    wallScale = next;
    soloId = null;
    beep("audio2");
    reflowWall();
    renderLiveCol();
  }

  function soloTile(id) {
    const all = wallTiles();
    const tile = all.find((t) => t.dataset.tile === id);
    if (!tile) return;
    soloId = (soloId === id) ? null : id;
    if (tile.dataset.fn) applyFunction(tile.dataset.fn);
    document.body.classList.toggle("solo-viz", !!soloId);
    beep("audio2");
    reflowWall();
    renderLiveCol();
    stampViewports();
  }

  function closeTile(id) {
    const tile = wallTiles().find((t) => t.dataset.tile === id);
    if (!tile) return;
    try { wallObs.unobserve(tile); } catch (_) { /* gone */ }
    tile.remove();
    if (soloId === id) soloId = null;
    if (!wallTiles().length) resetTiles();
    else {
      reflowWall();
      renderLiveCol();
    }
    beep("audio3");
  }

  function findSlot(cs, rs, used) {
    for (let r = 0; r <= WALL - rs; r++) {
      for (let c = 0; c <= WALL - cs; c++) {
        let ok = true;
        for (let dr = 0; dr < rs && ok; dr++) {
          for (let dc = 0; dc < cs && ok; dc++) {
            if (used[(r + dr) * WALL + (c + dc)]) ok = false;
          }
        }
        if (ok) return [c, r];
      }
    }
    return null;
  }

  function reflowWall() {
    if (!tileDeck) return;
    const all = [...tileDeck.querySelectorAll(".viz-tile")];
    const real = all.filter((t) => !t.dataset.placeholder);
    const slots = soloId ? 1 : wallScale * wallScale;
    let shown;
    if (!real.length) {
      shown = all;
    } else if (soloId) {
      shown = real.filter((t) => t.dataset.tile === soloId);
      if (!shown.length) shown = real.slice(-1);
    } else {
      shown = real.slice(-slots);
    }
    all.forEach((t) => { t.hidden = shown.indexOf(t) < 0; });
    // matrix fit: proportional to the 3200x1800 reference fullscreen
    // (tiles are exactly 632x396 there), capped by the deck's actual
    // fit so the matrix never overflows the chrome
    const refScale = Math.min(window.innerWidth / 3200,
                              window.innerHeight / 1800);
    const deckW = tileDeck.clientWidth || FB_MATRIX.w;
    const deckH = tileDeck.clientHeight || FB_MATRIX.h;
    const fitScale = Math.min(deckW / FB_MATRIX.w, deckH / FB_MATRIX.h);
    const fbScale = Math.max(0.1, Math.min(refScale, fitScale));
    const tileW = Math.max(40, Math.round(FB_TILE.w * fbScale));
    const tileH = Math.max(25, Math.round(FB_TILE.h * fbScale));
    tileDeck.style.gridTemplateColumns = `repeat(4, ${tileW}px)`;
    tileDeck.style.gridTemplateRows = `repeat(4, ${tileH}px)`;
    tileDeck.style.justifyContent = "center";
    tileDeck.style.alignContent = "center";
    const span = (soloId || shown.length <= 1) ? WALL : spanOfScale();
    const used = new Array(WALL * WALL).fill(false);
    shown.forEach((tile) => {
      let spanC = span, spanR = span;
      const want = Number((tile._fig && tile._fig.span)
        || (fn && fn.span) || 0);
      if (want >= 4) { spanC = WALL; spanR = WALL; }
      else if (want === 2) {
        spanC = Math.max(span, 2);
        spanR = Math.max(span, 2);
      }
      let pos = findSlot(spanC, spanR, used);
      if (!pos) {
        spanC = 1; spanR = 1;
        pos = findSlot(1, 1, used) || [0, 0];
      }
      const [c, r] = pos;
      tile.style.gridColumn = (c + 1) + " / span " + spanC;
      tile.style.gridRow = (r + 1) + " / span " + spanR;
      tile.hidden = false;
      for (let dr = 0; dr < spanR; dr++) {
        for (let dc = 0; dc < spanC; dc++) {
          used[(r + dr) * WALL + (c + dc)] = true;
        }
      }
    });
    requestAnimationFrame(() => {
      shown.forEach((tile) => {
        if (tile._fig && tile._fig.type !== "html") {
          const canvas = tile.querySelector("canvas");
          if (canvas) fitCanvas(canvas, tile._fig, tile);
        }
      });
      renderWires();
    });
  }

  function watchTile(tile) {
    if (!tile || tile._watched) return;
    tile._watched = true;
    wallObs.observe(tile);
    // click a visualizer tile -> expand it to fill the display area
    // (solo; other viewports + rails hidden). Click again to restore.
    const body = tile.querySelector(".vp-body");
    if (!body) return;
    let down = null;
    body.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY };
    });
    body.addEventListener("pointerup", (e) => {
      if (!down) return;
      const moved = Math.abs(e.clientX - down.x)
        + Math.abs(e.clientY - down.y);
      down = null;
      if (moved > 8) return;             // drag-rotate, not a click
      // control layers sit ABOVE the framebuffer: touching controls
      // manipulates them instead of maximizing the display tile
      if (e.target.closest(".vp-controls")
          || e.target.closest(".vp-expand")) return;
      // build/edit mode: clicking a matrix cell configures it
      // (wrapper i/o, aux channels, controls, chains) via popup
      if (appEdit && !tile.dataset.placeholder) {
        openTileConfig(tile, body);
        return;
      }
      // HID tiles (3D navigation) reserve clicks for gestures —
      // expansion is the explicit full-screen button only
      if (tile._fig && tile._fig.type === "space3d") return;
      soloTile(tile.dataset.tile);
      document.body.classList.toggle("solo-viz", !!soloId);
    });
  }

  function fitCanvas(canvas, fig, tile) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(32, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(32, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    // framebuffer standard: the tile canvas is the viewport's
    // framebuffer at its fraction of full output resolution — any
    // signal (charts, scopes, 3D, video, emulator frames) seeds it
    if (tile && canvas.tagName === "CANVAS") {
      tile._fb = tile._fb || {};
      tile._fb.canvas = canvas;
      tile._fb.w = w;
      tile._fb.h = h;
    }
    if (fig) {
      try {
        drawFigure(canvas, fig, tile);
      } catch (e) {
        console.warn("vp draw:", e.message);
      }
    }
  }

  // ---- framebuffer visualization presets -----------------------------
  // lcd: png-to-lcd-style 1-bit matrix (green-on-black cells)
  function drawLcd(canvas, fig) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0a0f0a";
    ctx.fillRect(0, 0, w, h);
    const grid = fig.grid || [];
    const rows = grid.length || 1;
    const cols = Math.max(...grid.map((r) => r.length), 1);
    const cell = Math.max(2, Math.floor(Math.min(w / cols, h / rows)));
    const ox = Math.floor((w - cell * cols) / 2);
    const oy = Math.floor((h - cell * rows) / 2);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < (grid[y] || []).length; x++) {
        const on = grid[y][x];
        ctx.fillStyle = on > 0 ? "#22c55e"
          : on < 0 ? "#7f1d1d" : "#050805";
        ctx.fillRect(ox + x * cell, oy + y * cell,
                     Math.max(1, cell - 1), Math.max(1, cell - 1));
      }
    }
  }

  // video: a media file seeded into the framebuffer (multimodal
  // viewport — the element decodes, the canvas presents)
  function drawVideo(canvas, fig, tile) {
    let vid = tile._vid;
    if (!vid || vid.dataset.src !== fig.src) {
      if (vid) { vid.pause(); vid.remove(); }
      vid = document.createElement("video");
      vid.dataset.src = fig.src || "";
      vid.src = fig.src || "";
      vid.muted = true;
      vid.loop = true;
      vid.playsInline = true;
      vid.style.display = "none";
      document.body.appendChild(vid);
      tile._vid = vid;
      vid.play().catch(() => { /* autoplay policy */ });
    }
    const ctx = canvas.getContext("2d");
    if (vid.readyState >= 2) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
    }
    if (tile._fb) tile._fb.video = vid;
  }

  function upsertFn(spec) {
    if (!spec || !spec.id) return;
    const i = catalogFns.findIndex((f) => f.id === spec.id);
    if (i >= 0) catalogFns[i] = spec;
    else catalogFns.push(spec);
  }

  function pushIo(tile) {
    const frame = tile && tile._frame;
    if (!frame || !frame.contentWindow) return;
    try {
      frame.contentWindow.postMessage(
        { type: "io", fields: { ...collect(), ...controlDefaults() } }, "*");
    } catch (_) { /* sandbox */ }
  }

  // initial values for shell-owned controls: programs read control
  // ports from frame one, so the manifest defaults ride the first
  // io push (fader/encoder defaults are already in real units)
  function controlDefaults() {
    const out = {};
    for (const c of (fn && fn.io && fn.io.controls) || []) {
      const port = String(c.name).toLowerCase();
      if (c.type === "fader" || c.type === "encoder") {
        const lo = c.min !== undefined ? c.min : 0;
        const hi = c.max !== undefined ? c.max : 1;
        out[port] = c.default !== undefined ? c.default
          : lo + 0.5 * (hi - lo);
      } else if (c.type === "xy") {
        out[port + "_x"] = 0.5;
        out[port + "_y"] = 0.5;
      } else if (c.type === "button") {
        out[port] = 0;
      }
    }
    return out;
  }

  // ---- app-level IO routing (MicroFX viewport matrix) ----------------
  // MicroFX app iframes emit {type:"app-out"} scalars and
  // {type:"app-figure"} payloads; figures land in their own deck tiles
  // keyed by app title, so any wall slot can show an app's output.
  function triples(flat) {
    const pts = [];
    for (let i = 0; i + 2 < flat.length; i += 3) {
      pts.push([flat[i], flat[i + 1], flat[i + 2]]);
    }
    return pts;
  }

  function upsertAppTile(key, fig, caption) {
    if (!key || !fig) return;
    let tile = tileDeck.querySelector(`[data-app="${CSS.escape(key)}"]`);
    const isNew = !tile;
    if (isNew) {
      tile = makeTile(caption || key);
      tile.dataset.app = key;
      if (!tile.dataset.tile) tile.dataset.tile = String(++tileSeq);
      tileDeck.appendChild(tile);
      while (tileDeck.querySelectorAll(".viz-tile").length > TILE_CAP) {
        const first = tileDeck.querySelector(
          ".viz-tile:not([data-app])");
        if (!first) break;
        if (soloId && first.dataset.tile === soloId) soloId = null;
        try { wallObs.unobserve(first); } catch (_) { /* gone */ }
        tileDeck.removeChild(first);
      }
    }
    const cap = tile.querySelector(".graph-caption");
    if (cap && caption) cap.textContent = caption;
    tile._fig = fig;
    const body = tile.querySelector(".vp-body");
    if (body && !body.querySelector("canvas")) {
      body.innerHTML = "<canvas></canvas>";
    }
    const canvas = tile.querySelector("canvas");
    if (canvas) fitCanvas(canvas, fig, tile);
    attachOverlay(tile);
    watchTile(tile);
    if (isNew) reflowWall();
  }

  // control CV patch point: cables start here (edit mode)
  function ctlPad(port, title) {
    const el = document.createElement("span");
    el.className = "io-pad out ctl-pad";
    el.dataset.port = `ctl:${port}`;
    el.title = title || `ctl ${port}`;
    return el;
  }

  function extPortsOf(c) {
    const nm = String(c.name).toLowerCase();
    if (c.input === "mouse_xy") return [`${nm}_x`, `${nm}_y`];
    if (c.input === "gamepad_axes") {
      return [0, 1, 2, 3].map((i) => `${nm}_a${i}`);
    }
    if (c.input === "gamepad_buttons") {
      return [0, 1, 2, 3, 4, 5, 6, 7].map((i) => `${nm}_b${i}`);
    }
    return [nm];
  }

  // Helpers for Rack-style grid placement — see Rack/include/app/common.hpp
  // RACK_GRID_SIZE and src/app/ModuleWidget.cpp grid snapping. Controls
  // are Widgets with box {pos,size}; we snap pos/size to RACK_GRID and
  // allow the user to drag/resize them inside the tile's vp-controls
  // rack (like Rack's ModuleWidget::onDragMove).
  function rackControlBox(c, idx) {
    if (c.box && typeof c.box.x === "number") {
      return {
        x: rackSnap(c.box.x), y: rackSnap(c.box.y),
        w: Math.max(RACK_GRID * 2, rackSnap(c.box.w)),
        h: Math.max(RACK_GRID * 2, rackSnap(c.box.h)),
      };
    }
    const presets = {
      fader: { w: 30, h: 105 },
      encoder: { w: 45, h: 60 },
      xy: { w: 90, h: 90 },
      button: { w: 60, h: 30 },
      ext: { w: 120, h: 28 },
    };
    const base = presets[c.type] || presets.fader;
    const cols = 4;
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    return {
      x: rackSnap(col * 80 + 10),
      y: rackSnap(row * 75 + 18),
      w: rackSnap(base.w),
      h: rackSnap(base.h),
    };
  }
  function persistControlBoxes() {
    if (!fn || !fn.io) return;
    const patch = { controls: fn.io.controls };
    // debounce via persistIo (already debounced by server)
    persistIo({ controls: patch.controls });
  }
  function makeResizeHandle(widget, c) {
    const h = document.createElement("div");
    h.className = "rack-resize-handle";
    h.title = "drag to resize — 15px grid";
    h.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      e.preventDefault();
      const startBox = { ...c.box };
      const sx = e.clientX, sy = e.clientY;
      widget.classList.add("resizing");
      const onMove = (ev) => {
        let nw = startBox.w + (ev.clientX - sx);
        let nh = startBox.h + (ev.clientY - sy);
        nw = Math.max(RACK_GRID * 2, rackSnap(nw));
        nh = Math.max(RACK_GRID * 2, rackSnap(nh));
        // clamp to tile bounds minus pos
        const ov = widget.parentElement;
        if (ov) {
          const maxW = ov.clientWidth - startBox.x - 2;
          const maxH = ov.clientHeight - startBox.y - 2;
          nw = Math.min(nw, rackSnap(maxW));
          nh = Math.min(nh, rackSnap(maxH));
        }
        widget.style.width = nw + "px";
        widget.style.height = nh + "px";
        c.box.w = nw; c.box.h = nh;
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        widget.classList.remove("resizing");
        persistControlBoxes();
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
    return h;
  }
  function makeDragMover(widget, c) {
    const header = widget.querySelector(".rack-widget-label");
    const target = header || widget;
    target.style.cursor = "move";
    target.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".rack-resize-handle") ||
          e.target.closest(".io-pad") ||
          e.target.closest("input") ||
          e.target.closest("button")) return;
      if (!appEdit) return;
      e.stopPropagation();
      const startBox = { ...c.box };
      const sx = e.clientX, sy = e.clientY;
      widget.classList.add("selected");
      const onMove = (ev) => {
        let nx = startBox.x + (ev.clientX - sx);
        let ny = startBox.y + (ev.clientY - sy);
        nx = rackSnap(nx); ny = rackSnap(ny);
        const ov = widget.parentElement;
        if (ov) {
          nx = Math.max(0, Math.min(nx, ov.clientWidth - c.box.w));
          ny = Math.max(0, Math.min(ny, ov.clientHeight - c.box.h));
        }
        widget.style.left = nx + "px";
        widget.style.top = ny + "px";
        c.box.x = nx; c.box.y = ny;
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        widget.classList.remove("selected");
        persistControlBoxes();
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
  }
  // Rack-inspired primitives — mimic componentlibrary.hpp:
  // RoundBlackKnob (circular shadow + SVG face + framebuffer),
  // VCVSlider / BefacoSlidePot (track + handle), XY pad, VCVButton.
  function knobAngle(norm) {
    return RACK_KNOB_MIN_ANGLE + norm * (RACK_KNOB_MAX_ANGLE - RACK_KNOB_MIN_ANGLE);
  }
  function getModSpeed(ev) {
    const ctrl = ev.ctrlKey || ev.metaKey;
    const shift = ev.shiftKey;
    if (ctrl && shift) return 0.01;
    if (ctrl) return 0.1;
    if (shift) return 4;
    return 1;
  }
  function createRackKnob(c, tile, port, box) {
    const lo = c.min !== undefined ? c.min : 0;
    const hi = c.max !== undefined ? c.max : 1;
    const range = hi - lo;
    let norm = c.default !== undefined ? (c.default - lo) / range : 0.5;
    norm = Math.max(0, Math.min(1, norm));
    const el = document.createElement("div");
    el.className = "rack-widget rack-knob vp-ctl";
    el.dataset.port = port;
    el.innerHTML = `
      <div class="rack-widget-label">${c.label || c.name}</div>
      <div class="rack-knob-body">
        <div class="rack-knob-face">
          <div class="rack-knob-indicator" style="transform: rotate(${knobAngle(norm)}rad)"></div>
        </div>
      </div>
      <div class="rack-widget-value">${((lo + norm * range)).toFixed(2)}</div>
    `;
    const indicator = el.querySelector(".rack-knob-indicator");
    const valueEl = el.querySelector(".rack-widget-value");
    let dragging = false;
    let startY = 0, startNorm = norm;
    let oldValue = lo + norm * range;
    const update = (n) => {
      norm = Math.max(0, Math.min(1, n));
      const v = lo + norm * range;
      indicator.style.transform = `rotate(${knobAngle(norm)}rad)`;
      valueEl.textContent = v.toFixed(2);
      pushIoField(tile, port, v);
    };
    el.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".rack-resize-handle")) return;
      e.stopPropagation();
      dragging = true;
      startY = e.clientY;
      startNorm = norm;
      oldValue = lo + norm * range;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      let delta = (startY - e.clientY) * 0.005 * getModSpeed(e);
      // snap handling like Knob.cpp snapDelta
      if (c.snap) {
        delta = Math.trunc(delta * 10) / 10;
      }
      update(startNorm + delta * (range / range));
    });
    el.addEventListener("pointerup", (e) => {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const d = e.deltaY > 0 ? -0.02 : 0.02;
      update(norm + d * getModSpeed(e));
    }, { passive: false });
    el.appendChild(ctlPad(port, `${c.label || c.name} cv out`));
    el.appendChild(makeResizeHandle(el, c));
    makeDragMover(el, c);
    // sync initial push
    queueMicrotask(() => update(norm));
    return el;
  }
  function createRackSlider(c, tile, port, box) {
    const lo = c.min !== undefined ? c.min : 0;
    const hi = c.max !== undefined ? c.max : 1;
    const range = hi - lo;
    let norm = c.default !== undefined ? (c.default - lo) / range : 0.5;
    norm = Math.max(0, Math.min(1, norm));
    const el = document.createElement("div");
    el.className = "rack-widget rack-slider vp-ctl";
    el.dataset.port = port;
    el.innerHTML = `
      <div class="rack-widget-label">${c.label || c.name}</div>
      <div class="rack-slider-track">
        <div class="rack-slider-fill" style="height:${norm * 100}%"></div>
        <div class="rack-slider-handle" style="bottom: calc(${norm * 100}% - 6px)"></div>
      </div>
      <div class="rack-widget-value">${((lo + norm * range)).toFixed(2)}</div>
    `;
    const fill = el.querySelector(".rack-slider-fill");
    const handle = el.querySelector(".rack-slider-handle");
    const valueEl = el.querySelector(".rack-widget-value");
    const track = el.querySelector(".rack-slider-track");
    const update = (n) => {
      norm = Math.max(0, Math.min(1, n));
      const v = lo + norm * range;
      fill.style.height = (norm * 100) + "%";
      handle.style.bottom = `calc(${norm * 100}% - 6px)`;
      valueEl.textContent = v.toFixed(2);
      pushIoField(tile, port, v);
    };
    let dragging = false;
    const posToNorm = (clientY) => {
      const r = track.getBoundingClientRect();
      return 1 - Math.max(0, Math.min(1, (clientY - r.top) / r.height));
    };
    track.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      dragging = true;
      update(posToNorm(e.clientY));
      track.setPointerCapture(e.pointerId);
    });
    track.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      update(posToNorm(e.clientY));
    });
    track.addEventListener("pointerup", (e) => {
      dragging = false;
      try { track.releasePointerCapture(e.pointerId); } catch(_){}
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const d = e.deltaY > 0 ? -0.02 : 0.02;
      update(norm + d);
    }, { passive: false });
    el.appendChild(ctlPad(port, `${c.label || c.name} cv out`));
    el.appendChild(makeResizeHandle(el, c));
    makeDragMover(el, c);
    queueMicrotask(() => update(norm));
    return el;
  }
  function createRackXY(c, tile, port) {
    const el = document.createElement("div");
    el.className = "rack-widget rack-xy-pad vp-ctl";
    el.dataset.port = port;
    el.innerHTML = `
      <div class="rack-widget-label">${c.label || c.name}</div>
      <div class="rack-xy-wrapper" style="flex:1; position:relative; width:100%; min-height:40px;">
        <div class="rack-xy-grid"></div>
        <div class="rack-xy-crosshair"></div>
        <div class="rack-xy-knob"></div>
      </div>
    `;
    const wrapper = el.querySelector(".rack-xy-wrapper");
    const knob = el.querySelector(".rack-xy-knob");
    let nx = 0.5, ny = 0.5;
    const place = (x, y) => {
      nx = Math.max(0, Math.min(1, x));
      ny = Math.max(0, Math.min(1, y));
      knob.style.left = (nx * 100) + "%";
      knob.style.top = (ny * 100) + "%";
      pushIoField(tile, `${port}_x`, nx);
      pushIoField(tile, `${port}_y`, 1 - ny);
    };
    const toNorm = (e) => {
      const r = wrapper.getBoundingClientRect();
      return {
        x: (e.clientX - r.left) / r.width,
        y: (e.clientY - r.top) / r.height,
      };
    };
    let dragging = false;
    wrapper.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      dragging = true;
      const p = toNorm(e);
      place(p.x, p.y);
      wrapper.setPointerCapture(e.pointerId);
    });
    wrapper.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const p = toNorm(e);
      place(p.x, p.y);
    });
    wrapper.addEventListener("pointerup", (e) => {
      dragging = false;
      try { wrapper.releasePointerCapture(e.pointerId);}catch(_){}
    });
    // initial
    knob.style.left = "50%"; knob.style.top = "50%";
    el.appendChild(ctlPad(`${port}_x`, "x cv out"));
    el.appendChild(ctlPad(`${port}_y`, "y cv out"));
    el.appendChild(makeResizeHandle(el, c));
    makeDragMover(el, c);
    return el;
  }
  function createRackButton(c, tile, port) {
    const el = document.createElement("div");
    el.className = "rack-widget rack-button vp-ctl" + (c.buttonType === "toggle" ? " toggle" : " momentary");
    el.dataset.port = port;
    const label = c.label || c.name;
    el.innerHTML = `
      <div class="rack-widget-label">${label}</div>
      <div class="rack-button-face">${label}</div>
      <div class="rack-button-light"></div>
    `;
    const face = el.querySelector(".rack-button-face");
    const light = el.querySelector(".rack-button-light");
    let on = false;
    const set = (v) => {
      on = !!v;
      el.classList.toggle("on", on);
      pushIoField(tile, port, on ? 1 : 0);
    };
    if (c.buttonType === "toggle") {
      face.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        set(!on);
      });
    } else {
      face.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        set(true);
        face.setPointerCapture(e.pointerId);
      });
      const off = (e) => { set(false); try{face.releasePointerCapture(e.pointerId);}catch(_){}};
      face.addEventListener("pointerup", off);
      face.addEventListener("pointerleave", off);
      face.addEventListener("pointercancel", off);
    }
    el.appendChild(ctlPad(port, `${label} cv out`));
    el.appendChild(makeResizeHandle(el, c));
    makeDragMover(el, c);
    return el;
  }
  function createRackExt(c, tile, port) {
    const el = document.createElement("div");
    el.className = "rack-widget rack-ext vp-ctl";
    el.dataset.port = port;
    el.innerHTML = `
      <div class="rack-widget-label">${c.label || c.name}</div>
      <div style="font-size:0.58rem; color:#8af; text-align:center; padding:2px;">${c.input || ""}${c.key ? " ["+c.key+"]" : ""}</div>
      <div class="rack-widget-value" style="font-size:0.5rem; opacity:0.7;">ext</div>
    `;
    for (const sub of extPortsOf(c)) {
      el.appendChild(ctlPad(sub, `${sub} cv out`));
    }
    el.appendChild(makeResizeHandle(el, c));
    makeDragMover(el, c);
    return el;
  }

  // Control layer ABOVE the framebuffer: on-tile Rack widgets
  // modulate the owning app's io ports (postMessage bridge) without
  // expanding or leaving the viewport. Each widget is a Widget with
  // box {pos,size} snapped to RACK_GRID, resizable via corner handle.
  function attachOverlay(tile) {
    // controls modulate the owning app's iframe program — but in
    // edit mode the manifest cells exist before any program mounts:
    // fall back to the selected/first materialized cell so the CV
    // pads (and the controls themselves) are visible and patchable
    if (!tile || !tile.isConnected) {
      tile = (lastSelTile && lastSelTile.isConnected && lastSelTile)
        || tileDeck.querySelector(`[data-fn="${CSS.escape((fn && fn.id)
            || "")}"]`)
        || tileDeck.querySelector("[data-mtile]")
        || null;
    }
    if (!tile) return;
    const body = tile.querySelector(".vp-body");
    if (!body) return;
    const controls = (fn && fn.io && fn.io.controls) || [];
    const wantsHid = tile._fig && tile._fig.type === "space3d";
    let ov = tile.querySelector(".vp-controls");
    if (!controls.length && !wantsHid) {
      if (ov) ov.remove();
      const xb = tile.querySelector(".vp-expand");
      if (xb) xb.remove();
      return;
    }
    if (!ov) {
      ov = document.createElement("div");
      ov.className = "vp-controls";
      body.appendChild(ov);
    }
    ov.innerHTML = "";
    // right-click context menu for control elements (operator spec)
    ov.addEventListener("contextmenu", (e) => {
      const w = e.target.closest(".rack-widget");
      if (!w || !appEdit) return;
      e.preventDefault();
      const port = w.dataset.port || "";
      const ctlName = port.replace("ctl:", "");
      const menu = document.createElement("div");
      menu.className = "vp-ctx-menu";
      menu.innerHTML = `<div style="font-size:10px;padding:4px;background:#1a1610;border:1px solid #8b72aa;color:#fcc19f;">control: <b>${ctlName}</b><br/>port: ${port}<br/>right-click options active</div>`;
      menu.style.position = "fixed"; menu.style.zIndex = "100";
      menu.style.left = e.clientX + "px"; menu.style.top = e.clientY + "px";
      document.body.appendChild(menu);
      const closeCtx = () => { menu.remove(); document.removeEventListener("pointerdown", closeCtx); };
      document.addEventListener("pointerdown", closeCtx);
    });
    controls.forEach((c, idx) => {
      const port = String(c.name).toLowerCase();
      // ensure Rack box exists — like ModuleWidget::box.size init
      const box = rackControlBox(c, idx);
      c.box = box;
      let widget;
      if (c.type === "fader") {
        widget = createRackSlider(c, tile, port, box);
      } else if (c.type === "encoder") {
        widget = createRackKnob(c, tile, port, box);
      } else if (c.type === "xy") {
        widget = createRackXY(c, tile, port);
      } else if (c.type === "ext") {
        widget = createRackExt(c, tile, port);
      } else if (c.type === "button") {
        widget = createRackButton(c, tile, port);
      } else {
        // fallback: treat unknown as fader
        widget = createRackSlider(c, tile, port, box);
      }
      widget.style.left = box.x + "px";
      widget.style.top = box.y + "px";
      widget.style.width = box.w + "px";
      widget.style.height = box.h + "px";
      ov.appendChild(widget);
    });
    // explicit full-screen: HID tiles (drag-rotate) and control
    // overlays reserve clicks, so expansion is a deliberate button
    if (wantsHid || controls.length) {
      let xb = tile.querySelector(".vp-expand");
      if (!xb) {
        xb = document.createElement("button");
        xb.type = "button";
        xb.className = "vp-expand";
        xb.title = "full screen";
        xb.textContent = "⤢";
        xb.addEventListener("pointerup", (e) => {
          e.stopPropagation();
          soloTile(tile.dataset.tile);
          document.body.classList.toggle("solo-viz", !!soloId);
        });
        body.appendChild(xb);
      }
    }
  }

  function pushIoField(tile, name, value) {
    // controls are CV sources: honor explicit patch wires first,
    // fall back to the plain port name
    routeValue(`ctl:${name}`, value);
  }

  function frameTile(source) {
    for (const fr of tileDeck.querySelectorAll("iframe.miniapp-frame")) {
      if (fr.contentWindow === source) {
        return fr.closest(".viz-tile");
      }
    }
    return null;
  }

  function handleAppMessage(e) {
    const d = e.data;
    if (!d || typeof d !== "object") return;
    const owner = frameTile(e.source);
    if (d.type === "app-figure" && d.figure && d.app) {
      const cap = `${d.app} · ${d.name || ""}`.replace(/ · $/, "");
      let fig = null;
      if (d.figure.kind === "series") {
        fig = { type: "series", series: d.figure.data || [], caption: cap };
      } else if (d.figure.kind === "points3d") {
        fig = { type: "space3d",
                planets: [],
                paths: { [d.name || "curve"]: triples(d.figure.data || []) },
                caption: cap };
      }
      // output parameter: auto-route only when enabled
      const o = { name: d.name || "" };
      const mine = owner && fn && owner.dataset.fn === fn.id;
      const routed = mine ? outRoutingFor(o)
        : outRouting[`${owner ? owner.dataset.fn : "?"}.${d.name}`] !== false;
      if (fig && routed) upsertAppTile(String(d.app), fig, cap);
      if (d.figure.kind === "fb-frame" && owner && d.bitmap) {
        const cv = owner.querySelector(".vp-body canvas");
        if (cv) {
          const ctx = cv.getContext("2d");
          ctx.drawImage(d.bitmap, 0, 0, cv.width, cv.height);
          if (owner._fb) owner._fb.source = "app";
        }
      }
      if (mine && ioRight) {
        const st = ioRight.querySelector(
          `.out-row[data-out="${CSS.escape(d.name)}"] .out-status`);
        if (st) {
          st.textContent = routed
            ? `${d.figure.count} pts · viewport`
            : `${d.figure.count} pts`;
        }
      }
    } else if (d.type === "app-out") {
      // scalar readout: right-rail value + emitting tile caption
      const num = typeof d.value === "number" && Number.isFinite(d.value);
      const mine = owner && fn && owner.dataset.fn === fn.id;
      if (mine && ioRight) {
        const val = ioRight.querySelector(
          `.out-row[data-out="${CSS.escape(d.name)}"] .out-val`);
        if (val && num) {
          val.textContent = String(parseFloat(d.value.toPrecision(5)));
        }
      }
      if (owner && num) {
        const cap = owner.querySelector(".graph-caption");
        if (cap) {
          cap.textContent = `${cap.textContent.split(" | ")[0]} | ` +
            `${d.name}=${parseFloat(d.value.toPrecision(5))}`;
        }
      }
      // outputs can loop back into the input side once set: an out
      // row with loop:<input> feeds its value into that io port
      if (owner && fn && owner.dataset.fn === fn.id && num) {
        const out = ((fn.io || {}).outs || [])
          .find((o) => o.name === d.name && o.loop);
        if (out && owner.querySelector("iframe.miniapp-frame")) {
          try {
            owner.querySelector("iframe.miniapp-frame")
              .contentWindow.postMessage(
                { type: "io",
                  fields: { [out.loop]: d.value } }, "*");
          } catch (_) { /* sandbox */ }
        }
      }
    }
  }
  window.addEventListener("message", handleAppMessage);

  function mountHtml(body, fig, tile) {
    const html = fig.html || "";
    if (fig.sandbox === false) {
      body.innerHTML = html;
      tile._frame = null;
      return;
    }
    const frame = document.createElement("iframe");
    frame.className = "miniapp-frame";
    frame.setAttribute("sandbox", "allow-scripts");
    frame.setAttribute("title", fig.caption || "miniapp");
    frame.srcdoc = html;
    body.innerHTML = "";
    body.appendChild(frame);
    tile._frame = frame;
    frame.addEventListener("load", () => pushIo(tile));
  }

  // viewport spawn default: 50% tile size; resizable via .rack-widget-style drag handle
  function makeTile(caption) {
    const tile = document.createElement("article");
    tile.className = "viz-tile";
    tile.innerHTML =
      '<p class="graph-caption"></p>' +
      '<div class="vp-body" style="width:50%;height:50%;position:relative;"><canvas style="width:100%;height:100%;"></canvas></div>' +
      '<span class="rack-resize-handle" title="resize viewport (50% default)" style="position:absolute;bottom:0;right:0;width:12px;height:12px;cursor:nwse-resize;z-index:10;"></span>' +
      '<button class="tile-remove" title="remove tile" style="position:absolute;top:2px;right:2px;z-index:10;font-size:10px;padding:0 3px;">×</button>';
    tile.querySelector(".graph-caption").textContent = caption || "";
    // wire remove + drag-to-reposition (tile-level, not control-level)
    const cap = tile.querySelector(".graph-caption");
    if (cap) {
      cap.style.cursor = "move";
      cap.title = "drag to reposition tile";
      cap.addEventListener("pointerdown", (e) => {
        if (!appEdit) return;
        e.preventDefault();
        const startX = e.clientX, startY = e.clientY;
        const startLeft = tile.offsetLeft || 0, startTop = tile.offsetTop || 0;
        const move = (ev) => {
          tile.style.position = "absolute";
          tile.style.left = (startLeft + ev.clientX - startX) + "px";
          tile.style.top = (startTop + ev.clientY - startY) + "px";
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    }
    const remBtn = tile.querySelector(".tile-remove");
    if (remBtn) remBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const name = tile.dataset.mtile || tile.dataset.tile;
      tile.remove();
      if (fn && fn.io && fn.io.tiles) {
        const next = fn.io.tiles.filter(t => t.name !== name);
        persistIoWires({ tiles: next, wires: (fn.io.wires || []).filter(w => !String(w.from || w.to).includes(name)) });
      }
    });
    return tile;
  }

  function addTile(fig, title) {
    const caption = (fig && fig.caption) || title || "";
    let tile = tileDeck.querySelector("[data-placeholder]");
    if (tile) {
      tile.removeAttribute("data-placeholder");
      const cap = tile.querySelector(".graph-caption");
      if (cap) cap.textContent = caption;
    } else {
      tile = makeTile(caption);
      tileDeck.appendChild(tile);
      while (tileDeck.querySelectorAll(".viz-tile").length > TILE_CAP) {
        const first = tileDeck.querySelector(".viz-tile");
        if (!first) break;
        if (soloId && first.dataset.tile === soloId) soloId = null;
        try { wallObs.unobserve(first); } catch (_) { /* gone */ }
        tileDeck.removeChild(first);
      }
    }
    if (!tile.dataset.tile) tile.dataset.tile = String(++tileSeq);
    tile.dataset.fn = (fn && fn.id) || "";
    const prev = tile._fig;
    tile._fig = fig || null;
    const body = tile.querySelector(".vp-body");
    if (fig && fig.type === "html") {
      if (body) {
        if (tile._frame && prev && prev.html === fig.html) {
          pushIo(tile);
        } else {
          mountHtml(body, fig, tile);
        }
      }
    } else {
      if (body && !body.querySelector("canvas")) {
        body.innerHTML = "<canvas></canvas>";
      }
      const canvas = tile.querySelector("canvas");
      if (canvas && fig) fitCanvas(canvas, fig, tile);
    }
    watchTile(tile);
    // app program mounted: control overlays live on the iframe tile
    if (fig && fig.type === "html") attachOverlay(tile);
    reflowWall();
    renderLiveCol();
    stampViewports();
  }

  function visibleFields(spec) {
    const fields = spec.fields || [];
    const values = {};
    for (const f of fields) {
      const el = ioEl.querySelector(`[name="${f.name}"]`);
      values[f.name] = el ? el.value : f.value;
    }
    return fields.filter((f) => {
      if (!f.when) return true;
      const got = String(values[f.when.name] ?? "");
      const ok = (f.when.in || []).map(String);
      return ok.includes(got);
    });
  }

  // ---- popup windows -------------------------------------------------
  // One popup layer for menus, io pickers + tile configuration:
  // fixed position centered on the screen (clamped to the viewport),
  // title bar, click-outside / Escape / x to dismiss.
  let mfxPopup = null;
  function closePopup() {
    if (mfxPopup) {
      if (mfxPopup._ro) { mfxPopup._ro.disconnect(); mfxPopup._ro = null; }
      if (mfxPopup._dismiss) {
        document.removeEventListener("mousedown", mfxPopup._dismiss);
        mfxPopup._dismiss = null;
      }
      if (mfxPopup._esc) {
        document.removeEventListener("keydown", mfxPopup._esc);
        mfxPopup._esc = null;
      }
      mfxPopup.remove();
      mfxPopup = null;
    }
  }
  function openPopup(title, anchor, build) {
    closePopup();
    const pop = document.createElement("div");
    pop.className = "mfx-popup";
    const bar = document.createElement("div");
    bar.className = "mfx-popup-bar";
    const span = document.createElement("span");
    span.textContent = title;
    const x = document.createElement("button");
    x.type = "button";
    x.className = "mfx-popup-x";
    x.textContent = "x";
    x.addEventListener("click", closePopup);
    bar.append(span, x);
    const bodyEl = document.createElement("div");
    bodyEl.className = "mfx-popup-body";
    pop.append(bar, bodyEl);
    document.body.appendChild(pop);
    const center = () => {
      const pw = pop.offsetWidth, ph = pop.offsetHeight;
      const left = Math.max(8, Math.min((innerWidth - pw) / 2,
                                        innerWidth - pw - 8));
      const top = Math.max(8, Math.min((innerHeight - ph) / 2,
                                       innerHeight - ph - 8));
      pop.style.left = Math.round(left) + "px";
      pop.style.top = Math.round(top) + "px";
    };
    pop.style.visibility = "hidden";
    requestAnimationFrame(() => {
      center();
      pop.style.visibility = "visible";
    });
    // menus load content async — keep them centered as they grow
    const ro = new ResizeObserver(center);
    ro.observe(pop);
    pop._ro = ro;
    const dismiss = (e) => {
      if (pop.contains(e.target)) return;
      closePopup();
    };
    const escDismiss = (e) => {
      if (e.key === "Escape") closePopup();
    };
    // tracked so closePopup can REMOVE them — a stale dismiss from a
    // detached popup treats every later mousedown as outside and kills
    // whatever popup is currently open
    pop._dismiss = dismiss;
    pop._esc = escDismiss;
    document.addEventListener("mousedown", dismiss);
    document.addEventListener("keydown", escDismiss);
    mfxPopup = pop;
    build(bodyEl);
    return pop;
  }

  // ---- build/edit-mode io assembly --------------------------------
  // sidebar empty-space click = add io; populated row click = edit.
  // Chooser rows write through POST /api/library/app/{id}/io.
  let ioPicker = null;
  function closeIoPicker() {
    if (ioPicker) { ioPicker.remove(); ioPicker = null; }
  }
  function persistIo(ioPatch, done) {
    if (!fn || !fn.id) return;
    fetch(`/api/library/app/${encodeURIComponent(fn.id)}/io`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ioPatch),
    }).then((r) => r.json()).then((data) => {
      if (data.error || data.detail) {
        console.warn("io save:", data.error || data.detail);
        return;
      }
      if (!data.io) return;
      fn.io = data.io;
      if (data.spec) upsertFn(data.spec);
      renderForm();
      if (done) done();
    }).catch((e) => console.warn("io save:", e));
  }
  function openIoPicker(side, editName, anchor) {
    closeIoPicker();
    const host = null; // popups render in the top layer now
    if (!fn) return;
    const pop = openPopup(
      (editName ? "edit " + editName : "add " +
        (side === "left" ? "input source" : "output destination")),
      anchor || (side === "left" ? ioEl : ioRight),
      (box) => {
        const ioPicker2 = box;
        buildIoOptions(side, editName, ioPicker2,
          () => { closePopup(); renderForm(); });
      });
    ioPicker = pop;
  }

  function buildIoOptions(side, editName, ioPicker, done) {
    const add = (label, pick) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "io-pick";
        b.textContent = label;
        b.addEventListener("click", () => { closePopup(); pick(); done && done(); });
        ioPicker.appendChild(b);
      };
    if (side === "left") {
      const ins = (fn.io || {}).ins || [];
      fetch("/api/wizard/catalog").then((r) => r.json()).then((cat) => {
        for (const s of cat.sources || []) {
          add(s.label, () => {
            const name = "sig" + (ins.length ? ins.length : "");
            const entry = { name, kind: s.id };
            if (s.needs === "topic") {
              entry.topic = (($("wizFeed") || {}).value
                || s.hint || "ship/vllm/toks")
                .replace(/^e\.g\.\s*/, "");
            }
            const next = editName
              ? ins.map((x) => x.name === editName ? { ...x, ...entry } : x)
              : [...ins, entry];
            persistIo({ ins: next });
          });
        }
      }).catch(() => {});
      return;
    }
    const outs = (fn.io || {}).outs || [];
    const insNames = ((fn.io || {}).ins || []).map((x) => x.name);
    add("series figure out", () => {
      const name = "out" + (outs.length ? outs.length : "");
      const entry = { name, kind: "series" };
      const next = editName
        ? outs.map((x) => x.name === editName ? { ...x, ...entry } : x)
        : [...outs, entry];
      persistIo({ outs: next });
    });
    for (const inName of insNames) {
      add(`loop back -> input ${inName}`, () => {
        const entry = { name: "loop" + outs.length, kind: "number",
                        loop: inName };
        const next = editName
          ? outs.map((x) => x.name === editName ? { ...x, ...entry } : x)
          : [...outs, entry];
        persistIo({ outs: next });
      });
    }
  }
  function openTileConfig(tile, anchor) {
    if (!fn) return;
    lastSelTile = tile;
    openPopup(`configure · ${tile.dataset.app || tile.dataset.mtile
               || "tile"}`, anchor,
      (box) => {
        const info = document.createElement("p");
        info.className = "settings-sub";
        const t = manifestTiles();
        info.textContent = `tiles ${t.length}/16 · controls ${
          (fn.io || {}).controls?.length || 0}`;
        box.appendChild(info);
        const row = (label) => {
          const p = document.createElement("p");
          p.className = "settings-sub";
          p.style.color = "#778";
          p.textContent = label;
          box.appendChild(p);
        };
        // standardized wrapper: aux channel count per cell
        const mt = tile.dataset.mtile;
        if (mt) {
          const entry = manifestTiles().find((x) => x.name === mt);
          if (entry) {
            row(`${mt} · aux i/o channels (free assign)`);
            const arow = document.createElement("div");
            arow.className = "mfx-as-row";
            const minus = document.createElement("button");
            minus.type = "button"; minus.className = "io-pick";
            minus.textContent = "−";
            const plus = document.createElement("button");
            plus.type = "button"; plus.className = "io-pick";
            plus.textContent = "+";
            const cnt = document.createElement("span");
            const paintAux = () => { cnt.textContent =
              ` ${entry.aux ?? 2} `; };
            paintAux();
            const bump = (d) => {
              const cur = Number(entry.aux ?? 2);
              const next = Math.max(0, Math.min(8, cur + d));
              if (next === cur) return;
              entry.aux = next;
              persistIoWires({ tiles: manifestTiles().map((x) =>
                x.name === mt ? { ...x, aux: next } : x) });
              paintAux();
            };
            minus.addEventListener("click", () => bump(-1));
            plus.addEventListener("click", () => bump(1));
            arow.append(minus, cnt, plus);
            box.appendChild(arow);
          }
        }
        row("add control (cv source on this cell)");
        const crow = document.createElement("div");
        crow.className = "cfg-grid";
        for (const [cid, label, entry] of [
          ["fader", "fader", { type: "fader", min: 0, max: 1,
                               default: 0.5, step: 0.01 }],
          ["knob", "knob", { type: "encoder", min: 0, max: 1 }],
          ["xy pad", "xy", { type: "xy" }],
          ["momentary", "button", { type: "button",
                                    buttonType: "momentary" }],
          ["toggle", "button", { type: "button",
                                 buttonType: "toggle" }],
        ]) {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "io-pick";
          b.textContent = label === "button" ? cid : label;
          b.addEventListener("click", () => {
            const controls = (fn.io || {}).controls || [];
            const name = `${entry.type}_${controls.length}`;
            const boxPos = rackControlBox({ type: entry.type }, controls.length);
            persistIoWires({ controls: [...controls, {
              ...entry, type: entry.type, name, label: name, box: boxPos }] }, () => {
                attachOverlay(tile);
              });
            closePopup();
          });
          crow.appendChild(b);
        }
        box.appendChild(crow);
        row("add external input");
        const xrow = document.createElement("div");
        xrow.className = "cfg-grid";
        for (const [inp, label] of EXT_INPUTS) {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "io-pick";
          b.textContent = label;
          b.addEventListener("click", () => {
            const controls = (fn.io || {}).controls || [];
            const name = `ext_${controls.length}`;
            const boxPos = rackControlBox({ type: "ext" }, controls.length);
            const entry = inp === "keyboard"
              ? { type: "ext", input: inp, key: "space",
                  name, label, box: boxPos }
              : { type: "ext", input: inp, name, label, box: boxPos };
            persistIoWires({ controls: [...controls, entry] }, () => {
              attachOverlay(tile);
            });
            closePopup();
          });
          xrow.appendChild(b);
        }
        box.appendChild(xrow);
        // control management (remove)
        const ctrls = (fn.io || {}).controls || [];
        if (ctrls.length) {
          row("controls (click x to remove)");
          const mrow = document.createElement("div");
          mrow.className = "mfx-as-row";
          for (const c of ctrls) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "io-pick";
            chip.textContent = c.name;
            chip.addEventListener("click", () => {
              persistIoWires({ controls:
                ctrls.filter((x) => x.name !== c.name) }, () => {
                  attachOverlay(tile);
                });
              closePopup();
            });
            mrow.appendChild(chip);
          }
          box.appendChild(mrow);
        }
        row("add visualizer tile (inner signal path)");
        const vrow = document.createElement("div");
        vrow.className = "cfg-grid";
        for (const [vid, kind] of [["series", "series"],
                                   ["scope3d", "points3d"]]) {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "io-pick";
          b.textContent = vid;
          b.addEventListener("click", () => {
            const tiles = manifestTiles();
            if (tiles.length >= 16) { closePopup(); return; }
            const name = `tile${tiles.length}`;
            persistIoWires({ tiles: [...tiles, { name, kind, aux: 2 }] },
                          () => attachOverlay(tile));
            closePopup();
          });
          vrow.appendChild(b);
        }
        box.appendChild(vrow);
        if (mt) {
          row("cell management");
          const capB = document.createElement("button");
          capB.type = "button";
          capB.className = "io-pick";
          capB.textContent = `capture ${mt} as custom tile`;
          capB.addEventListener("click", () => {
            const entry = manifestTiles().find((x) => x.name === mt);
            if (!entry) { closePopup(); return; }
            const store = customStore();
            store[mt] = { ...entry };
            try {
              localStorage.setItem("mfx-custom-tiles",
                                   JSON.stringify(store));
            } catch (_) { /* private */ }
            closePopup();
          });
          box.appendChild(capB);
          const rm = document.createElement("button");
          rm.type = "button";
          rm.className = "io-pick mfx-del";
          rm.textContent = `remove tile ${mt} from matrix`;
          rm.addEventListener("click", () => {
            const wires = ((fn.io || {}).wires || []).filter((w) =>
              !normPort(w.from).startsWith(`tile:${mt}:`)
              && !normPort(w.to).startsWith(`tile:${mt}:`));
            try { wallObs.unobserve(tile); } catch (_) { /* ok */ }
            tile.remove();
            if (soloId === tile.dataset.tile) soloId = null;
            if (lastSelTile === tile) lastSelTile = null;
            persistIoWires({
              tiles: manifestTiles().filter((x) => x.name !== mt),
              wires,
            }, () => { reflowWall(); renderLiveCol(); });
            closePopup();
          });
          box.appendChild(rm);
        }
        row("function library (click to add a chain tile)");
        fnListBox(box, (fid) => {
          const tiles = manifestTiles();
          if (tiles.length >= 16) { closePopup(); return; }
          const name = `fn_${fid}_${tiles.length}`;
          persistIoWires({ tiles: [...tiles, { name, kind: "series",
                                          functions: [fid], aux: 2 }] },
                    () => attachOverlay(tile));
        });
      });
  }

  // ---- patch wires · standardized tile i/o wrapper --------------------
  // Every matrix cell exposes BUS i/o (bus_in -> [cell] -> bus_out;
  // cells process sequentially so signal changes accumulate along the
  // chain) plus free AUX channels whose direction is assigned by the
  // cable itself (modulator, source, or direct cell->cell bypass).
  // Sidebar rails expose in:/out: bus points; controls expose ctl:
  // sources. Wires RENDER ONLY IN EDIT MODE — run mode shows just the
  // viewports.
  let wireSvg = null;
  const NS = "http://www.w3.org/2000/svg";
  const normPort = (p) => String(p || "").trim().toLowerCase();

  function figKind(fig) {
    if (!fig) return "series";
    if (fig.type === "space3d") return "points3d";
    if (fig.type === "lcd") return "lcd";
    if (fig.type === "video") return "video";
    if (fig.type === "html") return "number";
    return "series";
  }

  // keep io.tiles[] aligned with what is actually on the wall: every
  // displayed cell gets a manifest identity (name/kind/aux), and every
  // manifest tile gets a matrix cell (empty framebuffer slot until its
  // inner chain generates a program) so edit mode can wire the full
  // layout before programs exist.
  function syncManifestPads() {
    if (!tileDeck || !fn) return;
    const tiles = manifestTiles();
    let dirty = false;
    for (const entry of tiles) {
      let cell = tileDeck.querySelector(
        `[data-mtile="${CSS.escape(entry.name)}"]`);
      if (!cell && appEdit) {
        // materialize the slot: framebuffer display driven by kind
        cell = makeTile(`${entry.name} · ${entry.kind || "series"}`);
        cell.dataset.mtile = entry.name;
        cell.dataset.tile = String(++tileSeq);
        const body2 = cell.querySelector(".vp-body");
        body2.innerHTML = "<canvas></canvas>";
        cell._fig = { type: entry.kind === "points3d" ? "space3d"
                      : entry.kind === "lcd" ? "lcd"
                      : entry.kind === "video" ? "video"
                      : "series", series: [],
                      caption: `${entry.name} · ${entry.kind}` };
        tileDeck.appendChild(cell);
        watchTile(cell);
        dirty = true;
      }
      if (!cell) continue;
      attachWrapperPads(cell, entry.name,
                        entry.aux === undefined
                        ? 2 : Number(entry.aux) || 0);
    }
    if (dirty) { reflowWall(); stampViewports(); }
    // controls declared before any cell existed get their overlay now
    if (appEdit && manifestTiles().length) attachOverlay();
  }

  function attachWrapperPads(tile, mt, auxN) {
    const body = tile.querySelector(".vp-body");
    if (!body) return;
    let wrap = tile.querySelector(".vp-io");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "vp-io";
      tile.appendChild(wrap);
    }
    const mk = (cls, port, title) => {
      const el = document.createElement("span");
      el.className = `io-pad ${cls}`;
      el.dataset.port = port;
      el.title = title;
      return el;
    };
    const bin = `tile:${mt}:bus_in`;
    const bout = `tile:${mt}:bus_out`;
    // rebuild unconditionally: aux count can shrink, and the old
    // pads must go with it
    wrap.innerHTML = "";
    wrap.appendChild(mk("in", bin, `${mt} · bus in`));
    wrap.appendChild(mk("out", bout, `${mt} · bus out`));
    for (let k = 0; k < auxN; k++) {
      const pad = mk("aux", `tile:${mt}:aux${k}`,
                     `${mt} · aux ${k}`);
      pad.style.left =
        `${((k + 1) / (auxN + 1)) * 100}%`;
      wrap.appendChild(pad);
    }
  }

  function ensureWireLayer() {
    if (wireSvg) return wireSvg;
    wireSvg = document.createElementNS(NS, "svg");
    wireSvg.id = "wireLayer";
    document.body.appendChild(wireSvg);
    wireSvg.addEventListener("click", (e) => {
      const hit = e.target.closest(".wire-hit");
      if (!hit || !fn || !appEdit) return;
      const i = Number(hit.dataset.idx);
      const wires = ((fn.io || {}).wires || []).filter(
        (_w, j) => j !== i);
      persistIoWires({ wires });
    });
    return wireSvg;
  }

  function padPoint(port) {
    const el = document.querySelector(
      `[data-port="${CSS.escape(normPort(port))}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (!r.width && !r.height) return null;
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  function wirePath(a, b) {
    const dx = Math.max(28, Math.min(120, Math.abs(b.x - a.x) * 0.45));
    return `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ` +
           `${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
  }

  function seg(a, b, cls, label) {
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", wirePath(a, b));
    p.setAttribute("class", cls);
    return p;
  }

  // implicit sequential chain: sidebar ins feed the head cell, each
  // cell's bus_out continues into the next cell's bus_in (changes
  // accumulate per tile), the tail taps into every sidebar out.
  function defaultChain(explicitTargets) {
    const out2 = [];
    const tiles = wallTiles().filter((t) => t.dataset.mtile)
      .sort((x, y) => (Number(x.dataset.tile) || 0)
                   - (Number(y.dataset.tile) || 0));
    const head = tiles[0];
    const skip = (to) => explicitTargets.has(normPort(to));
    for (const s of (fn && fn.io && fn.io.ins) || []) {
      if (head && !skip(`tile:${head.dataset.mtile}:bus_in`)) {
        out2.push({ from: `in:${s.name}`, dashed: true,
                    to: `tile:${head.dataset.mtile}:bus_in` });
      }
    }
    for (let i = 0; i < tiles.length - 1; i++) {
      const a = `tile:${tiles[i].dataset.mtile}:bus_out`;
      const b = `tile:${tiles[i + 1].dataset.mtile}:bus_in`;
      if (!skip(b)) out2.push({ from: a, to: b, dashed: true });
    }
    const tail = tiles[tiles.length - 1];
    for (const o of (fn && fn.io && fn.io.outs) || []) {
      if (tail && !skip(`out:${o.name}`)) {
        out2.push({ from: `tile:${tail.dataset.mtile}:bus_out`,
                    to: `out:${o.name}`, dashed: true });
      }
    }
    return out2;
  }

  function renderWires() {
    const svg = ensureWireLayer();
    const show = appEdit && mode === "viz" && fn;
    svg.style.display = show ? "" : "none";
    if (!show) return;
    syncManifestPads();
    svg.innerHTML = "";
    const wires = (fn.io || {}).wires || [];
    const targets = new Set(wires.map((w) => normPort(w.to)));
    const draw = (list, base) => {
      list.forEach((w, i) => {
        const a = padPoint(w.from);
        const b = padPoint(w.to);
        if (!a || !b) return;
        svg.appendChild(seg(a, b,
          w.dashed ? "wire-default" : "wire"));
        const hit = seg(a, b, "wire-hit");
        if (!w.dashed) hit.dataset.idx = String(i - base);
        svg.appendChild(hit);
      });
    };
    draw(defaultChain(targets), -1);
    draw(wires, 0);
    if (wireDraft && wireDraft.ghost) svg.appendChild(wireDraft.ghost);
  }

  // routing enforcement: a patched value goes to its explicit wire
  // targets (namespaced cell ports `<tile>.<port>`), and unwired
  // sources keep the legacy plain-port mux so existing programs work.
  function postToAppFields(fields) {
    if (!fn || !fn.id) return;
    const appTile = tileDeck.querySelector(
      `[data-fn="${CSS.escape(fn.id)}"] iframe.miniapp-frame`);
    if (appTile && appTile.contentWindow) {
      try {
        appTile.contentWindow.postMessage({ type: "io", fields }, "*");
      } catch (_) { /* sandbox */ }
    }
  }

  function deliverPort(toPort, v) {
    const parts = normPort(toPort).split(":");
    if (parts[0] === "tile" && parts.length === 3) {
      postToAppFields({ [`${parts[1]}.${parts[2]}`]: v });
    } else if (parts.length === 2) {
      postToAppFields({ [parts[1]]: v });
    }
  }

  function routeValue(fromPort, v) {
    const from = normPort(fromPort);
    let wired = false;
    for (const w of (fn && fn.io && fn.io.wires) || []) {
      if (normPort(w.from) === from) {
        wired = true;
        deliverPort(w.to, v);
      }
    }
    if (!wired) deliverPort(from, v);   // legacy mux fallback
  }

  // ---- external inputs (joystick / keyboard / mouse as CV) ------------
  let extRunning = false;
  const extLast = {};
  const keysDown = {};
  let mousePos = { x: innerWidth / 2, y: innerHeight / 2 };
  function extSend(name, v) {
    const key = normPort(name);
    if (extLast[key] === v) return;
    extLast[key] = v;
    routeValue(`ctl:${key}`, v);
  }
  function feedExt(c) {
    const nm = String(c.name).toLowerCase();
    if (c.input === "mouse_xy") {
      extSend(`${nm}_x`,
        (mousePos.x / innerWidth) * 2 - 1);
      extSend(`${nm}_y`,
        (mousePos.y / innerHeight) * 2 - 1);
      return;
    }
    if (c.input === "keyboard") {
      extSend(nm, keysDown[String(c.key || "space").toLowerCase()]
        ? 1 : 0);
      return;
    }
    const pads = navigator.getGamepads
      ? [...navigator.getGamepads()].filter(Boolean) : [];
    const gp = pads[0];
    if (!gp) return;
    if (c.input === "gamepad_axes") {
      gp.axes.forEach((v, i) => extSend(`${nm}_a${i}`,
                                        Math.round(v * 1000) / 1000));
    } else if (c.input === "gamepad_buttons") {
      gp.buttons.forEach((b, i) =>
        extSend(`${nm}_b${i}`, b.pressed ? 1 : 0));
    }
  }
  function startExtInputs() {
    if (extRunning) return;
    extRunning = true;
    const hasExt = () => (fn && fn.io && fn.io.controls || [])
      .some((c) => c.type === "ext");
    window.addEventListener("pointermove", (e) => {
      mousePos = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener("keydown", (e) => {
      keysDown[e.key.toLowerCase()] = true;
    });
    window.addEventListener("keyup", (e) => {
      keysDown[e.key.toLowerCase()] = false;
    });
    const tick = () => {
      if (hasExt()) {
        for (const c of fn.io.controls || []) {
          if (c.type === "ext") feedExt(c);
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // drag-to-connect: pointerdown on any i/o pad drafts a cable; drop
  // on another pad commits it into the persisted wires manifest.
  document.addEventListener("pointerdown", (e) => {
    const pad = e.target.closest && e.target.closest(".io-pad");
    if (!pad || !appEdit || mode !== "viz") return;
    e.stopPropagation();
    e.preventDefault();
    const ghost = document.createElementNS(NS, "path");
    ghost.setAttribute("class", "wire-ghost");
    wireDraft = { from: pad.dataset.port, ghost,
                  at: padPoint(pad.dataset.port) };
    ensureWireLayer().appendChild(ghost);
  }, true);
  document.addEventListener("pointermove", (e) => {
    if (!wireDraft) return;
    wireDraft.ghost.setAttribute("d",
      wirePath(wireDraft.at, { x: e.clientX, y: e.clientY }));
  });
  document.addEventListener("pointerup", (e) => {
    if (!wireDraft) return;
    const draft = wireDraft;
    wireDraft = null;
    // stacked probe: cable hit-strokes and overlays sit above the
    // pads, so the pad may not be elementFromPoint's first answer
    const stack = document.elementsFromPoint(e.clientX, e.clientY);
    const pad = stack.map((el) =>
      el.closest && el.closest(".io-pad")).find(Boolean);
    if (!pad || !fn) { renderWires(); return; }
    const to = pad.dataset.port;
    const from = draft.from;
    if (normPort(to) === normPort(from)) { renderWires(); return; }
    const wires = ((fn.io || {}).wires || []).filter((w) =>
      normPort(w.from) !== normPort(from)
      || normPort(w.to) !== normPort(to));
    wires.push({ from, to });
    persistIoWires({ wires });
  }, true);


  // ---- App Mode menus (popup layer, mode-switched) -------------------
  // TR = app management (identical run/build). BL = settings in run
  // mode (theme + matrix shader; harness is terminal-only), component
  // library while the build switch is active.
  function popRow(box, label, cb, cls) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls || "io-pick mfx-menu-row";
    b.textContent = label;
    b.addEventListener("click", () => { beep("audio2"); cb(b); });
    box.appendChild(b);
    return b;
  }

  function mountSpec(spec, figure) {
    if (!spec || !spec.id) return;
    upsertFn(spec);
    applyFunction(spec.id);
    if (figure) addTile(figure, spec.title);
    else if (spec.html) {
      addTile({ type: "html", html: spec.html,
                caption: spec.title, sandbox: true }, spec.title);
    }
    closePopup();
  }

  function ensureDraftApp(done) {
    if (isStreamApp(fn) && fn.id) { if (done) done(fn); return; }
    fetch("/api/library/new", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "draft app" }),
    }).then((r) => r.json()).then((data) => {
      if (data.error || !data.spec) return;
      upsertFn(data.spec);
      applyFunction(data.spec.id);
      requestAnimationFrame(() => renderWires());
      if (done) done(data.spec);
    }).catch(() => {});
  }

  function openAppMenu(anchor) {
    openPopup("app management", anchor, (box) => {
      const nameIn = document.createElement("input");
      nameIn.className = "mfx-name-in";
      nameIn.placeholder = "app name (enter to rename)";
      nameIn.value = (fn && fn.title) || "";
      nameIn.addEventListener("keydown", (e) => {
        if (e.key !== "Enter" || !fn || !fn.id) return;
        e.preventDefault();
        const title = nameIn.value.trim();
        if (!title) return;
        fetch(`/api/library/app/${encodeURIComponent(fn.id)}/rename`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        }).then((r) => r.json()).then((data) => {
          if (data.error || data.detail) return;
          if (data.spec) upsertFn(data.spec);
          if (fn) fn.title = title;
          nameIn.blur();
          const t = $("maxTitle");
          if (t) t.textContent = (appEdit ? title + " · edit"
                                          : title).slice(0, 80);
        }).catch(() => {});
      });
      box.appendChild(nameIn);
      const status = document.createElement("p");
      status.className = "settings-sub";
      box.appendChild(status);
      const say = (txt) => { status.textContent = txt; };
      popRow(box, "new app · build wizard", () =>
        openBuildWizard(anchor));
      popRow(box, "save app", () => {
        if (!fn || !fn.id) { say("nothing to save"); return; }
        fetch(`/api/library/app/${encodeURIComponent(fn.id)}`)
          .then((r) => say(r.ok ? `saved · ${fn.id}` : "not in library"))
          .catch(() => say("library unreachable"));
      });
      // save as…: inline title + clone
      const asWrap = document.createElement("div");
      asWrap.className = "mfx-as-row";
      const asIn = document.createElement("input");
      asIn.placeholder = "save as… new name";
      const asBtn = document.createElement("button");
      asBtn.type = "button";
      asBtn.className = "io-pick";
      asBtn.textContent = "clone";
      asBtn.addEventListener("click", () => {
        if (!fn || !fn.id) { say("nothing to save"); return; }
        const title = asIn.value.trim();
        if (!title) { say("name the copy first"); return; }
        fetch(`/api/library/app/${encodeURIComponent(fn.id)}/clone`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        }).then((r) => r.json()).then(async (data) => {
          if (data.error || data.detail) {
            say(data.error || data.detail); return;
          }
          // switch the workspace to the copy: remount with the
          // clone's program so io/wires/controls target it
          const full = await fetch(`/api/library/app/${
            encodeURIComponent(data.spec.id)}`).then((x) => x.json());
          mountSpec(full, full.html
            ? { type: "html", html: full.html, caption: full.title,
                sandbox: true } : null);
        }).catch(() => say("clone failed"));
      });
      asWrap.append(asIn, asBtn);
      box.appendChild(asWrap);
      const loadWrap = document.createElement("div");
      loadWrap.className = "mfx-load-list";
      let loaded = false;
      popRow(box, "load app ▾", () => {
        if (loaded) { loadWrap.innerHTML = ""; loaded = false; return; }
        loaded = true;
        loadWrap.innerHTML = "";
        fetch("/api/library").then((r) => r.json()).then((lib) => {
          for (const a of lib.apps || []) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "io-pick";
            b.textContent = `${a.title} · ${a.kernel_kind || ""}`;
            b.addEventListener("click", async () => {
              try {
                const spec = await fetch(`/api/library/app/${
                  encodeURIComponent(a.id)}`).then((x) => x.json());
                mountSpec(spec.public_spec || spec,
                          spec.html ? { type: "html",
                                        html: spec.html,
                                        caption: spec.title,
                                        sandbox: true } : null);
              } catch (e) { console.warn("load:", e); }
            });
            loadWrap.appendChild(b);
          }
          if (!loadWrap.childElementCount) {
            const p = document.createElement("p");
            p.className = "settings-sub";
            p.textContent = "library empty";
            loadWrap.appendChild(p);
          }
        }).catch(() => {});
      });
      box.appendChild(loadWrap);
    });
  }

  // New App build wizard: signal path (source → feed → visualizer)
  // or a blank canvas assembled in edit mode.
  let bwState = { source: "sensor", viz: "series" };
  function openBuildWizard(anchor) {
    openPopup("new app · build wizard", anchor, (box) => {
      const sub = (t) => {
        const p = document.createElement("p");
        p.className = "settings-sub";
        p.style.color = "#778";
        p.textContent = t;
        box.appendChild(p);
      };
      sub("start from");
      const blank = document.createElement("button");
      blank.type = "button";
      blank.className = "io-pick mfx-menu-row";
      blank.textContent = "blank canvas (edit mode assembly)";
      blank.addEventListener("click", () => {
        beep("audio2");
        fetch("/api/library/new", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }).then((r) => r.json()).then((data) => {
          if (data.error || !data.spec) return;
          closePopup();
          upsertFn(data.spec);
          applyFunction(data.spec.id);
          if (mode === "split") setMode("viz");
          if (!appEdit) setEditMode(true);
          openLibraryPopup(cBL);
        }).catch(() => {});
      });
      box.appendChild(blank);
      sub("signal path · source");
      const srcRow = document.createElement("div");
      srcRow.className = "cfg-grid";
      box.appendChild(srcRow);
      sub("stream / url (when the source needs one)");
      const feed = document.createElement("input");
      feed.placeholder = "bus topic or media url";
      box.appendChild(feed);
      sub("visualizer");
      const vizRow = document.createElement("div");
      vizRow.className = "cfg-grid";
      box.appendChild(vizRow);
      const chainNote = document.createElement("p");
      chainNote.className = "settings-sub";
      box.appendChild(chainNote);
      const create = document.createElement("button");
      create.type = "button";
      create.className = "fabric-engage";
      create.textContent = "Create signal app";
      create.addEventListener("click", async () => {
        const body = {
          source: bwState.source, visualizer: bwState.viz,
          topic: feed.value || undefined,
          url: feed.value || undefined,
        };
        const r = await fetch("/api/wizard/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await r.json().catch(() => ({}));
        if (data.error || !data.spec) {
          chainNote.textContent = data.error || "generation failed";
          return;
        }
        if (mode === "split") setMode("viz");
        mountSpec(data.spec, data.figure);
      });
      box.appendChild(create);
      const paint = (cat) => {
        srcRow.innerHTML = "";
        vizRow.innerHTML = "";
        for (const s of cat.sources || []) {
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.id = s.id;
          b.textContent = s.label + (s.available === false ? " *" : "");
          if (s.id === bwState.source) b.classList.add("active");
          b.addEventListener("click", () => {
            bwState.source = s.id;
            const valid = new Set((cat.recipes || [])
              .filter((x) => x.source === s.id).map((x) => x.visualizer));
            if (valid.size && !valid.has(bwState.viz)) {
              bwState.viz = [...valid][0];
            }
            paint(cat);
          });
          srcRow.appendChild(b);
        }
        const valid = new Set((cat.recipes || [])
          .filter((x) => x.source === bwState.source)
          .map((x) => x.visualizer));
        for (const v of cat.visualizers || []) {
          if (valid.size && !valid.has(v.id)) continue;
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.id = v.id;
          b.textContent = v.label + (v.available === false ? " *" : "");
          if (v.id === bwState.viz) b.classList.add("active");
          b.addEventListener("click", () => {
            bwState.viz = v.id;
            paint(cat);
          });
          vizRow.appendChild(b);
        }
        const rec = (cat.recipes || []).find((x) =>
          x.source === bwState.source && x.visualizer === bwState.viz);
        chainNote.textContent = rec
          ? `functions: ${rec.functions.join(", ") || "none"} · controls: ${
              rec.controls.join(", ") || "none"}`
          : "no signal path for this pair";
        const src = (cat.sources || []).find(
          (s) => s.id === bwState.source);
        feed.placeholder = src && src.needs === "topic"
          ? "bus topic e.g. ship/vllm/toks"
          : src && src.needs === "url" ? "media url" : "n/a for this source";
      };
      fetch("/api/wizard/catalog").then((r) => r.json())
        .then(paint).catch(() => {});
    });
  }

  function openSettingsPopup(anchor) {
    openPopup("settings", anchor, (box) => {
      const sub = (t) => {
        const p = document.createElement("p");
        p.className = "settings-sub";
        p.style.color = "#778";
        p.textContent = t;
        box.appendChild(p);
      };
      sub("theme ui color");
      const grid = document.createElement("div");
      grid.className = "cfg-grid theme-grid-pop";
      const cur = currentTheme();
      THEMES.forEach((t) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = t.label;
        b.style.background = t.swatch;
        if (t.id === cur) b.classList.add("active");
        b.addEventListener("click", () => {
          beep("audio2");
          applyTheme(t.id);
          [...grid.children].forEach((x) => x.classList.remove("active"));
          b.classList.add("active");
        });
        grid.appendChild(b);
      });
      box.appendChild(grid);
      sub("display shader — whole viewport matrix");
      const fx = (label, key) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "io-pick mfx-fx-toggle";
        b.textContent = `${label}: ${shader[key] ? "on" : "off"}`;
        b.addEventListener("click", () => {
          shader[key] = !shader[key];
          b.textContent = `${label}: ${shader[key] ? "on" : "off"}`;
          applyShader();
        });
        return b;
      };
      box.appendChild(fx("scanlines", "scanlines"));
      box.appendChild(fx("glow", "glow"));
      const brightWrap = document.createElement("label");
      brightWrap.className = "mfx-bright-row";
      brightWrap.textContent = "brightness";
      const br = document.createElement("input");
      br.type = "range";
      br.min = 0.5; br.max = 1.8; br.step = 0.05;
      br.value = shader.bright ?? 1;
      br.addEventListener("input", () => {
        shader.bright = parseFloat(br.value);
        applyShader();
      });
      brightWrap.appendChild(br);
      box.appendChild(brightWrap);
      sub("harness options live in terminal mode only");
    });
  }

  // ---- component library (BL while the build switch is active) -------
  let libTab = "functions";
  function openLibraryPopup(anchor) {
    openPopup("component library", anchor, (box) => {
      ensureDraftApp();
      const tabs = document.createElement("div");
      tabs.className = "mfx-lib-tabs";
      const body2 = document.createElement("div");
      body2.className = "mfx-lib-body";
      const paint = () => {
        [...tabs.children].forEach((b) =>
          b.classList.toggle("active",
                             b.dataset.tab === libTab));
        body2.innerHTML = "";
        if (libTab === "functions") libFunctions(body2);
        else if (libTab === "controls") libControls(body2);
        else if (libTab === "viewports") listViewports(body2);
        else libCustom(body2);
      };
      for (const t of ["functions", "controls", "viewports",
                       "custom tiles"]) {
        const b = document.createElement("button");
        b.type = "button";
        b.dataset.tab = t.split(" ")[0];
        b.textContent = t;
        b.addEventListener("click", () => {
          libTab = b.dataset.tab;
          paint();
        });
        tabs.appendChild(b);
      }
      box.append(tabs, body2);
      paint();
    });
  }

  function manifestTiles() {
    return (fn && fn.io && fn.io.tiles) || [];
  }

  function persistIoWires(patch, done) {
    persistIo(patch, () => {
      syncManifestPads();
      renderWires();
      if (done) done();
    });
  }

  // scrollable function list (one-click add) — replaces chip grids
  function fnListBox(box, onPick) {
    const sel = document.createElement("select");
    sel.className = "mfx-select";
    sel.size = 8;
    sel.multiple = false;
    fetch("/api/library").then((r) => r.json()).then((lib) => {
      const fns = ((lib.functions || {}).nodes || {}).function || [];
      for (const f of fns) {
        const o = document.createElement("option");
        o.value = f.id;
        o.textContent = f.id;
        o.title = `in:[${f.inputs.join(",") || "-"}] ` +
                  `out:[${f.outputs.join(",") || "-"}]` +
                  (f.multi_in ? " ·multi" : "");
        sel.appendChild(o);
      }
    }).catch(() => {});
    sel.addEventListener("change", () => {
      const v = sel.value;
      sel.selectedIndex = -1;
      if (v) onPick(v);
    });
    box.appendChild(sel);
    return sel;
  }

  // Category 1 — functions: drop into the selected tile's inner chain
  // or add a fresh chain tile.
  function libFunctions(box) {
    const sel = lastSelTile && lastSelTile.dataset.mtile
      ? lastSelTile.dataset.mtile : null;
    const head = document.createElement("p");
    head.className = "settings-sub";
    head.style.color = "#778";
    head.textContent = sel
      ? `target tile: ${sel} (chain append)`
      : "target: new tile (click a viewport tile first to append)";
    box.appendChild(head);
    fnListBox(box, (fid) => {
      const tiles = manifestTiles();
      if (tiles.length >= 16) return;
      if (sel) {
        const next = tiles.map((t) => t.name === sel
          ? { ...t, functions: [...(t.functions || []), fid] } : t);
        persistIoWires({ tiles: next });
      } else {
        persistIoWires({ tiles: [...tiles, {
          name: `tile${tiles.length}`, kind: "series",
          functions: [fid], aux: 2 }] });
      }
    });
  }

  // Category 2 — controls: TouchOSC-style composition — controls are
  // declared by TYPE group (faders / knobs / pads / buttons / external
  // inputs), then wired to functions via their CV points.
  const EXT_INPUTS = [
    ["mouse_xy", "mouse xy"],
    ["keyboard", "keyboard key"],
    ["gamepad_axes", "joystick axes"],
    ["gamepad_buttons", "joystick buttons"],
  ];
  function libControls(box) {
    // Category-based list: controls grouped by type (faders, knobs, pads,
    // buttons, external inputs, gates). Gates live ONLY here per spec.
    const cats = [
      { label: "faders", items: [{type:"fader",name:"slider",label:"slider",min:0,max:1,default:0.5}] },
      { label: "knobs", items: [{type:"encoder",name:"knob",label:"knob",min:0,max:1,default:0.5}] },
      { label: "pads", items: [{type:"xy",name:"pad",label:"xy pad"}] },
      { label: "buttons", items: [{type:"button",name:"btn_mom",label:"momentary",buttonType:"momentary"},
                             {type:"button",name:"btn_tog",label:"toggle",buttonType:"toggle"}] },
      { label: "external inputs", items: EXT_INPUTS.map(([k,l]) => ({type:"ext",name:k,input:k,label:l,key:k==="keyboard"?"space":""})) },
      { label: "gates (control library)", items: [
        {type:"gate",name:"gate_buffer",label:"gate_buffer"},
        {type:"gate",name:"gate_and",label:"gate_and"},
        {type:"gate",name:"gate_or",label:"gate_or"},
        {type:"gate",name:"gate_xor",label:"gate_xor"},
        {type:"gate",name:"gate_not",label:"gate_not"},
      ] },
    ];
    const header = document.createElement("p");
    header.className = "settings-sub"; header.style.color = "#778";
    header.textContent = "category-based control library · gates included";
    box.appendChild(header);
    for (const cat of cats) {
      const sub = document.createElement("p");
      sub.className = "settings-sub"; sub.style.color = "#778";
      sub.textContent = cat.label;
      box.appendChild(sub);
      const row = document.createElement("div"); row.className = "cfg-grid";
      for (const entry of cat.items) {
        const b = document.createElement("button"); b.type = "button";
        b.className = "io-pick"; b.textContent = entry.label || entry.name || entry.type;
        b.addEventListener("click", () => addCtl(entry));
        row.appendChild(b);
      }
      box.appendChild(row);
    }
    const addCtl = (entry) => {
      const controls = (fn && fn.io && fn.io.controls) || [];
      const name = entry.name ||
        `${entry.type}_${controls.length}`;
      const boxPos = rackControlBox({ type: entry.type }, controls.length);
      persistIoWires({ controls: [...controls,
                                  { ...entry, name, box: boxPos }] },
        () => attachOverlay(lastSelTile ||
          tileDeck.querySelector(`[data-fn="${CSS.escape(fn.id)}"]`) ||
          tileDeck.querySelector("[data-mtile]")));
    };
    const group = (label, entries) => {
      const sub = document.createElement("p");
      sub.className = "settings-sub";
      sub.style.color = "#778";
      sub.textContent = label;
      box.appendChild(sub);
      const row = document.createElement("div");
      row.className = "cfg-grid";
      for (const [label2, entry] of entries) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "io-pick";
        b.textContent = label2;
        b.addEventListener("click", () => addCtl(entry));
        row.appendChild(b);
      }
      box.appendChild(row);
    };
  }

  // Category 3 — viewports: framebuffer display presets + named
  // routing presets (tiles + wires snapshot).
  const VP_PRESETS_STORE = "mfx-vp-presets";
  function listViewports(box) {
    const row = document.createElement("div");
    row.className = "cfg-grid";
    for (const [kind, label] of [["series", "chart"],
                                 ["points3d", "3d scope"],
                                 ["lcd", "lcd matrix"],
                                 ["video", "video player"]]) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "io-pick";
      b.textContent = label;
      b.addEventListener("click", () => {
        const tiles = manifestTiles();
        if (tiles.length >= 16) return;
        persistIoWires({ tiles: [...tiles, {
          name: `tile${tiles.length}`, kind, aux: 2 }] });
      });
      row.appendChild(b);
    }
    box.appendChild(row);
    const sub = document.createElement("p");
    sub.className = "settings-sub";
    sub.style.color = "#778";
    sub.textContent = "routing presets (tiles + wires)";
    box.appendChild(sub);
    const prow = document.createElement("div");
    prow.className = "mfx-as-row";
    const pin = document.createElement("input");
    pin.placeholder = "preset name";
    const saveB = document.createElement("button");
    saveB.type = "button";
    saveB.className = "io-pick";
    saveB.textContent = "save";
    saveB.addEventListener("click", () => {
      const nm = (pin.value || "").trim();
      if (!nm || !fn) return;
      let all = {};
      try {
        all = JSON.parse(localStorage.getItem(VP_PRESETS_STORE)
                         || "{}") || {};
      } catch (_) { /* private */ }
      all[nm] = { tiles: manifestTiles(),
                  wires: (fn.io || {}).wires || [] };
      try {
        localStorage.setItem(VP_PRESETS_STORE, JSON.stringify(all));
      } catch (_) { /* private */ }
      paintPresets();
    });
    prow.append(pin, saveB);
    box.appendChild(prow);
    const list = document.createElement("div");
    list.className = "mfx-load-list";
    box.appendChild(list);
    const paintPresets = () => {
      list.innerHTML = "";
      let all = {};
      try {
        all = JSON.parse(localStorage.getItem(VP_PRESETS_STORE)
                         || "{}") || {};
      } catch (_) { /* private */ }
      for (const [nm, snap] of Object.entries(all)) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "io-pick";
        b.textContent = `${nm} · ${snap.tiles.length} tiles`;
        b.addEventListener("click", () => {
          persistIoWires({ tiles: snap.tiles, wires: snap.wires || [] });
        });
        list.appendChild(b);
      }
    };
    paintPresets();
  }

  // Category 4 — custom tiles: capture a display tile (kind + inner
  // chain + aux count) as a reusable block.
  function customStore() {
    try {
      return JSON.parse(localStorage.getItem("mfx-custom-tiles")
                        || "{}") || {};
    } catch (_) { return {}; }
  }
  function libCustom(box) {
    const cap = document.createElement("button");
    cap.type = "button";
    cap.className = "io-pick mfx-menu-row";
    cap.textContent = lastSelTile && lastSelTile.dataset.mtile
      ? `capture tile ${lastSelTile.dataset.mtile}`
      : "capture selected tile (click one first)";
    cap.addEventListener("click", () => {
      const mt = lastSelTile && lastSelTile.dataset.mtile;
      if (!mt) return;
      const entry = manifestTiles().find((t) => t.name === mt);
      if (!entry) return;
      const store = customStore();
      store[mt] = { ...entry };
      try {
        localStorage.setItem("mfx-custom-tiles",
                             JSON.stringify(store));
      } catch (_) { /* private */ }
      openLibraryPopup(cBL);   // repaint with the new entry
    });
    box.appendChild(cap);
    const list = document.createElement("div");
    list.className = "mfx-load-list";
    box.appendChild(list);
    for (const [nm, snap] of Object.entries(customStore())) {
      const wrap = document.createElement("div");
      wrap.className = "mfx-as-row";
      const b = document.createElement("button");
      b.type = "button";
      b.className = "io-pick";
      b.textContent = `${nm} · ${(snap.functions || []).join("+")
        || snap.kind}`;
      b.addEventListener("click", () => {
        const tiles = manifestTiles();
        if (tiles.length >= 16) return;
        const name = `${nm}_${tiles.length}`;
        persistIoWires({ tiles: [...tiles,
                                 { ...snap, name }] });
      });
      const del = document.createElement("button");
      del.type = "button";
      del.className = "io-pick mfx-del";
      del.textContent = "x";
      del.addEventListener("click", () => {
        const store = customStore();
        delete store[nm];
        try {
          localStorage.setItem("mfx-custom-tiles",
                               JSON.stringify(store));
        } catch (_) { /* private */ }
        openLibraryPopup(cBL);
      });
      wrap.append(b, del);
      list.appendChild(wrap);
    }
  }

  ioEl.addEventListener("click", (e) => {
    if (!appEdit || !fn) return;
    if (e.target.closest(".mfx-popup")) return;
    const sigRow = e.target.closest(".mfx-sig");
    if (sigRow) { openIoPicker("left", sigRow.dataset.sig); return; }
    if (e.target === ioEl) openIoPicker("left");
  });
  // backstop: empty space in the left column outside the form itself
  if (boxCol) boxCol.addEventListener("click", (e) => {
    if (!appEdit || !fn) return;
    if (e.target.closest(".mfx-popup")) return;
    if (e.target.closest("button, input, label, textarea, select")) return;
    if (e.target.closest("#io")) return;
    openIoPicker("left");
  });
  if (ioRight) ioRight.addEventListener("click", (e) => {
    if (!appEdit || !fn) return;
    if (e.target.closest(".mfx-popup")) return;
    if (e.target.closest("input,button,label")) return;
    const outRow = e.target.closest(".out-row");
    if (outRow) { openIoPicker("right", outRow.dataset.out); return; }
    if (e.target === ioRight) openIoPicker("right");
  });

  function renderForm() {
    ioEl.innerHTML = "";
    stopTriggerSources();
    if (!fn) { ioEl.hidden = true; renderRight(); return; }
    ioEl.hidden = false;
    // live input signals (left rail, read-only rows): ports the shell
    // keeps fed from the bus; the app samples them per frame.
    for (const sig of (fn.io && fn.io.ins) || []) {
      const row = document.createElement("div");
      row.className = "mfx-sig";
      row.dataset.sig = sig.name;
      const nm = document.createElement("span");
      nm.className = "mfx-sig-name";
      nm.textContent = sig.name;
      const topic = document.createElement("span");
      topic.className = "mfx-sig-topic";
      topic.textContent = sig.topic || "";
      const val = document.createElement("span");
      val.className = "mfx-sig-val";
      val.textContent = "—";
      // bus i/o point: patch cables leave the rail from here
      const pad = document.createElement("span");
      pad.className = "io-pad out";
      pad.dataset.port = `in:${String(sig.name).toLowerCase()}`;
      pad.title = `bus out · ${sig.name}`;
      row.append(nm, topic, val, pad);
      ioEl.appendChild(row);
    }
    // permanent layout rule: LEFT rail = inputs — app triggers first
    // (manual function triggers are inputs), then parameter fields.
    for (const t of (fn.io && fn.io.triggers) || []) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "mfx-trigger";
      b.dataset.trigger = t.name;
      b.textContent = t.label || t.name;
      b.title = `src: ${srcDesc(t.source)}`;
      if (isClockSrc(t)) {
        // clock-sourced: the button toggles the GATE. The clock is
        // already running; open passes ticks through, closed blocks.
        const key = gateKey(t);
        const paint = () => {
          const open = !!gateOpen[key];
          b.classList.toggle("gate-open", open);
          b.textContent = (t.label || t.name) + (open ? " · open" : "");
        };
        paint();
        b.addEventListener("click", () => {
          gateOpen[key] = !gateOpen[key];
          paint();
        });
      } else {
        b.addEventListener("click", () => fireTrigger(t.name, b));
      }
      ioEl.appendChild(b);
      if (t.source && t.source !== "manual") {
        const badge = document.createElement("span");
        badge.className = "mfx-trigger-src";
        badge.textContent = srcDesc(t.source);
        ioEl.appendChild(badge);
      }
    }
    startTriggerSources();
    // stream-i/o convention: MicroFX signal apps are driven through
    // the bus rails — the v1 parameter form stays out of App Mode
    const streamUi = mode !== "term" && isStreamApp(fn);
    if (!streamUi) {
      const fields = fn.fields || [];
      for (const f of fields) {
        const lab = document.createElement("label");
        lab.dataset.field = f.name;
        lab.appendChild(document.createTextNode(f.name.replace(/_/g, " ")));
        let inp;
        if (f.type === "select") {
          inp = document.createElement("input");
          inp.type = "hidden";
          inp.name = f.name;
          inp.value = f.value ?? "";
          ioEl.appendChild(inp);
          continue;
        } else if (f.type === "textarea" || (f.type === "text" && (f.name === "message" || f.name === "expr"))) {
          inp = document.createElement("textarea");
          inp.rows = 2;
          inp.value = f.value ?? "";
        } else {
          inp = document.createElement("input");
          inp.type = f.type === "number" ? "number" : "text";
          if (f.type === "number") inp.step = "any";
          inp.value = f.value ?? "";
        }
        inp.name = f.name;
        lab.appendChild(inp);
        ioEl.appendChild(lab);
      }
      const go = document.createElement("button");
      go.type = "submit";
      go.className = "fabric-engage";
      go.textContent = "Apply";
      ioEl.appendChild(go);
    }
    applyWhen();
    renderRight();
  }

  function applyWhen() {
    if (!fn) return;
    const vis = new Set(visibleFields(fn).map((f) => f.name));
    for (const lab of ioEl.querySelectorAll("label[data-field]")) {
      lab.hidden = !vis.has(lab.dataset.field);
    }
  }

  // Fire a named app trigger: the shell owns trigger buttons (left
  // rail); the iframe consumes them via its rising-edge tracker.
  function fireTrigger(name, btn) {
    const tile = fn && fn.id
      ? tileDeck.querySelector(`[data-fn="${CSS.escape(fn.id)}"]`) : null;
    const frame = tile && tile.querySelector("iframe.miniapp-frame");
    if (frame && frame.contentWindow) {
      try {
        frame.contentWindow.postMessage(
          { type: "io", fields: { [name]: 1 } }, "*");
      } catch (_) { /* sandbox */ }
      if (btn) {
        btn.classList.add("fired");
        setTimeout(() => btn.classList.remove("fired"), 200);
      }
    }
  }

  // ---- trigger sources ------------------------------------------------
  // A trigger is a gate on an io port. For EXTERNAL CLOCK sources the
  // clock runs persistently from app mount (before any manual action);
  // the manual button does not inject a pulse — it OPENS/CLOSES the
  // gate, and the clock passes through to the app for as long as the
  // gate stays open. Manual-only triggers keep one-shot pulses.
  const gateOpen = {};   // "fnId.trigger" -> bool (clock gates)
  let trigSources = [];   // active {stop()} handles for the current fn

  function gateKey(t) {
    return `${(fn && fn.id) || "?"}.${t.name}`;
  }

  function stopTriggerSources() {
    for (const s of trigSources) {
      try { s.stop(); } catch (_) { /* gone */ }
    }
    trigSources = [];
  }

  function srcDesc(src) {
    if (!src || src === "manual") return "manual";
    if (src.event === "clock") return `clock ${src.every_s}s · gated`;
    if (src.event === "sensor") {
      return `sensor ${src.topic} ${src.op || ">"} ${src.value}`;
    }
    if (src.event === "app") return `app ${src.app}.${src.out}`;
    return "manual";
  }

  function isClockSrc(t) {
    return !!(t.source && t.source !== "manual"
              && t.source.event === "clock");
  }

  function startTriggerSources() {
    stopTriggerSources();
    if (!fn || !fn.io) return;
    // live INPUT signals: sample the bus continuously and write the
    // latest value into the port — no trigger involved. Inputs are
    // always live captured signals, read per frame via input('name').
    for (const sig of fn.io.ins || []) {
      if (!sig.topic) continue;
      const port = String(sig.name).toLowerCase();
      const poll = async () => {
        try {
          const r = await fetch(`/api/sensors/history?topic=${
            encodeURIComponent(sig.topic)}&seconds=30`).then(
            (x) => x.json());
          const s = r.series || [];
          if (s.length) {
            const v = s[s.length - 1][1];
            const valEl = ioEl && ioEl.querySelector(
              `.mfx-sig[data-sig="${CSS.escape(sig.name)}"] .mfx-sig-val`);
            if (valEl) valEl.textContent = String(v);
            // bus -> wire targets (namespaced cell ports) or legacy mux
            routeValue(`in:${port}`, v);
          }
        } catch (_) { /* bus offline */ }
      };
      poll();
      const iv = setInterval(poll, 5000);
      trigSources.push({ stop: () => clearInterval(iv) });
    }
    for (const t of fn.io.triggers || []) {
      const src = t.source || "manual";
      if (src === "manual") continue;
      // the gate check lives at fire time: clock ticks persistently;
      // closed gates swallow them, open gates pass through to the port
      const fire = () => {
        if (isClockSrc(t) && !gateOpen[gateKey(t)]) return;
        fireTrigger(t.name, null);
      };
      if (src.event === "clock") {
        const iv = setInterval(fire,
          Math.max(100, (src.every_s || 1) * 1000));
        trigSources.push({ stop: () => clearInterval(iv) });
      } else if (src.event === "sensor") {
        let prev = false;
        const op = src.op || ">";
        const cmp = (a, b) => op === ">" ? a > b : op === ">=" ? a >= b
          : op === "<" ? a < b : a <= b;
        const poll = async () => {
          try {
            const r = await fetch(`/api/sensors/history?topic=${
              encodeURIComponent(src.topic)}&seconds=30`).then(
              (x) => x.json());
            const s = r.series || [];
            const cur = s.length ? cmp(s[s.length - 1][1], src.value)
              : false;
            if (cur && !prev) fire();   // rising transition of the gate
            prev = cur;
          } catch (_) { /* bus offline */ }
        };
        poll();
        const iv = setInterval(poll, 5000);
        trigSources.push({ stop: () => clearInterval(iv) });
      } else if (src.event === "app") {
        // cross-app: fires on each output message from the named app
        const h = (e) => {
          const d = e.data;
          if (d && d.type === "app-out" && d.name === src.out
              && d.app && String(d.app).toLowerCase()
                .includes(String(src.app).toLowerCase())) {
            fire();
          }
        };
        window.addEventListener("message", h);
        trigSources.push({
          stop: () => window.removeEventListener("message", h),
        });
      }
    }
  }

  function collect() {
    const vis = visibleFields(fn);
    const body = {};
    for (const f of vis) {
      const el = ioEl.querySelector(`[name="${f.name}"]`);
      if (!el) continue;
      let v = el.value;
      if (f.type === "number") v = v === "" ? null : Number(v);
      body[f.name] = v;
    }
    return body;
  }

  function drawSpace(canvas, fig, tile) {
    if (window.THREE) return drawSpace3(canvas, fig, tile);
    drawSpace2(canvas, fig, tile);
  }

  // three.js renderer (vendored /ui/vendor/three.min.js): real 3D
  // scene per viewport tile — lines for paths, spheres for bodies.
  function drawSpace3(canvas, fig, tile) {
    const w = Math.max(2, canvas.clientWidth);
    const h = Math.max(2, canvas.clientHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    if (!tile._three || tile._three.figRef !== fig) {
      const renderer = tile._three && tile._three.renderer;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);
      const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 2000);
      const group = new THREE.Group();
      scene.add(group);
      let extent = 1;
      const allPts = [];
      for (const name of Object.keys(fig.paths || {})) {
        const pts = (fig.paths[name] || [])
          .filter((p) => p.every(Number.isFinite));
        for (const p of pts) allPts.push(p);
        if (pts.length < 2) continue;
        const geom = new THREE.BufferGeometry().setFromPoints(
          pts.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
          color: 0xbaa4e5, transparent: true, opacity: 0.55 }));
        group.add(line);
      }
      for (const pl of fig.planets || []) {
        if (![pl.x, pl.y, pl.z].every(Number.isFinite)) continue;
        allPts.push([pl.x, pl.y, pl.z]);
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.02, 12, 12),
          new THREE.MeshBasicMaterial({ color: 0xeba75a }));
        mesh.position.set(pl.x, pl.y, pl.z);
        group.add(mesh);
      }
      for (const p of allPts) {
        extent = Math.max(extent, Math.abs(p[0]), Math.abs(p[1]),
                          Math.abs(p[2]));
      }
      tile._three = { renderer, scene, camera, group, extent,
                      figRef: fig };
    }
    const t3 = tile._three;
    if (!t3.renderer || t3.renderer.domElement !== canvas) {
      // three r149 wants webgl2; a FAILED acquisition poisons the
      // canvas for good (first draw can race layout on software-GL
      // heads), so swap in a fresh canvas and acquire there
      let cv = canvas;
      let gl = cv.getContext("webgl2", { antialias: true })
        || cv.getContext("webgl", { antialias: true });
      if (!gl) {
        cv = document.createElement("canvas");
        canvas.parentNode.replaceChild(cv, canvas);
        if (tile._fb) tile._fb.canvas = cv;
        tile._dragged = false;
        gl = cv.getContext("webgl2", { antialias: true });
      }
      t3.renderer = new THREE.WebGLRenderer(
        gl ? { canvas: cv, antialias: true, context: gl }
           : { canvas: cv, antialias: true });
    }
    const view = t3.renderer.domElement;
    if (view !== canvas) {
      // recompute the framebuffer geometry for the swapped canvas
      canvas = view;
      const cw = Math.max(2, canvas.clientWidth);
      const ch = Math.max(2, canvas.clientHeight);
      W = Math.round(cw * dpr);
      H = Math.round(ch * dpr);
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
    }
    t3.renderer.setSize(W, H, false);
    t3.renderer.setPixelRatio(dpr);
    t3.camera.aspect = W / H;
    t3.camera.updateProjectionMatrix();
    if (!tile._rot) tile._rot = { yaw: 0.6, pitch: 0.35 };
    const rot = tile._rot;
    const d = t3.extent * 2.6;
    t3.camera.position.set(
      d * Math.cos(rot.pitch) * Math.sin(rot.yaw),
      d * Math.sin(rot.pitch),
      d * Math.cos(rot.pitch) * Math.cos(rot.yaw));
    t3.camera.lookAt(0, 0, 0);
    t3.renderer.render(t3.scene, t3.camera);
    if (!tile._dragged) {
      const onDown = (e) => {
        tile._drag = { x: e.clientX, y: e.clientY };
        const move = (ev) => {
          if (!tile._drag) return;
          rot.yaw += (ev.clientX - tile._drag.x) * 0.008;
          rot.pitch += (ev.clientY - tile._drag.y) * 0.006;
          tile._drag = { x: ev.clientX, y: ev.clientY };
          drawSpace3(canvas, tile._fig, tile);
        };
        const up = () => {
          tile._drag = null;
          window.removeEventListener("mousemove", move);
          window.removeEventListener("mouseup", up);
        };
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", up);
      };
      canvas.addEventListener("mousedown", onDown);
      tile._dragged = true;
    }
  }

  // canvas-2D fallback (no WebGL / three unavailable)
  function drawSpace2(canvas, fig, tile) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!tile._rot) tile._rot = { yaw: 0.6, pitch: 0.35 };
    const rot = tile._rot;
    if (!tile._dragged) {
      const onDown = (e) => {
        tile._drag = { x: e.clientX, y: e.clientY };
        const move = (ev) => {
          if (!tile._drag) return;
          rot.yaw += (ev.clientX - tile._drag.x) * 0.008;
          rot.pitch += (ev.clientY - tile._drag.y) * 0.006;
          tile._drag = { x: ev.clientX, y: ev.clientY };
          fitCanvas(canvas, tile._fig, tile);
        };
        const up = () => {
          tile._drag = null;
          window.removeEventListener("mousemove", move);
          window.removeEventListener("mouseup", up);
        };
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", up);
      };
      canvas.addEventListener("mousedown", onDown);
      tile._dragged = true;
    }
    const cy = Math.cos(rot.yaw), sy = Math.sin(rot.yaw);
    const cp = Math.cos(rot.pitch), sp = Math.sin(rot.pitch);
    const scale = (Math.min(w, h) / 2) * 0.42 / 32; // Neptune ~30 AU
    const cx = w / 2, cyy = h / 2;
    const proj = (x, y, z) => {
      let x1 = x * cy - y * sy;
      let y1 = x * sy + y * cy;
      let z1 = z;
      let y2 = y1 * cp - z1 * sp;
      let z2 = y1 * sp + z1 * cp;
      return [cx + x1 * scale, cyy + z2 * scale];
    };
    // orbits
    ctx.strokeStyle = "rgba(150,140,200,.35)";
    for (const name of Object.keys(fig.paths || {})) {
      ctx.beginPath();
      (fig.paths[name] || []).forEach((p, i) => {
        const [px, py] = proj(p[0], p[1], p[2]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.closePath();
      ctx.stroke();
    }
    // sun
    ctx.fillStyle = "#ffd27f";
    ctx.beginPath();
    ctx.arc(cx, cyy, 4, 0, Math.PI * 2);
    ctx.fill();
    // planets
    ctx.font = "11px Antonio, sans-serif";
    for (const pl of fig.planets || []) {
      const [px, py] = proj(pl.x, pl.y, pl.z);
      ctx.fillStyle = "#eba75a";
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#baa4e5";
      ctx.fillText(pl.name, px + 5, py - 3);
    }
    ctx.fillStyle = "#778";
    ctx.fillText("drag to rotate", w - 90, h - 6);
  }

  function figureFrom(data) {
    if (!data || typeof data !== "object") return null;
    if (data.figure) return data.figure;
    if (Array.isArray(data.planets) && data.paths) {
      return { type: "space3d", planets: data.planets,
               paths: data.paths, caption: data.date || "solar system" };
    }
    if (Array.isArray(data.series)) {
      return { type: "series", series: data.series,
               ylabel: data.key || "", caption: data.topic || "series" };
    }
    if (Array.isArray(data.devices)) {
      const rows = data.devices.slice(0, 40).map((d) =>
        "<tr><td>" + [d.ip, d.name, d.conn, d.signal]
          .map((c) => String(c ?? "").replace(/[<&]/g, "_")).join("</td><td>")
          + "</td></tr>").join("");
      return { type: "html",
        html: "<style>.lantab{width:100%;border-collapse:collapse;" +
              "font:12px Antonio,sans-serif;color:#eba75a}" +
              ".lantab th{color:#baa4e5;text-align:left}" +
              ".lantab td{padding:1px 6px;border-bottom:1px solid #333}</style>" +
              "<table class='lantab'><tr><th>ip</th><th>name</th><th>conn</th>" +
              "<th>sig</th></tr>" + rows + "</table>",
        caption: "LAN · " + data.total + " devices" };
    }
    if (Array.isArray(data.readings)) {
      const rows = data.readings.slice(0, 40).map((r) => {
        let v = r.payload;
        if (typeof v === "object" && v !== null) {
          const k = Object.keys(v)[0];
          v = (k === "val" ? v.val : JSON.stringify(v)) ?? "";
        }
        return "<tr><td>" + [r.topic, String(v), String(r.ts || "").slice(11, 19)]
          .map((c) => String(c ?? "").replace(/[<&]/g, "_")).join("</td><td>")
          + "</td></tr>";
      }).join("");
      return { type: "html",
        html: "<style>.bus{width:100%;border-collapse:collapse;" +
              "font:12px Antonio,sans-serif;color:#eba75a}" +
              ".bus th{color:#baa4e5;text-align:left}" +
              ".bus td{padding:1px 6px;border-bottom:1px solid #333}</style>" +
              "<table class='bus'><tr><th>topic</th><th>value</th><th>ts</th></tr>"
              + rows + "</table>",
        caption: "sensor bus · " + data.readings.length };
    }
    if (data.services && data.host !== undefined) {
      const rows = Object.entries(data.services).map(([k, v]) =>
        "<tr><td>" + k + "</td><td style='color:" +
        ((v === true || (v && v.up)) ? "#7fbf7f" : "#f66") + "'>" +
        (v === true ? "up" : v === false ? "down" : JSON.stringify(v))
        .replace(/[<&]/g, "_") + "</td></tr>").join("");
      return { type: "html",
        html: "<style>.svc{width:100%;border-collapse:collapse;" +
              "font:12px Antonio,sans-serif;color:#eba75a}" +
              ".svc td{padding:1px 8px}</style><table class='svc'>" +
              rows + "</table>", caption: data.host };
    }
    if (data.dist_m && data.elev_m) {
      return { type: "xy", x: data.dist_m, y: data.elev_m,
        xlabel: "m", ylabel: "elev m", caption: data.verdict || "survey" };
    }
    if (data.azimuths_deg && data.distances_m) {
      const mx = Math.max(...data.distances_m, 1);
      return { type: "polar", theta_deg: data.azimuths_deg,
        power_norm: data.distances_m.map((d) => d / mx),
        caption: "horizon" };
    }
    if (Array.isArray(data.gpus)) {
      return { type: "bars",
        items: data.gpus.map((g) => ({
          label: g.card || "gpu",
          value: g.power_w ?? g.power_cap_w ?? 0,
          cap: g.power_cap_w,
        })),
        caption: "gpu power W" };
    }
    if (Array.isArray(data.heightmap) && data.heightmap.length) {
      return { type: "heatmap", grid: data.heightmap,
        caption: data.name || data.display || "terrain" };
    }
    return null;
  }

  function lcarsRamp(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [
      [0, 0, 0],
      [207, 79, 79],
      [235, 148, 58],
      [252, 193, 159],
      [186, 164, 229],
      [136, 153, 255],
    ];
    const x = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    return [
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f,
    ];
  }

  function drawFigure(canvas, fig, tile) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!fig) return;
    const square = fig.type === "matrix" || fig.type === "heatmap" || fig.type === "polar";
    const side = square ? Math.min(w, h) : 0;
    const ox = square ? Math.floor((w - side) / 2) : 0;
    const oy = square ? Math.floor((h - side) / 2) : 0;
    const dw = square ? side : w;
    const dh = square ? side : h;
    const pad = Math.max(12, Math.round(Math.min(dw, dh) * 0.04));
    if (fig.type === "series") {
      drawSeries(canvas, fig);
      return;
    }
    if (fig.type === "space3d") {
      drawSpace(canvas, fig, tile);
      return;
    }
    if (fig.type === "matrix" || fig.type === "heatmap") {
      const grid = fig.grid || [];
      const rows = grid.length, cols = (grid[0] || []).length;
      if (!rows || !cols) return;
      let mn = Infinity, mx = -Infinity;
      for (const row of grid) for (const v of row) {
        if (v < mn) mn = v; if (v > mx) mx = v;
      }
      const span = (mx - mn) || 1;
      const cw = (dw - pad * 2) / cols, ch = (dh - pad * 2) / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const t = (grid[r][c] - mn) / span;
          const [R, G, B] = lcarsRamp(t);
          ctx.fillStyle = `rgb(${R|0},${G|0},${B|0})`;
          ctx.fillRect(ox + pad + c * cw, oy + pad + r * ch, cw + 0.5, ch + 0.5);
        }
      }
      return;
    }
    if (fig.type === "xy") {
      const xs = fig.x || [], ys = fig.y || [];
      const n = Math.min(xs.length, ys.length);
      if (n < 2) return;
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      const ymin = Math.min(...ys), ymax = Math.max(...ys);
      const sx = (w - pad * 2) / ((xmax - xmin) || 1);
      const sy = (h - pad * 2) / ((ymax - ymin) || 1);
      ctx.strokeStyle = "#eb943a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = pad + (xs[i] - xmin) * sx;
        const y = h - pad - (ys[i] - ymin) * sy;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = "#baa4e5";
      ctx.font = "14px Antonio, sans-serif";
      ctx.fillText(fig.xlabel || "", pad, h - 6);
      ctx.fillText(fig.ylabel || "", 4, pad);
      return;
    }
    if (fig.type === "polar") {
      const th = fig.theta_deg || [];
      const p = fig.power_norm || [];
      const n = Math.min(th.length, p.length);
      const cx = ox + dw / 2, cy = oy + dh / 2, r = Math.min(dw, dh) * 0.38;
      ctx.strokeStyle = "#333";
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = "#eb943a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const a = (th[i] * Math.PI) / 180;
        const rr = r * Math.max(0, p[i]);
        const x = cx + rr * Math.sin(a);
        const y = cy - rr * Math.cos(a);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      if (n) ctx.closePath();
      ctx.stroke();
      return;
    }
    if (fig.type === "lcd") {
      drawLcd(canvas, fig);
      return;
    }
    if (fig.type === "video") {
      drawVideo(canvas, fig, tile);
      return;
    }
    if (fig.type === "bars") {
      const items = fig.items || [];
      const n = items.length || 1;
      const bw = (w - pad * 2) / n * 0.6;
      const gap = (w - pad * 2) / n;
      const mx = Math.max(...items.map((it) => it.cap || it.value || 1), 1);
      items.forEach((it, i) => {
        const bh = ((it.value || 0) / mx) * (h - pad * 2);
        const x = pad + i * gap + (gap - bw) / 2;
        ctx.fillStyle = i % 2 ? "#89f" : "#eb943a";
        ctx.fillRect(x, h - pad - bh, bw, bh);
        ctx.fillStyle = "#fcc19f";
        ctx.font = "13px Antonio, sans-serif";
        ctx.fillText(it.label, x, h - 8);
      });
    }
  }

  function scrollBottom(el) {
    if (el) { el.scrollTop = el.scrollHeight; }
  }

  function chatStatus(text) {
    if (termEl.firstChild && termEl.firstChild.nodeType === 3) {
      termEl.textContent = "";
    }
    const line = document.createElement("div");
    line.className = "status";
    line.textContent = text;
    termEl.appendChild(line);
    while (termEl.childElementCount > 80) termEl.removeChild(termEl.firstChild);
    scrollBottom(termEl);
    return line;
  }

  // Lane tag for the terminal — shows which harness the line is
  // coming from (chat/omp/pi/swarm/trace/atomize/improve/atomic/etc.).
  const LANE_BADGES = {
    duty:      ["duty",     "or"],
    chat:      ["chat",     "a2"],
    loop:      ["loop",     "a4"],
    omp:       ["omp",      "a5"],
    pi:        ["pi",       "a3"],
    code:      ["pi",       "a3"],
    pipecat:   ["voice",    "a7"],
    bicameral: ["biceph",   "a6"],
    swarm:     ["swarm",    "a8"],
    ui:        ["ui",       "a9"],
    command:   ["cmd",      "a1"],
    optimize:  ["opt",      "a4"],
  };

  function laneTag(lane) {
    const tag = LANE_BADGES[lane] || [lane || "duty", "or"];
    const span = document.createElement("span");
    span.className = "lane-tag lane-" + tag[1];
    span.textContent = "[" + tag[0] + "]";
    return span;
  }

  function chatStatusLane(lane, text) {
    if (termEl.firstChild && termEl.firstChild.nodeType === 3) {
      termEl.textContent = "";
    }
    const line = document.createElement("div");
    line.className = "status";
    line.appendChild(laneTag(lane));
    const t = document.createElement("span");
    t.className = "lane-text";
    t.textContent = text;
    line.appendChild(t);
    termEl.appendChild(line);
    while (termEl.childElementCount > 80) termEl.removeChild(termEl.firstChild);
    scrollBottom(termEl);
    return line;
  }

  async function engageChatStream(body) {
    const r = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const ct = r.headers.get("content-type") || "";
    if (!r.ok || !r.body || !ct.includes("event-stream")) {
      const e = new Error("stream unavailable");
      e.streamFallback = true;
      throw e;
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let pendTool = null;
    let finalData = null;
    const handle = (ev) => {
      if (ev.t === "llm") {
        chatStatusLane("chat", "▸ duty officer thinking · leg " + ev.round);
      } else if (ev.t === "repair") {
        chatStatusLane("chat", "▸ repairing truncated tool call");
      } else if (ev.t === "ctx") {
        chatStatusLane("chat", ev.retry
          ? "▸ context overflow — trimmed, retrying"
          : "▸ context trimmed to fit window");
      } else if (ev.t === "capped") {
        chatStatusLane("chat", "▸ stopped at " + ev.rounds + " legs (tool limit)");
      } else if (ev.t === "tool") {
        const args = String(ev.args || "");
        pendTool = chatStatusLane("chat", "▸ " + ev.name + "("
          + (args.length > 90 ? args.slice(0, 90) + "…" : args) + ") …");
      } else if (ev.t === "result" && pendTool) {
        pendTool.textContent = "";
        pendTool.appendChild(laneTag("chat"));
        const t = document.createElement("span");
        t.className = "lane-text";
        t.textContent = "▸ " + ev.name + " "
          + (ev.ok ? "ok" : "ERR") + " · " + (ev.ms / 1000).toFixed(1) + "s";
        pendTool.appendChild(t);
        pendTool.classList.add(ev.ok ? "ok" : "bad");
        pendTool = null;
      } else if (ev.t === "final") {
        finalData = ev;
      } else if (ev.t === "error") {
        throw new Error(ev.detail || "orchestrator failed");
      }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        const row = raw.split("\n").find((l) => l.startsWith("data: "));
        if (!row) continue;
        let ev;
        try { ev = JSON.parse(row.slice(6)); } catch { continue; }
        handle(ev);
      }
    }
    return finalData;
  }

  function ompLine(text, cls) {
    const line = document.createElement("div");
    if (cls) line.className = cls;
    line.textContent = text;
    termEl.appendChild(line);
    while (termEl.childElementCount > 200) termEl.removeChild(termEl.firstChild);
    scrollBottom(termEl);
    return line;
  }

  function ompLineLane(text, cls) {
    const line = document.createElement("div");
    if (cls) line.className = cls;
    line.appendChild(laneTag("omp"));
    const t = document.createElement("span");
    t.className = "lane-text";
    t.textContent = text;
    line.appendChild(t);
    termEl.appendChild(line);
    while (termEl.childElementCount > 200) termEl.removeChild(termEl.firstChild);
    scrollBottom(termEl);
    return line;
  }

  async function engageOmp(body) {
    const r = await fetch("/api/omp/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || "omp run failed to start");
    }
    log("cmd", "▸ omp: " + (body.message || "").slice(0, 90));
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let session = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        const row = raw.split("\n").find((l) => l.startsWith("data: "));
        if (!row) continue;
        let ev;
        try { ev = JSON.parse(row.slice(6)); } catch { continue; }
        if (ev.t === "session") {
          session = ev.id;
          ompLineLane("▸ omp session " + String(ev.id).slice(0, 8), "status");
        } else if (ev.t === "tool") {
          ompLineLane("▸ " + (ev.name || "tool") + " "
                      + (ev.detail || "").slice(0, 80), "status");
        } else if (ev.t === "toolout") {
          ompLine("  └ " + String(ev.text || "")
                  .replace(/\n/g, " ").slice(0, 140));
        } else if (ev.t === "delta") {
          for (const seg of String(ev.text).split("\n")) ompLine(seg);
        } else if (ev.t === "final") {
          ompLineLane("▸ omp finished (exit " + ev.code + ")",
                      ev.code ? "err" : "ok");
        }
      }
    }
    return { reply: "OMP run complete" + (session ? " · session "
            + session.slice(0, 8) : ""), omp_session: session };
  }

  async function engage(ev) {
    if (ev) ev.preventDefault();
    if (!fn) return;
    beep("audio4");
    const go = ioEl.querySelector(".fabric-engage");
    if (go) go.disabled = true;
    const body = collect();
    let url = fn.url;
    const opt = { method: fn.method, headers: {} };
    if (fn.url === "/chat") {
      opt.headers["Content-Type"] = "application/json";
      opt.body = JSON.stringify({
        message: body.message || "",
        department: dept ? dept.ext : null,
      });
    } else if (fn.method === "GET") {
      const q = new URLSearchParams();
      for (const [k, v] of Object.entries(body)) {
        if (v === null || v === undefined || v === "") continue;
        q.set(k, v);
      }
      const qs = q.toString();
      if (qs) url += (url.includes("?") ? "&" : "?") + qs;
    } else {
      opt.headers["Content-Type"] = "application/json";
      opt.body = JSON.stringify(body);
    }
    try {
      let data = null;
      if (fn.id === "omp") {
        data = await engageOmp(JSON.parse(opt.body));
      } else if (fn.url === "/chat") {
        try {
          data = await engageChatStream(JSON.parse(opt.body));
        } catch (se) {
          if (!se || !se.streamFallback) throw se;
          data = null;
        }
      }
      if (!data) {
        const r = await fetch(url, opt);
        data = await r.json();
        if (!r.ok) {
          log("err", data.detail || data, true);
          beep("audio3");
          return;
        }
      }
      log(fn.id, data);
      if (data.clarify || data.wizard) {
        if (data.spec && data.spec.id) {
          upsertFn(data.spec);
          applyFunction(data.spec.id);
        }
        if (mode === "split") setMode("viz");
        await openSettings(data.wizard ? "wizard" : "library", data);
        return;
      }
      if (data.spec && data.spec.id) upsertFn(data.spec);
      if (data.microapp && data.microapp.id) upsertFn(data.microapp);
      if (fn.id === "compose" && data.spec) applyFunction(data.spec.id);
      else if (data.microapp && data.microapp.id) applyFunction(data.microapp.id);
      const fig = figureFrom(data);
      if (fig) {
        if (fn && fn.span) fig.span = fn.span;
        if (fn && fn.live && fig.type === "series") {
          fig.live = { url: fn.url, interval: fn.live };
        }
        addTile(fig, (data.spec && data.spec.title) || fn.title);
        if (fig.live) {
          const lt = tileDeck.querySelector(".viz-tile:last-child");
          if (lt) registerLive(lt, fig);
        }
        if (data.gates && data.gates.pass === false) {
          if (mode === "split") setMode("viz");
          await openSettings("wizard", data);
        } else if (data.spec || data.microapp) {
          postTrace({
            query: lastQuery || (fn && fn.title) || "",
            action: "accept",
            spec: data.spec || data.microapp,
            gates: data.gates || { pass: true },
          });
        }
      } else if (fn.id === "chat" && data.reply) {
        addTile({ type: "html",
          html: "<pre class='miniapp-text'></pre>",
          caption: "duty" }, "duty");
        const pre = tileDeck.querySelector(".viz-tile:last-child .miniapp-text");
        if (pre) pre.textContent = data.reply;
        if (dutyLog) {
          const line = document.createElement("div");
          line.textContent = data.reply;
          dutyLog.appendChild(line);
          scrollBottom(dutyLog);
        }
      } else if (data.result != null && fn.id === "sage") {
        addTile({ type: "html",
          html: "<pre class='miniapp-text'></pre>",
          caption: "cas" }, "cas");
        const pre = tileDeck.querySelector(".viz-tile:last-child .miniapp-text");
        if (pre) pre.textContent = String(data.result);
      }
    } catch (e) {
      log("err", String(e), true);
      beep("audio3");
    } finally {
      if (go) go.disabled = false;
    }
  }

  function renderLiveCol() {
    if (!liveCol) return;
    liveCol.innerHTML = "";
    const all = wallTiles();
    live = all.map((tile) => ({
      id: tile.dataset.fn || "tile",
      tile: tile.dataset.tile,
      title: (tile.querySelector(".graph-caption") || {}).textContent || tile.dataset.fn || "tile",
    }));
    all.forEach((tile) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = (tile.querySelector(".graph-caption") || {}).textContent
        || tile.dataset.fn || "tile";
      if (soloId ? tile.dataset.tile === soloId : (fn && tile.dataset.fn === fn.id)) {
        b.classList.add("active");
      }
      b.addEventListener("click", () => soloTile(tile.dataset.tile));
      b.addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        closeTile(tile.dataset.tile);
      });
      liveCol.appendChild(b);
    });
  }

  function syncVizRails() {
    const cap = $("maxCapR");
    if (!cap) return;
    const has = ioRight && !ioRight.hidden && ioRight.childElementCount > 0;
    cap.hidden = mode === "viz" && has;
  }

  function outKey(o) {
    return `${(fn && fn.id) || "?"}.${o.name}`;
  }

  function outRoutingFor(o) {
    // output parameter: auto-route figure emissions to deck viewports
    return outRouting[outKey(o)] !== false;
  }

  function renderRight() {
    if (!ioRight) return;
    ioRight.innerHTML = "";
    // permanent layout rule: RIGHT rail = outputs + their parameters
    const outs = (fn && fn.io && fn.io.outs) || [];
    const selects = (fn && fn.fields || []).filter((f) => f.type === "select");
    if (!outs.length && !selects.length) {
      ioRight.hidden = mode === "viz" ? false : true;
      syncVizRails();
      return;
    }
    ioRight.hidden = false;
    if (outs.length) {
      const head = document.createElement("p");
      head.className = "right-label";
      head.textContent = "outputs";
      ioRight.appendChild(head);
      for (const o of outs) {
        const row = document.createElement("div");
        row.className = "out-row";
        row.dataset.out = o.name;
        // bus i/o point: patch cables enter the output rail here
        const pad = document.createElement("span");
        pad.className = "io-pad in";
        pad.dataset.port = `out:${String(o.name).toLowerCase()}`;
        pad.title = `bus in · ${o.name}`;
        row.appendChild(pad);
        const nm = document.createElement("span");
        nm.className = "out-name";
        nm.textContent = o.name.replace(/_/g, " ");
        row.appendChild(nm);
        const kind = document.createElement("span");
        kind.className = "out-badge";
        kind.textContent = o.kind || "number";
        row.appendChild(kind);
        if ((o.kind || "number") === "number") {
          const val = document.createElement("span");
          val.className = "out-val";
          val.textContent = "—";
          row.appendChild(val);
        } else {
          const st = document.createElement("span");
          st.className = "out-status";
          st.textContent = "idle";
          row.appendChild(st);
          const route = document.createElement("label");
          route.className = "out-route";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = outRoutingFor(o);
          cb.addEventListener("change", () => {
            outRouting[outKey(o)] = cb.checked;
            try {
              localStorage.setItem("mfx-out-routing",
                                   JSON.stringify(outRouting));
            } catch (_) { /* private */ }
          });
          route.appendChild(cb);
          route.appendChild(document.createTextNode("viewport"));
          row.appendChild(route);
        }
        ioRight.appendChild(row);
      }
    }
    if (!(mode !== "term" && isStreamApp(fn))) {
      selects.forEach((f) => {
      if (f.when) {
        const gate = ioEl.querySelector(`[name="${f.when.name}"]`);
        const ok = (f.when.in || []).map(String);
        if (!gate || !ok.includes(String(gate.value))) return;
      }
      const lab = document.createElement("div");
      lab.className = "right-label";
      lab.textContent = f.name.replace(/_/g, " ");
      ioRight.appendChild(lab);
      const cur = ioEl.querySelector(`[name="${f.name}"]`);
      (f.options || []).forEach((opt) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "lcars-btn";
        b.textContent = opt;
        if (cur && String(cur.value) === String(opt)) b.classList.add("active");
        b.addEventListener("click", () => {
          beep("audio2");
          if (cur) cur.value = opt;
          renderRight();
          applyWhen();
        });
        ioRight.appendChild(b);
      });
    });
    }
    if (!ioRight.childElementCount) {
      ioRight.hidden = mode === "viz" ? false : true;
    }
    syncVizRails();
  }

  function renderCtx() {
    ctxNav.innerHTML = "";
    depts.forEach((d, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `${PALETTE[i % PALETTE.length]} ctx`;
      if (dept && d.ext === dept.ext) b.classList.add("active");
      b.textContent = d.name;
      b.title = d.scope + (d.wired ? "" : " (standby)");
      b.addEventListener("click", () => { beep("audio2"); selectDept(d.ext); });
      ctxNav.appendChild(b);
    });
  }

  function currentTheme() {
    return document.documentElement.dataset.theme || "classic";
  }

  function applyTheme(id) {
    const spec = THEMES.find((t) => t.id === id) || THEMES[0];
    document.documentElement.dataset.theme = spec.id;
    try { localStorage.setItem(THEME_STORE, spec.id); } catch (_) { /* private */ }
    renderThemeGrid();
  }

  function renderThemeGrid() {
    const grid = $("themeGrid");
    if (!grid) return;
    grid.innerHTML = "";
    const cur = currentTheme();
    THEMES.forEach((t) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t.label;
      b.style.background = t.swatch;
      if (t.id === cur) b.classList.add("active");
      b.addEventListener("click", () => {
        beep("audio2");
        applyTheme(t.id);
      });
      grid.appendChild(b);
    });
  }

  function setSettingsTab(tab) {
    const known = { library: 1, wizard: 1, palette: 1, harness: 1 };
    if (!known[tab]) tab = "library";
    settingsTab = tab;
    const lib = $("libPane");
    const wizEl = $("wizPane");
    const pal = $("palettePane");
    const har = $("harnessPane");
    if (lib) lib.hidden = settingsTab !== "library";
    if (wizEl) wizEl.hidden = settingsTab !== "wizard";
    if (pal) pal.hidden = settingsTab !== "palette";
    if (har) har.hidden = settingsTab !== "harness";
    document.querySelectorAll("#settingsTabs [data-settab]").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-settab") === settingsTab);
    });
    const h = $("settingsH");
    if (h) {
      h.textContent = settingsTab === "wizard" ? "new miniapp"
        : settingsTab === "library" ? "library"
          : settingsTab === "harness" ? "harness" : "settings";
    }
  }

  function fillCfg(el, items, onpick) {
    if (!el) return;
    el.innerHTML = "";
    (items || []).forEach((it) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.id = it.id || "";
      b.textContent = it.label || it.id;
      if (it.id && fn && (fn.id === it.id || fn.kernel === it.id)) b.classList.add("active");
      b.addEventListener("click", () => { beep("audio2"); onpick(it); });
      el.appendChild(b);
    });
  }

  async function loadKit() {
    if (kitCache) return kitCache;
    try {
      kitCache = await fetch("/api/microapps/kit").then((r) => r.json());
    } catch (_) {
      kitCache = { libraries: [], kernels: [], templates: [] };
    }
    return kitCache;
  }

  // library: constructed APPS (loadable into viewports) + the
  // function/control catalog for build/edit mode
  async function renderLibrary(filter) {
    const el = $("libList");
    if (!el) return;
    const q = (filter || "").toLowerCase();
    el.innerHTML = "";
    let data = null;
    try {
      data = await fetch("/api/library").then((r) => r.json());
    } catch (_) { data = { apps: [], functions: { nodes: {} } }; }

    const head = (txt) => {
      const p = document.createElement("p");
      p.className = "settings-sub";
      p.textContent = txt;
      el.appendChild(p);
    };
    head("apps — load into a viewport");
    const apps = (data.apps || []).filter((a) =>
      !q || (a.title || "").toLowerCase().includes(q)
      || (a.id || "").toLowerCase().includes(q));
    for (const a of apps) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "cfg-btn lib-app";
      b.dataset.id = a.id;
      b.textContent = `${a.title} · ${a.kernel_kind || a.kernel || ""}`;
      b.addEventListener("click", async () => {
        try {
          const spec = await fetch(
            `/api/library/app/${encodeURIComponent(a.id)}`)
            .then((r) => r.json());
          upsertFn(spec.public_spec || spec);
          applyFunction(spec.id, spec.fields || []);
          addTile({ type: "html", html: spec.html || "",
                    caption: spec.title, sandbox: true }, spec.title);
          if (settings) settings.hidden = true;
          if (mode === "split") setMode("viz");
        } catch (e) { console.warn("library load:", e); }
      });
      el.appendChild(b);
    }
    if (!apps.length) {
      const p = document.createElement("p");
      p.className = "settings-sub";
      p.textContent = "no apps match";
      el.appendChild(p);
    }

    head("function + control library (build/edit)");
    const nodes = (data.functions || {}).nodes || {};
    for (const cat of ["source", "function", "control", "visualizer"]) {
      const list = nodes[cat] || [];
      if (!list.length) continue;
      const p = document.createElement("p");
      p.className = "settings-sub";
      p.style.color = "#778";
      p.textContent = `· ${cat} (${list.length})`;
      el.appendChild(p);
      for (const n of list) {
        if (q && !(n.id.includes(q) || n.title.toLowerCase().includes(q)))
          continue;
        const chip = document.createElement("div");
        chip.className = "cfg-gates lib-node";
        chip.dataset.node = n.id;
        chip.textContent =
          `${n.title} — in:[${n.inputs.join(",") || "-"}] ` +
          `out:[${n.outputs.join(",") || "-"}]` +
          (n.multi_in ? " ·multi" : "");
        el.appendChild(chip);
      }
    }
  }

  function routePreview(kit) {
    const rec = (kit.routes || []).find((r) =>
      r.dept === wiz.dept && r.input === wiz.inputs[0] && r.output === wiz.outputs[0]);
    const vp = wiz.viewports;
    const bind = rec ? (rec.bind || rec.kernel || (rec.stub ? "standby" : "?")) : "no route";
    return `${wiz.dept} · ${wiz.inputs[0]} → ${wiz.outputs[0]} · ${vp} · ${bind}`;
  }

  // ---- AppWizard: signal-path assembly (source → functions →
  // visualizer; control path rides the recipe) ----
  let sigCatalog = null;
  async function renderSignalWizard() {
    if (!sigCatalog) {
      try {
        sigCatalog = await fetch("/api/wizard/catalog").then((r) => r.json());
      } catch (_) { sigCatalog = { sources: [], visualizers: [], recipes: [] }; }
    }
    const cat = sigCatalog;
    fillCfg($("wizSource"), (cat.sources || []).map((s) => ({
      kind: "sigsource", id: s.id,
      label: s.label + (s.available === false ? " *" : ""),
    })), wizPick);
    markActive($("wizSource"), wiz.sigSource);
    const valid = new Set((cat.recipes || [])
      .filter((r) => r.source === wiz.sigSource)
      .map((r) => r.visualizer));
    const vizs = (cat.visualizers || []).map((v) => ({
      kind: "sigviz", id: v.id,
      label: v.label + (v.available === false ? " *" : ""),
    })).filter((v) => !valid.size || valid.has(v.id)
              || v.id === wiz.sigViz);
    fillCfg($("wizSigViz"), vizs, wizPick);
    if (!vizs.some((v) => v.id === wiz.sigViz) && vizs.length) {
      wiz.sigViz = vizs[0].id;
      markActive($("wizSigViz"), wiz.sigViz);
    } else {
      markActive($("wizSigViz"), wiz.sigViz);
    }
    const recipe = (cat.recipes || []).find((r) =>
      r.source === wiz.sigSource && r.visualizer === wiz.sigViz);
    const chain = $("wizSigChain");
    if (chain) {
      chain.textContent = recipe
        ? `functions: ${recipe.functions.join(", ") || "none"} · ` +
          `controls: ${recipe.controls.join(", ") || "none"}`
        : "no signal path for this pair";
    }
    const feed = $("wizFeed");
    if (feed) {
      const src = (cat.sources || []).find((s) => s.id === wiz.sigSource);
      feed.hidden = !(src && src.needs);
      feed.placeholder = src && src.needs === "topic"
        ? "bus topic e.g. ship/vllm/toks"
        : src && src.needs === "url" ? "media url" : "";
    }
  }

  async function wizSignalCreate() {
    const body = {
      source: wiz.sigSource,
      visualizer: wiz.sigViz,
      topic: ($("wizFeed") || {}).value || undefined,
      url: ($("wizFeed") || {}).value || undefined,
      title: undefined,
    };
    const r = await fetch("/api/wizard/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (data.error || !data.spec) {
      const prev = $("wizPreview");
      if (prev) { prev.hidden = false;
        prev.textContent = data.error || "generation failed"; }
      return;
    }
    if (settings) settings.hidden = true;
    upsertFn(data.spec);
    applyFunction(data.spec.id, data.spec.fields || []);
    addTile(data.figure, data.spec.title);
    if (mode === "split") setMode("viz");
  }

  async function renderWizard(clarify) {
    renderSignalWizard();
    const kit = await loadKit();
    if (clarify) {
      clarifyState = clarify;
      if (clarify.query) lastQuery = clarify.query;
    }
    const prev = $("wizPreview");
    if (prev) prev.textContent = (clarify && clarify.hint) || routePreview(kit);
    const g = (clarify && clarify.gates) || null;
    const gatesEl = $("cfgGates");
    if (gatesEl) {
      if (!g) gatesEl.hidden = true;
      else {
        gatesEl.hidden = false;
        const keys = ["intention", "static", "dynamic", "left_io"];
        gatesEl.textContent = keys.map((k) =>
          (k === "left_io" ? "left io" : k) + " " + (g[k] ? "ok" : "fail")
        ).join(" · ");
        gatesEl.classList.toggle("fail", g.pass === false);
      }
    }
    const depts = kit.departments || [];
    fillCfg($("wizDept"), depts.map((d) => ({
      kind: "dept", id: d.id, label: d.name + (d.wired ? "" : " *"),
    })), wizPick);
    markActive($("wizDept"), wiz.dept);
    fillCfg($("wizIn"), (kit.inputs || []).map((x) => ({
      kind: "input", id: x.id, label: x.label + (x.wired ? "" : " *"),
    })), wizPick);
    markActive($("wizIn"), wiz.inputs[0]);
    const allowed = (kit.routes || [])
      .filter((r) => r.dept === wiz.dept && r.input === wiz.inputs[0])
      .map((r) => r.output);
    const outs = (kit.outputs || []).filter((o) =>
      !allowed.length || allowed.indexOf(o.id) >= 0);
    fillCfg($("wizOut"), outs.map((x) => ({
      kind: "output", id: x.id,
      label: x.id + (x.wired === false ? " *" : ""),
    })), wizPick);
    markActive($("wizOut"), wiz.outputs[0]);
    fillCfg($("wizVp"), (kit.viewports || []).map((x) => ({
      kind: "viewports", id: String(x.id), label: x.label,
    })), wizPick);
    markActive($("wizVp"), String(wiz.viewports));
  }

  function markActive(el, id) {
    if (!el) return;
    [...el.children].forEach((b) => {
      b.classList.toggle("active", String(b.dataset.id) === String(id));
    });
  }

  async function openSettings(tab, clarify) {
    if (!settings) return;
    settings.hidden = false;
    const next = tab || (mode === "viz" ? "library" : "palette");
    setSettingsTab(next);
    if (settingsTab === "palette") renderThemeGrid();
    if (settingsTab === "library") renderLibrary(($("libSearch") || {}).value);
    if (settingsTab === "wizard") await renderWizard(clarify);
    if (settingsTab === "harness") await renderHarness();
  }

  async function renderHarness() {
    const status = $("harnessStatus");
    let data;
    try {
      data = await fetch("/api/harness").then((r) => r.json());
    } catch (e) {
      if (status) { status.hidden = false; status.textContent = "harness unreachable"; }
      return;
    }
    fillCfg($("harnessOrch"), (data.providers || []).map((p) => ({
      kind: "orch", id: p.id,
      label: p.label + (p.id === data.orchestrator ? " · on" : "")
        + (p.external && p.model ? " · " + p.model : ""),
    })), pickOrch);
    markActive($("harnessOrch"), data.orchestrator);
    fillCfg($("harnessWorkers"), (data.workers || []).map((w) => ({
      kind: "worker", id: w.id,
      label: (w.up ? "up · " : "down · ") + (w.role || w.id),
    })), () => {});
    const fields = {
      grok: { key: "keyGrok", model: "modelGrok", list: "modelsGrok" },
      kimi: { key: "keyKimi", model: "modelKimi", list: "modelsKimi" },
      "opencode-zen": { key: "keyZen", model: "modelZen", list: "modelsZen" },
    };
    (data.providers || []).forEach((p) => {
      const ids = fields[p.id];
      if (!ids) return;
      const keyEl = $(ids.key);
      if (keyEl) {
        keyEl.value = "";
        keyEl.placeholder = p.key_set ? ("set · " + (p.key_tail || "****")) : "paste key";
      }
      const modelEl = $(ids.model);
      if (modelEl) {
        modelEl.value = p.model || "";
        modelEl.placeholder = p.model || "model id";
      }
      const listEl = $(ids.list);
      if (listEl) fillModelList(listEl, p.models || []);
    });
    if (status) {
      status.hidden = false;
      const on = (data.providers || []).find((p) => p.id === data.orchestrator);
      const name = on && on.label ? on.label : "duty vLLM";
      const mid = on && on.external && on.model ? " · " + on.model : "";
      status.textContent = name + mid + " orchestrates · fetching live models…";
      status.classList.toggle("fail", false);
    }
    await overlayLiveModels();
  }

  function fillModelList(listEl, models) {
    listEl.innerHTML = "";
    (models || []).forEach((m) => {
      const o = document.createElement("option");
      o.value = m;
      listEl.appendChild(o);
    });
  }

  async function overlayLiveModels() {
    const status = $("harnessStatus");
    let live;
    try {
      live = await fetch("/api/harness/models").then((r) => r.json());
    } catch (_) {
      if (status) status.textContent = (status.textContent || "").replace(
        "fetching live models…", "catalog models (live fetch failed)");
      return;
    }
    const fields = {
      grok: { model: "modelGrok", list: "modelsGrok" },
      kimi: { model: "modelKimi", list: "modelsKimi" },
      "opencode-zen": { model: "modelZen", list: "modelsZen" },
    };
    const rows = live.providers || {};
    let liveN = 0;
    Object.keys(fields).forEach((pid) => {
      const row = rows[pid];
      if (!row) return;
      const ids = fields[pid];
      const listEl = $(ids.list);
      if (listEl && (row.models || []).length) fillModelList(listEl, row.models);
      if (row.live) liveN += 1;
    });
    if (status) {
      const base = (status.textContent || "").replace(" · fetching live models…", "");
      status.textContent = liveN
        ? base + " · live models from " + liveN + " provider" + (liveN === 1 ? "" : "s")
        : base + " · catalog models (add a key to refresh live lists)";
    }
  }

  async function pickOrch(it) {
    if (!it || !it.id) return;
    beep("audio2");
    const r = await fetch("/api/harness", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ orchestrator: it.id }),
    });
    const data = await r.json();
    if (!r.ok) {
      const status = $("harnessStatus");
      if (status) { status.hidden = false; status.textContent = data.detail || "save failed"; status.classList.add("fail"); }
      return;
    }
    await renderHarness();
  }

  async function saveHarness(ev) {
    if (ev) ev.preventDefault();
    const providers = {};
    [["grok", "keyGrok", "modelGrok"],
     ["kimi", "keyKimi", "modelKimi"],
     ["opencode-zen", "keyZen", "modelZen"]].forEach(([id, kid, mid]) => {
      const row = {};
      const keyEl = $(kid);
      const modelEl = $(mid);
      const key = keyEl && keyEl.value ? keyEl.value.trim() : "";
      const model = modelEl && modelEl.value ? modelEl.value.trim() : "";
      if (key) row.api_key = key;
      if (model) row.model = model;
      if (Object.keys(row).length) providers[id] = row;
    });
    const r = await fetch("/api/harness", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ providers }),
    });
    const data = await r.json();
    const status = $("harnessStatus");
    if (status) {
      status.hidden = false;
      status.textContent = r.ok ? "keys + models saved on the ship" : (data.detail || "save failed");
      status.classList.toggle("fail", !r.ok);
    }
    await renderHarness();
  }

  async function testHarness() {
    const status = $("harnessStatus");
    if (status) { status.hidden = false; status.textContent = "testing…"; status.classList.remove("fail"); }
    const r = await fetch("/api/harness/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await r.json();
    if (status) {
      status.textContent = data.ok
        ? ("ok · " + (data.provider || "") + " · " + (data.reply || "pong").slice(0, 80))
        : ("fail · " + (data.error || "no reply"));
      status.classList.toggle("fail", !data.ok);
    }
  }

  async function postTrace(payload) {
    try {
      await fetch("/api/microapps/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (_) { /* offline */ }
  }

  async function pickChoice(it) {
    if (!it) return;
    await postTrace({
      query: lastQuery || "",
      action: "choose",
      choice: it,
      spec: fn ? { id: fn.id, title: fn.title, kernel: fn.kernel, fields: fn.fields } : null,
    });
    clarifyState = null;
    if (settings) settings.hidden = true;
    const id = it.id;
    const mapped = catalogFns.find((f) => f.id === id);
    if (mapped) {
      applyFunction(mapped.id, it.fields || {});
      if (mode === "split") setMode("viz");
      await engage();
      return;
    }
    log("cfg", it.label || id);
  }

  function wizPick(it) {
    if (!it) return;
    if (it.kind === "dept") wiz.dept = it.id;
    if (it.kind === "input") wiz.inputs = [it.id];
    if (it.kind === "output") wiz.outputs = [it.id];
    if (it.kind === "viewports") wiz.viewports = Number(it.id) || 1;
    if (it.kind === "sigsource") wiz.sigSource = it.id;
    if (it.kind === "sigviz") wiz.sigViz = it.id;
    renderWizard(clarifyState);
  }

  async function wizCreate() {
    const q = lastQuery || "new miniapp";
    const body = {
      query: q,
      route: {
        dept: wiz.dept,
        inputs: wiz.inputs,
        outputs: wiz.outputs,
        viewports: wiz.viewports,
      },
    };
    const r = await fetch("/api/microapps/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.wizard && data.clarify && !data.spec && !data.bind) {
      await renderWizard(data);
      return;
    }
    if (settings) settings.hidden = true;
    if (data.bind && data.spec) {
      upsertFn(data.spec);
      applyFunction(data.spec.id, data.fields || {});
      if (mode === "split") setMode("viz");
      await engage();
      return;
    }
    if (data.spec) {
      upsertFn(data.spec);
      applyFunction(data.spec.id, data.fields || {});
    }
    const fig = figureFrom(data);
    if (fig) addTile(fig, (data.spec && data.spec.title) || q);
    if (mode === "split") setMode("viz");
  }

  function paintMaxTitle() {
    const title = $("maxTitle");
    if (!title) return;
    if (mode !== "viz") {
      title.textContent = "";
      title.dataset.fn = "";
      return;
    }
    title.textContent = ((fn && fn.title) || "STANDBY").toUpperCase();
    title.dataset.fn = (fn && fn.id) || "";
  }

  function paintLabels() {
    lblDept.replaceChildren(document.createTextNode((dept && dept.name || "STANDBY").toUpperCase()));
    lblDept.appendChild(hop("\u00a0" + (dept ? dept.ext : "000")));
    lblTask.replaceChildren(document.createTextNode((fn && fn.title || "COMMAND").toUpperCase()));
    lblTask.appendChild(hop("-" + (fn ? fn.id.slice(0, 2).toUpperCase() : "IN")));
    if (lblLive) {
      lblLive.textContent = (fn && fn.title || "STANDBY").toUpperCase();
    }
    paintMaxTitle();
  }

  function applyFunction(id, fields) {
    fn = catalogFns.find((f) => f.id === id) || fn;
    if (!fn) return;
    const ext = fn.group === "command" ? 100 : 400;
    dept = depts.find((d) => d.ext === ext) || dept;
    renderForm();
    if (fields) {
      for (const [k, v] of Object.entries(fields)) {
        const el = ioEl.querySelector(`[name="${k}"]`);
        if (el) el.value = v;
      }
      applyWhen();
    }
    renderLiveCol();
    renderRight();
    renderCtx();
    paintLabels();
    persist();
  }

  async function selectDept(ext) {
    ext = aliases[ext] || ext;
    dept = depts.find((d) => d.ext === ext) || depts[0];
    renderCtx();
    paintLabels();
    persist();
    log(dept.id, dept.wired ? dept.scope : (dept.note || "standby"));
    if (cmdIn) cmdIn.focus();
  }

  async function slashFetch(got) {
    const lane = got.lane || "command";
    chatStatusLane(lane, "running " + (got.lane || got.url || "") + "…");
    const opt = { method: got.method || "GET", headers: {} };
    let url = got.url;
    if ((got.method || "GET") === "POST") {
      opt.headers["Content-Type"] = "application/json";
      opt.body = JSON.stringify(got.body || {});
    }
    const r = await fetch(url, opt);
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("event-stream") && r.body) {
      // OMP lanes (/goal /code /omp): stream progress into the terminal
      // so long runs show life instead of a frozen "running omp…".
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", reply = "", finalEv = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dl = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (!dl) continue;
          let ev;
          try { ev = JSON.parse(dl.slice(5).trim()); } catch (e) { continue; }
          if (ev.t === "tool") {
            chatStatusLane(lane, "▸ " + ev.name + " "
              + String(ev.detail || "").slice(0, 90));
          } else if (ev.t === "goal") {
            chatStatusLane(lane, "▸ goal #" + ev.id + " registered");
          } else if (ev.t === "delta") {
            reply = ev.text;
            chatStatusLane(lane, "▸ "
              + String(ev.text || "").replace(/\s+/g, " ").slice(0, 120));
          } else if (ev.t === "final") {
            finalEv = ev;
          }
        }
      }
      if (!reply) reply = finalEv && finalEv.err
        ? "omp exited " + finalEv.code + ": " + finalEv.err
        : "(no OMP reply)";
      const fig = figureFrom({}) || {
        type: "html", html: "<pre class='miniapp-text'></pre>",
        caption: got.command || "omp", sandbox: false,
      };
      addTile(fig, got.command || "omp");
      const pre = tileDeck.querySelector(".viz-tile:last-child .miniapp-text");
      if (pre) pre.textContent = reply;
      if (dutyLog) {
        const line = document.createElement("div");
        line.appendChild(laneTag(lane));
        const t = document.createElement("span");
        t.className = "lane-text";
        t.textContent = reply;
        line.appendChild(t);
        dutyLog.appendChild(line);
        scrollBottom(dutyLog);
      }
      return;
    }
    const data = await r.json();
    let reply = (data.reply && String(data.reply).trim()) || data.error || null;
    if (!reply && Array.isArray(data.goals)) {
      reply = data.goals.length
        ? data.goals.map((g) => (g.status || "") + " " + (g.text || "")).join("\n")
        : "(no goals yet)";
    }
    if (!reply) {
      // Compact JSON for terminal display
      reply = JSON.stringify(data, null, 0).slice(0, 1500);
    }
    const fig = figureFrom(data) || {
      type: "html", html: "<pre class='miniapp-text'></pre>",
      caption: got.command || "pi", sandbox: false,
    };
    addTile(fig, got.command || "pi");
    const pre = tileDeck.querySelector(".viz-tile:last-child .miniapp-text");
    if (pre) pre.textContent = reply;
    if (dutyLog) {
      const line = document.createElement("div");
      line.appendChild(laneTag(lane));
      const t = document.createElement("span");
      t.className = "lane-text";
      t.textContent = reply;
      line.appendChild(t);
      dutyLog.appendChild(line);
      scrollBottom(dutyLog);
    }
  }

  function renderSlash(token) {
    const menu = $("slashMenu");
    if (!menu) return;
    const f = (token || "").toLowerCase();
    const items = slashCmds.filter((c) =>
      !f || c.id.startsWith(f) || (c.usage || "").toLowerCase().includes(f));
    if (!items.length) { menu.hidden = true; return; }
    slashIdx = Math.max(0, Math.min(slashIdx, items.length - 1));
    menu.innerHTML = "";
    items.forEach((c, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "option");
      const u = document.createElement("span");
      u.className = "u";
      u.textContent = c.usage;
      const bl = document.createElement("span");
      bl.className = "b";
      bl.textContent = c.blurb;
      b.appendChild(u);
      b.appendChild(bl);
      if (i === slashIdx) b.classList.add("active");
      b.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        pickSlash(c);
      });
      menu.appendChild(b);
    });
    menu.hidden = false;
    placeSlashMenu();
  }

  function placeSlashMenu() {
    const menu = $("slashMenu");
    if (!menu || menu.hidden || !cmdIn) return;
    const r = cmdIn.getBoundingClientRect();
    menu.style.left = r.left + "px";
    menu.style.width = r.width + "px";
    const below = window.innerHeight - r.bottom;
    if (below < 200 && r.top > below) {
      menu.style.top = "auto";
      menu.style.bottom = (window.innerHeight - r.top + 4) + "px";
    } else {
      menu.style.bottom = "auto";
      menu.style.top = (r.bottom + 4) + "px";
    }
  }

  function pickSlash(c) {
    if (!cmdIn || !c) return;
    const needs = /[<\[]/.test(c.usage || "");
    cmdIn.value = needs ? ("/" + c.id + " ") : ("/" + c.id);
    const menu = $("slashMenu");
    if (menu) menu.hidden = true;
    cmdIn.focus();
  }

  async function runCommand(q) {
    const text = (q || "").trim();
    if (!text) return;
    lastQuery = text;
    log("cmd", text);
    const prefer = (mode === "term" && !text.startsWith("/")) ? "&prefer_chat=1" : "";
    const r = await fetch("/api/resolve?q=" + encodeURIComponent(text) + prefer);
    const got = await r.json();
    if (got.ui === "library") {
      if (mode === "split") setMode("viz");
      await openSettings("library");
      return;
    }
    if (got.ui === "theme") {
      if (got.args) applyTheme(got.args);
      else {
        if (mode === "split") setMode("term");
        await openSettings("palette");
      }
      return;
    }
    if (got.ui === "harness") {
      if (mode === "split") setMode("term");
      await openSettings("harness");
      return;
    }
    if (got.slash && got.url && got.autorun) {
      await slashFetch(got);
      return;
    }
    if (got.fallback === "help") {
      log("index", (got.candidates || []).map((c) => c.title || c.id).join(" · "));
      return;
    }
    if (got.hint && !got.match && !got.url && !got.wizard && !got.ui) {
      log("hint", got.hint);
      return;
    }
    if (got.wizard || got.clarify) {
      if (got.match) applyFunction(got.match.id, got.fields || {});
      if (mode === "split") setMode("viz");
      await openSettings("wizard", got.clarify || got);
      if (cmdIn) cmdIn.focus();
      return;
    }
    if (got.match) {
      applyFunction(got.match.id, got.fields || {});
      if (got.autorun) await engage();
      else if (cmdIn) cmdIn.focus();
      return;
    }
    log("miss", "no match");
  }

  ioEl.addEventListener("submit", engage);
  const ctlTerm = $("ctlTerm");
  const ctlViz = $("ctlViz");
  if (ctlTerm) ctlTerm.addEventListener("click", () => { beep("audio2"); setMode("term"); });
  if (ctlViz) ctlViz.addEventListener("click", () => { beep("audio2"); setMode("viz"); });
  const cTL = $("cTL"), cTR = $("cTR"), cBL = $("cBL"), cBR = $("cBR");
  if (cBR) cBR.addEventListener("click", () => {
    beep("audio2");
    if (appEdit) setEditMode(false);
    closePopup();
    setMode("split");
  });
  if (cBL) cBL.addEventListener("click", () => {
    beep("audio2");
    if (mode !== "viz") {
      // terminal mode: full settings pane (harness lives here only)
      if (!settings) return;
      if (!settings.hidden) {
        settings.hidden = true;
        return;
      }
      openSettings(mode === "term" ? "harness" : "palette");
      return;
    }
    // App Mode: settings in run mode, component library in build/edit
    if (appEdit) openLibraryPopup(cBL);
    else openSettingsPopup(cBL);
  });
  document.querySelectorAll("#settingsTabs [data-settab]").forEach((b) => {
    b.addEventListener("click", () => {
      beep("audio2");
      const tab = b.getAttribute("data-settab");
      openSettings(tab, clarifyState);
    });
  });
  const harnessForm = $("harnessForm");
  if (harnessForm) harnessForm.addEventListener("submit", saveHarness);
  const harnessTest = $("harnessTest");
  if (harnessTest) harnessTest.addEventListener("click", () => { beep("audio2"); testHarness(); });
  const libSearch = $("libSearch");
  if (libSearch) {
    libSearch.addEventListener("input", () => renderLibrary(libSearch.value));
  }
  const appNameIn = $("appName");
  if (appNameIn) {
    appNameIn.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || !fn || !fn.id) return;
      e.preventDefault();
      const title = appNameIn.value.trim();
      if (!title) return;
      fetch(`/api/library/app/${encodeURIComponent(fn.id)}/rename`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }).then((r) => r.json()).then((data) => {
        if (data.error) { console.warn("rename:", data.error); return; }
        if (data.spec) upsertFn(data.spec);
        if (fn) fn.title = title;
        const t = $("maxTitle");
        if (t) t.textContent = appEdit ? title + " · edit" : title;
        appNameIn.blur();
      }).catch((err) => console.warn("rename:", err));
    });
  }
  const wizBtn = $("wizCreate");
  const wizSigBtn = $("wizSignalCreate");
  if (wizSigBtn) wizSigBtn.addEventListener("click", () => {
    wizSignalCreate().catch((e) => console.warn("signal wizard:", e));
  });
  if (wizBtn) wizBtn.addEventListener("click", () => {
    beep("audio4");
    wizCreate().catch((e) => log("err", String(e), true));
  });
  // Frame corners (App Mode): TL = build/edit switch, TR = app
  // management (new/save-as/load/rename; identical in both modes),
  // BL = settings — component library while the build switch is
  // active, BR = exit to mainUI/split.
  function setEditMode(on) {
    appEdit = on;
    document.body.classList.toggle("edit-mode", on);
    closePopup();
    const t = $("maxTitle");
    if (t && on) t.textContent = (t.textContent.replace(/ · edit$/, "")
      + " · edit").slice(0, 80);
    else if (t) t.textContent = (t.textContent || "")
      .replace(/ · edit$/, "");
    if (on && !isStreamApp(fn)) ensureDraftApp();
    requestAnimationFrame(() => { renderWires(); });
  }
  if (cTL) cTL.addEventListener("click", () => {
    beep("audio2");
    if (mode === "term") { cycleTerm(-1); return; }
    setEditMode(!appEdit);
  });
  if (cTR) cTR.addEventListener("click", () => {
    beep("audio2");
    if (mode === "term") { cycleTerm(1); return; }
    openAppMenu(cTR);
  });
  if (maxBox) {
    maxBox.querySelectorAll(".max-bar .seg[data-corner]").forEach((seg) => {
      seg.addEventListener("click", () => {
        const btn = $(seg.getAttribute("data-corner"));
        if (btn) btn.click();
      });
    });
  }
  const scaleDown = $("scaleDown");
  const scaleUp = $("scaleUp");
  if (scaleDown) {
    scaleDown.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (mode === "viz") bumpScale(-1);
    });
  }
  if (scaleUp) {
    scaleUp.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (mode === "viz") bumpScale(1);
    });
  }
  cmdForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const menu = $("slashMenu");
    if (menu) menu.hidden = true;
    beep("audio4");
    const q = cmdIn.value;
    cmdIn.value = "";
    runCommand(q).catch((e) => log("err", String(e), true));
  });
  if (cmdIn) {
    cmdIn.addEventListener("input", () => {
      const v = cmdIn.value;
      if (v.startsWith("/")) {
        slashIdx = 0;
        renderSlash(v.slice(1).split(" ")[0]);
      } else {
        const menu = $("slashMenu");
        if (menu) menu.hidden = true;
      }
    });
    cmdIn.addEventListener("keydown", (ev) => {
      const menu = $("slashMenu");
      if (!menu || menu.hidden) return;
      const buttons = [...menu.querySelectorAll("button")];
      if (!buttons.length) return;
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        slashIdx = (slashIdx + 1) % buttons.length;
        buttons.forEach((b, i) => b.classList.toggle("active", i === slashIdx));
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        slashIdx = (slashIdx - 1 + buttons.length) % buttons.length;
        buttons.forEach((b, i) => b.classList.toggle("active", i === slashIdx));
      } else if (ev.key === "Escape") {
        menu.hidden = true;
      } else if ((ev.key === "Tab" || ev.key === "Enter")
                 && !cmdIn.value.slice(1).includes(" ")) {
        ev.preventDefault();
        const hit = slashCmds.filter((c) => {
          const f = cmdIn.value.slice(1).toLowerCase();
          return !f || c.id.startsWith(f) || (c.usage || "").toLowerCase().includes(f);
        })[slashIdx];
        if (hit) pickSlash(hit);
        if (ev.key === "Enter" && hit && !/[<\[]/.test(hit.usage || "")) {
          cmdForm.requestSubmit();
        }
      }
    });
  }

  async function boot() {
    try {
      const [deptCat, fnCat, cmdCat] = await Promise.all([
        fetch("/api/departments").then((r) => r.json()),
        fetch("/api/console").then((r) => r.json()),
        fetch("/api/commands").then((r) => r.json()).catch(() => ({ commands: [] })),
      ]);
      slashCmds = cmdCat.commands || [];
      const sm = $("slashMenu");
      if (sm && sm.parentNode !== document.body) document.body.appendChild(sm);
      window.addEventListener("resize", placeSlashMenu);
      let resizeRaf = 0;
      window.addEventListener("resize", () => {
        cancelAnimationFrame(resizeRaf);
        resizeRaf = requestAnimationFrame(() => reflowWall());
      });
      rememberHomes();
      renderThemeGrid();
      applyShader();
      startExtInputs();
      if (tileDeck) {
        tileDeck.classList.add("wall");
        resetTiles();
      }
      depts = deptCat.departments || [];
      aliases = deptCat.aliases || {};
      catalogFns = fnCat.functions || [];
      const saved = restore();
      dept = depts.find((d) => d.ext === (aliases[saved.ext] || saved.ext || 400))
        || depts[0];
      renderCtx();
      renderLiveCol();
      renderRight();
      paintLabels();
      const q = new URLSearchParams(location.search);
      const th = q.get("theme");
      if (th) applyTheme(th);
      const mx = q.get("mode");
      if (mx === "viz" || mx === "term") setMode(mx);
      else if (saved.mode && saved.mode !== "split") setMode(saved.mode);
      if (cmdIn) cmdIn.focus();
      const seed = q.get("q") || q.get("fn");
      if (seed) {
        if (q.get("fn") && !q.get("q")) {
          applyFunction(q.get("fn"));
          if (q.get("engage") === "1") await engage();
        } else {
          await runCommand(seed);
        }
      }
      const st = q.get("settings");
      if (st === "library" || st === "wizard" || st === "palette" || st === "harness") {
        await openSettings(st);
      }
      if (q.get("slash") === "1" && cmdIn) {
        cmdIn.value = "/";
        slashIdx = 0;
        renderSlash("");
      }
    } catch (e) {
      log("err", "boot failed: " + e, true);
    }
  }
  // headless-verification handle (read-mostly; drives e2e probes)
  window.__fabric = {
    get fn() { return fn; },
    get appEdit() { return appEdit; },
    get shader() { return shader; },
    ioEl, tileDeck, ioRight,
    upsertFn, applyFunction, addTile,
    openSettings, wizSignalCreate, setMode, setEditMode,
    openAppMenu: () => openAppMenu(cTR),
    openSettingsPopup: () => openSettingsPopup(cBL),
    openLibraryPopup: () => openLibraryPopup(cBL),
    renderWires, applyShader, routeValue,
  };
  boot();
})();
