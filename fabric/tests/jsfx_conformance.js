// MicroFX/EEL2-subset conformance vectors (run via node).
// Usage: node fabric/tests/jsfx_conformance.js
//   optional env MFX_FIXTURES=path.json supplies module sources/params.
const fs = require("fs");
const path = require("path");
const M = require(__dirname + "/../web/jsfx.js");

let FIX = { source: {}, params: {} };
try {
  FIX = JSON.parse(fs.readFileSync(
    process.env.MFX_FIXTURES ||
    path.join(__dirname, "microfx_modules.json"), "utf8"));
  M.registerModules(FIX.source, FIX.params);
} catch (_) { /* patch vectors will fail visibly below */ }

const evalProg = (src) => {
  const it = new M.Interpreter({});
  let got = 0;
  for (const st of M.parse(M.lex(src))) got = it.evalNode(st);
  return got;
};

let pass = 0, total = 0;
function check(label, got, want, tol) {
  total++;
  const ok = tol !== undefined
    ? Math.abs(got - want) <= tol
    : got === want;
  if (!ok) console.error("FAIL", label, "got", got, "want", want);
  else pass++;
}

// ------------------------------------------------- language core
[
  ["1 + 2 * 3;", 7],
  ["x = 5; x += 3; x;", 8],
  ["a = 2; a > 1 ? (b = 10; b;) : (b = 20; b;);", 10],
  ["v = 0; loop(5, v += 4;); v;", 20],
  ["w = 1; while(w < 100)( w *= 3; ); w;", 243],
  ["buf = 0; buf[0] = 42; buf[0];", 42],
  ["p = 100; p[5] = 7; mem[105];", 7],
  ["gmem[3] = 9; gmem[3];", 9],
  ["$x10 + 1;", 17],
  ["max(3,9) + min(2,6);", 11],
  ["y = -(-5); y;", 5],
  ["2 ^ 3 ^ 1;", 8],
  ["7 % 3 + 1;", 2],
  ["(a=1;b=2;a+b;)*10;", 30],
  ["c = 0; loop(3, c = c == 0 ? 5 : c + 1;); c;", 7],
  ["n = 0; i = 0; loop(4, i += 1; n += i;); n;", 10],
].forEach(([src, want]) => {
  let got;
  try { got = evalProg(src); }
  catch (e) { console.error("ERR", JSON.stringify(src), e.message); return; }
  check("lang " + src.slice(0, 28), got, want);
});

// ------------------------------------------------- app-level io
// trigger(): rising-edge semantics shared with runProgram
{
  const inputs = { go: 0 };
  const edge = M.makeEdgeTracker((k) => inputs[k]);
  let fires = 0, seriesKind = null, seriesLen = -1, seriesSum = -1;
  const it = new M.Interpreter({
    input: (nm) => inputs[String(nm).toLowerCase()] || 0,
    trigger: (nm) => {
      const r = edge(String(nm));
      if (r) fires += 1;
      return r;
    },
    outData: (kind, nm, start, count) => {
      if (nm === "trace") {
        seriesKind = kind;
        seriesLen = count;
        seriesSum = 0;
        for (let i = 0; i < count; i++) seriesSum += it.mem[start + i];
      }
    },
  });
  it.mem[10] = 3;
  it.mem[11] = 4;
  it.mem[12] = 5;
  const prog = M.parse(M.lex(
    "trigger('go') ? output('hit', 1) : 0; out_series('trace', 10, 3);"));
  const tick = () => { for (const st of prog) it.evalNode(st); };
  inputs.go = 1; tick();   // rise -> fires
  inputs.go = 1; tick();   // held -> no fire
  inputs.go = 0; tick();   // falls
  inputs.go = 7; tick();   // rises again (any >0 is high)
  check("trigger fires once per rise", fires, 2);
  check("out_series kind", seriesKind, "series");
  check("out_series len", seriesLen, 3);
  check("out_series mem slice sum", seriesSum, 12);
}
// out_points3d: metadata + mem slice
{
  let meta = null, sum = -1;
  const it = new M.Interpreter({
    outData: (kind, nm, start, count) => {
      if (nm === "scene") {
        meta = { kind, start, count };
        sum = 0;
        for (let i = 0; i < count; i++) sum += it.mem[start + i];
      }
    },
  });
  for (const st of M.parse(M.lex(
      "mem[100]=1; mem[101]=2; mem[102]=3;"
      + "out_points3d('scene', 100, 3);"))) it.evalNode(st);
  check("points3d meta", JSON.stringify(meta),
        JSON.stringify({ kind: "points3d", start: 100, count: 3 }));
  check("points3d mem sum", sum, 6);
}

// ------------------------------------------------- mdct/imdct
const N = 128, H = 64;
[
  [`buf=0;i0=0;
    loop(128, buf[i0]=cos((3.14159/128)*(i0+32.5)*19)*8; i0+=1;);
    mdct(0,128);
    best=0;bi=-1;k=0;loop(64,m=abs(buf[k]);m>best?(best=m;bi=k):0;k+=1;);bi;`,
   9, "tone->bin"],
  // TDAC perfect reconstruction; coeffs stashed before time-domain restore
  [`SIG=500000; A=700000; CA=800000; B=900000; CB=950000;
    i0=0;
    loop(192, mem[SIG+i0]=1+sin(i0*0.31)+cos(i0*0.07)*0.5; i0+=1;);
    i0=0; loop(128, mem[A+i0]=mem[SIG+i0]; i0+=1;);
    mdct(A, 128);
    i0=0; loop(64, mem[CA+i0]=mem[A+i0]; i0+=1;);
    i0=0; loop(128, mem[A+i0]=mem[SIG+i0]; i0+=1;);
    i0=0; loop(128, mem[B+i0]=mem[SIG+64+i0]; i0+=1;);
    mdct(B, 128);
    i0=0; loop(64, mem[CB+i0]=mem[B+i0]; i0+=1;);
    i0=0; loop(128, mem[B+i0]=mem[SIG+64+i0]; i0+=1;);
    imdct(CA, 128);
    imdct(CB, 128);
    worst=0;i0=0;
    loop(64,
      d=abs(mem[CA+64+i0]+mem[CB+i0]-mem[SIG+64+i0]);
      d>worst?worst=d:0;i0+=1;);
    worst;`, 0, "TDAC recon"],
].forEach(([src, want, label]) => {
  let got;
  try { got = evalProg(src); }
  catch (e) { console.error("ERR", label, e.message); return; }
  check("mdct " + label, got, want, 1e-6);
});

// validation throws
for (const [call, frag] of [
  ["mdct(0,100)", "64|128"],
  ["mdct(65500,128)", "boundary"]]) {
  total++;
  try { evalProg(call); console.error("FAIL validate ran:", call); }
  catch (e) {
    if (e.message.includes(frag)) pass++;
    else console.error("FAIL validate msg:", call, e.message);
  }
}

// ------------------------------------------------- CV-module patches
if (!FIX.source || !Object.keys(FIX.source).length) {
  console.error("SKIP patch vectors (no fixtures)");
} else {
  const patches = [
    { label: "const->gain", ticks: 120,
      check: (r) => Math.abs(r.final["g1.cv"] - 10) < 0.01,
      patch: { modules: [
        { id: "c1", primitive: "const", params: { value: 5 } },
        { id: "g1", primitive: "gain", params: { factor: 2 } }],
        wires: [{ from: "c1.cv", to: "g1.in" }],
        views: [{ module: "g1", as: "series" }] } },
    { label: "smooth converges", ticks: 200,
      check: (r) => Math.abs(r.final["s1.cv"] - 7) < 0.05,
      patch: { modules: [
        { id: "k", primitive: "const", params: { value: 7 } },
        { id: "s1", primitive: "smooth", params: { alpha: 0.5 } }],
        wires: [{ from: "k.cv", to: "s1.in" }], views: [] } },
    { label: "mdct_flux finite", ticks: 150,
      check: (r) => Number.isFinite(r.final["f.flux"]),
      patch: { modules: [
        { id: "k2", primitive: "const", params: { value: 3 } },
        { id: "f", primitive: "mdct_flux", params: {} }],
        wires: [{ from: "k2.cv", to: "f.in" }], views: [] } },
    // counter app: two taps -> edge counter counts 2 (one input node)
    { label: "counter via taps", ticks: 60,
      ui_taps: [5, 30],
      check: (r) => r.final["acc.acc"] === 2,
      patch: { modules: [
        { id: "ui", primitive: "tap", params: {} },
        { id: "acc", primitive: "accum", params: { per_tick: 1 } }],
        wires: [{ from: "ui.tap", to: "acc.in" }],
        views: [] } },
    // Rack stackable inputs: two cables into one input SUM
    // (engine/Engine.cpp Engine_stepFrameCables)
    { label: "stacked inputs sum", ticks: 5,
      check: (r) => Math.abs(r.final["g.cv"] - 8) < 0.01,
      patch: { modules: [
        { id: "c1", primitive: "const", params: { value: 5 } },
        { id: "c2", primitive: "const", params: { value: 3 } },
        { id: "g", primitive: "gain", params: { factor: 1 } }],
        wires: [
          { from: "c1.cv", to: "g.in" },
          { from: "c2.cv", to: "g.in" }],
        views: [] } },
    // fan-out: one output feeds two nodes freely
    { label: "fan-out", ticks: 5,
      check: (r) => r.final["g1.cv"] === 5 && r.final["g2.cv"] === 5,
      patch: { modules: [
        { id: "c", primitive: "const", params: { value: 5 } },
        { id: "g1", primitive: "gain", params: { factor: 1 } },
        { id: "g2", primitive: "gain", params: { factor: 1 } }],
        wires: [
          { from: "c.cv", to: "g1.in" },
          { from: "c.cv", to: "g2.in" }],
        views: [] } },
  ];
  var pp_count = 0;
  var patch_total = patches.length;
  let pp = 0;
  for (const pc of patches) {
    const r = M.evaluatePatch(pc.patch, pc.ticks,
      Object.assign({ dt: 1 / 30 }, pc.ui_taps
        ? { ui_taps: pc.ui_taps } : {}));
    if (pc.check(r)) { pp++; pp_count++; }
    else console.error("FAIL patch", pc.label,
      "->", JSON.stringify(r.final).slice(0, 160));
  }
  console.log(pp + "/" + patches.length + " patches ok");
}

const allPass = pass === total;
console.log(pass + "/" + total + " ok" +
  (Object.keys(FIX.source).length ? " +" + pp_count + " patches" : ""));
process.exit(allPass && pp_count === patch_total ? 0 : 1);

// ------------------------------------------ logic gate truth tables
// Boolean convention: in > 0.5 is true; outputs 1/0. Quantum gates act
// on bipolar (±1) streams.
{
  const gates = {
    gate_buffer: (a, b) => a, gate_not: (a, b) => 1 - a,
    gate_and: (a, b) => a & b, gate_or: (a, b) => a | b,
    gate_nand: (a, b) => 1 - (a & b), gate_nor: (a, b) => 1 - (a | b),
    gate_xor: (a, b) => a ^ b, gate_xnor: (a, b) => 1 - (a ^ b),
    gate_imply: (a, b) => (a && b) || (!a) ? 1 : 0,
    gate_nimply: (a, b) => (a && !b) ? 1 : 0,
  };
  for (const [name, want] of Object.entries(gates)) {
    const it = new M.Interpreter({ has: () => true });
    const vals = {};
    it.host.output = (nm, v) => { vals[nm] = v; };
    const secs = M.splitSections(
      (FIX.source && FIX.source[name]) || "");
    const tick = M.parse(M.lex(secs.tick || ""));
    let okAll = true;
    for (const a of [0, 1]) for (const b of [0, 1]) {
      it.host.input = (nm) => nm === "a" ? a : nm === "b" ? b : 0;
      if (secs.init) for (const st of M.parse(M.lex(secs.init))) it.evalNode(st);
      for (const st of tick) it.evalNode(st);
      if (vals.q !== want(a, b)) okAll = false;
    }
    check(`gate ${name} truth table`, okAll ? 1 : 0, 1);
  }
  // quantum: pauli flips bipolar; hadamard scales by 1/sqrt2
  {
    const it = new M.Interpreter({ has: () => true });
    let out = null;
    it.host.output = (nm, v) => { out = v; };
    for (const [name, v, want] of [
      ["pauli_x", 1, -1], ["pauli_x", -1, 1],
      ["pauli_z", 1, -1],
      ["hadamard_gate", 1, Math.SQRT1_2],
      ["phase_s", 1, 1], ["phase_t", -1, -1],
    ]) {
      const secs = M.splitSections(FIX.source[name]);
      it.host.input = () => v;
      for (const st of M.parse(M.lex(secs.tick))) it.evalNode(st);
      check(`qgate ${name}(${v})`, Math.abs(out - want) < 1e-9 ? 1 : 0, 1);
    }
  }
  // cnot: target flips (bipolar sign) only when control is high
  {
    const it = new M.Interpreter({ has: () => true });
    const vals = {};
    it.host.output = (nm, v) => { vals[nm] = v; };
    const secs = M.splitSections(FIX.source.cnot);
    const tick = M.parse(M.lex(secs.tick));
    for (const [c, t, want] of [[1, 1, -1], [1, -1, 1], [0, 1, 1]]) {
      it.host.input = (nm) => nm === "c" ? c : t;
      for (const st of tick) it.evalNode(st);
      check(`cnot c=${c} t=${t}`, vals.out === want ? 1 : 0, 1);
    }
  }
}
