/* MicroFX — JSFX/EEL2-inspired runtime for fabric MiniApps.
 *
 * Programs are declared as JSON: {id,title,params[],io{},source}
 * where source contains REAPER-JSFX-style sections (@init/@slider/@gfx)
 * written in the EEL2 subset implemented here:
 *   - undeclared double variables, case-insensitive, namespaced (a.b)
 *   - $pi/$e/$phi, $xHEX, $'c'
 *   - [] local memory (1M slots), gmem[] shared per page
 *   - full operator set incl. ?: with ( ) statement blocks
 *   - loop(count,code) / while(cond)(code), 1e6 iteration cap
 *   - math builtins + time()/time_precise()
 *   - host surface: params become globals; gfx_* drawing; mouse_x/y/cap;
 *     input(name); output(name,v); store/load (localStorage JSON)
 *
 * Node-compatible: the interpreter exports itself when `module` exists,
 * so pytest can run conformance vectors without a browser.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.MicroFX = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  let __MFX_MODULES_SOURCE = {};
  let __MFX_MODULES_PARAMS = {};
  function registerModules(sourceMap, paramsMap) {
    __MFX_MODULES_SOURCE = sourceMap;
    __MFX_MODULES_PARAMS = paramsMap;
  }

  // ---------------------------------------------------------------- lexer
  function lex(src) {
    const toks = [];
    let i = 0;
    const n = src.length;
    while (i < n) {
      const c = src[i];
      if (c === " " || c === "\t" || c === "\n" || c === "\r") { i++; continue; }
      if (c === "/" && src[i + 1] === "/") { while (i < n && src[i] !== "\n") i++; continue; }
      if (c === "/" && src[i + 1] === "*") {
        i += 2;
        while (i < n && !(src[i] === "*" && src[i + 1] === "/")) i++;
        i += 2; continue;
      }
      if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(src[i + 1] || ""))) {
        let j = i;
        while (j < n && /[0-9._]/.test(src[j])) j++;
        toks.push({ t: "num", v: parseFloat(src.slice(i, j).replace(/_/g, "")) });
        i = j; continue;
      }
      if (c === "$") {
        if (src[i + 1] === "'") {
          const ch = src[i + 2];
          toks.push({ t: "num", v: (src[i + 3] === "\\")
            ? src.charCodeAt(i + 4) : src.charCodeAt(i + 2) });
          i += (src[i + 3] === "\\") ? 5 : 4; continue;
        }
        let j = i + 1;
        if (src[j] === "x" || src[j] === "X") {
          j++;
          while (j < n && /[0-9a-fA-F]/.test(src[j])) j++;
          toks.push({ t: "num", v: parseInt(src.slice(i + 2, j), 16) });
          i = j; continue;
        }
        if (src[j] === "~") {
          j++;
          while (j < n && /[0-9a-fA-F]/.test(src[j])) j++;
          toks.push({ t: "num", v: ~parseInt(src.slice(i + 2, j), 16) });
          i = j; continue;
        }
        j = i + 1;
        while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
        const name = src.slice(i + 1, j).toLowerCase();
        const consts = { pi: Math.PI, e: Math.E, phi: 1.6180339887498948 };
        toks.push({ t: "num", v: consts[name] !== undefined ? consts[name] : 0 });
        i = j; continue;
      }
      if (c === "\"" || c === "'") {
        const q = c;
        let j = i + 1, out = "";
        while (j < n && src[j] !== q) {
          out += src[j] === "\\" ? src[j + 1] || "" : src[j];
          j += src[j] === "\\" ? 2 : 1;
        }
        toks.push({ t: "str", v: out });
        i = j + 1; continue;
      }
      if (/[A-Za-z_.]/.test(c)) {
        let j = i;
        while (j < n && /[A-Za-z0-9_.]/.test(src[j])) j++;
        toks.push({ t: "id", v: src.slice(i, j) });
        i = j; continue;
      }
      const three = src.substr(i, 3);
      const two = src.substr(i, 2);
      if (["===", "!=="].includes(three)) { toks.push({ t: three }); i += 3; continue; }
      if (["==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=",
           "%=", "^=", "&=", "|=", "~="].includes(two)) {
        toks.push({ t: two }); i += 2; continue;
      }
      toks.push({ t: c }); i += 1;
    }
    return toks;
  }

  // ------------------------------------------------------------- parser
  // Grammar: statements sep by ';'. Expression grammar with ?:
  // statement-blocks inside ( ). Returns AST nodes:
  //   {k:"num"|"id",v} {k:"mem",base,idx} {k:"bin",op,a,b}
  //   {k:"un",op,a} {k:"assign",target,op,val} {k:"cond",c,a,b}
  //   {k:"block",[stmts]} {k:"call",name,args} {k:"loop",count,body}
  //   {k:"while",cond,body} {k:"whileC",body,cond}
  function parse(toks) {
    let p = 0;
    const peek = () => toks[p];
    const next = () => toks[p++];
    const expect = (t) => {
      const tok = next();
      if (!tok || tok.t !== t) throw new Error("expected " + t + " got " + (tok && tok.t));
      return tok;
    };

    function stmtListUntilClose() {
      const stmts = [];
      for (;;) {
        while (peek() && peek().t === ";") next();
        if (!peek() || peek().t === ")") break;
        stmts.push(parseStmt());
      }
      expect(")");
      return stmts.length === 1 ? stmts[0] : { k: "block", s: stmts };
    }

    function parseBlockOrExpr() {
      if (peek() && peek().t === "(") {
        next();
        const stmts = [];
        while (peek() && peek().t !== ")") stmts.push(parseStmt());
        expect(")");
        return stmts.length === 1 ? stmts[0] : { k: "block", s: stmts };
      }
      return parseExpr();
    }

    function primary() {
      const tok = peek();
      if (!tok) throw new Error("unexpected EOF");
      if (tok.t === "num") { next(); return { k: "num", v: tok.v }; }
      if (tok.t === "str") { next(); return { k: "str", v: tok.v }; }
      if (tok.t === "(") {
        next();
        const stmts = [];
        while (peek() && peek().t !== ")") stmts.push(parseStmt());
        expect(")");
        return stmts.length === 1 ? stmts[0] : { k: "block", s: stmts };
      }
      if (tok.t === "!") { next(); return { k: "un", op: "!", a: parseUnary() }; }
      if (tok.t === "-") { next(); return { k: "un", op: "-", a: parseUnary() }; }
      if (tok.t === "+") { next(); return parseUnary(); }
      if (tok.t === "#" || tok.t === "@") { next(); return parseUnary(); } // str ops: stub 0
      if (tok.t === "id") {
        next();
        const lname = tok.v.toLowerCase();
        let node = { k: "id", v: tok.v };
        while (peek() && peek().t === "[") {
          next();
          const idx = parseExpr();
          expect("]");
          node = { k: "mem", base: node, idx };
        }
        if (peek() && peek().t === "(") {
          next();
          if (lname === "loop") {
            const count = parseExpr();
            expect(",");
            const body = stmtListUntilClose();
            return { k: "loop", count, body };
          }
          if (lname === "while") {
            // while( stmts )  -> last stmt is condition
            // while( cond )( body )
            const inner = stmtListUntilClose();
            if (peek() && peek().t === "(") {
              next();
              const body = stmtListUntilClose();
              return { k: "whileC", cond: inner, body };
            }
            return { k: "whileB", body: inner };
          }
          const args = [];
          if (peek() && peek().t !== ")") {
            args.push(parseExpr());
            while (peek() && peek().t === ",") { next(); args.push(parseExpr()); }
          }
          expect(")");
          return { k: "call", name: lname, args };
        }
        return node;
      }
      throw new Error("unexpected token " + tok.t);
    }

    function parseUnary() {
      const tok = peek();
      if (tok && (tok.t === "!" || tok.t === "-")) {
        next();
        return { k: "un", op: tok.t, a: parseUnary() };
      }
      return postfix();
    }

    function postfix() {
      let node = primary();
      while (peek() && peek().t === "[") {
        next();
        const idx = parseExpr();
        expect("]");
        node = { k: "mem", base: node, idx };
      }
      return node;
    }

    const BIN_LEVELS = [
      ["&&", "||"],
      ["===", "!==", "==", "!=", "<=", ">=", "<", ">"],
      ["&", "|", "~"],
      ["<<", ">>"],
      ["+", "-"],
      ["*", "/", "%"],
      ["^"],
    ];
    const ASSIGN_OPS = ["+=", "-=", "*=", "/=", "%=", "^=", "&=", "|=", "~="];

    function parseExpr(level) {
      level = level || 0;
      if (level >= BIN_LEVELS.length) return parseUnaryTarget();
      let left = parseExpr(level + 1);
      for (;;) {
        const tok = peek();
        if (!tok) break;
        const op = BIN_LEVELS[level].find((o) => tok.t === o);
        if (!op) break;
        next();
        const right = parseExpr(level + 1);
        left = { k: "bin", op, a: left, b: right };
      }
      if (level === 0 && peek() && peek().t === "?") {
        next();
        const a = parseBlockOrExpr();
        let b = null;
        if (peek() && peek().t === ":") { next(); b = parseBlockOrExpr(); }
        return { k: "cond", c: left, a, b };
      }
      return left;
    }

    function parseUnaryTarget() {
      const node = parseUnary();
      const tok = peek();
      if (tok && ASSIGN_OPS.includes(tok.t)) {
        next();
        const val = parseExpr(0);
        return { k: "assign", target: node, op: tok.t[0], val };
      }
      if (tok && tok.t === "=") {
        next();
        const val = parseExpr(0);
        return { k: "assign", target: node, op: "=", val };
      }
      return node;
    }

    function parseStmt() {
      const st = parseExpr(0);
      if (peek() && peek().t === ";") next();
      return st;
    }

    const prog = [];
    while (p < toks.length) prog.push(parseStmt());
    return prog;
  }

  // ------------------------------------------------------------ evaluator
  const MATH1 = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin,
    acos: Math.acos, atan: Math.atan, sqr: (x) => x * x, sqrt: Math.sqrt,
    exp: Math.exp, log: Math.log, log10: Math.log10, abs: Math.abs,
    sign: Math.sign, floor: Math.floor, ceil: Math.ceil,
    invsqrt: (x) => 1 / Math.sqrt(x),
  };

  function Interpreter(host) {
    this.vars = Object.create(null);
    this.mem = new Float64Array(1 << 20);
    this.gmem = host.gmem || new Float64Array(1 << 20);
    this.host = host;
    this.deadline = Infinity;
  }

  Interpreter.prototype.var = function (name) {
    const key = name.toLowerCase();
    return this.vars[key] || 0;
  };

  Interpreter.prototype.setVar = function (name, val) {
    this.vars[name.toLowerCase()] = val;
  };

  Interpreter.prototype.evalNode = function (n) {
    switch (n.k) {
      case "num": return n.v;
      case "str": return n.v;
      case "id": {
        const key = n.v.toLowerCase();
        if (key in this.vars) return this.vars[key];
        const h = this.host;
        if (h.globals && key in h.globals)
          return h.globals[key];
        return this.var(key);
      }
      case "un": {
        const v = this.evalNode(n.a);
        return n.op === "!" ? (Math.abs(v) < 1e-5 ? 1 : 0) : -v;
      }
      case "mem": {
        const base = this.evalNode(n.base);
        const idx = n.idx ? this.evalNode(n.idx) : 0;
        const addr = Math.floor(base + idx + 0.00001);
        return this.mem[addr | 0] || 0;
      }
      case "block": {
        let last = 0;
        for (const s of n.s) last = this.evalNode(s);
        return last;
      }
      case "cond": {
        const c = this.evalNode(n.c);
        if (Math.abs(c) > 1e-5) return n.a ? this.evalNode(n.a) : c;
        return n.b ? this.evalNode(n.b) : 0;
      }
      case "bin": return this.binop(n.op, n.a, n.b);
      case "assign": return this.assign(n);
      case "call": return this.call(n);
      case "loop": {
        const count = this.evalNode(n.count);
        const cap = Math.min(count | 0, 1000000);
        let last = 0;
        for (let i = 0; i < cap; i++) {
          last = this.evalNode(n.body);
          if (time.now() > this.deadline) throw new Error("loop deadline");
        }
        return last;
      }
      case "whileB": {
        let guard = 1000000;
        let last = 0;
        while (guard-- > 0) {
          last = this.evalNode(n.body);
          if (time.now() > this.deadline) throw new Error("loop deadline");
          if (Math.abs(last) < 1e-5) break;
        }
        return last;
      }
      case "whileC": {
        let guard = 1000000;
        let last = 0;
        while (guard-- > 0) {
          if (Math.abs(this.evalNode(n.cond)) < 1e-5) break;
          last = this.evalNode(n.body);
          if (time.now() > this.deadline) throw new Error("loop deadline");
        }
        return last;
      }
      default: throw new Error("bad node " + n.k);
    }
  };

  Interpreter.prototype.binop = function (op, an, bn) {
    // short-circuit pair
    if (op === "&&") {
      const a = this.evalNode(an);
      return Math.abs(a) < 1e-5 ? 0 : this.evalNode(bn);
    }
    if (op === "||") {
      const a = this.evalNode(an);
      return Math.abs(a) > 1e-5 ? 1 : this.evalNode(bn);
    }
    const a = this.evalNode(an), b = this.evalNode(bn);
    switch (op) {
      case "+": return a + b;
      case "-": return a - b;
      case "*": return a * b;
      case "/": return a / b;
      case "%": return Math.trunc(Math.abs(a)) % Math.trunc(Math.abs(b));
      case "^": return Math.pow(a, b);
      case "<<": return (a | 0) << (b | 0);
      case ">>": return (a | 0) >> (b | 0);
      case "&": return (a | 0) & (b | 0);
      case "|": return (a | 0) | (b | 0);
      case "~": return (a | 0) ^ (b | 0);
      case "==": return (typeof a === "string" || typeof b === "string")
        ? a === b ? 1 : 0 : Math.abs(a - b) < 1e-5 ? 1 : 0;
      case "===": return a === b ? 1 : 0;
      case "!=": return Math.abs(a - b) >= 1e-5 ? 1 : 0;
      case "!==": return a !== b ? 1 : 0;
      case "<": return a < b ? 1 : 0;
      case ">": return a > b ? 1 : 0;
      case "<=": return a <= b ? 1 : 0;
      case ">=": return a >= b ? 1 : 0;
      default: throw new Error("bad op " + op);
    }
  };

  Interpreter.prototype.targetRef = function (n) {
    if (n.k === "id") {
      const self = this;
      return { get: () => self.var(n.v), set: (v) => self.setVar(n.v, v) };
    }
    if (n.k === "mem") {
      const self = this;
      const addr = () => {
        const base = self.evalNode(n.base);
        const idx = n.idx ? self.evalNode(n.idx) : 0;
        return Math.floor(base + idx + 0.00001) | 0;
      };
      return { get: () => self.mem[addr()] || 0, set: (v) => { self.mem[addr()] = v; } };
    }
    if (n.k === "cond") {
      // (a ? b : c) = v lvalue form
      const self = this;
      return {
        get: () => Math.abs(self.evalNode(n.c)) > 1e-5
          ? self.evalNode(n.a) : self.evalNode(n.b),
        set: (v) => {
          const t = Math.abs(self.evalNode(n.c)) > 1e-5 ? n.a : n.b;
          const ref = self.targetRef(t);
          ref.set(v);
        },
      };
    }
    throw new Error("bad assignment target");
  };

  Interpreter.prototype.assign = function (n) {
    const ref = this.targetRef(n.target);
    const cur = ref.get();
    let v = this.evalNode(n.val);
    if (n.op !== "=") {
      switch (n.op) {
        case "+": v = cur + v; break;
        case "-": v = cur - v; break;
        case "*": v = cur * v; break;
        case "/": v = cur / v; break;
        case "%": v = Math.trunc(Math.abs(cur)) % Math.trunc(Math.abs(v)); break;
        case "^": v = Math.pow(cur, v); break;
        case "&": v = (cur | 0) & (v | 0); break;
        case "|": v = (cur | 0) | (v | 0); break;
        case "~": v = (cur | 0) ^ (v | 0); break;
      }
    }
    ref.set(v);
    return v;
  };

  Interpreter.prototype.call = function (n) {
    const a = n.args;
    const ev = (i) => this.evalNode(a[i]);
    const h = this.host;
    switch (n.name) {
      case "sin": case "cos": case "tan": case "asin": case "acos":
      case "atan": case "sqr": case "sqrt": case "exp": case "log":
      case "log10": case "abs": case "sign": case "floor": case "ceil":
      case "invsqrt":
        return MATH1[n.name](ev(0));
      case "atan2": return Math.atan2(ev(0), ev(1));
      case "pow": return Math.pow(ev(0), ev(1));
      case "min": return Math.min(ev(0), ev(1));
      case "max": return Math.max(ev(0), ev(1));
      case "rand": return Math.random() * (ev(0) || 1);
      case "mdct": case "imdct": {
        // Faithful subset of EEL2 mdct/imdct(start,size):
        //   size in {64,128,256,512,1024,2048,4096}; windowing built in;
        //   forward: size samples -> size/2 coeffs (first half replaced);
        //   inverse: size/2 coeffs -> size windowed samples;
        //   windows must not cross a 65536-item boundary.
        const MALLOW = { 64: 1, 128: 1, 256: 1, 512: 1,
                         1024: 1, 2048: 1, 4096: 1 };
        const off = Math.floor(ev(0));
        const N = Math.floor(ev(1));
        if (!MALLOW[N]) throw new Error(
          "mdct size must be 64|128|256|512|1024|2048|4096");
        const seg = off % 65536;
        if (seg + N > 65536) throw new Error(
          "mdct window crosses 64k boundary");
        const half = N >> 1;
        const win = (nn) => Math.sin(Math.PI / N * (nn + 0.5));
        const ker = (nn, k) =>
          Math.cos(Math.PI / N * (nn + (half + 1) / 2) * (2 * k + 1));
        const rd = (i) => this.mem[(off + i) | 0] || 0;
        const INV_SQRT2 = Math.SQRT1_2;
        if (n.name === "mdct") {
          // 1/sqrt2 normalization: orthonormal-class coefficient scale
          const tmp = new Float64Array(half);
          for (let k = 0; k < half; k++) {
            let acc = 0;
            for (let nn = 0; nn < N; nn++)
              acc += rd(nn) * win(nn) * ker(nn, k);
            tmp[k] = INV_SQRT2 * acc;
          }
          for (let k = 0; k < half; k++) this.mem[(off + k) | 0] = tmp[k];
          return 0;
        }
        // imdct: compensate sqrt2, window, caller overlaps-and-adds
        const tmp2 = new Float64Array(N);
        for (let nn = 0; nn < N; nn++) {
          let acc = 0;
          for (let k = 0; k < half; k++) acc += rd(k) * ker(nn, k);
          tmp2[nn] = (4 / N) * Math.SQRT2 * acc * win(nn);
        }
        for (let nn = 0; nn < N; nn++) this.mem[(off + nn) | 0] = tmp2[nn];
        return 0;
      }
      case "time": return Date.now() / 1000 | 0;
      case "time_precise": return performance.now() / 1000;
      case "input": return h.input ? (h.input(String(ev(0))) || 0) : 0;
      case "output": if (h.output) h.output(String(ev(0)), ev(1)); return 0;
      case "store": if (h.store) h.store(String(ev(0)), ev(1)); return 0;
      case "load": return h.load ? (h.load(String(ev(0))) || 0) : 0;
      // app-level IO (one level above module CV): named trigger edges
      // and data-plane outputs sliced out of mem[].
      case "trigger":
        return h.trigger ? (h.trigger(String(ev(0))) || 0) : 0;
      case "out_series":
        if (h.outData) h.outData("series", String(ev(0)),
                                 ev(1) | 0, ev(2) | 0);
        return 0;
      case "out_points3d":
        if (h.outData) h.outData("points3d", String(ev(0)),
                                 ev(1) | 0, ev(2) | 0);
        return 0;
      // Writable n-gram PLE (27B): MV2-backed verbs, frame-cached via
      // the host bridge (fetch to /api/ngram/*).  Lookup returns 0
      // until the row lands in the frame cache — read per frame,
      // like sensor ins.  ngram_neighbors stays 0 (vec index disabled
      // until an embedder is chosen).
      case "ngram_lookup":
        return h.ngramLookup
          ? (h.ngramLookup(ev(0) | 0, ev(1) | 0, ev(2) | 0,
                           a.length > 3 ? ev(3) | 0 : 0) || 0)
          : 0;
      case "ngram_store":
        return h.ngramStore
          ? (h.ngramStore(ev(0) | 0, ev(1) | 0, ev(2) | 0,
                          a.length > 3 ? ev(3) | 0 : 0) || 0)
          : 0;
      case "ngram_neighbors":
        return h.ngramNeighbors
          ? (h.ngramNeighbors(ev(0) | 0, ev(1) | 0, ev(2) | 0,
                              a.length > 3 ? ev(3) | 0 : 8) || 0)
          : 0;
      default:
        // unknown call = 0 (lenient, like missing gfx on headless)
        return h.callable ? (h.callable(n.name, a, this) || 0) : 0;
    }
  };

  const perfNow = () =>
    (typeof performance !== "undefined"
      ? performance.now() : Date.now()) / 1000;
  const time = { now: perfNow };

  // Rising-edge tracker over named io fields: fires exactly once per
  // rise. Used for app-level triggers (button clicks and external
  // {type:"io"} messages share the same path).
  function makeEdgeTracker(getValue) {
    const prev = Object.create(null);
    return (nm) => {
      const k = String(nm).toLowerCase();
      const cur = getValue(k) || 0;
      const was = prev[k] || 0;
      prev[k] = cur;
      return cur > 0 && !(was > 0) ? 1 : 0;
    };
  }

  // ------------------------------------------------------- gfx host (DOM)
  function GfxHost(canvas, interp, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.interp = interp;
    this.fps = opts.fps || 30;
    this.bg = opts.background || "#000";
    ctxSetup(this);
  }

  function ctxSetup(g) {
    const ctx = g.canvas.getContext("2d");
    g.ctx = ctx;
    g.color = { r: 1, g: 1, b: 1, a: 1 };
    g.fontsize = 12;
    g.mouse = { x: 0, y: 0, cap: 0 };
    const c = g.canvas;
    c.addEventListener("mousemove", (e) => {
      const r = c.getBoundingClientRect();
      g.mouse.x = (e.clientX - r.left) / r.width * c.width;
      g.mouse.y = (e.clientY - r.top) / r.height * c.height;
    });
    c.addEventListener("mousedown", () => {
      g.mouse.cap = 1;
      g.onTap && g.onTap();
    });
    window.addEventListener("mouseup", () => { g.mouse.cap = 0; });
  }

  GfxHost.prototype.bindGlobals = function () {
    const g = this;
    const m = g.mouse;
    return {
      globals: {
        mouse_x: m.x, mouse_y: m.y, mouse_cap: m.cap,
        gfx_w: g.canvas.width, gfx_h: g.canvas.height,
        gfx_r: g.color.r, gfx_g: g.color.g, gfx_b: g.color.b,
        gfx_a: g.color.a, gfx_texth: g.fontsize,
      },
      callable: (name, argNodes, interp) => {
        const v = (i) => interp.evalNode(argNodes[i] || { k: "num", v: 0 });
        switch (name) {
          case "gfx_clear": {
            g.ctx.fillStyle = cssColor(v(0), v(1), v(2), v(3) === 0 ? 1 : v(3));
            g.ctx.fillRect(0, 0, g.canvas.width, g.canvas.height);
            return 0;
          }
          case "gfx_set": {
            g.color = { r: v(0), g: v(1), b: v(2), a: a.length > 3 ? v(3) : 1 };
            g.fontsize = g.color.fontsize || g.fontsize;
            return 0;
          }
          case "gfx_rect": {
            const x = v(0), y = v(1), ww = v(2), hh = v(3);
            const filled = a.length > 4 ? !!v(4) : true;
            g.ctx.fillStyle = cssColor(g.color.r, g.color.g, g.color.b, g.color.a);
            if (filled) g.ctx.fillRect(x, y, ww, hh);
            else { g.ctx.strokeStyle = g.ctx.fillStyle; g.ctx.strokeRect(x, y, ww, hh); }
            return 0;
          }
          case "gfx_rectto": {
            const x = g.mouse.x, y = g.mouse.y; // legacy rectto uses gfx_x/gfy — rare; treat as rect from origin
            void x; void y;
            return 0;
          }
          case "gfx_line": {
            g.ctx.strokeStyle = rgbaStr(g.color.r, g.color.g, g.color.b,
                                        g.color.a);
            g.ctx.lineWidth = a.length > 4 ? v(4) : 1;
            g.ctx.beginPath();
            g.ctx.moveTo(v(0), v(1));
            g.ctx.lineTo(v(2), v(3));
            g.ctx.stroke();
            return 0;
          }
          case "gfx_circle": {
            g.ctx.fillStyle = g.ctx.strokeStyle =
              rgbaStr(g.color.r, g.color.g, g.color.b, g.color.a);
            g.ctx.beginPath();
            g.ctx.arc(v(0), v(1), v(2), 0, Math.PI * 2);
            if (a.length > 3 ? !!v(3) : true) g.ctx.fill();
            else g.ctx.stroke();
            return 0;
          }
          case "gfx_setpixel": {
            g.ctx.fillStyle = rgbaStr(v(0), v(1), v(2), 1);
            g.ctx.fillRect(v(3) | 0, v(4) | 0, 1, 1);
            return 0;
          }
          case "gfx_setfont": g.fontsize = v(0) || 12; return 0;
          case "gfx_printf":
          case "gfx_drawstr": {
            const s = String(v(0));
            const x = v(1), y = v(2);
            g.ctx.fillStyle = rgbaStr(g.color.r, g.color.g, g.color.b,
                                      g.color.a);
            g.ctx.font = g.fontsize + "px Antonio, monospace";
            const just = a.length > 3 ? v(3) : 0;
            const wid = g.ctx.measureText(s).width;
            g.ctx.fillText(s, x - (just === 1 ? wid / 2 : just === 2 ? wid : 0),
                           y + g.fontsize);
            return 0;
          }
          case "gfx_drawnumber": {
            const num = v(0), x = v(1), y = v(2);
            g.ctx.fillStyle = rgbaStr(g.color.r, g.color.g, g.color.b,
                                      g.color.a);
            g.ctx.font = g.fontsize + "px Antonio, monospace";
            g.ctx.fillText(String(num), x, y + g.fontsize);
            return 0;
          }
          case "gfx_measurestr": {
            g.ctx.font = g.fontsize + "px Antonio, monospace";
            const ref = argNodes[1];
            const wid = g.ctx.measureText(String(v(0))).width;
            if (ref && ref.k === "mem") {
              const base = interp.evalNode(ref.base);
              const slot = Math.floor(base + (ref.idx ? interp.evalNode(ref.idx) : 0) + 0.00001) | 0;
              interp.mem[slot] = wid;
            }
            return wid;
          }
          case "gfx_getchar": return 0;
          default: return 0;
        }
      },
    };
  };

  function rgbaStr(r, g, b, a) {
    const cl = (x) => Math.max(0, Math.min(255, Math.round(x * 255)));
    return "rgba(" + cl(r) + "," + cl(g) + "," + cl(b) + ","
      + (a === undefined ? 1 : a) + ")";
  }
  function cssColor(r, g, b, a) { return rgbaStr(r, g, b, a); }

  // a.length hack: callable receives argNodes; provide length via closure
  // (callable above reads `a.length`; bind real nodes array there)


  // ------------------------------------------ Rack-style engine core
  // Modeled on VCV Rack v2 (engine/Module.hpp, engine/Cable.hpp,
  // dsp/SchmittTrigger.hpp, dsp/PulseGenerator.hpp):
  //   - Module: params/inputs/outputs bags + tick(dt)
  //   - Cable: explicit out->in copy performed BEFORE each tick pass
  //   - SchmittTrigger: hysteresis state machine
  //   - PulseGenerator: fixed-duration gate
  class SchmittTrigger {
    constructor(lo, hi) { this.lo = lo; this.hi = hi; this.state = 0; }
    process(v) {
      if (v >= this.hi) this.state = 1;
      else if (v <= this.lo) this.state = 0;
      return this.state;
    }
    reset() { this.state = 0; }
  }

  class PulseGenerator {
    constructor(duration) { this.duration = duration; this.remaining = 0; }
    trigger(duration) {
      this.remaining = duration !== undefined ? duration : this.duration;
    }
    process(dt) {
      if (this.remaining > 0) {
        this.remaining -= dt;
        return 1;
      }
      return 0;
    }
  }

  class CvModule {
    constructor(paramDefaults) {
      this.params = Object.assign({}, paramDefaults);
      this.inputs = {};
      this.outputs = {};
      this.id = "";
    }
    init() {}
    tick(dt) {}
  }

  class Engine {
    constructor() {
      this.modules = new Map();
      this.cables = [];
      this.t = 0;
      this.dt = 1 / 30;
    }
    add(id, module) {
      module.id = id;
      this.modules.set(id, module);
      return module;
    }
    connect(fromId, outId, toId, inId) {
      this.cables.push({ fromId, outId, toId, inId });
    }
    param(id, name, value) {
      const m = this.modules.get(id);
      if (m && m.params && name in m.params) m.params[name] = value;
    }
    step(dt) {
      dt = dt || this.dt;
      // Rack v2 engine/Engine.cpp Engine_stepFrameCables: cables are
      // grouped by destination input; a single cable COPIES its voltage,
      // multiple cables into one input SUM (stackable inputs). Values
      // are finitized like Rack's. Transfer uses the previous tick's
      // outputs (one-tick latency on feedback loops).
      const incoming = new Map();
      for (const c of this.cables) {
        const src = this.modules.get(c.fromId);
        if (!src) continue;
        let v = src.outputs[c.outId] || 0;
        if (!Number.isFinite(v)) v = 0;
        const key = c.toId + "\x00" + c.inId;
        incoming.set(key, (incoming.get(key) || 0) + v);
      }
      for (const [key, sum] of incoming) {
        const z = key.indexOf("\x00");
        const dst = this.modules.get(key.slice(0, z));
        if (dst) dst.inputs[key.slice(z + 1)] = sum;
      }
      for (const [, m] of this.modules) m.tick(dt);
      this.t += dt;
    }
    run(ticks) {
      for (let i = 0; i < ticks; i++) this.step();
      const flat = {};
      for (const [id, m] of this.modules)
        for (const o of Object.keys(m.outputs))
          flat[id + "." + o] = m.outputs[o];
      return flat;
    }
  }

  // ------------------------------------------------ patch engine (CV)
  // Built on the Rack-style core: Engine provides cable copy + ordered
  // ticks; SchmittTrigger/PulseGenerator provide trigger math. Module
  // BODIES are authored in the EEL2 subset via an interpreter wrapper.

  function _paramDefaults(name) {
    const out = {};
    for (const prm of (getPrm()[name] || []))
      out[prm.name] = prm.default;
    return out;
  }

  function makeModule(id, name, paramOverrides, env) {
    const params = Object.assign({}, _paramDefaults(name),
                                 paramOverrides || {});
    const base = new CvModule(params);
    base.id = id;
    let initDone = false;

    // ---- native host implementations ----
    if (name === "threshold") {
      const st = new SchmittTrigger(params.lo, params.hi);
      return Object.assign(base, {
        tick(dt) {
          this.outputs.gate =
            st.process(Number(this.inputs.in) || 0);
        },
      });
    }
    if (name === "clock_bpm") {
      const pg = new PulseGenerator(0.02);
      return Object.assign(base, {
        tick(dt) {
          const period = 60 / Math.max(0.1, Number(this.params.bpm) || 60);
          if (env.t >= this._next === false) {
            /* fallthrough */
          }
          if (env.t >= (this._next || 0)) {
            this._next = (this._next || env.t) + period;
            pg.trigger(0.01);
          }
          this.outputs.trig = pg.process(dt);
        },
      });
    }
    if (name === "tap") {
      return Object.assign(base, {
        tick(dt) {
          this.outputs.trig = env.uiTapAt(Math.round(env.t / dt)) ? 1 : 0;
        },
      });
    }
    if (name === "sensor") {
      return Object.assign(base, {
        tick(dt) {
          void dt;
          this.outputs.cv = env.sensorValue(String(this.params.topic || ""));
        },
      });
    }

    // ---- generic EEL2 body ----
    const srcSrc = getSrc()[name] || "";
    const compiled = compile({ source: srcSrc });
    const interp = new Interpreter({
      globals: new Proxy({}, {
        get: (_t, k) => {
          k = String(k);
          if (k in params) return Number(params[k]) || 0;
          if (k === "dt") return env.dt || 1 / 30;
          return undefined;
        },
        has: () => true,
      }),
      input: (nm) => Number(base.inputs[String(nm).toLowerCase()] || 0),
      output: (nm, v) => { base.outputs[String(nm).toLowerCase()] = v; },
      __has: () => true,
      store: (k, v) => { (base.state ||= {})[k] = v; },
      load: (k) => (base.state || {})[k] || 0,
    });
    interp.deadline = Infinity;
    return Object.assign(base, {
      tick(dt) {
        try {
          if (!initDone && compiled.init) {
            interp.evalNode(compiled.init);
          }
          initDone = true;
          if (compiled.tick) interp.evalNode(compiled.tick);
        } catch (e) {
          this.outputs.error = e.message;
        }
      },
    });
  }

  function buildEngine(patch, opts) {
    opts = opts || {};
    const engine = new Engine();
    engine.dt = opts.dt || 1 / 30;
    const taps = new Set(opts.ui_taps || []);
    const env = {
      t: 0,
      dt: engine.dt,
      uiTapAt: (tickIdx) => taps.has(tickIdx) ? 1 : 0,
      sensorValue: (topic) =>
        (opts.sensorValues || {})[topic] !== undefined
          ? opts.sensorValues[topic] : 0,
    };
    for (const mod of patch.modules || []) {
      engine.add(mod.id,
                 makeModule(mod.id, mod.primitive, mod.params, env));
    }
    for (const w of patch.wires || []) {
      try {
        engine.connect(w.from.split(".")[0], w.from.split(".")[1],
                       w.to.split(".")[0], w.to.split(".")[1]);
      } catch (_) {}
    }
    const viewKeys = (patch.views || [])
      .filter((v) => v.as === "series")
      .map((v) => v.module + "." + (v.output || "cv"));
    return { engine, viewKeys };
  }

  function paintSeries(canvas, data) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!data.length) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) { lo -= 1; hi += 1; }
    ctx.strokeStyle = "#eba75a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1 || 1)) * (w - 8) + 4;
      const y = h - 6 - ((v - lo) / (hi - lo)) * (h - 16);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px Antonio, sans-serif";
    ctx.fillText(String(data[data.length - 1]), 6, 16);
  }

  function startPatch(container, patch, opts) {
    opts = opts || {};
    const fps = opts.fps || 30;
    const { engine, viewKeys } = buildEngine(patch, opts);
    const views = [];
    const bar = document.createElement("div");
    bar.className = "jsfx-params";
    container.appendChild(bar);
    for (const key of viewKeys) {
      const wrap = document.createElement("div");
      wrap.className = "view";
      const cv = document.createElement("canvas");
      cv.width = opts.width || 480;
      cv.height = opts.height || 140;
      wrap.appendChild(cv);
      container.appendChild(wrap);
      views.push({ key, cv, data: [] });
    }
    for (const mod of patch.modules || []) {
      for (const prm of MODULES_PARAMS[mod.primitive] || []) {
        const key = mod.id + "." + prm.name;
        const lab = document.createElement("label");
        lab.textContent = key;
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = prm.min !== undefined ? prm.min : 0;
        inp.max = prm.max !== undefined ? prm.max : 1;
        inp.step = prm.step || ((inp.max - inp.min) / 100) || 0.01;
        const cur = (mod.params || {})[prm.name];
        inp.value = cur !== undefined ? cur
          : prm.default !== undefined ? prm.default : inp.min;
        inp.addEventListener("input", () => {
          engine.param(mod.id, prm.name, parseFloat(inp.value));
        });
        lab.appendChild(inp);
        bar.appendChild(lab);
      }
    }
    const timer = setInterval(() => {
      engine.step();
      for (const v of views) {
        const [mid, port] = v.key.split(".");
        const m = engine.modules.get(mid);
        const val = (m && m.outputs[port]) || 0;
        v.data.push(val);
        if (v.data.length > 512) v.data.shift();
        paintSeries(v.cv, v.data);
      }
    }, 1000 / fps);
    return { stop: () => clearInterval(timer), engine };
  }

  // ------------------------------------------------ patch engine (CV)
  // patch: {modules:[{id,primitive,params{}}], wires:[{from,to}],
  //         views:[{module,output?,as:"series"|"number"}]}
  // Pure modules evaluate via EEL2 @tick; host sources (clock_bpm)
  // are fed by the engine; sensor sources poll an endpoint.
  function makeModuleInstance(mod) {
    const name = mod.primitive;
    const def = (this && this.MODULES) || null;
    const inst = {
      id: mod.id,
      primitive: name,
      params: Object.assign({}, mod.params || {}),
      inputs: {},
      outputs: {},
      interp: null,
      tick: null,
      hostState: {},
    };
    if (name === "clock_bpm") {
      inst.host = true;
      inst.tickFn = function (bus, tSec) {
        const period = 60 / Math.max(0.1, inst.params.bpm || 60);
        if (!inst.hostState.last) inst.hostState.last = tSec;
        let n = 0;
        while (tSec - inst.hostState.last >= period) {
          inst.hostState.last += period;
          n++;
        }
        bus[inst.id + ".trig"] = n > 0 ? 1 : 0;
      };
      return inst;
    }
    if (name === "sensor") {
      inst.host = true;
      inst.topic = mod.params && mod.params.topic;
      inst.tickFn = function (bus) { /* values pushed via setValue */ };
      return inst;
    }
    const src = (this && this.sources && this.sources[name]) ||
                getSrc()[name];
    const compiled = compile(lex(src));
    const interpreter = new Interpreter({
      globals: new Proxy({}, {
        get: (_t, k) => {
          k = String(k);
          if (k in inst.params) return Number(inst.params[k]) || 0;
          return undefined;
        },
        has: () => true,
      }),
      input: (nm) => inst.inputs[String(nm).toLowerCase()] || 0,
      output: (nm, v) => { bus[inst.id + "." + nm] = v; },
      store: (k, v) => { inst.hostState[k] = v; },
      load: (k) => inst.hostState[k] || 0,
    });
    interpreter.deadline = Infinity;
    inst.interp = interpreter;
    inst.compiled = compiled;
    inst.tickFn = function () {
      if (!compiled.tick) return;
      try {
        if (compiled.init) interpreter.evalNode(compiled.init);
        interpreter.evalNode(compiled.tick);
      } catch (e) { inst.error = e.message; }
    };
    return inst;
  }

  function evaluatePatch(patch, ticks, opts) {
    opts = opts || {};
    const dt = opts.dt || 1 / 30;
    const modules = (patch.modules || []).map((mod) => ({
      id: mod.id,
      primitive: mod.primitive,
      params: Object.assign(
        {}, defaultsFor(mod.primitive), mod.params || {}),
      inputs: {}, outputs: {}, hostState: {},
    }));
    const byId = {};
    modules.forEach((m) => { byId[m.id] = m; });
    const wires = patch.wires || [];
    const bus = {};
    const series = {};
    for (const view of patch.views || []) {
      if (view.as === "series")
        series[view.module + "." + (view.output || "cv")] = [];
    }
    // compile each pure module once with per-instance interpreter
    const instances = modules.map((m) => {
      const name = m.primitive;
      if (HOST_SOURCES_SET.has(name)) {
        return { id: m.id, hostTick: (t) => {
          if (name === "clock_bpm") {
            const bpm = Number(m.params.bpm) || 60;
            const period = 60 / Math.max(0.1, bpm);
            m.hostState.acc = (m.hostState.acc || 0) + dt;
            if (m.hostState.acc >= period) {
              m.hostState.acc -= period;
              bus[m.id + ".trig"] = 1;
            } else bus[m.id + ".trig"] = 0;
          }
        }};
      }
      const srcSrc = getSrc()[name] || "";
      const compiled = srcSrc ? compile({ source: srcSrc })
        : { init: null, tick: null };
      const interp = new Interpreter({
        globals: new Proxy({}, {
          get: (_t, k) => {
            k = String(k);
            if (k in m.params) return Number(m.params[k]) || 0;
            return undefined;
          },
          has: () => true,
        }),
        input: (nm) => m.inputs[String(nm).toLowerCase()] || 0,
        output: (nm, v) => { bus[m.id + "." + nm] = v; },
        store: (k, v) => { m.hostState[k] = v; },
        load: (k) => m.hostState[k] || 0,
      });
      interp.deadline = Infinity;
      return { id: m.id, interp, compiled };
    });
    const taps = new Set(opts.ui_taps || []);
    for (let t = 0; t < ticks; t++) {
      bus["ui.tap"] = taps.has(t) ? 1 : 0;
      for (const inst of instances) {
        if (inst.hostTick) { inst.hostTick(t * dt); continue; }
        try {
          if (inst.compiled.init && t === 0)
            inst.interp.evalNode(inst.compiled.init);
          if (inst.compiled.tick)
            inst.interp.evalNode(inst.compiled.tick);
        } catch (e) { bus[inst.id + ".error"] = e.message; }
      }
      // propagate wires — Rack v2 stackable-input rule
      // (engine/Engine.cpp Engine_stepFrameCables): cables into one
      // input SUM; non-finite voltages finitize to 0.
      const sums = new Map();
      for (const w of wires) {
        let v = bus[w.from];
        if (v === undefined) continue;
        v = Number(v);
        if (!Number.isFinite(v)) v = 0;
        sums.set(w.to, (sums.get(w.to) || 0) + v);
      }
      for (const [key, sum] of sums) {
        const [mid, inp] = key.split(".");
        const target = modules.find((x) => x.id === mid);
        if (target) target.inputs[inp.toLowerCase()] = sum;
      }
      for (const key of Object.keys(series)) {
        const v = bus[key];
        if (v !== undefined) {
          series[key].push(v);
          if (series[key].length > 512) series[key].shift();
        }
      }
    }
    return { bus, series,
             final: Object.fromEntries(Object.entries(bus)) };
  }

  function defaultsFor(name) {
    const out = {};
    for (const prm of (getPrm()[name] || []))
      out[prm.name] = prm.default;
    return out;
  }

  // --------------------------------------------------------------- engine
  const getSrc = () => __MFX_MODULES_SOURCE;
  const getPrm = () => __MFX_MODULES_PARAMS;
  const HOST_SOURCES_SET = new Set(["clock_bpm", "sensor"]);

  function compile(program) {
    const sections = splitSections(program.source || "");
    const out = {};
    for (const k of Object.keys(sections))
      out[k] = { k: "block", s: parse(lex(sections[k])) };
    return out;
  }

  function splitSections(source) {
    const re = /@(init|slider|tick|block|sample|gfx|serialize)\b/g;
    const secs = {};
    let m;
    const marks = [];
    while ((m = re.exec(source))) marks.push({ name: m[1], at: m.index });
    for (let i = 0; i < marks.length; i++) {
      const end = i + 1 < marks.length ? marks[i + 1].at : source.length;
      const name = marks[i].name;
      secs[name] = (secs[name] || "") + "\n" + source.slice(
        source.indexOf("\n", marks[i].at) + 1, end);
    }
    return secs;
  }

  function runProgram(container, program, opts) {
    opts = opts || {};
    const compiled = compile(program);
    const canvas = document.createElement("canvas");
    canvas.className = "jsfx-canvas";
    canvas.width = opts.width || 480;
    canvas.height = opts.height || 300;
    container.appendChild(canvas);

    const inputs = Object.assign({}, opts.inputs || {});
    // framed apps are driven by the shell (left-rail trigger buttons
    // post {type:"io"} messages); standalone renders its own buttons.
    let framed = false;
    try { framed = window.parent && window.parent !== window; } catch (_) {}
    // TouchOSC-style control surface: controls are live signal sources
    // writing io ports (fader/button/xy/encoder; hexler scripting-api
    // vocabulary). Values normalized 0..1, scaled by min/max at the
    // port boundary. xy pads own two ports (name_x / name_y).
    const io = program.io || {};
    const controlPorts = new Set();
    if (Array.isArray(io.controls) && io.controls.length) {
      const sbar = document.createElement("div");
      sbar.className = "jsfx-controls";
      for (const c of io.controls) {
        const port = String(c.name).toLowerCase();
        if (c.type === "fader" || c.type === "encoder") {
          const lo = c.min !== undefined ? c.min : 0;
          const hi = c.max !== undefined ? c.max : 1;
          controlPorts.add(port);
          const wrap = document.createElement("label");
          wrap.className = "jsfx-control";
          wrap.textContent = c.name;
          const inp = document.createElement("input");
          inp.type = "range";
          inp.min = 0;
          inp.max = 1;
          inp.step = 0.001;
          inp.value = c.default !== undefined ? c.default : 0.5;
          const push = () => {
            inputs[port] = lo + parseFloat(inp.value) * (hi - lo);
          };
          inp.addEventListener("input", push);
          push();
          wrap.appendChild(inp);
          sbar.appendChild(wrap);
        } else if (c.type === "button") {
          controlPorts.add(port);
          const b = document.createElement("button");
          b.type = "button";
          b.className = "jsfx-control-btn";
          b.textContent = c.label || c.name;
          const toggle = c.buttonType === "toggle";
          let on = false;
          b.addEventListener("pointerdown", () => {
            on = toggle ? !on : true;
            b.classList.toggle("on", on);
            inputs[port] = on ? 1 : 0;
          });
          if (!toggle) {
            const release = () => {
              on = false;
              b.classList.remove("on");
              inputs[port] = 0;
            };
            b.addEventListener("pointerup", release);
            b.addEventListener("pointerleave", release);
          }
          sbar.appendChild(b);
        } else if (c.type === "xy") {
          controlPorts.add(port + "_x");
          controlPorts.add(port + "_y");
          const pad = document.createElement("canvas");
          pad.className = "jsfx-xy";
          pad.width = 150;
          pad.height = 150;
          const pctx = pad.getContext("2d");
          const draw = (x, y) => {
            pctx.fillStyle = "#0d0b12";
            pctx.fillRect(0, 0, 150, 150);
            pctx.strokeStyle = "#baa4e5";
            pctx.strokeRect(0.5, 0.5, 149, 149);
            pctx.fillStyle = "#eba75a";
            pctx.beginPath();
            pctx.arc(4 + x * 142, 4 + y * 142, 6, 0, Math.PI * 2);
            pctx.fill();
          };
          draw(0.5, 0.5);
          inputs[port + "_x"] = 0.5;
          inputs[port + "_y"] = 0.5;
          let active = false;
          const set = (e) => {
            const r = pad.getBoundingClientRect();
            const x = Math.min(1, Math.max(0,
              (e.clientX - r.left) / r.width));
            const y = Math.min(1, Math.max(0,
              (e.clientY - r.top) / r.height));
            inputs[port + "_x"] = x;
            inputs[port + "_y"] = y;
            draw(x, y);
          };
          pad.addEventListener("pointerdown", (e) => {
            active = true;
            try { pad.setPointerCapture(e.pointerId); } catch (_) {}
            set(e);
          });
          pad.addEventListener("pointermove", (e) => {
            if (active) set(e);
          });
          pad.addEventListener("pointerup", () => { active = false; });
          sbar.appendChild(pad);
        }
      }
      container.appendChild(sbar);
    }
    // trigger ports auto-clear at end of frame so event sources can
    // rise again; control-owned ports manage their own levels
    const trigNames = (Array.isArray(io.triggers) ? io.triggers : [])
      .map((t) => String(t.name).toLowerCase())
      .filter((n) => !controlPorts.has(n));
    if (!framed && trigNames.length) {
      const tbar = document.createElement("div");
      tbar.className = "jsfx-triggers";
      for (const t of io.triggers) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = (t.label || t.name || "run");
        b.addEventListener("click", () => {
          inputs[String(t.name).toLowerCase()] = 1;
        });
        tbar.appendChild(b);
      }
      container.appendChild(tbar);
    }

    // param UI
    const paramState = {};
    if ((program.params || []).length) {
      const bar = document.createElement("div");
      bar.className = "jsfx-params";
      for (const prm of program.params) {
        const lab = document.createElement("label");
        lab.textContent = prm.label || prm.name;
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = prm.min !== undefined ? prm.min : 0;
        inp.max = prm.max !== undefined ? prm.max : 1;
        inp.step = prm.step || ((inp.max - inp.min) / 100);
        inp.value = prm.default !== undefined ? prm.default : inp.min;
        paramState[prm.name.toLowerCase()] = parseFloat(inp.value);
        inp.addEventListener("input", () => {
          paramState[prm.name.toLowerCase()] = parseFloat(inp.value);
        });
        lab.appendChild(inp);
        bar.appendChild(lab);
      }
      container.appendChild(bar);
    }

    const storageKey = "mfx-" + (program.id || "prog");
    const interp = new Interpreter({
      globals: paramState,
      input: (nm) => inputs[String(nm).toLowerCase()] || 0,
      output: (nm, v) => { if (opts.onOutput) opts.onOutput(nm, v); },
      store: (k, v) => {
        try {
          const bag = JSON.parse(localStorage.getItem(storageKey) || "{}");
          bag[k] = v;
          localStorage.setItem(storageKey, JSON.stringify(bag));
        } catch (_) {}
      },
      load: (k) => {
        try {
          return JSON.parse(localStorage.getItem(storageKey) || "{}")[k] || 0;
        } catch (_) { return 0; }
      },
    });

    const gfx = new GfxHost(canvas, interp, opts);
    // merge mouse/gfx globals into the same global view as params
    const baseGlobals = gfx.bindGlobals().globals;
    const baseCallable = gfx.bindGlobals().callable;
    // rising-edge tracker over io fields: shell buttons AND external
    // {type:"io"} messages feed the same named-trigger path.
    const onEdge = makeEdgeTracker((k) => inputs[k]);
    // when framed by the shell, app outputs route to the parent page
    // (scalar outs + figure payloads for the viewport matrix)
    if (framed && !opts.onOutput) {
      opts.onOutput = (nm, v) => {
        try { window.parent.postMessage(
          { type: "app-out", name: String(nm), value: v }, "*"); } catch (_) {}
      };
    }
    if (framed && !opts.onFigure) {
      opts.onFigure = (nm, fig) => {
        try { window.parent.postMessage(
          { type: "app-figure", name: String(nm),
            app: program.title || program.id || "microapp",
            figure: fig }, "*"); } catch (_) {}
      };
    }
    interp.host = {
      globals: new Proxy({}, {
        get: (_t, k) => {
          k = String(k);
          if (paramState && k in paramState) return paramState[k];
          if (k in baseGlobals) return baseGlobals[k];
          // EEL2 semantics: uninitialized variables read 0 — an
          // undefined here poisons arithmetic with NaN (shell-owned
          // control ports may not have been pushed yet)
          return 0;
        },
        has: () => true,
      }),
      input: (nm) => inputs[String(nm).toLowerCase()] || 0,
      output: (nm, v) => { if (opts.onOutput) opts.onOutput(nm, v); },
      trigger: (nm) => onEdge(nm),
      outData: (kind, nm, start, count) => {
        if (!opts.onFigure || count <= 0) return;
        const data = new Array(count);
        for (let i = 0; i < count; i++) data[i] = interp.mem[start + i];
        opts.onFigure(nm, { kind, count,
                            data: Array.isArray(data) ? data : [] });
      },
      store: (k, v) => {
        try {
          const bag = JSON.parse(localStorage.getItem(storageKey) || "{}");
          bag[k] = v;
          localStorage.setItem(storageKey, JSON.stringify(bag));
        } catch (_) {}
      },
      load: (k) => {
        try {
          return JSON.parse(localStorage.getItem(storageKey) || "{}")[k] || 0;
        } catch (_) { return 0; }
      },
      callable: baseCallable,
      // Writable n-gram PLE bridge (Step 3): frame-cached MV2 access.
      // ngramLookup(memStart, token, layer, pos) fills 160 floats into
      // mem[] and returns 1 once the row is cached; first frames return
      // 0 while the fetch is in flight (live-signal semantics: read
      // every frame).  ngramStore(token, layer, memStart, pos) writes
      // 160 floats from mem[] to the shard (WAL; server flushes).
      ngramLookup: (() => {
        const cache = new Map();      // "layer:token:pos" -> number[160]
        const inflight = new Set();
        return (memStart, token, layer, pos) => {
          const key = layer + ":" + token + ":" + (pos || 0);
          const hit = cache.get(key);
          if (hit) {
            for (let i = 0; i < 160; i++) interp.mem[memStart + i] = hit[i];
            return 1;
          }
          if (!inflight.has(key) && typeof fetch === "function") {
            inflight.add(key);
            fetch("/api/ngram/lookup?layer=" + layer + "&token=" + token +
                  "&pos=" + (pos || 0))
              .then((r) => (r.ok ? r.json() : null))
              .then((j) => {
                if (j && j.found && Array.isArray(j.vector))
                  cache.set(key, j.vector);
              })
              .catch(() => {})
              .finally(() => inflight.delete(key));
          }
          return 0;
        };
      })(),
      ngramStore: (() => {
        const cache = new Map();
        return (token, layer, memStart, pos) => {
          const v = new Array(160);
          for (let i = 0; i < 160; i++) v[i] = interp.mem[memStart + i] || 0;
          cache.set(layer + ":" + token + ":" + (pos || 0), v);
          if (typeof fetch === "function") {
            fetch("/api/ngram/store", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ layer, token, pos: pos || 0,
                                     vector: v }),
            }).catch(() => {});
          }
          return 1;
        };
      })(),
      // vec index disabled (memvid unknown-embedder landmine) — the
      // verb exists so programs compile; returns 0 = no neighbors.
      ngramNeighbors: () => 0,
    };

    try { if (compiled.init) interp.evalNode(compiled.init); }
    catch (e) { console.warn("mfx @init:", e.message); }
    try { if (compiled.slider) interp.evalNode(compiled.slider); }
    catch (e) { console.warn("mfx @slider:", e.message); }

    // left-rail io bridge: {type:"io", fields:{name:value}}
    window.addEventListener("message", (e) => {
      if (!e.data || e.data.type !== "io") return;
      const f = e.data.fields || {};
      for (const k of Object.keys(f)) {
        const v = parseFloat(f[k]);
        if (!Number.isNaN(v)) inputs[k.toLowerCase()] = v;
      }
    });

    let stopped = false;
    function frame() {
      if (stopped) return;
      interp.deadline = performance.now() / 1000 + 0.05;
      try { if (compiled.gfx) interp.evalNode(compiled.gfx); }
      catch (e) {
        console.warn("mfx @gfx:", e.message);
        stopped = true;
      }
      // consumed triggers fall back to 0 so the next click/event rises
      for (const k of trigNames) inputs[k] = 0;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    return { stop: () => { stopped = true; }, canvas, interp };
  }

  // expose argNodes-length fix: wrap callable invocation context
  const _origCall = Interpreter.prototype.call;
  Interpreter.prototype.call = function (n) {
    if (n.name && n.name.startsWith("gfx_")) {
      this._argnodes = n.args;
      const h = this.host;
      const saved = h.callable;
      if (typeof saved !== "function") return 0; // headless: no canvas
      h.callable = (name, _a, _i) => saved(name, n.args, this);
      const r = _origCall.call(this, n);
      h.callable = saved;
      return r;
    }
    return _origCall.call(this, n);
  };


  function startPatch(container, patch, opts) {
    opts = opts || {};
    const fps = opts.fps || 30;
    const views = [];
    const bar = document.createElement("div");
    bar.className = "jsfx-params";
    container.appendChild(bar);
    for (const view of patch.views || []) {
      if (view.as === "number") continue;
      const wrap = document.createElement("div");
      wrap.className = "view";
      const cv = document.createElement("canvas");
      cv.width = opts.width || 480;
      cv.height = opts.height || 140;
      wrap.appendChild(cv);
      container.appendChild(wrap);
      views.push({ view, cv });
    }
    const dt = 1 / fps;
    const state = { t: 0 };
    // params UI per module
    for (const mod of patch.modules || []) {
      for (const prm of MODULES_PARAMS[mod.primitive] || []) {
        const key = mod.id + "." + prm.name;
        const lab = document.createElement("label");
        lab.textContent = key;
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = prm.min !== undefined ? prm.min : 0;
        inp.max = prm.max !== undefined ? prm.max : 1;
        inp.step = prm.step || ((inp.max - inp.min) / 100);
        const cur = (mod.params || {})[prm.name];
        inp.value = cur !== undefined ? cur
          : prm.default !== undefined ? prm.default : inp.min;
        inp.addEventListener("input", () => {
          mod.params = mod.params || {};
          mod.params[prm.name] = parseFloat(inp.value);
        });
        lab.appendChild(inp);
        bar.appendChild(lab);
      }
    }
    const timer = setInterval(() => {
      state.t += dt;
      const res = evaluatePatch(patch, 1,
        Object.assign({ dt }, opts, { startT: state.t }));
      for (const { view, cv } of views) {
        const key = view.module + "." + (view.output || "cv");
        const seriesData = res.series[key] || [];
        paintSeries(cv, seriesData);
      }
      if (opts.ticks && state.t * fps >= opts.ticks) stop();
    }, 1000 / fps);
    function stop() { clearInterval(timer); }
    return { stop };
  }

  function paintSeries(canvas, data) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!data.length) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) { lo -= 1; hi += 1; }
    ctx.strokeStyle = "#eba75a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1 || 1)) * (w - 8) + 4;
      const y = h - 6 - ((v - lo) / (hi - lo)) * (h - 16);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px Antonio, sans-serif";
    ctx.fillText(String(data[data.length - 1]), 6, 16);
  }

  return { lex, parse, Interpreter, compile, splitSections, runProgram,
           GfxHost, registerModules, evaluatePatch, startPatch,
           makeEdgeTracker,
           SchmittTrigger, PulseGenerator, CvModule, Engine };
}));
