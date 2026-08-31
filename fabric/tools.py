from __future__ import annotations

import json
import re

from . import status_sources
from .log_store import get_log_store


def ship_status() -> dict:
    return status_sources.ship_status()


def log_append(text: str, source: str = "operator", level: str = "info") -> dict:
    return get_log_store().append(text, source=source, level=level)


def log_recent(limit: int = 20, level: str = None, source: str = None) -> list:
    return get_log_store().recent(limit=limit, level=level, source=source)


def terrain_los(tx: list[float] = None, rx: list[float] = None,
                f_mhz: float = 5800.0,
                tx_lat: float = None, tx_lon: float = None, tx_h: float = 15.0,
                rx_lat: float = None, rx_lon: float = None, rx_h: float = 15.0) -> dict:
    """Line-of-sight check over the station terrain (resident Terrarium tile).

    Prefer world-metre endpoints tx/rx = [x, y, z]. Geographic endpoints
    (tx_lat/tx_lon + rx_lat/rx_lon, heights AGL) are accepted when both
    fall inside the resident tile. Returns verdict, diffraction, Fresnel.
    """
    from . import geo

    if tx_lat is not None and tx_lon is not None and rx_lat is not None and rx_lon is not None:
        return geo.los_latlon(tx_lat, tx_lon, rx_lat, rx_lon, tx_h, rx_h, f_mhz)
    if not tx or not rx or len(tx) < 3 or len(rx) < 3:
        return {"error": "tx and rx need [x,y,z] metres, or lat/lon pairs"}
    return geo.los(tx, rx, f_mhz)


def terrain_survey(tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
                   tx_h: float = 15.0, rx_h: float = 15.0,
                   f_mhz: float = 5800.0) -> dict:
    """Great-circle Terrarium site survey between two lat/lon antennas.
    Crosses tile boundaries. Returns Deygout verdict, diffraction, profile."""
    from . import geo

    return geo.survey(tx_lat, tx_lon, rx_lat, rx_lon, tx_h, rx_h, f_mhz)


def terrain_horizon(center: list[float] = None, h_agl: float = 25.0,
                    n_az: int = 72, elev_deg: float = -2.0) -> dict:
    """Azimuth horizon sweep from a sensor position: first-terrain-hit
    distance per azimuth (coverage/radio-horizon analysis)."""
    from . import geo

    meta = geo.terrain_meta()
    if not center or len(center) < 2:
        half = meta["span_m"] / 2.0
        center = [half, half]
    n_az = int(max(8, min(int(n_az), 360)))
    res = geo.horizon(center, h_agl, n_az, elev_deg)
    ds = [d for d in res["distances_m"] if d > 0]
    res["hit_count"] = len(ds)
    res["min_m"] = round(min(ds), 1) if ds else None
    res["max_m"] = round(max(ds), 1) if ds else None
    if n_az > 72:
        step = max(1, n_az // 72)
        res["azimuths_deg"] = res["azimuths_deg"][::step]
        res["distances_m"] = res["distances_m"][::step]
    return res


def show_place(q: str = "", lat: float = None, lon: float = None,
               zoom: int = 12, view: str = "terrain") -> dict:
    """Load a named city/place onto the resident Terrarium DEM and return
    a terrain heatmap or horizon fan. Use this for 'show me X' / 'city of X'."""
    from . import geo

    return geo.show_place(q, lat=lat, lon=lon, zoom=zoom, view=view)


def terrain_set_origin(lat: float, lon: float, zoom: int = 12) -> dict:
    """Load a Terrarium DEM tile as the resident station terrain, centred
    on (lat, lon) at the given slippy-map zoom (default 12, ~6 km at 52°N)."""
    from . import geo

    try:
        origin = geo.load_origin(lat, lon, zoom)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    meta = geo.terrain_meta()
    return {**origin, "hmin_m": meta["hmin_m"], "hmax_m": meta["hmax_m"],
            "engine": meta["engine"]}


def hadamard_build(n: int, method: str = "auto") -> dict:
    """Build and verify a Hadamard matrix of order n (library of 810
    verified orders + Sylvester Kronecker construction)."""
    from . import lab

    return lab.hadamard_build(n, method)


def hoa_encode(azimuths: list[float], elevations: list[float],
               gains: list[float] = None, order: int = 4) -> dict:
    """Encode point sources into ACN/SN3D higher-order ambisonics and
    analyse: peak direction, per-order energy, power map."""
    from . import lab

    return lab.hoa_encode(azimuths, elevations, gains, order)


def orbital_probe(n: int, l: int, m: int, n_r: int = 48) -> dict:
    """Hydrogenic orbital |psi|^2 slice (y=0 plane) with quantum numbers
    n, l, m — atom simulator view."""
    from . import lab

    return lab.orbital_probe(n, l, m, n_r)


def antenna_pattern(kind: str, f_mhz: float, phi_deg: float = 0.0) -> dict:
    """Antenna pattern cut for kind in dipole/monopole/loop/patch/helix/yagi
    at f_mhz: gain, dimensions, impedance, front-to-back ratio."""
    from . import lab

    return lab.antenna_pattern(kind, f_mhz, phi_deg)


def filter_response(kind: str, f_c_mhz: float, n: int = 5,
                    proto: str = "butterworth", f_lo_mhz: float = None,
                    f_hi_mhz: float = None) -> dict:
    """Microstrip stepped-impedance filter design + S21 sweep for kind in
    lpf/hpf/bpf/bsf."""
    from . import lab

    return lab.filter_response(kind, f_c_mhz, n, proto, f_lo_mhz, f_hi_mhz)


def link_budget(p_tx_dbw: float, g_tx_dbi: float, g_rx_dbi: float,
                f_mhz: float, d_m: float, medium: str = "air") -> dict:
    """Friis link budget in a lossy medium: received power, path loss,
    link margins."""
    from . import lab

    return lab.link_budget(p_tx_dbw, g_tx_dbi, g_rx_dbi, f_mhz, d_m, medium)


def sensor_query(prefix: str = "ship/", since: float = None,
                 limit: int = 20) -> dict:
    """Query the sensor bus (MQTT-shaped topics). prefix filters topics
    (e.g. ship/gpu/). since is a unix timestamp; omit for the latest."""
    from . import bus

    rows = bus.query(prefix=prefix, since=since, limit=limit)
    return {"n": len(rows), "broker": bus.broker_status(), "readings": rows}


def room_impulse(src: list[float], lst: list[float],
                 box: list[float] = None, n_rays: int = 512,
                 max_bounce: int = 3, absorption: float = 0.2,
                 order: int = 3, furniture: list = None) -> dict:
    """Shoebox room impulse response for HOA decode: specular acoustic
    rays vs the room BVH. src/lst are [x,y,z] metres inside box=[Lx,Ly,Lz]
    (Y up). Returns first-arrival delay, HOA peak direction, W-channel IR."""
    from . import room

    out = room.room_impulse(src, lst, box=box, n_rays=n_rays,
                            max_bounce=max_bounce, absorption=absorption,
                            order=order, furniture=furniture)
    if "error" in out:
        return out
    ana = out.get("analysis") or {}
    sab = out.get("sabine") or {}
    return {
        "engine": out.get("engine"),
        "n_arrivals": out.get("n_arrivals"),
        "trace_ms": out.get("trace_ms"),
        "box_m": out.get("box_m"),
        "src_m": out.get("src_m"),
        "lst_m": out.get("lst_m"),
        "peak_delay_s": ana.get("peak_delay_s"),
        "peak_az_deg": ana.get("peak_az_deg"),
        "peak_el_deg": ana.get("peak_el_deg"),
        "direct_delay_error_s": out.get("direct_delay_error_s"),
        "image_direct_delay_s": out.get("image_direct_delay_s"),
        "n_channels": ana.get("n_channels"),
        "direct_occluded": out.get("direct_occluded"),
        "t20_s": ana.get("schroeder_t20_s"),
        "t60_sabine_s": sab.get("t60_sabine_s"),
        "n_furniture": out.get("n_furniture"),
        "caption": (out.get("figure") or {}).get("caption"),
    }


def hoa_decode(azimuths: list[float], elevations: list[float],
               gains: list[float] = None, order: int = 4) -> dict:
    """Round-trip HOA: encode sources then sample the field at those directions."""
    from . import lab

    return lab.hoa_decode(azimuths, elevations, gains, order)


def hoa_rotate(azimuths: list[float], elevations: list[float],
               yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0,
               order: int = 4) -> dict:
    """Wigner-D rotate an encoded HOA scene; returns new peak direction."""
    from . import lab

    return lab.hoa_rotate(azimuths, elevations, yaw_deg, pitch_deg, roll_deg,
                          order=order)


def fdtd_probe(f_mhz: float = 150.0, medium: str = "air", n: int = 16) -> dict:
    """Small Yee FDTD dipole radiation probe (air/water)."""
    from . import lab

    return lab.fdtd_probe(f_mhz, medium, n)


def materials_design(kind: str = "cloth", order: int = 8) -> dict:
    """H.8 flux-tile materials layout: cloth, touchpad, or metamaterial."""
    from . import lab

    return lab.materials_design(kind, order)


def actual_size_scales(eps: float = 0.003) -> dict:
    """Press chemistry-fraction length/mass/temperature scales."""
    from . import lab

    return lab.actual_size_scales(eps)


def crown_psf(n: int = 32) -> dict:
    """Crown-aperture point-spread function magnitude (closed-form)."""
    from . import lab

    return lab.crown_psf(n)


def sage_eval(expr: str) -> dict:
    """Computer-algebra evaluate. SageMath if installed, else SymPy."""
    from . import sage

    return sage.sage_eval(expr)


def jellyfin_status() -> dict:
    """Media stack reachability on the NAS host: Jellyfin, Sonarr,
    Radarr, Prowlarr, Bazarr, qBittorrent, Jellyseerr."""
    from . import media

    return media.stack_status()


def jellyfin_search(query: str, limit: int = 10) -> dict:
    """Search the Jellyfin library: movies/series/episodes/music by
    title. Returns item ids for jellyfin_play."""
    from . import media

    return media.jellyfin_search(query, limit)


def jellyfin_play(item_id: str, client: str = "") -> dict:
    """Play a Jellyfin item on an active session. NON-IDEMPOTENT:
    confirm with the operator before calling. client matches the
    player device name when several sessions are live."""
    from . import media

    return media.jellyfin_play(item_id, client)


def arr_queue() -> dict:
    """Sonarr + Radarr download queue counts."""
    from . import media

    return media.arr_queue()


def ha_state(entity_id: str) -> dict:
    """Read a Home Assistant entity state (light.sensor.climate...).
    Honest error while D09 is offline."""
    from . import ha

    return ha.ha_state(entity_id)


def ha_call(domain: str, service: str, entity_id: str = "",
            data: dict | None = None) -> dict:
    """Call a Home Assistant service (light/turn_on, scene/turn_on...).
    NON-IDEMPOTENT — confirm with the operator first. D09 offline
    returns an error until HA reconnects."""
    from . import ha

    return ha.ha_call(domain, service, entity_id=entity_id, data=data)


def backlot_status() -> dict:
    """AI Backlot service health: job spine, vLLM and ComfyUI flags."""
    from . import backlot

    return backlot.backlot_status()


def backlot_programs() -> dict:
    """List Backlot studio programs (image/sphere/mesh/upscale/sample/
    sfx) and available workflow templates."""
    from . import backlot

    return backlot.programs()


def backlot_submit(program: str, prompt: str = "",
                   params: dict | None = None) -> dict:
    """Queue a Backlot studio generation job (durable worker; returns
    job_id immediately). NON-IDEMPOTENT — confirm with the operator
    first. Poll with backlot_job_status."""
    from . import backlot

    return backlot.submit(program, prompt, params)


def backlot_job_status(job_id: int) -> dict:
    """Durable status/result paths of one Backlot generation job."""
    from . import backlot

    return backlot.job_status(job_id)


def backlot_train_submit(params: dict) -> dict:
    """Submit a Backlot LoRA training job (musubi/ACE-Step). Needs
    name/target_arch/dataset_path in params. NON-IDEMPOTENT — confirm
    with the operator first."""
    from . import backlot

    return backlot.train_submit(params)


def backlot_train_jobs(limit: int = 10) -> dict:
    """List recent Backlot LoRA training jobs and statuses."""
    from . import backlot

    return backlot.train_jobs(limit)


def backlot_speak(text: str, character: str = "") -> dict:
    """Piper TTS via Backlot; saves a WAV under fabric/data/speak/ and
    returns its path."""
    from . import backlot

    return backlot.speak(text, character)


def backlot_transcribe(path: str, model: str = "",
                       language: str = "", task: str = "transcribe") -> dict:
    """Whisper STT via Backlot for a local audio/video file. Returns
    the transcript text."""
    from . import backlot

    return backlot.transcribe(
        path, model=model or None, language=language or None, task=task)


def backlot_stems(path: str, model: str = "htdemucs") -> dict:
    """Queue demucs stem separation on Backlot (durable worker).
    NON-IDEMPOTENT — confirm with the operator first. Poll with
    backlot_job_status."""
    from . import backlot

    return backlot.stems_submit(path, model)


def backlot_sample_export(fmt: str, paths: list, name: str = "") -> dict:
    """Export samples via Backlot as wavset / Korg Volca Sample 2 /
    Ableton Simpler bundle. NON-IDEMPOTENT — confirm first."""
    from . import backlot

    return backlot.sample_export(fmt, paths, name)


def backlot_mesh_upgrade(job_id: int, params: dict | None = None) -> dict:
    """Re-mesh a previous Backlot mesh job at higher quality.
    NON-IDEMPOTENT — confirm with the operator first."""
    from . import backlot

    return backlot.mesh_upgrade(job_id, params)


def backlot_stepseq(params: dict) -> dict:
    """Queue a ComfyUI step-sequencer image grid on Backlot. Needs
    prompt + steps[] in params. NON-IDEMPOTENT — confirm first."""
    from . import backlot

    return backlot.stepseq(params)


def backlot_transcode_submit(path: str, quality: int = 0) -> dict:
    """Transcode a Backlot outputs video to AV1 (QSV). SYNCHRONOUS —
    may run minutes. NON-IDEMPOTENT — confirm with the operator first."""
    from . import backlot

    return backlot.transcode_submit(path, quality or None)


def backlot_transcode_status() -> dict:
    """Backlot AV1/QSV transcoder capability flags."""
    from . import backlot

    return backlot.transcode_status()


def port_status() -> dict:
    """Backend service port sweep on this host: fabric, duty vLLM,
    Backlot, ComfyUI ×2, pi-bridge, MQTT broker."""
    from . import status_sources

    ports = {
        "fabric": 18093,
        "duty_vllm": 8000,
        "backlot": 8001,
        "comfyui": 8188,
        "comfyui_h3": 8189,
        "pi_bridge": 8642,
        "mqtt_broker": 1883,
    }
    rows = {}
    for name, port in sorted(ports.items(), key=lambda kv: kv[1]):
        rows[name] = {"port": port,
                      "up": status_sources._port_up(port)}
    return {"host": "127.0.0.1", "services": rows,
            "all_up": all(r["up"] for r in rows.values())}


def stock_quote(symbols: str) -> dict:
    """Keyless stock quotes (stooq). symbols like 'AAPL,MSFT,NVDA'."""
    from .stocks import quotes

    return quotes(symbols)


def firefly_net_worth() -> dict:
    """Firefly III net worth (read-only asset accounts). Needs
    FIREFLY_TOKEN stored in secrets; honest error until then."""
    from .stocks import net_worth

    return net_worth()


def microfx_patch(patch: dict, title: str = "") -> dict:
    """Instantiate a MicroFX CV-module patch as a wall tile.

    patch = {modules:[{id,primitive,params{}}], wires:[{from,to}],
             views:[{module,output?,as}]}. Catalog: const/clock_bpm/
    sine_lfo/gain/bias/smooth/threshold/moving_avg/clamp/mdct_flux.
    NON-IDEMPOTENT persistence — confirm with operator first."""
    from . import microfx

    built = microfx.build_patch_html(patch, title or None)
    if "error" in built:
        return built
    from .microapps import compose

    return compose(patch.get("title") or title or "MicroFX patch",
                   html=built["html"], fields=None)


def latent_loop(prompt: str, max_iterations: int = 3) -> dict:
    """Looping-transformer PoC: iterative self-refinement with spectral
    tracking. Generates, captures hidden states, applies FWHT for energy
    analysis, self-critiques, and refines until complete or capped."""
    from .latent import latent_loop as _run

    return _run(prompt, max_iterations)


def omp_code(task: str) -> dict:
    """Delegate a coding task to OMP (oh-my-pi). Use for multi-step
    code writing, debugging, refactoring, or anything needing more
    than 2-3 tool rounds. NOT for status/lab/media queries."""
    import subprocess
    import os as _os

    repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    omp_bin = _os.environ.get(
        "OMP_BIN", _os.path.expanduser("~/.local/bin/omp"))
    cmd = [omp_bin, "--mode", "json", "-p", task,
           "--cwd", repo, "--max-time", "10m",
           "--approval-mode", "yolo",
           "--no-skills", "--no-rules", "--no-extensions",
           "--tools", "read,edit,write,bash,grep,glob,todo",
           "--no-session"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=600, cwd=repo,
                           stdin=subprocess.DEVNULL)
        # extract final assistant text from JSONL events
        final = ""
        session_id = ""
        for raw in r.stdout.splitlines():
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            if ev.get("type") == "session":
                session_id = ev.get("id", "")
            m = ev.get("message") or {}
            if (ev.get("type") == "message_end"
                    and m.get("role") == "assistant"):
                parts = m.get("content") or []
                text = " ".join(p.get("text", "") for p in parts
                                if isinstance(p, dict))
                if text.strip():
                    final = text.strip()
        return {"ok": r.returncode == 0, "result": final[:4000],
                "session": session_id[:12]}
    except subprocess.TimeoutExpired:
        return {"error": "OMP timed out after 10 minutes"}
    except FileNotFoundError:
        return {"error": "omp binary not found"}


def dept_memory_add(dept: str, text: str, title: str = "") -> dict:
    """Save one memory card into a department's .mv2 shard
    (persona/knowledge/facts). Persistent across sessions."""
    from . import dept_memory as dm

    return dm.add(dept, text, title or None)


def dept_memory_search(dept: str, query: str, k: int = 4) -> dict:
    """Recall from a department memory shard (memvid lex/vec)."""
    from . import dept_memory as dm

    return dm.search(dept, query, k)


def net_devices() -> dict:
    """LAN device map from the Orbi router: name/ip/mac/connection/
    signal for every attached host. Read-only."""
    from . import net

    return net.net_devices()


def code_index(query: str) -> dict:
    """Single-symbol repo index: identifier or heading word -> file:line.
    Use before grep/read. Not prose phrases; misses return word hints."""
    from . import code_index as ci

    q = (query or "").strip()
    if not q:
        return {"error": "missing query"}
    hits = ci.find(q)
    out = {
        "query": q,
        "n": len(hits),
        "hits": [f"{h['file']}:{h['line']} {h['kind']} {h['name']}"
                 for h in hits],
    }
    if not hits:
        hints: list[str] = []
        seen: set[str] = set()
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", q):
            if word.lower() == q.lower():
                continue
            for h in ci.find(word, limit=4):
                row = f"{h['file']}:{h['line']} {h['kind']} {h['name']}"
                if row not in seen:
                    seen.add(row)
                    hints.append(row)
            if len(hints) >= 8:
                break
        if hints:
            out["word_hits"] = hints
            out["note"] = ("no phrase match; index is single symbols/"
                           "headings — word-level matches above")
        else:
            out["note"] = ("no match; index covers def/class/const names "
                           "and md headings only — use grep")
    return out


def pi_code(prompt: str) -> dict:
    """Run a coding task on the Pi agent (olympus-pi-bridge :8642).

    Use for file/read/edit/bash in the M1Multitronic tree. Not for ship
    status or lab calculators — those stay on duty tools.
    """
    from . import pi_client

    return pi_client.code_task(prompt)


def pi_goal(goal: str) -> dict:
    """Store a goal statement and have Pi work the first safe step."""
    from . import pi_client

    return pi_client.run_goal(goal)


def pipecat_status() -> dict:
    """Pipecat realtime voice-leg status (wake/barge-in/speak)."""
    from . import voice

    return voice.pipecat_status()


def harness_status() -> dict:
    """Harness orchestrator + worker lanes (keys masked)."""
    from . import harness

    return harness.public()


def _repo_out(text) -> dict:
    if isinstance(text, dict):
        return text
    err = str(text).startswith("error") or str(text).startswith("refused")
    return {"ok": not err, "text": text}


def ls(path: str = ".") -> dict:
    """List a directory in the M1Multitronic working tree."""
    from . import pi_client

    return _repo_out(pi_client._dispatch("ls", {"path": path}))


def grep(pattern: str, path: str = ".") -> dict:
    """Search file contents under the working tree."""
    from . import pi_client

    return _repo_out(pi_client._dispatch("grep", {"pattern": pattern, "path": path}))


def find(name: str = "*", path: str = ".") -> dict:
    """Find files by glob name under the working tree."""
    from . import pi_client

    return _repo_out(pi_client._dispatch("find", {"name": name, "path": path}))


def read(path: str, offset: int = 0, limit: int = 200) -> dict:
    """Read a text file in the working tree (line offset/limit)."""
    from . import pi_client

    return _repo_out(pi_client._dispatch(
        "read", {"path": path, "offset": offset, "limit": limit}))


def write(path: str, contents: str) -> dict:
    """Create or overwrite a text file in the working tree. Not secrets."""
    from . import pi_client

    return _repo_out(pi_client._dispatch(
        "write", {"path": path, "contents": contents}))


def edit(path: str, old: str, new: str) -> dict:
    """Replace one unique string in a working-tree file."""
    from . import pi_client

    return _repo_out(pi_client._dispatch(
        "edit", {"path": path, "old": old, "new": new}))


def generate_microapp(query: str, html: str = "", title: str = "",
                      dept: str = "", input: str = "", output: str = "",
                      viewports: int = 1) -> dict:
    """Register a MiniApp. Prefer a wizard route (dept/input/output/viewports)
    over free HTML. Named places use show_place. Returns spec or wizard=true.
    """
    from . import microapps

    route = None
    if dept or input or output:
        route = {
            "dept": dept or "command",
            "inputs": [input or "text"],
            "outputs": [output or "html"],
            "viewports": viewports or 1,
        }
    return microapps.compose(query, html=html or None, title=title or None,
                             route=route)


def list_microapps() -> dict:
    """List generated MiniApps currently registered on the wall catalog."""
    from . import microapps

    return {"apps": microapps.list_public()}


# ---------------------------------------------------------------------------
# Qwen-Flash-Next buffer (BICAMERAL_FRAMEWORK.md §6.6) — three MicroFX
# outputs: compaction_90 sensor, layer_swap trigger, mv2_tap trigger.
# Read-only inspection is always safe; publish/trigger are confirm-gated
# in the docstring because they affect the live bus state.
# ---------------------------------------------------------------------------

def buffer_state() -> dict:
    """Snapshot the Qwen-FN buffer state: bus values (compaction_90,
    qsa_keep_frac, igram progress, current layer) plus the last 16
    events (layer_swap / mv2_tap).  Read-only.
    """
    from .qwen_buffer import snapshot
    return snapshot()


def buffer_publish(field: str, value: float, budget: int = 2048,
                   total: int = 1) -> dict:
    """Publish one buffer field to the MicroFX bus.  AFFECTS LIVE STATE.

    field:
      compaction_90   - per-iteration spectral compaction (0..1)
      qsa_keep        - selected blocks count (budget defaults 2048)
      igram_progress  - n-gram shards loaded (total defaults 1)
    """
    from . import qwen_buffer
    if field == "compaction_90":
        qwen_buffer.publish_compaction(value)
    elif field == "qsa_keep":
        qwen_buffer.publish_qsa_keep(int(value), budget=budget)
    elif field == "igram_progress":
        qwen_buffer.publish_igram_progress(int(value), total=total)
    else:
        return {"error": f"unknown field '{field}'"}
    return qwen_buffer.snapshot()


def buffer_trigger(name: str, layer: int = -1, note: str = "") -> dict:
    """Fire one buffer trigger (rising-edge in jsfx.js).  AFFECTS LIVE STATE.

    name:
      layer_swap  - per-layer hot-swap event (set layer=<int>)
      mv2_tap     - latent-loop iteration complete (note=<string>)
    """
    from . import qwen_buffer
    if name == "layer_swap":
        qwen_buffer.fire_layer_swap(layer)
    elif name == "mv2_tap":
        qwen_buffer.fire_mv2_tap(note)
    else:
        return {"error": f"unknown trigger '{name}'"}
    return qwen_buffer.snapshot()


# ---------------------------------------------------------------------------
# BICAMERAL Goals 1, 9: DMA trace + bicameral pipeline
# ---------------------------------------------------------------------------

def dma_trace_start() -> dict:
    """Start recording per-frame data-flow (Goal 1).  AFFECTS LIVE STATE."""
    from . import dma_trace
    dma_trace.start_trace()
    return {"ok": True, "active": True}


def dma_trace_stop() -> dict:
    """Stop recording and return the snapshot."""
    from . import dma_trace
    dma_trace.stop_trace()
    return dma_trace.snapshot()


def dma_trace_snapshot() -> dict:
    """Return the current DMA trace (active flag + frames ring)."""
    from . import dma_trace
    return dma_trace.snapshot()


def bicameral_run(rows: list | None = None, n_frames: int = 60,
                  alpha: float = 0.2, record: bool = True) -> dict:
    """Run the two-card test stand (Goal 9).  When the Qwen-FN weights
    land, swap rows=<np.array from /v1/hidden_states capture> for real."""
    from . import bicameral_pipeline
    import numpy as np
    if rows:
        arr = np.array(rows, dtype=np.float16)
    else:
        arr = np.random.default_rng(0).standard_normal(
            (64, 2560)).astype(np.float16)
    return bicameral_pipeline.run_pipeline(
        arr, n_frames=n_frames, alpha=alpha, record_to_trace=record)


# ---------------------------------------------------------------------------
# BICAMERAL Goal 10: swarm
# ---------------------------------------------------------------------------

def swarm_run(prompt: str, personas: list[str] | None = None,
              n_per_persona: int = 1, max_tokens: int = 512) -> dict:
    """Run a multi-agent swarm against the duty LLM (Goal 10)."""
    from . import swarm
    return swarm.run_swarm(
        prompt=prompt, personas=personas,
        n_per_persona=n_per_persona, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# BICAMERAL Goals 5, 6: domain router + mv2 export/import
# ---------------------------------------------------------------------------

def domain_route(intent: str, top_k: int = 1) -> dict:
    """Route an intent to a domain shard and swap (Goal 5).
    AFFECTS LIVE STATE — fires layer_swap trigger in the buffer bus."""
    from . import domain_router
    return domain_router.swap_for_intent(intent, k=top_k)


def domain_state() -> dict:
    """Current domain shard state + the catalog."""
    from . import domain_router
    return {
        "loaded": domain_router.list_loaded(),
        "catalog": domain_router.domain_summary(),
    }


def shard_list() -> dict:
    """List transferable domain .mv2 shards (Goal 6)."""
    from . import mv2_export
    return {"transferable": mv2_export.list_transferable()}


def shard_export(domain: str, note: str = "") -> dict:
    """Export a domain .mv2 shard for transfer.  AFFECTS FILESYSTEM."""
    from . import mv2_export
    return mv2_export.export_domain(domain, note=note)


def shard_import(src_path: str, manifest: dict | None = None) -> dict:
    """Install a transferred .mv2 shard locally.  AFFECTS FILESYSTEM."""
    from . import mv2_export
    return mv2_export.import_domain(src_path, manifest=manifest)


# ---------------------------------------------------------------------------
# BICAMERAL Goal 7: self-improvement loop
# ---------------------------------------------------------------------------

def self_improve_run(prompt: str, max_iterations: int = 3,
                     compact_rate: float = 0.5,
                     department: str = "Sciences",
                     session_tag: str = "") -> dict:
    """Run the self-improvement loop (Goal 7): generate → capture →
    analyze → score → store → critique → augment → repeat."""
    from . import self_improve
    cfg = self_improve.SelfImproveConfig(
        prompt=prompt, max_iterations=max_iterations,
        compact_rate=compact_rate, department=department,
        session_tag=session_tag,
    )
    return self_improve.run_self_improve(cfg)


# ---------------------------------------------------------------------------
# BICAMERAL Goals 2, 3, 4: atomic program + round-trip + teaching
# ---------------------------------------------------------------------------

def atomic_compile(name: str, blocks: list, wires: list,
                   description: str = "", tags: list | None = None,
                   target: str = "microfx") -> dict:
    """Compile an AtomicProgram to the specified target
    (microfx / eel2 / python / mermaid / summary).
    Goals 2, 4: 'from assembly description back to program'."""
    from . import atomic_program
    p = atomic_program.AtomicProgram(
        name=name, description=description,
        blocks=[atomic_program.AtomicBlock(**b) for b in blocks],
        wires=[atomic_program.Wire(**w) for w in wires],
        tags=tags or [],
    )
    return atomic_program.compile_program(p, target)


def atomic_from_description(description: str) -> dict:
    """Compile a natural-language program description into an
    AtomicProgram (Goal 2).  Calls the duty LLM with the teaching
    prompt; returns the compiled MicroFX patch."""
    from . import atomic_program
    return atomic_program.from_description(description)


def roundtrip_apply(app_id: str, modifications: dict | None = None) -> dict:
    """Load an app, decompose to AtomicProgram, apply modifications,
    recompile, and return the new patch (Goal 3).  AFFECTS LIBRARY
    on mount (caller composes the result into a POST /api/library/app)."""
    from . import round_trip
    return round_trip.round_trip(app_id, modifications=modifications)


def teach_examples(domain: str = "") -> dict:
    """Return the curated teaching examples (Goal 4)."""
    from . import teach_paths
    return {
        "n": len(teach_paths.TEACHING_EXAMPLES),
        "examples": teach_paths.TEACHING_EXAMPLES,
    }


def teach_prompt(intent: str) -> dict:
    """Return a system prompt that teaches the component model
    (Goal 4)."""
    from . import teach_paths
    return {"prompt": teach_paths.prompt_for_intent(intent)}


def atomize_run(source: str = "recent", dest: str = "atomic") -> dict:
    """/atomize <source> <dest> — turn a DMA trace into AtomicProgram
    or n-gram teacher examples (Goals 1, 4, 8).  AFFECTS LIBRARY when
    dest='atomic' (registers new compiled programs)."""
    import requests as _r
    r = _r.post(
        "http://127.0.0.1:18093/api/atomize/run",
        json={"source": source, "dest": dest}, timeout=60,
    )
    return r.json()


def optimize_analyze(focus: str = "") -> dict:
    """/optimize [focus] — self-analyse memory/process state.
    focus: shard | trace | buffer | domain | memory | process | teach
    (empty = all probes)."""
    import requests as _r
    body = {"focus": focus} if focus else {}
    r = _r.post("http://127.0.0.1:18093/api/optimize/analyze",
                json=body, timeout=30)
    return r.json()


TOOLS = [ship_status, log_append, log_recent, show_place, terrain_los,
         terrain_horizon,
         terrain_set_origin, terrain_survey, hadamard_build, hoa_encode,
         hoa_decode, hoa_rotate, orbital_probe, antenna_pattern,
         filter_response, link_budget, room_impulse, sensor_query,
         fdtd_probe, materials_design, actual_size_scales, crown_psf,
         sage_eval, generate_microapp, list_microapps, code_index,
         jellyfin_status, jellyfin_search, jellyfin_play, arr_queue,
         ha_state, ha_call,
         backlot_status, backlot_programs, backlot_submit,
         backlot_job_status, backlot_train_submit, backlot_train_jobs,
         backlot_speak, backlot_transcribe, backlot_stems,
         backlot_sample_export, backlot_mesh_upgrade, backlot_stepseq,
         backlot_transcode_submit, backlot_transcode_status,
         net_devices, port_status, stock_quote, firefly_net_worth,
         dept_memory_add, dept_memory_search,
         microfx_patch, latent_loop,
         omp_code,
         pi_code, pi_goal, pipecat_status, harness_status,
         buffer_state, buffer_publish, buffer_trigger,
         dma_trace_start, dma_trace_stop, dma_trace_snapshot,
         bicameral_run,
         swarm_run,
         domain_route, domain_state,
         shard_list, shard_export, shard_import,
         self_improve_run,
         atomic_compile, atomic_from_description, roundtrip_apply,
         teach_examples, teach_prompt, atomize_run, optimize_analyze,
         ls, grep, find, read, write, edit]
