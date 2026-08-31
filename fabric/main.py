import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from . import departments
from .tools import log_append, log_recent, ship_status
from .tracker import parse_sections


@asynccontextmanager
async def _lifespan(app):
    from . import bus

    bus.start_publisher(interval_s=5.0)
    try:
        yield
    finally:
        bus.stop_publisher()


app = FastAPI(title="woodfire-fabric", version="0.1.0", lifespan=_lifespan)

from .opencode_proxy import router as _oc_router  # noqa: E402

app.include_router(_oc_router)

_agent = None


class LogIn(BaseModel):
    text: str
    source: str = "operator"
    level: str = "info"


class ChatIn(BaseModel):
    message: str
    department: int | None = None


class OmpRun(BaseModel):
    message: str
    session: str | None = None


class GeoLosIn(BaseModel):
    tx: list[float]
    rx: list[float]
    f_mhz: float = 5800.0


class GeoHorizonIn(BaseModel):
    center: list[float] | None = None
    h_agl: float = 25.0
    n_az: int = 72
    elev_deg: float = -2.0


class GeoOriginIn(BaseModel):
    lat: float
    lon: float
    zoom: int = 12


class GeoPlaceIn(BaseModel):
    q: str = ""
    lat: float | None = None
    lon: float | None = None
    zoom: int = 12
    view: str = "terrain"
    h_agl: float = 25.0
    n_az: int = 72

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def _empty_coord(cls, v):
        if v == "" or v is None:
            return None
        return v


class GeoLosLlIn(BaseModel):
    tx_lat: float
    tx_lon: float
    rx_lat: float
    rx_lon: float
    tx_h: float = 15.0
    rx_h: float = 15.0
    f_mhz: float = 5800.0


class GeoSurveyIn(BaseModel):
    tx_lat: float
    tx_lon: float
    rx_lat: float
    rx_lon: float
    tx_h: float = 15.0
    rx_h: float = 15.0
    f_mhz: float = 5800.0
    n: int = 200
    zoom: int | None = None


class RoomIrIn(BaseModel):
    src: list[float] | str
    lst: list[float] | str
    box: list[float] | None = None
    n_rays: int = 1024
    max_bounce: int = 3
    absorption: float = 0.2
    order: int = 3
    furniture: list | None = None

    @field_validator("src", "lst", mode="before")
    @classmethod
    def _csv_vec(cls, v):
        if isinstance(v, str):
            return [float(x) for x in v.split(",") if x.strip()]
        return v


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.get("/ship_status")
def ship_status_route():
    return ship_status()


@app.get("/api/geo/terrain")
def geo_terrain_route():
    from . import geo

    return geo.terrain_meta()


@app.post("/api/geo/los")
def geo_los_route(body: GeoLosIn):
    from . import geo

    return geo.los(body.tx, body.rx, body.f_mhz)


@app.get("/api/geo/los/last")
def geo_los_last_route():
    from . import geo

    result = geo.last_los()
    if result is None:
        raise HTTPException(status_code=404, detail="no los query yet")
    return result


@app.post("/api/geo/horizon")
def geo_horizon_route(body: GeoHorizonIn):
    from . import geo

    center = body.center
    if not center:
        meta = geo.terrain_meta()
        half = float(meta["span_m"]) / 2.0
        center = [half, half]
    return geo.horizon(center, body.h_agl, body.n_az, body.elev_deg)


@app.post("/api/geo/place")
def geo_place_route(body: GeoPlaceIn):
    from . import geo

    return geo.show_place(body.q, lat=body.lat, lon=body.lon,
                          zoom=body.zoom, view=body.view,
                          h_agl=body.h_agl, n_az=body.n_az)


@app.post("/api/geo/origin")
def geo_origin_route(body: GeoOriginIn):
    from . import geo

    try:
        return geo.load_origin(body.lat, body.lon, body.zoom)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/geo/los/latlon")
def geo_los_latlon_route(body: GeoLosLlIn):
    from . import geo

    return geo.los_latlon(body.tx_lat, body.tx_lon, body.rx_lat, body.rx_lon,
                          body.tx_h, body.rx_h, body.f_mhz)


@app.post("/api/geo/survey")
def geo_survey_route(body: GeoSurveyIn):
    from . import geo

    return geo.survey(body.tx_lat, body.tx_lon, body.rx_lat, body.rx_lon,
                      body.tx_h, body.rx_h, body.f_mhz, n=body.n,
                      zoom=body.zoom)


@app.get("/api/sensors")
def sensors_route(prefix: str = "ship/", since: float | None = None,
                  limit: int = 50):
    from . import bus

    return {
        "broker": bus.broker_status(),
        "readings": bus.query(prefix=prefix, since=since, limit=limit),
    }


@app.get("/api/console")
def console_route(group: str | None = None):
    from . import console

    return console.catalog(group)


@app.get("/api/resolve")
def resolve_route(q: str = "", prefer_chat: bool = False):
    from . import intent

    return intent.resolve(q, prefer_chat=prefer_chat)


@app.get("/api/commands")
def commands_route():
    from . import commands

    return commands.catalog()


@app.get("/api/voice")
def voice_route():
    from . import voice

    return voice.pipecat_status()


@app.get("/api/pi")
def pi_health_route():
    from . import pi_client

    return pi_client.health()


@app.post("/api/pi/code")
def pi_code_route(body: dict):
    """SSE-stream an OMP coding run (same events as /api/omp/run).

    Used by the /code slash command; the web terminal reads the
    event-stream so long runs show live progress instead of hanging
    on "running omp…".  Falls back to plain JSON for clients that
    send ``{"stream": false}``.
    """
    from . import pi_client

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    if body.get("stream") is False:
        return pi_client.code_task(prompt)
    return _omp_stream_response(prompt, body.get("session"))


@app.post("/api/pi/goal")
def pi_goal_route(body: dict):
    """Register a goal and SSE-stream the OMP run that works it.

    Same events as /api/omp/run; when the run finishes the goal row
    is updated (done/error + result tail).  ``{"stream": false}``
    keeps the old blocking JSON behavior.
    """
    from . import pi_client
    from .log_store import get_log_store

    statement = (body.get("goal") or body.get("prompt") or "").strip()
    if not statement:
        return {"ok": False, "error": "empty goal"}
    if body.get("stream") is False:
        return pi_client.run_goal(statement)
    store = get_log_store()
    row = store.goal_add(statement)
    task = ("Active goal: " + statement + "\n"
            "Task: Work this goal in the repo. Small goals: do them. "
            "Large goals: at most an 8-step plan and only the "
            "first safe step.")

    def after(code: int, reply: str, err: str) -> None:
        status = "done" if code == 0 else "error"
        store.goal_update(row["id"], status=status,
                          last_result=(reply or err or "")[:4000])

    return _omp_stream_response(task, body.get("session"),
                                after=after,
                                extra_events=[{"t": "goal",
                                               "id": row["id"],
                                               "text": statement}])


@app.get("/api/goals")
def goals_route(limit: int = 20):
    from .log_store import get_log_store

    return {"goals": get_log_store().goals(limit=min(limit, 100))}


@app.get("/api/harness")
def harness_get():
    from . import harness

    return harness.public()


@app.put("/api/harness")
def harness_put(body: dict):
    from . import harness

    out = harness.update(body or {})
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    global _agent
    _agent = None
    return out


@app.post("/api/harness/test")
def harness_test(body: dict | None = None):
    from . import harness

    pid = (body or {}).get("provider") or (body or {}).get("orchestrator")
    return harness.ping(pid)


@app.get("/api/harness/models")
def harness_models_route(provider: str | None = None):
    from . import harness

    if provider:
        return harness.list_models(provider)
    return harness.fetch_all_models()


@app.get("/api/microapps/kit")
def microapp_kit_route():
    from . import microapps

    return microapps.kit()


@app.get("/api/microapps")
def microapp_list_route():
    from . import microapps

    return {"apps": microapps.list_public()}


@app.get("/api/microapps/propose")
def microapp_propose_route(q: str = ""):
    from . import microapps

    return microapps.propose(q)


@app.post("/api/microapps/trace")
def microapp_trace_route(body: dict):
    from . import microapps

    microapps.record_trace(
        query=body.get("query") or "",
        action=body.get("action") or "choose",
        choice=body.get("choice") or {},
        spec=body.get("spec"),
        gates=body.get("gates"),
    )
    return {"ok": True}


@app.get("/api/microapps/traces")
def microapp_traces_route(limit: int = 50, sft: bool = False,
                          passed: bool | None = None):
    from .log_store import get_log_store

    store = get_log_store()
    if sft:
        return {"examples": store.sft_examples(limit=min(limit, 500))}
    return {"traces": store.traces(limit=min(limit, 200), passed=passed)}


@app.get("/api/library")
def library_route():
    from . import library

    return {"apps": library.apps(), "functions": library.functions()}


@app.post("/api/library/cull")
def library_cull_route(body: dict | None = None):
    from . import library

    out = library.cull(confirm=bool((body or {}).get("confirm")))
    if not (body or {}).get("confirm"):
        return out
    return out


@app.post("/api/scriptwiz/convert")
def scriptwiz_convert_route(body: dict):
    from . import microapps

    out = microapps.compose_script(body or {})
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.get("/api/wizard/catalog")
def wizard_catalog_route():
    from . import appwiz

    return appwiz.wizard_catalog()


@app.post("/api/wizard/generate")
def wizard_generate_route(body: dict):
    from . import microapps

    out = microapps.compose_signal(body or {})
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/api/microapps/compose")
def microapp_compose_route(body: dict):
    from . import microapps

    out = microapps.compose(body.get("query") or "",
                            html=body.get("html") or None,
                            title=body.get("title") or None,
                            kernel=body.get("kernel") or None,
                            bind=body.get("bind") or None,
                            route=body.get("route") or None,
                            span=body.get("span") or None)
    if out.get("error") and not out.get("clarify"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/api/library/app/{app_id}/io")
def library_app_io_route(app_id: str, body: dict):
    from . import microapps

    out = microapps.update_io(app_id, body or {})
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/api/library/new")
def library_new_route(body: dict | None = None):
    """Blank app draft for the build wizard (New App)."""
    from . import microapps

    out = microapps.new_app((body or {}).get("title"))
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/api/library/app/{app_id}/clone")
def library_app_clone_route(app_id: str, body: dict | None = None):
    """Save App As: copy a stored app under a new id/title."""
    from . import microapps

    out = microapps.clone_app(app_id, (body or {}).get("title"))
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/api/library/app/{app_id}/rename")
def library_app_rename_route(app_id: str, body: dict):
    from . import microapps

    out = microapps.rename_app(app_id, (body or {}).get("title", ""))
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.get("/api/library/app/{app_id}")
def library_app_route(app_id: str):
    from . import microapps

    spec = microapps.get(app_id)
    if not spec:
        raise HTTPException(status_code=404, detail="no such app")
    return spec


@app.post("/api/microapps/{app_id}/run")
def microapp_run_route(app_id: str, body: dict | None = None):
    from . import microapps

    out = microapps.run(app_id, body or {})
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@app.get("/api/sage")
def sage_info_route():
    from . import sage

    return sage.sage_available()


@app.post("/api/sage")
def sage_route(body: dict):
    from . import sage

    return sage.sage_eval(body.get("expr") or "")


@app.get("/api/lab/hadamard")
def lab_hadamard_route(n: int, method: str = "auto"):
    from . import lab

    return lab.hadamard_build(n, method)


# ---- writable n-gram PLE (Step 3, docs/27b-writable-ngram-plan.md) ----
@app.get("/api/ngram/lookup")
def ngram_lookup_route(layer: int, token: int, pos: int = 0):
    from . import ngram_api

    return ngram_api.lookup(layer, token, pos)


@app.post("/api/ngram/lookup_batch")
def ngram_lookup_batch_route(body: dict):
    from . import ngram_api

    return ngram_api.lookup_batch(
        int(body.get("layer", 3)), body.get("tokens") or [],
        int(body.get("pos", 0)))


@app.post("/api/ngram/store")
def ngram_store_route(body: dict):
    from . import ngram_api

    rows = body.get("rows")
    if rows is None:  # single-row shorthand {token, vector}
        rows = [{"token": body.get("token"), "vector": body.get("vector"),
                 "graph_id": body.get("graph_id")}]
    return ngram_api.store_rows(
        int(body.get("layer", 3)), rows, int(body.get("pos", 0)))


@app.post("/api/ngram/flush")
def ngram_flush_route():
    from . import ngram_api

    return ngram_api.flush()


@app.get("/api/ngram/stats")
def ngram_stats_route():
    from . import ngram_api

    return ngram_api.stats()


def _csv_floats(v):
    if isinstance(v, str):
        return [float(x) for x in v.split(",") if x.strip()]
    return v


@app.post("/api/lab/hoa")
def lab_hoa_route(body: dict):
    from . import lab

    az = _csv_floats(body.get("azimuths"))
    el = _csv_floats(body.get("elevations"))
    return lab.hoa_encode(az, el, body.get("gains"), body.get("order", 4))


@app.post("/api/lab/hoa/decode")
def lab_hoa_decode_route(body: dict):
    from . import lab

    az = _csv_floats(body.get("azimuths"))
    el = _csv_floats(body.get("elevations"))
    return lab.hoa_decode(az, el, body.get("gains"), body.get("order", 4))


@app.get("/api/lab/orbital")
def lab_orbital_route(n: int, l: int, m: int, n_r: int = 48):
    from . import lab

    return lab.orbital_probe(n, l, m, n_r)


@app.get("/api/lab/antenna")
def lab_antenna_route(kind: str, f_mhz: float, phi_deg: float = 0.0,
                      n_theta: int = 91):
    from . import lab

    return lab.antenna_pattern(kind, f_mhz, phi_deg, n_theta)


@app.get("/api/lab/filter")
def lab_filter_route(kind: str, f_c_mhz: float, n: int = 5,
                     proto: str = "butterworth", f_lo_mhz: float = None,
                     f_hi_mhz: float = None, eps_r: float = 4.4,
                     h_mm: float = 1.6):
    from . import lab

    return lab.filter_response(kind, f_c_mhz, n, proto, f_lo_mhz, f_hi_mhz,
                               eps_r, h_mm)


@app.get("/api/lab/link")
def lab_link_route(p_tx_dbw: float, g_tx_dbi: float, g_rx_dbi: float,
                   f_mhz: float, d_m: float, medium: str = "air"):
    from . import lab

    return lab.link_budget(p_tx_dbw, g_tx_dbi, g_rx_dbi, f_mhz, d_m, medium)


@app.post("/api/lab/hoa/rotate")
def lab_hoa_rotate_route(body: dict):
    from . import lab

    az = _csv_floats(body.get("azimuths"))
    el = _csv_floats(body.get("elevations"))
    return lab.hoa_rotate(az, el, body.get("yaw_deg", 0),
                          body.get("pitch_deg", 0), body.get("roll_deg", 0),
                          order=body.get("order", 4))


@app.get("/api/lab/fdtd")
def lab_fdtd_route(f_mhz: float = 150.0, medium: str = "air", n: int = 16):
    from . import lab

    return lab.fdtd_probe(f_mhz, medium, n)


@app.get("/api/lab/materials")
def lab_materials_route(kind: str = "cloth", order: int = 8):
    from . import lab

    return lab.materials_design(kind, order)


@app.get("/api/lab/scales")
def lab_scales_route(eps: float = 0.003):
    from . import lab

    return lab.actual_size_scales(eps)


@app.get("/api/lab/crown")
def lab_crown_route(n: int = 32):
    from . import lab

    return lab.crown_psf(n)


@app.get("/api/lab/last")
def lab_last_route():
    from . import lab

    figure = lab.last_figure()
    if figure.get("figure") is None:
        raise HTTPException(status_code=404, detail="no lab figure yet")
    return figure


@app.post("/api/room/ir")
def room_ir_route(body: RoomIrIn):
    from . import room

    return room.room_impulse(body.src, body.lst, box=body.box,
                             n_rays=body.n_rays, max_bounce=body.max_bounce,
                             absorption=body.absorption, order=body.order,
                             furniture=body.furniture)


@app.get("/api/room/last")
def room_last_route():
    from . import room

    result = room.last_room()
    if result.get("figure") is None:
        raise HTTPException(status_code=404, detail="no room IR yet")
    return result


@app.get("/api/departments")
def departments_route():
    return departments.catalog()


@app.get("/api/departments/{ext}")
def department_route(ext: int):
    dept = departments.by_ext(ext)
    if dept is None:
        raise HTTPException(status_code=404, detail="unknown extension")
    dept["pane_order"] = departments.pane_order(ext)
    return dept


@app.post("/log")
def log_route(body: LogIn):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    return log_append(body.text, source=body.source, level=body.level)


@app.get("/log/recent")
def log_recent_route(limit: int = 50, level: str = None, source: str = None):
    return {"entries": log_recent(limit=min(limit, 500), level=level, source=source)}


def _tool_names(result) -> list[str]:
    names = []
    try:
        for msg in result.all_messages():
            for part in getattr(msg, "parts", []):
                tn = getattr(part, "tool_name", None)
                if tn and type(part).__name__ == "ToolCallPart":
                    names.append(tn)
    except Exception:
        pass
    return names


@app.post("/chat")
async def chat_route(body: ChatIn):
    global _agent
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="empty message")
    extra = ""
    dept = None
    if body.department is not None:
        dept = departments.by_ext(body.department)
        if dept:
            extra = (
                f"Active department: {dept['name']} (ext {dept['ext']}). "
                f"{dept['scope']}."
            )
    if _agent is None:
        from .agent import build_agent

        try:
            _agent = build_agent()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"agent init failed: {e}") from e
    try:
        if dept:
            try:
                from .dept_memory import context_for

                block = context_for(dept["name"], body.message)
                if block:
                    extra = (extra + "\n" if extra else "") + block
            except Exception:
                pass
        result = await _agent.run(
            body.message,
            instructions=extra or None,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"orchestrator failed: {e}") from e
    return _chat_payload(result, dept)


def _chat_payload(result, dept) -> dict:
    microapp = None
    figure = None
    clarify = None
    gates = None
    last = getattr(result, "last_result", None)
    if isinstance(last, dict):
        if isinstance(last.get("spec"), dict):
            microapp = last["spec"]
        if isinstance(last.get("figure"), dict):
            figure = last["figure"]
        if last.get("clarify"):
            clarify = {k: last[k] for k in
                       ("clarify", "choices", "hint", "gates", "draft")
                       if k in last}
        if isinstance(last.get("gates"), dict):
            gates = last["gates"]
    orch = getattr(_agent, "orchestrator", None) or "local"
    return {
        "reply": result.output,
        "department": None if dept is None else dept["ext"],
        "tools": _tool_names(result),
        "microapp": microapp,
        "figure": figure,
        "clarify": clarify,
        "gates": gates,
        "orchestrator": orch,
    }


@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """SSE progress for the duty loop: llm/tool/result events, then final.

    Final event carries the exact /chat response shape so the shell can
    reuse one renderer. Errors surface as {"t":"error","detail":…}.
    """
    global _agent
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="empty message")
    extra = ""
    dept = None
    if body.department is not None:
        dept = departments.by_ext(body.department)
        if dept:
            extra = (
                f"Active department: {dept['name']} (ext {dept['ext']}). "
                f"{dept['scope']}."
            )
    if _agent is None:
        from .agent import build_agent

        try:
            _agent = build_agent()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"agent init failed: {e}") from e

    async def gen():
        import asyncio
        import json as _json

        q: asyncio.Queue = asyncio.Queue()

        def sink(ev: dict) -> None:
            q.put_nowait(ev)

        async def drive():
            try:
                result = await _agent.run(
                    body.message, instructions=extra or None, on_event=sink)
            except Exception as e:  # noqa: BLE001
                await q.put({"t": "error", "detail": str(e)})
                return
            try:
                payload = _chat_payload(result, dept)
            except Exception as e:  # noqa: BLE001
                await q.put({"t": "error", "detail": str(e)})
                return
            await q.put({"t": "final", **payload})

        driver = asyncio.create_task(drive())
        try:
            while True:
                ev = await q.get()
                yield f"data: {_json.dumps(ev)}\n\n"
                if ev.get("t") in ("final", "error"):
                    break
        finally:
            if not driver.done():
                driver.cancel()

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/tracker")
def tracker_route():
    return parse_sections()


@app.get("/api/lab/planets")
def lab_planets(date: str = "", orbits: int = 1):
    from . import ephem

    out = ephem.planets(date)
    if orbits:
        sampled = ephem.orbits_sampled(date)
        out["paths"] = sampled["paths"]
    return out


@app.get("/api/news")
def news_route(limit: int = 20):
    from .news import latest

    return latest(limit)


@app.get("/api/stocks")
def stocks_route(symbols: str = "AAPL,MSFT"):
    from .stocks import quotes

    return quotes(symbols)


@app.get("/api/firefly/net-worth")
def firefly_networth_route():
    from .firefly import net_worth

    return net_worth()


@app.get("/api/sensors/history")
def sensors_history(topic: str = "ship/vllm/toks", seconds: float = 900,
                    key: str = "val", maxpoints: int = 180):
    """Windowed time-series from the sensor log (rolling charts)."""
    import time as _time

    from . import bus

    since = _time.time() - float(max(30, min(seconds, 86400)))
    rows = bus.query(prefix=topic, since=since,
                     limit=int(max(10, min(maxpoints * 4, 2000))))
    rows.reverse()
    step = max(1, len(rows) // int(max(20, min(maxpoints, 500))))
    series = []
    for r in rows[::step]:
        payload = r["payload"]
        if isinstance(payload, dict):
            val = payload.get(key)
        else:
            val = payload
        if isinstance(val, (int, float)):
            series.append([round(r["ts"], 2), round(float(val), 3)])
    return {"topic": topic, "key": key, "series": series,
            "window_s": seconds}


@app.get("/api/net/devices")
def net_devices_route():
    from .net import net_devices

    return net_devices()


@app.get("/api/ports")
def ports_route():
    from .tools import port_status

    return port_status()


@app.get("/api/media/stack")
def media_stack_route():
    from .media import stack_status

    return stack_status()


@app.get("/api/backlot/status")
def backlot_status_route():
    from .backlot import backlot_status

    return backlot_status()


# ---------------------------------------------------------------------------
# Qwen-Flash-Next buffer (BICAMERAL_FRAMEWORK.md §6.6)
# Three MicroFX outputs wired into the bus:
#   /api/buffer/state         GET   bus snapshot + event ring
#   /api/buffer/publish       POST  publish compaction_90 / qsa / igram
#   /api/buffer/trigger       POST  fire layer_swap / mv2_tap
# ---------------------------------------------------------------------------

class BufferPublish(BaseModel):
    field: str
    value: float


class BufferTrigger(BaseModel):
    name: str
    payload: dict | None = None


@app.get("/api/buffer/state")
def buffer_state_route():
    from .qwen_buffer import snapshot

    return snapshot()


@app.post("/api/buffer/publish")
def buffer_publish_route(body: BufferPublish):
    from . import qwen_buffer

    if body.field == "compaction_90":
        qwen_buffer.publish_compaction(body.value)
    elif body.field == "qsa_keep":
        budget = int((body.payload or {}).get("budget", 2048)) \
            if body.payload else 2048
        qwen_buffer.publish_qsa_keep(int(body.value), budget=budget)
    elif body.field == "igram_progress":
        total = int((body.payload or {}).get("total", 1)) \
            if body.payload else 1
        qwen_buffer.publish_igram_progress(int(body.value), total=total)
    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown field '{body.field}'")
    return qwen_buffer.snapshot()


@app.post("/api/buffer/trigger")
def buffer_trigger_route(body: BufferTrigger):
    from . import qwen_buffer

    if body.name == "layer_swap":
        layer = int((body.payload or {}).get("layer", -1))
        qwen_buffer.fire_layer_swap(layer)
    elif body.name == "mv2_tap":
        note = str((body.payload or {}).get("note", ""))
        qwen_buffer.fire_mv2_tap(note)
    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown trigger '{body.name}'")
    return qwen_buffer.snapshot()


def _omp_stream_response(msg: str, session: str | None = None,
                         after=None, extra_events: list | None = None):
    """Shared SSE wrapper around an oh-my-pi ``--mode json`` run.

    Events: {"t":"session","id"} · {"t":"tool","name","detail"} ·
    {"t":"delta","text"} (assistant text per message) ·
    {"t":"toolout","text"} (tool results, trimmed) · {"t":"final"}.
    cwd is pinned to the working tree; runs may take many minutes.

    ``after(code, reply, err_tail)`` runs once the process exits (goal
    bookkeeping); ``extra_events`` are yielded first (e.g. goal id).
    """
    import json as _json

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def gen():
        import subprocess

        for ev in extra_events or []:
            yield f"data: {_json.dumps(ev)}\n\n"
        omp_bin = os.environ.get(
            "OMP_BIN", os.path.expanduser("~/.local/bin/omp"))
        # Lean tool surface: OMP's default system prompt (~24k tok of
        # skills/rules/extensions) starves a 64k window before turn one.
        cmd = [omp_bin, "--mode", "json", "-p", msg,
               "--cwd", repo, "--max-time", "30m",
               "--approval-mode", "yolo",
               "--no-skills", "--no-rules", "--no-extensions",
               "--tools", "read,edit,write,bash,grep,glob,todo"]
        if session:
            cmd += ["-r", session]
        else:
            cmd.append("--no-session")
        err_f = open("/tmp/fabric-omp-stderr.log", "w")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_f,
            stdin=subprocess.DEVNULL,
            cwd=repo, text=True, bufsize=1)
        assert proc.stdout is not None
        last_reply = ""
        omp_sid = f"omp-{session}" if session else "omp-new"
        live = False
        try:
            from . import latent as _latent
            live = _latent._ngram_live_enabled()
        except Exception:  # noqa: BLE001
            live = False
        if live:
            # Activate this OMP session's rolling buffer as early as
            # possible — OMP's first prefill happens inside the CLI.
            _latent._PLE_BUFFER.activate(omp_sid)
        for raw in proc.stdout:
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = _json.loads(line)
            except Exception:
                continue
            etype = ev.get("type") or ""
            if etype == "session":
                sid = ev.get("id")
                if live and sid and f"omp-{sid}" != omp_sid:
                    omp_sid = f"omp-{sid}"
                    _latent._PLE_BUFFER.activate(omp_sid)
                yield f"data: {_json.dumps({'t': 'session', 'id': sid})}\n\n"
            elif etype == "tool_execution_start":
                yield f"data: {_json.dumps({'t': 'tool', 'name': ev.get('toolName') or '?', 'detail': str(ev.get('args') or '')[:120]})}\n\n"
            elif etype == "message_end":
                m = ev.get("message") or {}
                role = m.get("role")
                content = m.get("content")
                texts = []
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            texts.append(c.get("text") or "")
                text = "\n".join(texts).strip()
                if role == "assistant" and text:
                    last_reply = text
                    yield f"data: {_json.dumps({'t': 'delta', 'text': text[:4000]})}\n\n"
                elif role == "toolResult" and text:
                    yield f"data: {_json.dumps({'t': 'toolout', 'text': text[:400]})}\n\n"
        code = proc.wait()
        if live and msg.strip():
            # End-of-run feed: the run's prompt joins the session's
            # rolling buffer, so the next run resumes with the table warm.
            # User text only — feeding the assistant's own reply caused a
            # self-referential decode echo (intra-sentence stutter).
            def _feed() -> None:
                try:
                    _latent.live_note(
                        omp_sid,
                        f"User: {msg}")
                except Exception:  # noqa: BLE001
                    pass
            import threading
            threading.Thread(target=_feed, daemon=True).start()
        try:
            err_f.close()
        except Exception:
            pass
        tail = ""
        try:
            with open("/tmp/fabric-omp-stderr.log") as _f:
                tail = _f.read()[-300:]
        except Exception:
            pass
        if after is not None:
            try:
                after(code, last_reply, tail)
            except Exception:
                pass
        yield f"data: {_json.dumps({'t': 'final', 'code': code, 'err': tail})}\n\n"

    from fastapi.responses import StreamingResponse

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/omp/run")
async def omp_run(body: OmpRun):
    """Stream an oh-my-pi coding-agent run as SSE events.

    Events: {"t":"session","id"} · {"t":"tool","name","detail"} ·
    {"t":"delta","text"} (assistant text per message) ·
    {"t":"toolout","text"} (tool results, trimmed) · {"t":"final"}.
    cwd is pinned to the working tree; runs may take many minutes.
    """
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="empty message")
    return _omp_stream_response(msg, body.session)


_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lcars26 = os.path.join(_repo, "LCARS-26", "LCARS-26")
if os.path.isdir(_lcars26):
    app.mount("/ui/lcars26", StaticFiles(directory=_lcars26), name="lcars26")

_upstream_dir = os.path.join(_repo, "vendor", "lcars", "dist")
if os.path.isdir(_upstream_dir):
    app.mount("/ui/upstream", StaticFiles(directory=_upstream_dir, html=True), name="upstream")

_web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(_web_dir):
    app.mount("/ui", StaticFiles(directory=_web_dir, html=True), name="ui")


# ---------------------------------------------------------------------------
# BICAMERAL Goals 1, 9: DMA trace + bicameral pipeline
# ---------------------------------------------------------------------------

@app.post("/api/trace/start")
def trace_start():
    from . import dma_trace
    dma_trace.start_trace()
    return {"ok": True, "active": True}


@app.post("/api/trace/stop")
def trace_stop():
    from . import dma_trace
    dma_trace.stop_trace()
    return dma_trace.snapshot()


@app.get("/api/trace/snapshot")
def trace_snapshot():
    from . import dma_trace
    return dma_trace.snapshot()


@app.get("/api/trace/recent_bus")
def trace_recent_bus():
    from . import dma_trace
    return dma_trace.recent_bus()


@app.post("/api/bicameral/run")
def bicameral_run(body: dict | None = None):
    from . import bicameral_pipeline
    import numpy as np
    body = body or {}
    if body.get("rows"):
        rows = np.array(body["rows"], dtype=np.float16)
    else:
        rows = np.random.default_rng(0).standard_normal(
            (64, 2560)).astype(np.float16)
    return bicameral_pipeline.run_pipeline(
        rows,
        n_frames=int(body.get("n_frames", 60)),
        alpha=float(body.get("alpha", 0.2)),
        record_to_trace=bool(body.get("record", True)),
    )


# ---------------------------------------------------------------------------
# BICAMERAL Goal 10: swarm
# ---------------------------------------------------------------------------

@app.post("/api/swarm/run")
def swarm_run(body: dict | None = None):
    from . import swarm
    body = body or {}
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    return swarm.run_swarm(
        prompt=prompt,
        personas=body.get("personas"),
        n_per_persona=int(body.get("n_per_persona", 1)),
        max_tokens=int(body.get("max_tokens", 512)),
    )


# ---------------------------------------------------------------------------
# BICAMERAL Goals 5, 6: domain router + mv2 export/import
# ---------------------------------------------------------------------------

@app.post("/api/domain/route")
def domain_route(body: dict | None = None):
    from . import domain_router
    body = body or {}
    text = body.get("intent", "")
    k = int(body.get("top_k", 1))
    return domain_router.swap_for_intent(text, k=k)


@app.get("/api/domain/state")
def domain_state():
    from . import domain_router
    return {
        "loaded": domain_router.list_loaded(),
        "catalog": domain_router.domain_summary(),
    }


@app.get("/api/shard/list")
def shard_list():
    from . import mv2_export
    return {"transferable": mv2_export.list_transferable()}


@app.post("/api/shard/export")
def shard_export(body: dict | None = None):
    from . import mv2_export
    body = body or {}
    domain = body.get("domain", "")
    if not domain:
        raise HTTPException(status_code=400, detail="missing 'domain'")
    return mv2_export.export_domain(
        domain,
        origin=body.get("origin"),
        note=body.get("note", ""),
    )


@app.post("/api/shard/import")
def shard_import(body: dict | None = None):
    from . import mv2_export
    body = body or {}
    src = body.get("src_path", "")
    manifest = body.get("manifest")
    if not src:
        raise HTTPException(status_code=400, detail="missing 'src_path'")
    return mv2_export.import_domain(src, manifest=manifest)


# ---------------------------------------------------------------------------
# BICAMERAL Goal 7: self-improvement loop
# ---------------------------------------------------------------------------

@app.post("/api/improve/run")
def improve_run(body: dict | None = None):
    from . import self_improve
    body = body or {}
    cfg = self_improve.SelfImproveConfig(
        prompt=body.get("prompt", ""),
        max_iterations=int(body.get("max_iterations", 3)),
        compact_rate=float(body.get("compact_rate", 0.5)),
        department=body.get("department", "Sciences"),
        session_tag=body.get("session_tag", ""),
    )
    return self_improve.run_self_improve(cfg)


# ---------------------------------------------------------------------------
# BICAMERAL Goals 2, 3, 4: atomic program + round-trip + teacher examples
# ---------------------------------------------------------------------------

@app.post("/api/atomic/compile")
def atomic_compile(body: dict | None = None):
    from . import atomic_program
    body = body or {}
    if not body:
        raise HTTPException(status_code=400, detail="missing program body")
    p = atomic_program.AtomicProgram(
        name=body.get("name", "generated"),
        description=body.get("description", ""),
        blocks=[atomic_program.AtomicBlock(**b)
                for b in body.get("blocks", [])],
        wires=[atomic_program.Wire(**w)
                for w in body.get("wires", [])],
        tags=body.get("tags", []),
    )
    target = body.get("target", "microfx")
    return atomic_program.compile_program(p, target)


@app.get("/api/atomic/examples")
def atomic_examples(domain: str = "", limit: int = 20):
    from . import atomic_program
    return atomic_program.get_teacher_examples(domain=domain, limit=limit)


@app.post("/api/atomic/from_description")
def atomic_from_description(body: dict | None = None):
    from . import atomic_program
    desc = (body or {}).get("description", "")
    if not desc:
        raise HTTPException(status_code=400, detail="missing 'description'")
    return atomic_program.from_description(desc)


@app.post("/api/roundtrip/apply")
def roundtrip_apply(body: dict | None = None):
    from . import round_trip
    body = body or {}
    app_id = body.get("app_id", "")
    mods = body.get("modifications")
    return round_trip.round_trip(app_id, modifications=mods)


@app.get("/api/teach/examples")
def teach_examples(domain: str = ""):
    from . import teach_paths
    return {
        "n": len(teach_paths.TEACHING_EXAMPLES),
        "examples": teach_paths.TEACHING_EXAMPLES,
    }


@app.post("/api/teach/prompt")
def teach_prompt(body: dict | None = None):
    from . import teach_paths
    text = (body or {}).get("intent", "")
    return {"prompt": teach_paths.prompt_for_intent(text)}


@app.post("/api/optimize/analyze")
def optimize_analyze(body: dict | None = None):
    """Self-analysis — model examines its own state for memory/process
    optimizations.  body: {focus: shard|trace|buffer|domain|memory|
    process|teach} (empty = all).
    """
    from . import optimize as _opt
    body = body or {}
    focus = body.get("focus") or ""
    if focus:
        return _opt.analyze(focus=focus)
    return _opt.analyze()


@app.post("/api/atomize/run")
def atomize_run(body: dict | None = None):
    """Compile a DMA trace into AtomicProgram / n-gram teacher pairs.

    /atomize <source> <dest> in the terminal.
    source: live (currently recording) | recent (last snapshot) | trace (full export)
    dest:   atomic | ngram | both
    """
    from . import dma_trace
    body = body or {}
    source = body.get("source", "recent")
    dest = body.get("dest", "atomic")

    if source == "live":
        snap = dma_trace.snapshot()
        if not snap.get("recording"):
            raise HTTPException(status_code=409,
                detail="DMA trace not recording — /trace start first")
        frames = snap.get("frames", [])
    elif source in ("recent", "snapshot"):
        snap = dma_trace.snapshot()
        frames = snap.get("frames", [])
    else:
        frames = dma_trace.export_trace() or []

    if not frames:
        raise HTTPException(status_code=404,
            detail="No DMA trace frames found — run /trace start then interact")

    result: dict = {"source": source, "frames": len(frames)}

    if dest in ("atomic", "both"):
        from . import atomic_program
        chains = _trace_to_chains(frames)
        programs = []
        for chain in chains:
            try:
                modules = chain["modules"]
                wires = []
                for i in range(len(modules) - 1):
                    src_id = modules[i]["id"]
                    dst_id = modules[i + 1]["id"]
                    wires.append(atomic_program.Wire(
                        src=f"{src_id}.cv", dst=f"{dst_id}.in",
                    ))
                prog = atomic_program.AtomicProgram(
                    name="trace_" + chain["id"][:12],
                    description=f"Auto-generated from DMA trace — "
                                f"{len(modules)} modules",
                    blocks=[
                        atomic_program.AtomicBlock(
                            id=m["id"],
                            primitive=m.get("primitive") or m.get("type") or "unknown",
                            params={},
                            label=m.get("label", ""),
                            position=(0.0, 0.0),
                        )
                        for m in modules
                    ],
                    wires=wires,
                    tags=["atomized", source],
                )
                compiled = atomic_program.compile_program(prog, target="microfx")
                programs.append({
                    "program": prog.name,
                    "blocks": len(prog.blocks),
                    "wires": len(prog.wires),
                    "compiled": compiled.get("modules", compiled.get("error", ""))[:1] if isinstance(compiled, dict) else str(compiled)[:300],
                })
            except Exception as ex:
                programs.append({"error": str(ex), "chain": chain.get("id", "?")})
        result["atomic_programs"] = programs

    if dest in ("ngram", "both"):
        from . import teach_paths
        chains = _trace_to_chains(frames)
        examples = []
        for chain in chains:
            modules = chain.get("modules", [])
            if len(modules) < 2:
                continue
            diagram = "flowchart LR\n" + "\n".join(
                f"    {modules[i]['id']} -->|sig| {modules[i+1]['id']}"
                for i in range(len(modules) - 1)
            )
            desc = (f"Trace atomized from DMA — "
                    f"{modules[0].get('id','?')} feeds {modules[-1].get('id','?')}, "
                    f"{len(modules)} modules")
            example = teach_paths.TeachingExample(
                id="trace_" + chain.get("id", "general")[:12],
                domain=chain.get("id", "general")[:20],
                difficulty="intermediate",
                description=desc,
                assembly_mermaid=diagram,
                inputs=[m.get("id", "?") for m in modules[:1]],
                outputs=[m.get("id", "?") for m in modules[-1:]],
                blocks_used=[m.get("type", "?") for m in modules],
                tags=list(chain.get("tags", [])) + ["atomized", source],
            )
            examples.append({
                "id": example.id,
                "domain": example.domain,
                "description": example.description,
                "assembly_mermaid": example.assembly_mermaid,
                "tags": example.tags,
            })
        result["ngram_examples"] = examples

    return result


def _trace_to_chains(frames: list) -> list[dict]:
    """Flatten DMA trace frames into module chains grouped by tag prefix.

    A frame's tag carries the chain id (e.g. "chain_a:src1"); the
    chain key is the prefix before the colon, or the module_id if
    no prefix is set.
    """
    chains_by_source: dict = {}
    for frame in frames:
        src = frame.get("module_id") or "unknown"
        tag = frame.get("tag") or ""
        chain_key = tag.split(":", 1)[0] if ":" in tag else (tag or src)
        if chain_key not in chains_by_source:
            chains_by_source[chain_key] = {
                "id": chain_key,
                "modules": [],
                "tags": [tag] if tag else [],
            }
        chain = chains_by_source[chain_key]
        module = {
            "id": src,
            "type": frame.get("module_type", "unknown"),
            "primitive": frame.get("module_type", "unknown"),
            "label": frame.get("module_id", ""),
            "in_ports": frame.get("in_ports", {}),
            "out_ports": frame.get("out_ports", {}),
        }
        if module not in chain["modules"]:
            chain["modules"].append(module)
    return list(chains_by_source.values())
