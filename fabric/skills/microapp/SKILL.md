---
name: microapp
description: >
  Generate a new LCARS MiniApp (interactive HTML tile) for a user query
  that is not already a catalog function. Use when the operator says make /
  build / create an app, tracker, calculator, simulator, game, visualizer,
  widget, or microapp; when /chat should return a function rather than a
  paragraph; or when wiring generate_microapp. MiniAppBench (arxiv 2603.09652).
---

# Fabric MiniApp skill

Duty officer skill. The spoken reply stays one sentence; the artifact is a
catalog function whose center tile is a MiniApp.

Call `generate_microapp` with a wizard **route** whenever possible:
`dept` (command/science/…), `input` (text/parameters/place/mqtt/video),
`output` (heatmap/polar/xy/bars/text/html/webpage/3d/media), `viewports`
(1/4/16). Do not invent rails or backends. If the tool returns
`wizard=true`, stop — the operator finishes the four steps.

Do not write HTML unless they pasted some. Do not invent a Python backend.

## When to generate vs reuse

1. If `/api/resolve` already matches a real catalog id with score ≥ 4, call
   that tool instead (hadamard, yagi, sage, survey, ship_status, …).
2. If the query wants a *new* interactive surface — tracker, calculator,
   game, timer, converter, dashboard — call `generate_microapp`.
3. Named places (`show me the city of X`, map/terrain of X) call
   `show_place` (Terrarium DEM heatmap or horizon). Do not invent a
   caption MiniApp for a map.
4. If Intention / Static / Dynamic / left-rail I/O fail, return `clarify`
   (do not ship a caption tile). The viz BL elbow is the configurator.
   Every compose/choose/accept is traced at `GET /api/microapps/traces`
   (`?sft=1` is the later script-slot LoRA corpus).
5. Never generate CircuitJS / KiCad (D19 concept). Never run micromag,
   gerzon, or sudoku search on the device.

## World model (MiniAppBench)

HTML = world state. CSS = visual salience. JS = causal / temporal logic.
Score the result on three gates (the tool returns them):

| Gate | Pass when |
|---|---|
| Intention | Title + principle cover the query; a real-world rule is instantiated, not captioned |
| Static | Valid id, I/O fields, template; no remote script / iframe / cookie |
| Dynamic | At least one event listener; Apply posts `{type:"io", fields}` into the iframe |

## I/O positions (locked chrome)

| Slot | Where | What |
|---|---|---|
| command | bottom bar, Communications-aligned | the query; generates the box |
| left | 150px rail | number / text fields + Apply |
| right | viz rail | select options only |
| center | 4×4 wall tile | sandboxed iframe or canvas |
| term | terminal viewport | log line, not the MiniApp |

Templates: `html` (default iframe), `canvas` (bound library figure),
`text` (CAS/chat), `square` / `wide` / `solo` spans on the wall.

Libraries the MiniApp may bind (do not reimplement): `show_place`,
`hadamard_build`, `hoa_encode/decode/rotate`, `orbital_probe`,
`antenna_pattern`, `filter_response`, `link_budget`, `terrain_*`,
`room_impulse`, `sage_eval`, `ship_status`, `sensor_query`.
Full kit: `GET /api/microapps/kit`.

## After the tool returns

Speak: `"<title> is on the wall. Left rail is the I/O."`
The HTTP layer attaches `spec` so the shell upserts the catalog and mounts
the tile. Do not paste markup into the reply.
