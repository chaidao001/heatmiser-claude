"""Claude-powered natural-language control layer over a Heatmiser NeoHub.

Run:  uvicorn app:app --reload  (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from neohub import make_backend

load_dotenv(Path(__file__).parent / "conf" / ".env")

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
STATIC = Path(__file__).parent / "static"

# Log every control operation to the console. Backend writes to the hub don't
# always "take" (e.g. a setpoint is ignored while a zone is in standby), and the
# only way to see that after the fact is a trace of what was sent and how the hub
# responded. Set HEATMISER_LOG_LEVEL=DEBUG for before/after zone snapshots.
# A dedicated handler (not propagating to root) keeps our lines readable next to
# uvicorn's own logging.
LOG = logging.getLogger("heatmiser")
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOG.addHandler(_handler)
    LOG.propagate = False
LOG.setLevel(os.getenv("HEATMISER_LOG_LEVEL", "INFO").upper())

app = FastAPI(title="Heatmiser + Claude")
# make_backend() reads NEOHUB_TOKEN (and host/port) from the environment, which
# load_dotenv populated from conf/.env.
backend = make_backend()

# Claude authenticates with ANTHROPIC_API_KEY from the environment.
claude = AsyncAnthropic()

SYSTEM_PROMPT = """You control a home's Heatmiser Neo heating via the provided tools.

Guidelines:
- Always call list_zones first if you are unsure of the current state or zone names.
- Interpret vague comfort language sensibly: "warmer" = current target + ~1.5C,
  "cooler" = target - ~1.5C, "cosy/comfortable" ~= 21C, "cold/eco" ~= 17C.
- Temperatures are in degrees Celsius. Keep targets within a safe 5-30C range.
- When the user names a room, match it to a zone; "everywhere"/"the house" means all zones.
- Zones report a `mode` (Heating / Cooling / Vent), a `fan` level, and a `schedule`.
  A zone in `standby` is effectively Off - describe it that way rather than quoting its
  target. Refer to the system as heating or cooling according to the zone's mode.
- You can also change fan speed (set_fan_speed: Auto/High/Medium/Low/Off), heating/cooling
  mode (set_mode: Heating/Cooling/Vent/Auto), and turn a zone's schedule on or off
  (set_schedule).
- You can edit the schedule itself: set_schedule_period sets (or adds) the programmed
  target at a given time of day, keeping the rest of the schedule; clear_schedule_period
  removes the period at a time. Times are 24-hour "HH:MM" (interpret "3am" as "03:00",
  "half seven"/"7:30pm" as "19:30", etc.). Changes apply to the whole weekly program
  (this hub runs one schedule for every day). Use these for lasting schedule changes;
  use hold_temperature for a temporary override that reverts to the schedule.
- To turn the schedule "off" at a time (system does not heat or cool from then), call
  set_schedule_period with off=true - never pass a low temperature to mean off. Remember a
  zone in Cooling mode cools DOWN to its target, so a LOW target means MORE cooling; a low
  number is the opposite of off. "Remove"/"delete" a period means clear_schedule_period.
- After acting, reply in one or two short, friendly sentences stating what you changed
  and the new target(s). Do not invent zones or values you did not get from a tool.
"""

TOOLS = [
    {
        "name": "list_zones",
        "description": "List all heating zones with their current temperature, target "
        "temperature, whether they are calling for heat, and hold/away state.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_temperature",
        "description": "Permanently set the target temperature for one or more zones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Zone names, or ["all"] for every zone.',
                },
                "target_c": {"type": "number", "description": "Target temperature in Celsius (5-35; >=30 is effectively off for cooling)."},
            },
            "required": ["zones", "target_c"],
        },
    },
    {
        "name": "hold_temperature",
        "description": "Temporarily hold a target temperature for a set duration, after "
        "which the zone reverts to its schedule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "target_c": {"type": "number"},
                "hours": {"type": "integer", "minimum": 0},
                "minutes": {"type": "integer", "minimum": 0, "maximum": 59},
            },
            "required": ["zones", "target_c", "hours", "minutes"],
        },
    },
    {
        "name": "set_away",
        "description": "Turn frost/away (standby) protection on or off for one or more zones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "enable": {"type": "boolean"},
            },
            "required": ["zones", "enable"],
        },
    },
    {
        "name": "set_fan_speed",
        "description": "Set the fan speed for one or more zones (cooling / HVAC fan-coil units).",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "speed": {"type": "string", "enum": ["Auto", "High", "Medium", "Low", "Off"]},
            },
            "required": ["zones", "speed"],
        },
    },
    {
        "name": "set_mode",
        "description": "Set a zone's heating/cooling mode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["Heating", "Cooling", "Vent", "Auto"]},
            },
            "required": ["zones", "mode"],
        },
    },
    {
        "name": "set_schedule",
        "description": "Turn a zone's time schedule on (follow its program) or off (manual). "
        "This is the on/off switch only; use set_schedule_period to change the times/temperatures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "enable": {"type": "boolean"},
            },
            "required": ["zones", "enable"],
        },
    },
    {
        "name": "set_schedule_period",
        "description": "Set or add a programmed schedule period: at the given time of day, the "
        "zone's target becomes target_c (or, with off=true, the system is off - no heating/cooling "
        "from that time). Existing period at that time is updated; otherwise a new one is added. "
        "The rest of the schedule is preserved. Applies to the whole weekly program. Use this for "
        "lasting changes (not a temporary hold). To turn a period off, pass off=true - do NOT pass a "
        "low temperature (in cooling a low target means maximum cooling, the opposite of off).",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "time": {"type": "string", "description": '24-hour time of day, "HH:MM" (e.g. "03:00").'},
                "target_c": {"type": "number", "description": "Target temperature in Celsius (5-30). Omit when off=true."},
                "off": {"type": "boolean", "description": "If true, the period is off (no heating/cooling) instead of a temperature."},
            },
            "required": ["zones", "time"],
        },
    },
    {
        "name": "clear_schedule_period",
        "description": "Remove the programmed schedule period at the given time of day, keeping "
        "the rest of the schedule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zones": {"type": "array", "items": {"type": "string"}},
                "time": {"type": "string", "description": '24-hour time of day, "HH:MM".'},
            },
            "required": ["zones", "time"],
        },
    },
]


async def dispatch(name: str, args: dict) -> dict:
    """Execute a Claude tool call against the heating backend."""
    if name != "list_zones":
        LOG.info("Claude tool %s args=%s", name, dict(args))
    if name == "list_zones":
        return {"zones": [z.as_dict() for z in await backend.list_zones()]}
    if name == "set_temperature":
        target = max(5.0, min(35.0, float(args["target_c"])))
        changed = await backend.set_temperature(args["zones"], target)
        return {"changed": changed, "target_c": target}
    if name == "hold_temperature":
        target = max(5.0, min(30.0, float(args["target_c"])))
        changed = await backend.hold_temperature(
            args["zones"], target, int(args["hours"]), int(args["minutes"])
        )
        return {"changed": changed, "target_c": target, "hours": args["hours"], "minutes": args["minutes"]}
    if name == "set_away":
        changed = await backend.set_away(args["zones"], bool(args["enable"]))
        return {"changed": changed, "away": bool(args["enable"])}
    if name == "set_fan_speed":
        changed = await backend.set_fan(args["zones"], str(args["speed"]))
        return {"changed": changed, "fan": args["speed"]}
    if name == "set_mode":
        changed = await backend.set_mode(args["zones"], str(args["mode"]))
        return {"changed": changed, "mode": args["mode"]}
    if name == "set_schedule":
        changed = await backend.set_schedule(args["zones"], bool(args["enable"]))
        return {"changed": changed, "schedule_on": bool(args["enable"])}
    if name == "set_schedule_period":
        if args.get("off"):
            changed = await backend.set_schedule_period(args["zones"], str(args["time"]), off=True)
            return {"changed": changed, "time": str(args["time"]), "off": True}
        target = max(5.0, min(30.0, float(args["target_c"])))
        changed = await backend.set_schedule_period(args["zones"], str(args["time"]), target)
        return {"changed": changed, "time": str(args["time"]), "target_c": target}
    if name == "clear_schedule_period":
        changed = await backend.clear_schedule_period(args["zones"], str(args["time"]))
        return {"changed": changed, "time": str(args["time"])}
    return {"error": f"unknown tool {name}"}


class ChatIn(BaseModel):
    message: str


@app.get("/api/zones")
async def get_zones():
    return {"zones": [z.as_dict() for z in await backend.list_zones()]}


class SetIn(BaseModel):
    action: str
    zone: str
    value: Any = None


@app.post("/api/set")
async def set_control(body: SetIn):
    """Direct control from the dashboard widgets - one zone, one property."""
    LOG.info("UI /api/set action=%s zone=%s value=%r", body.action, body.zone, body.value)
    zones = [body.zone]
    try:
        if body.action == "target":
            await backend.set_temperature(zones, max(5.0, min(35.0, float(body.value))))
        elif body.action == "mode":
            await backend.set_mode(zones, str(body.value))
        elif body.action == "fan":
            await backend.set_fan(zones, str(body.value))
        elif body.action == "away":
            await backend.set_away(zones, bool(body.value))
        elif body.action == "schedule":
            await backend.set_schedule(zones, bool(body.value))
        elif body.action == "schedule_period":
            await backend.set_schedule_period(
                zones, str(body.value["time"]), float(body.value["target_c"])
            )
        elif body.action == "schedule_clear":
            await backend.clear_schedule_period(zones, str(body.value["time"]))
        else:
            return {"ok": False, "error": f"unknown action {body.action}"}
    except Exception as e:
        LOG.exception("UI /api/set failed action=%s zone=%s", body.action, body.zone)
        return {"ok": False, "error": str(e)}
    return {"ok": True, "zones": [z.as_dict() for z in await backend.list_zones()]}


@app.post("/api/chat")
async def chat(body: ChatIn):
    """Run the Claude tool-use loop for a single user message."""
    messages = [{"role": "user", "content": body.message}]
    actions: list[dict] = []

    for _ in range(8):  # generous cap on tool round-trips
        resp = await claude.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                try:
                    result = await dispatch(block.name, dict(block.input))
                except Exception as e:  # surface tool failures to Claude, not as success
                    result = {"error": str(e)}
                if block.name != "list_zones":
                    actions.append({"tool": block.name, "input": block.input, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    zones = [z.as_dict() for z in await backend.list_zones()]
    return {"reply": reply or "Done.", "actions": actions, "zones": zones}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
