"""Thin async backend over the Heatmiser NeoHub local API.

Exposes a small, tool-friendly surface (list zones, set temp, hold, away) that the
Claude tool loop in app.py maps natural language onto. A MockBackend implements the
same interface so the whole dashboard runs without a hub.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from dataclasses import asdict, dataclass, field

log = logging.getLogger("heatmiser.neohub")


@dataclass
class Zone:
    name: str
    current: float          # measured temperature, degrees C
    target: float           # set/target temperature, degrees C
    heating: bool           # is the zone currently calling for heat
    hold: bool = False       # a temporary hold is active
    standby: bool = False    # per-zone standby / frost-protection mode (shown as Off)
    away: bool = False       # hub-wide away mode
    active: bool = False     # currently calling for heat or cool
    mode: str = "Heating"    # Heating / Cooling / Vent
    fan: str = ""            # fan speed (Off/Low/Med/High), blank if not applicable
    fan_auto: bool = False   # fan speed chosen automatically
    schedule: str = "Manual"  # active profile name, or "Manual" when none
    schedule_on: bool = False  # zone is following its program (not a manual setpoint)
    sched_day: str = ""      # day/label the shown periods belong to ("" means today)
    periods: list = field(default_factory=list)  # [{time, value}] for that day

    def as_dict(self) -> dict:
        return asdict(self)


def _match_zones(requested: list[str] | str, zones: list[Zone]) -> list[Zone]:
    """Resolve caller-supplied zone names to real zones.

    Accepts the literal "all" (or "*") to mean every zone. Otherwise matches each
    requested name case-insensitively, preferring an exact match so "bedroom" does
    not also select "Master Bedroom"; only falls back to prefix/substring matches
    when there is no exact hit.
    """
    if requested in ("all", "*") or requested == ["all"]:
        return list(zones)
    if isinstance(requested, str):
        requested = [requested]
    wanted = [r.strip().lower() for r in requested]
    out: list[Zone] = []
    seen: set[str] = set()
    for w in wanted:
        exact = [z for z in zones if z.name.lower() == w]
        if exact:
            matches = exact
        else:
            starts = [z for z in zones if z.name.lower().startswith(w)]
            matches = starts if starts else [z for z in zones if w in z.name.lower()]
        for z in matches:
            if z.name not in seen:
                seen.add(z.name)
                out.append(z)
    return out


def _fmt_setpoint(value) -> str:
    """Format a schedule setpoint; large sentinel values (>=30) mean 'off'."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return "Off"
    return "Off" if c >= 30 else f"{c:g}°C"


def _norm_time(value: str) -> str:
    """Normalise a period time to zero-padded 'HH:MM'. Raises on garbage."""
    s = str(value).strip()
    if ":" not in s:
        raise ValueError(f"invalid time {value!r}; use HH:MM")
    hh, mm = s.split(":", 1)
    h, m = int(hh), int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"time out of range {value!r}; use 00:00-23:59")
    return f"{h:02d}:{m:02d}"


def _num(value: float) -> float | int:
    """Whole numbers as int (24.0 -> 24), otherwise keep the float (21.5)."""
    f = float(value)
    return int(f) if f == int(f) else f


def _mins(hhmm: str) -> int:
    """Minutes since midnight for 'HH:MM'; unparseable/24:00 sort to the end."""
    try:
        h, m = str(hhmm).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return 24 * 60


# Setpoint written to mark a schedule period "off" (no heating/cooling at that time).
# We use 35 - a value the hub accepts and stores as-is; being >=30 it renders as "Off"
# (see _fmt_setpoint), and for cooling it is high enough that the system never engages.
SCHED_OFF = 35


# The hub's SET_FAN_SPEED accepts only these tokens (uppercase, "MED" not "Medium").
_FAN_TOKENS = {
    "auto": "AUTO", "high": "HIGH", "medium": "MED",
    "med": "MED", "low": "LOW", "off": "OFF",
}


class BaseBackend:
    async def list_zones(self) -> list[Zone]:
        raise NotImplementedError

    async def set_temperature(self, zones, target: float) -> list[str]:
        raise NotImplementedError

    async def hold_temperature(self, zones, target: float, hours: int, minutes: int) -> list[str]:
        raise NotImplementedError

    async def set_away(self, zones, enable: bool) -> list[str]:
        raise NotImplementedError

    async def set_fan(self, zones, speed: str) -> list[str]:
        raise NotImplementedError

    async def set_mode(self, zones, mode: str) -> list[str]:
        raise NotImplementedError

    async def set_schedule(self, zones, enable: bool) -> list[str]:
        raise NotImplementedError

    async def set_schedule_period(
        self, zones, time: str, target_c: float | None = None, off: bool = False
    ) -> list[str]:
        """Set (or add) the schedule period at ``time`` to ``target_c`` (or off), keeping the rest."""
        raise NotImplementedError

    async def clear_schedule_period(self, zones, time: str) -> list[str]:
        """Remove the schedule period at ``time``, keeping the rest."""
        raise NotImplementedError

    async def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Mock backend: in-memory zones, no hardware required.
# --------------------------------------------------------------------------- #
class MockBackend(BaseBackend):
    def __init__(self) -> None:
        self._zones = [
            Zone("Lounge", 19.5, 21.0, heating=True),
            Zone("Kitchen", 20.0, 20.0, heating=False),
            Zone("Bedroom", 17.5, 18.0, heating=True),
            Zone("Bathroom", 21.0, 22.0, heating=True),
            Zone("Office", 18.0, 16.0, heating=False),
        ]
        # One "every day" schedule per zone: {name: {"on": bool, "periods": [[time, temp]]}}.
        # Times are sorted; a period sets the target from that time until the next one.
        self._sched: dict[str, dict] = {
            "Lounge": {"on": True, "periods": [["07:00", 21], ["09:00", 18], ["17:30", 21], ["22:30", 16]]},
            "Kitchen": {"on": True, "periods": [["07:00", 20], ["22:00", 16]]},
            "Bedroom": {"on": True, "periods": [["06:30", 20], ["22:00", 17]]},
            "Bathroom": {"on": True, "periods": [["07:00", 22], ["23:00", 18]]},
            "Office": {"on": False, "periods": [["08:00", 20], ["18:00", 16]]},
        }

    def _apply_schedule(self, z: Zone) -> None:
        """Populate a zone's schedule display fields from its stored program."""
        s = self._sched.get(z.name, {"on": False, "periods": []})
        z.schedule = "Every day"
        z.schedule_on = bool(s["on"])
        z.sched_day = "every day"
        z.periods = [
            {
                "slot": "",
                "time": t,
                "value": _fmt_setpoint(temp),
                "temp": None if float(temp) >= 30 else float(temp),
            }
            for t, temp in sorted(s["periods"])
        ]

    async def list_zones(self) -> list[Zone]:
        # Nudge current temps toward target so the dashboard feels alive.
        for z in self._zones:
            if z.standby:
                z.heating = z.current < 7.0
            elif z.current < z.target - 0.1:
                z.current = round(z.current + 0.2, 1)
                z.heating = True
            elif z.current > z.target + 0.1:
                z.current = round(z.current - 0.2, 1)
                z.heating = False
            else:
                z.heating = False
            z.active = z.heating
            self._apply_schedule(z)
        return list(self._zones)

    async def set_temperature(self, zones, target: float) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            z.target = float(target)
            z.standby = False
            z.hold = False
        return [z.name for z in touched]

    async def hold_temperature(self, zones, target: float, hours: int, minutes: int) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            z.target = float(target)
            z.hold = True
            z.standby = False
        return [z.name for z in touched]

    async def set_away(self, zones, enable: bool) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            z.standby = bool(enable)
            if enable:
                z.hold = False
        return [z.name for z in touched]

    async def set_fan(self, zones, speed: str) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            if speed.lower() == "auto":
                z.fan_auto, z.fan = True, ""
            else:
                z.fan_auto, z.fan = False, speed.title()
        return [z.name for z in touched]

    async def set_mode(self, zones, mode: str) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            z.mode = mode.title()
        return [z.name for z in touched]

    async def set_schedule(self, zones, enable: bool) -> list[str]:
        touched = _match_zones(zones, self._zones)
        for z in touched:
            self._sched.setdefault(z.name, {"on": False, "periods": []})["on"] = bool(enable)
        return [z.name for z in touched]

    async def set_schedule_period(
        self, zones, time: str, target_c: float | None = None, off: bool = False
    ) -> list[str]:
        time = _norm_time(time)
        target = SCHED_OFF if off else _num(max(5.0, min(30.0, float(target_c))))
        touched = _match_zones(zones, self._zones)
        for z in touched:
            s = self._sched.setdefault(z.name, {"on": True, "periods": []})
            for period in s["periods"]:
                if period[0] == time:  # update an existing period at this time
                    period[1] = target
                    break
            else:
                s["periods"].append([time, target])
            s["periods"].sort()
        return [z.name for z in touched]

    async def clear_schedule_period(self, zones, time: str) -> list[str]:
        time = _norm_time(time)
        touched = _match_zones(zones, self._zones)
        for z in touched:
            s = self._sched.get(z.name)
            if s:
                s["periods"] = [p for p in s["periods"] if p[0] != time]
        return [z.name for z in touched]


# --------------------------------------------------------------------------- #
# Real backend: wraps the neohubapi library.
# --------------------------------------------------------------------------- #
class NeoHubBackend(BaseBackend):
    def __init__(self, host: str, port: int, token: str | None) -> None:
        from neohubapi.neohub import NeoHub

        kwargs = {"host": host, "port": port}
        if token:
            kwargs["token"] = token
        self._hub = NeoHub(**kwargs)
        self._sched_cache: dict = {}
        self._sched_at: float = 0.0

    async def _thermostats(self):
        # Gen 2/3 hubs talk over a WebSocket that may need an explicit connect.
        connect = getattr(self._hub, "connect", None)
        if connect and not getattr(self, "_connected", False):
            result = connect()
            if hasattr(result, "__await__"):
                await result
            self._connected = True
        # neohubapi 3.x: get_devices_data() -> {"neo_devices": [NeoStat, ...]}
        data = await self._hub.get_devices_data()
        return data["neo_devices"]

    @staticmethod
    def _to_zone(t) -> Zone:
        hc = str(getattr(t, "hc_mode", "") or "").upper()
        mode = {"COOLING": "Cooling", "HEATING": "Heating", "VENT": "Vent"}.get(
            hc, hc.title() or "Heating"
        )
        profile = getattr(t, "active_profile", 0) or 0
        # In cooling mode the setpoint is cool_temp; in heating it is target_temperature.
        raw_target = (
            getattr(t, "cool_temp", None)
            if mode == "Cooling"
            else getattr(t, "target_temperature", None)
        )
        try:
            target = float(raw_target)
        except (TypeError, ValueError):
            target = 0.0
        return Zone(
            name=t.name,
            current=float(getattr(t, "temperature", 0) or 0),
            target=target,
            heating=bool(getattr(t, "heat_on", False)),
            hold=bool(getattr(t, "hold_on", False)),
            standby=bool(getattr(t, "standby", False)),
            away=bool(getattr(t, "away", False)),
            active=bool(getattr(t, "heat_on", False) or getattr(t, "cool_on", False)),
            mode=mode,
            fan=str(getattr(t, "fan_speed", "") or ""),
            fan_auto=str(getattr(t, "fan_control", "")).lower().startswith("auto"),
            schedule="Manual" if not profile else f"Profile {profile}",
            schedule_on=not bool(getattr(t, "manual_off", True)),
        )

    _DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    _PERIODS = ["wake", "leave", "return", "sleep"]

    async def list_zones(self) -> list[Zone]:
        thermos = await self._thermostats()
        sched = await self._today_schedules(thermos)
        zones = []
        for t in thermos:
            z = self._to_zone(t)
            info = sched.get(t.name, {})
            z.sched_day = info.get("day", "")
            z.periods = info.get("periods", [])
            zones.append(z)
        return zones

    async def _today_schedules(self, thermos) -> dict:
        """Each zone's schedule for today, falling back to the next programmed day.

        Cached for 5 minutes - profiles change rarely and reading them is slow.
        """
        now = time.time()
        if self._sched_cache and now - self._sched_at < 300:
            return self._sched_cache
        try:
            fmt = str(getattr(await self._hub.get_system(), "FORMAT", ""))
        except Exception:
            fmt = ""
        same_every_day = fmt.endswith("ONE")  # ScheduleFormat.ONE = same schedule daily
        today_idx = datetime.datetime.now().weekday()  # Monday = 0
        order = self._DAYS[today_idx:] + self._DAYS[:today_idx]
        result = {}
        for t in thermos:
            idx = 2 if str(getattr(t, "hc_mode", "")).upper() == "COOLING" else 1
            try:
                prof = await self._hub.get_profile_0(t.name)
                daymap = prof.profiles[0]
            except Exception:
                daymap = None
            candidates = self._DAYS if same_every_day else order
            chosen, periods = None, []
            for dname in candidates:
                day = getattr(daymap, dname, None) if daymap is not None else None
                if day is None:
                    continue
                ps = []
                for per in self._PERIODS:
                    e = getattr(day, per, None)
                    if e and e[0] and e[0] != "24:00":
                        try:
                            temp = float(e[idx])
                        except (TypeError, ValueError):
                            temp = None
                        ps.append({
                            "slot": per,
                            "time": e[0],
                            "value": _fmt_setpoint(e[idx]),
                            "temp": None if (temp is None or temp >= 30) else temp,
                        })
                if ps:
                    ps.sort(key=lambda p: p["time"])
                    chosen, periods = dname, ps
                    break
            if same_every_day:
                label = "every day" if periods else ""
            else:
                label = "" if chosen == self._DAYS[today_idx] else (chosen.capitalize() if chosen else "")
            result[t.name] = {"day": label, "periods": periods}
        self._sched_cache = result
        self._sched_at = now
        return result

    async def _resolve(self, zones):
        thermos = await self._thermostats()
        as_zones = [self._to_zone(t) for t in thermos]
        wanted = {z.name for z in _match_zones(zones, as_zones)}
        return [t for t in thermos if t.name in wanted]

    # Control-relevant fields, logged before/after a write to diagnose setpoints
    # that don't "take" (the classic being a write ignored while a zone is off).
    _SNAP_FIELDS = (
        "hc_mode", "standby", "away", "manual_off", "hold_on",
        "cool_on", "heat_on", "target_temperature", "cool_temp",
    )

    @classmethod
    def _snapshot(cls, t) -> str:
        return " ".join(f"{k}={getattr(t, k, '?')!r}" for k in cls._SNAP_FIELDS)

    async def set_temperature(self, zones, target: float) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
            mode = str(getattr(t, "hc_mode", "")).upper()
            log.info("set_temperature %s -> %s | before: %s", t.name, target, self._snapshot(t))
            # Choosing a target means "run this zone at this temperature", so an
            # off/standby zone must come out of frost first: the hub ignores
            # setpoint writes while a zone is in standby, so the value would snap
            # straight back to "off" on the next refresh.
            if bool(getattr(t, "standby", False)):
                log.info("set_temperature %s: clearing standby before writing setpoint", t.name)
                await t.set_frost(False)
            if mode == "COOLING":
                reply = await t.set_cool_temp(float(target))
            else:
                reply = await t.set_target_temperature(float(target))
            log.info("set_temperature %s: hub reply %r", t.name, reply)
            if log.isEnabledFor(logging.DEBUG):
                after = await self._resolve([t.name])
                if after:
                    log.debug("set_temperature %s | after: %s", t.name, self._snapshot(after[0]))
        return [t.name for t in touched]

    async def hold_temperature(self, zones, target: float, hours: int, minutes: int) -> list[str]:
        touched = await self._resolve(zones)
        if touched:
            log.info("hold_temperature %s -> %s for %dh%02dm", [t.name for t in touched], target, hours, minutes)
            # neohubapi 3.x: hold is a hub-level call taking a list of NeoStats.
            reply = await self._hub.set_hold(float(target), int(hours), int(minutes), touched)
            log.info("hold_temperature: hub reply %r", reply)
        return [t.name for t in touched]

    async def set_away(self, zones, enable: bool) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
            log.info("set_away %s enable=%s", t.name, enable)
            reply = await t.set_frost(bool(enable))
            log.info("set_away %s: hub reply %r", t.name, reply)
        return [t.name for t in touched]

    async def set_fan(self, zones, speed: str) -> list[str]:
        token = _FAN_TOKENS.get(str(speed).strip().lower())
        if token is None:
            raise ValueError(f"invalid fan speed {speed!r}; use Auto/High/Medium/Low/Off")
        touched = await self._resolve(zones)
        for t in touched:
            log.info("set_fan %s -> %s (%s)", t.name, speed, token)
            reply = await t.set_fan_speed(token)
            err = getattr(reply, "error", None)
            if err:
                log.warning("set_fan %s: hub error %s", t.name, err)
                raise ValueError(str(err))
        return [t.name for t in touched]

    async def set_mode(self, zones, mode: str) -> list[str]:
        from neohubapi.enums import HCMode

        try:
            hc = HCMode(str(mode).strip().upper())
        except ValueError:
            raise ValueError(f"invalid mode {mode!r}; use Heating/Cooling/Vent/Auto")
        touched = await self._resolve(zones)
        for t in touched:
            log.info("set_mode %s -> %s", t.name, hc)
            reply = await t.set_hc_mode(hc)
            err = getattr(reply, "error", None)
            if err:
                log.warning("set_mode %s: hub error %s", t.name, err)
                raise ValueError(str(err))
        return [t.name for t in touched]

    async def set_schedule(self, zones, enable: bool) -> list[str]:
        touched = await self._resolve(zones)
        if touched:
            log.info("set_schedule %s enable=%s", [t.name for t in touched], enable)
            # set_manual(True) = manual (schedule off); False = follow schedule.
            reply = await self._hub.set_manual(not enable, touched)
            log.info("set_schedule: hub reply %r", reply)
            self._sched_cache = {}
        return [t.name for t in touched]

    # --- schedule editing (comfort levels) ---------------------------------- #
    #
    # Each zone's own schedule is "profile 0": day -> {wake,leave,return,sleep},
    # every slot a [time, heat, cool, flag] comfort level ("24:00" = unused).
    # neohubapi can read it (get_profile_0) but cannot write it, so we send the
    # raw SET_COMFORT_LEVELS command. Two things learned from the hub:
    #  1. WebSocketClient serialises the command with str(), so a Python bool
    #     renders as True/False and the hub rejects it as "Invalid Json" - every
    #     value must be a str or number, hence flags are written as 1/0.
    #  2. It is a full replace, so we read-modify-write and resend all the days.

    @staticmethod
    def _clean_slot(e) -> list:
        """A comfort level as [time, heat, cool, flag] with only str/number values."""
        vals = list(e) if e is not None else []
        vals += ["24:00", 19, 0, 0][len(vals):]

        def n(x, default):
            try:
                return _num(x)
            except (TypeError, ValueError):
                return default

        return [str(vals[0] or "24:00"), n(vals[1], 19), n(vals[2], 0), 1 if vals[3] else 0]

    def _day_slots(self, daymap, dname) -> dict:
        day = getattr(daymap, dname, None)
        return {p: self._clean_slot(getattr(day, p, None)) for p in self._PERIODS}

    @classmethod
    def _base_day(cls, daymap, present: list[str]) -> str:
        """Canonical day to edit - the first present day that has a real period."""
        for d in present:
            day = getattr(daymap, d, None)
            if day and any(
                getattr(day, p, ["24:00"])[0] not in ("24:00", "", None) for p in cls._PERIODS
            ):
                return d
        return present[0]

    async def _edit_schedule(self, zones, time: str, target_c, clear: bool, off: bool = False) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
            idx = 2 if str(getattr(t, "hc_mode", "")).upper() == "COOLING" else 1
            prof = await self._hub.get_profile_0(t.name)
            daymap = prof.profiles[0]
            present = [d for d in self._DAYS if getattr(daymap, d, None) is not None]
            if not present:
                continue
            # Edit one canonical day (the one the dashboard shows) and write it to
            # every present day. This app treats the schedule as a single "every
            # day" program, so keeping the days identical is both correct and
            # avoids the drift that per-day writes caused when days disagreed.
            slots = self._day_slots(daymap, self._base_day(daymap, present))
            # Pick the slot: exact time match, else (when setting) a free 24:00
            # slot, else the nearest existing one.
            slot = next((p for p in self._PERIODS if slots[p][0] == time), None)
            if slot is None and not clear:
                slot = next((p for p in self._PERIODS if slots[p][0] == "24:00"), None)
                if slot is None:
                    slot = min(self._PERIODS, key=lambda p: abs(_mins(slots[p][0]) - _mins(time)))
            if slot is None:
                continue  # clearing a time that isn't programmed - nothing to do
            if clear:
                slots[slot] = ["24:00", slots[slot][1], slots[slot][2], 0]
            else:
                s = slots[slot]
                s[0], s[idx], s[3] = time, (SCHED_OFF if off else _num(target_c)), 1
                slots[slot] = s
            info = {d: {p: list(slots[p]) for p in self._PERIODS} for d in present}
            log.info(
                "edit_schedule %s time=%s slot=%s %s: %s",
                t.name, time, slot,
                "clear" if clear else ("off" if off else f"target={target_c}"),
                slots[slot],
            )
            reply = await self._hub._send({"SET_COMFORT_LEVELS": [info, [t.name]]})
            log.info("edit_schedule %s: hub reply %r", t.name, reply)
        self._sched_cache = {}
        return [t.name for t in touched]

    async def set_schedule_period(
        self, zones, time: str, target_c: float | None = None, off: bool = False
    ) -> list[str]:
        target = None if off else max(5.0, min(30.0, float(target_c)))
        return await self._edit_schedule(zones, _norm_time(time), target, clear=False, off=off)

    async def clear_schedule_period(self, zones, time: str) -> list[str]:
        return await self._edit_schedule(zones, _norm_time(time), None, clear=True)

    async def close(self) -> None:
        close = getattr(self._hub, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result


def make_backend() -> BaseBackend:
    if os.getenv("NEOHUB_MOCK", "0") == "1":
        return MockBackend()
    host = os.environ["NEOHUB_HOST"]
    port = int(os.getenv("NEOHUB_PORT", "4242"))
    token = os.getenv("NEOHUB_TOKEN") or None
    return NeoHubBackend(host, port, token)
