"""Thin async backend over the Heatmiser NeoHub local API.

Exposes a small, tool-friendly surface (list zones, set temp, hold, away) that the
Claude tool loop in app.py maps natural language onto. A MockBackend implements the
same interface so the whole dashboard runs without a hub.
"""

from __future__ import annotations

import datetime
import os
import time
from dataclasses import asdict, dataclass, field


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
        return [z.name for z in _match_zones(zones, self._zones)]


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
                        ps.append({"time": e[0], "value": _fmt_setpoint(e[idx])})
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

    async def set_temperature(self, zones, target: float) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
            if str(getattr(t, "hc_mode", "")).upper() == "COOLING":
                await t.set_cool_temp(float(target))
            else:
                await t.set_target_temperature(float(target))
        return [t.name for t in touched]

    async def hold_temperature(self, zones, target: float, hours: int, minutes: int) -> list[str]:
        touched = await self._resolve(zones)
        if touched:
            # neohubapi 3.x: hold is a hub-level call taking a list of NeoStats.
            await self._hub.set_hold(float(target), int(hours), int(minutes), touched)
        return [t.name for t in touched]

    async def set_away(self, zones, enable: bool) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
            if enable:
                await t.set_frost(True)
            else:
                await t.set_frost(False)
        return [t.name for t in touched]

    async def set_fan(self, zones, speed: str) -> list[str]:
        token = _FAN_TOKENS.get(str(speed).strip().lower())
        if token is None:
            raise ValueError(f"invalid fan speed {speed!r}; use Auto/High/Medium/Low/Off")
        touched = await self._resolve(zones)
        for t in touched:
            reply = await t.set_fan_speed(token)
            err = getattr(reply, "error", None)
            if err:
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
            reply = await t.set_hc_mode(hc)
            err = getattr(reply, "error", None)
            if err:
                raise ValueError(str(err))
        return [t.name for t in touched]

    async def set_schedule(self, zones, enable: bool) -> list[str]:
        touched = await self._resolve(zones)
        if touched:
            # set_manual(True) = manual (schedule off); False = follow schedule.
            await self._hub.set_manual(not enable, touched)
            self._sched_cache = {}
        return [t.name for t in touched]

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
