"""Thin async backend over the Heatmiser NeoHub local API.

Exposes a small, tool-friendly surface (list zones, set temp, hold, away) that the
Claude tool loop in app.py maps natural language onto. A MockBackend implements the
same interface so the whole dashboard runs without a hub.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


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

    def as_dict(self) -> dict:
        return asdict(self)


def _match_zones(requested: list[str] | str, zones: list[Zone]) -> list[Zone]:
    """Resolve caller-supplied zone names to real zones.

    Accepts the literal "all" (or "*") to mean every zone, otherwise matches
    case-insensitively, allowing partial names ("lounge" -> "Lounge").
    """
    if requested in ("all", "*") or requested == ["all"]:
        return list(zones)
    if isinstance(requested, str):
        requested = [requested]
    wanted = [r.strip().lower() for r in requested]
    out: list[Zone] = []
    for z in zones:
        name = z.name.lower()
        if any(w == name or w in name for w in wanted):
            out.append(z)
    return out


class BaseBackend:
    async def list_zones(self) -> list[Zone]:
        raise NotImplementedError

    async def set_temperature(self, zones, target: float) -> list[str]:
        raise NotImplementedError

    async def hold_temperature(self, zones, target: float, hours: int, minutes: int) -> list[str]:
        raise NotImplementedError

    async def set_away(self, zones, enable: bool) -> list[str]:
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
        return Zone(
            name=t.name,
            current=float(getattr(t, "temperature", 0) or 0),
            target=float(getattr(t, "target_temperature", 0) or 0),
            heating=bool(getattr(t, "heat_on", False)),
            hold=bool(getattr(t, "hold_on", False)),
            standby=bool(getattr(t, "standby", False)),
            away=bool(getattr(t, "away", False)),
            active=bool(getattr(t, "heat_on", False) or getattr(t, "cool_on", False)),
            mode=mode,
            fan=str(getattr(t, "fan_speed", "") or ""),
            fan_auto=str(getattr(t, "fan_control", "")).lower().startswith("auto"),
            schedule="Manual" if not profile else f"Profile {profile}",
        )

    async def list_zones(self) -> list[Zone]:
        return [self._to_zone(t) for t in await self._thermostats()]

    async def _resolve(self, zones):
        thermos = await self._thermostats()
        as_zones = [self._to_zone(t) for t in thermos]
        wanted = {z.name for z in _match_zones(zones, as_zones)}
        return [t for t in thermos if t.name in wanted]

    async def set_temperature(self, zones, target: float) -> list[str]:
        touched = await self._resolve(zones)
        for t in touched:
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
