# Heatmiser + Claude

A local web dashboard that controls your **Heatmiser Neo** heating with natural language,
powered by the **Claude API**.
Say "make the lounge warmer" or "set everywhere to 20 for two hours" and Claude translates
it into structured commands against your **NeoHub**.

## How it works

```
Browser (dashboard) ──HTTP──> FastAPI (app.py)
                                 ├── Claude API (tool use)  ── interprets language
                                 └── neohub.py ── NeoHub local API (or mock)
```

Claude is given four tools - `list_zones`, `set_temperature`, `hold_temperature`,
`set_away` - and decides which to call. The backend executes them against the hub and
returns fresh zone state, which the dashboard renders as live cards.

## Quick start (no hub needed)

```bash
cd heatmiser-claude
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp conf/.env.example conf/.env
# edit conf/.env: set NEOHUB_HOST etc. Leave NEOHUB_MOCK=1 to use fake zones.
# (ANTHROPIC_API_KEY comes from your environment; the NeoHub token is NEOHUB_TOKEN in conf/.env.)

python app.py          # serves on http://127.0.0.1:8765 by default
```

Open <http://127.0.0.1:8765>. The dashboard starts with five mock zones you can drive
entirely by chat. Change the port with `PORT=... python app.py` (or `HOST=0.0.0.0` to
expose it on your LAN).

## Connecting your real NeoHub

In `.env`:

- Set `NEOHUB_MOCK=0`.
- Set `NEOHUB_HOST` to the hub's LAN IP (find it in your router, or the neoApp under
  *Settings → System → About*).
- **Legacy hub:** `NEOHUB_PORT=4242`, leave `NEOHUB_TOKEN` blank.
- **Gen 2 / Mini hub:** `NEOHUB_PORT=4243` and create an API token in the neoApp
  (*Settings → API/Connections*), then set `NEOHUB_TOKEN`.

The hub speaks a local API, so this runs entirely on your LAN - no cloud round-trip for the
heating commands (only the language understanding calls Claude).

## Notes

- `neohubapi` method names have shifted slightly across versions. `neohub.py` targets the
  2.x `get_live_data()` / `NeoStat` surface; if you hit an attribute error, check
  `pip show neohubapi` and adjust `NeoHubBackend`.
- Safety clamp: targets are limited to 5-35 °C in `app.py` (>=30 reads as "off" for cooling).
- Model is set by `CLAUDE_MODEL` (default `claude-sonnet-4-6`) - a good latency/cost fit
  for this short tool loop.
