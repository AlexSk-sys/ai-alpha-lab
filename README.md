# ai-alpha-lab

A deterministic momentum agent and a **risk-envelope study** on 520 Binance USDT perpetuals:
15,160 monthly observations covering January 2020 – August 2026 (80 months).

## What it answers

- Is **100,000 ₽ → $1M** achievable? 250× at 25 ₽/$ happened, but exactly once and it cannot be
  selected in advance. 1000× at 100 ₽/$ never occurs on this panel.
- The decision function is held **identical** across every track — same bars, same seed, same
  signal. The tracks diverge only in the `RiskOverlay` the runner applies, so the experiment
  isolates the risk envelope instead of mixing strategy and risk into one result.

The full result and the method are in [`CEILING_REPORT.md`](CEILING_REPORT.md).

## Files

- `agent.py` — deterministic momentum agent (pure seed → parameters mapping)
- `broker.py` — fill/fee/position accounting
- `ceiling.py`, `ceiling_stats.py`, `ceiling_study.py` — the risk-envelope runs
- `CEILING_REPORT.md` — result and caveats

## Reproduce

```powershell
python -m pytest -q          # the deterministic agent/broker/risk tests
python ceiling_study.py      # the risk-envelope run
```

The panel construction is deterministic; the report records the exact window and counts.
`RESULTS.md` holds the earlier two-track results.
