# Maths slides: evidence

Every number on `../eisengrau-maths-slides.pdf` comes from `deck_numbers.json`, computed on slide 5's sample (Elexon imbalance price, 1 Sep–23 Oct 2026, 28-day warm-up from 4 Aug) with origin/main's registered algorithms and adapters. The run reproduces every figure in `docs/gridpulse/MODELLING_NOTES.md`, so it uses the same data as the Tiger dev runs.

```bash
python fetch_prices_parallel.py <data_dir> 2026-08-04 2026-10-23
PYTHONPATH=<repo>/gridpulse/backend/src uv run --no-project --with pandas --with numpy --with tzdata \
  python deck_numbers.py <data_dir> deck_numbers.json
```
