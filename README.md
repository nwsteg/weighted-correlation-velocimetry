# weighted-correlation-velocimetry

Patch-based weighted-correlation velocimetry for estimating convection velocity from image sequences (e.g., high-speed PLIF).

## Install

```bash
pip install -e .
```

## Core API

```python
from wcv import (
    GridSpec,
    EstimationOptions,
    estimate_single_seed_velocity,
    estimate_velocity_map,
    estimate_velocity_map_streaming,
)
```

- `estimate_single_seed_velocity`: one seed region
- `estimate_velocity_map`: full map (materialized correlation path)
- `estimate_velocity_map_streaming`: lower-memory streaming path
- `estimate_velocity_map_hybrid`: backward-compatible alias of the streaming estimator

## Interactive explorer

```bash
python -m wcv.interactive --mode gui
```

or in notebooks:

```python
from wcv import make_plif_interactive_widget
make_plif_interactive_widget()
```

The loader prompts for TIFF source data, frame range, sample rate, and an `extent.txt` file.

## Documentation

Full docs live in `docs/` and can be served with MkDocs:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Key pages:
- `docs/usage.md`
- `docs/implementation.md`
- `docs/theory.md`

## Profiling

```bash
python scripts/profile_velocity_map.py --shape 30,128,128 --bin-sizes 4,8 --mode all
```
