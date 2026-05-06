# Contributing

## Setup

```bash
git clone https://github.com/cgodoyle/losneomrade.git
cd losneomrade
pip install -e ".[dev]"
```

## Running tests

```bash
# Fast offline tests (~4 seconds)
pytest tests/ -m "not slow"

# All tests including network (hits NVE/Høydedata APIs)
pytest tests/

# Single test
pytest tests/test_retrogression.py::TestRetrogressionOffline::test_run_retrogression_basic
```

## Code quality

```bash
# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
ty check src/
```

## Conventions

- **Python ≥ 3.12** with modern type annotations (`list[...]`, `X | None`)
- **Google-style docstrings** for all public functions
- **Trailing commas** in function signatures (ruff format handles expansion)
- **Logging** instead of `print()` — use `logger = logging.getLogger(__name__)`
- **CRS**: Always EPSG:25833
- **Bounds order**: All modules use standard GIS `(xmin, ymin, xmax, ymax)`.

## Project structure

```
src/losneomrade/
├── config.py                # Settings dataclass
├── hoydedata.py             # DEM fetching from Høydedata.no
├── masks.py                 # MSML/AR5 mask fetching from NVE
├── retrogression.py         # Step-wise retrogression
├── terrain_criteria.py      # Pixel-based slope verification
├── profile_retrogression.py # Profile-based retrogression
├── utils.py                 # Shared utilities
└── types.py                 # Type aliases
```

## Commit messages

- Concise, imperative mood, one line when possible
- Breaking changes should be noted in the commit body

## Architecture decisions

- **Masks are external**: The package doesn't fetch masks automatically.
  Users call `masks.get_msml_mask()` and pass the result to the analysis functions.
- **DEM fetching is optional**: All functions accept `custom_raster` to work offline.
- **Config is a singleton**: `from losneomrade.config import settings` gives you the global instance.
