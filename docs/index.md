# losneomrade

Calculate maximum retrogression of landslides in quick-clay slopes, based on
**NVE Kvikkleireveileder 1/2019**.

The core assumption: a slope of **1:15** is the conservative limit for landslide retrogression.

## What does this package do?

Given a source area (where a landslide starts) and a Digital Elevation Model (DEM),
this package calculates how far back the landslide could propagate. It provides
three different methods for this:

| Method | Module | Best for |
|--------|--------|----------|
| **Terrain Criteria** | `terrain_criteria` | Fast pixel-by-pixel slope check over a DEM |
| **Retrogression** | `retrogression` | Step-wise propagation from an initial release area |
| **Profile Retrogression** | `profile_retrogression` | Cross-section analysis along terrain profiles |

## Quick links

- [Quick Start](quickstart.md) — Get running in 5 minutes
- [Theory](theory.md) — How each method works
- [API Overview](api.md) — Function signatures and parameters
- [Configuration](configuration.md) — Customize endpoints and settings
- [Contributing](contributing.md) — How to help improve this package

## Breaking changes in this update

This modernization pass intentionally includes a few breaking API changes:

- **Bounds order is now standard GIS order**: `(xmin, ymin, xmax, ymax)` in all modules.
- **MSML clipping is now explicit**: use `mask=...` instead of the old `clip_to_msml` flag.
- **Mask utilities moved**: MSML- and AR5-related helpers now live in `losneomrade.masks`.
- **MSML source changed**: MSML is now fetched from the NVE MapServer endpoint.

Check the [Quick Start](quickstart.md) and [API Overview](api.md) pages before
updating older scripts.
