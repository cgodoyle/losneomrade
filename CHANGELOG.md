# Changelog

This project follows **Semantic Versioning** and uses **Conventional Commits**
for changelog-friendly history from the `v2.0.0` release line onward.

## [Unreleased]

### Changed

- Future entries should be grouped from Conventional Commits (`feat`, `fix`,
  `docs`, `refactor`, `test`, `ci`, `chore`).

## [2.0.0] - 2026-05-07

This release covers the modernization work since **`v1.1.3`** (baseline commit
`610b5a8489143301bbce8d25ffce152ecc4a12bc`).

### Added

- Integrated the `profile_retrogression` workflow into the package.
- Added a pytest-based test suite with fast offline coverage and optional slow
  network tests.
- Added Zensical documentation with quick start, theory, configuration, API
  overview, and contributor guidance.
- Added dedicated `masks` helpers and documentation for the new NVE MSML
  MapServer source.
- Added GitHub Actions workflows for CI, docs publishing, and tag-based releases.
- Added an initial changelog and release process for the new major-version line.

### Changed

- Modernized the project for Python 3.12 and updated dependency constraints.
- Replaced `print()`-style progress reporting with standard library logging.
- Extracted Høydedata access into a dedicated module with structured config.
- Modernized type annotations, docstrings, and formatting across the package.
- Normalized all `bounds` handling to standard GIS order:
  `(xmin, ymin, xmax, ymax)`.
- Standardized contributor guidance around Conventional Commits and Semantic
  Versioning for future releases.
- Updated GitHub Actions workflows to use `uv` for faster install/build steps.

### Removed

- Removed the old implicit MSML clipping flow based on `clip_to_msml`.

### Breaking Changes

- `bounds` now always use standard GIS order `(xmin, ymin, xmax, ymax)`.
- Analysis functions now take an explicit `mask=` GeoDataFrame instead of the
  old `clip_to_msml` boolean flow.
- Mask-related helpers now live in `losneomrade.masks`.
- MSML is now fetched from the NVE MapServer endpoint used by the new masks
  module.

### Migration Notes

- Treat `v1.1.3` as the baseline before the modernization and API cleanup.
- Existing callers should update `bounds` ordering and replace
  `clip_to_msml=True` with an explicit mask fetched from `losneomrade.masks`.
- The package version for this release line should be tagged as **`v2.0.0`**.

## [1.1.3]

- Baseline version before the Python 3.12 modernization and breaking API
  cleanup that lead to `v2.0.0`.
