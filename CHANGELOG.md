# Changelog

All notable changes to Code Atlas will be documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Read compiled BEAM debug data for high-fidelity Elixir callsites, with
  compiler-backed `.exs` and `.eex` analysis and the existing xref/static
  fallbacks.
- Resolve in-repository Erlang remote calls, imported calls, and function
  references.
- Cache semantic analysis under `.git/code-atlas/`, including safe keys for
  clean and dirty worktrees and analyzer versions.
- Filter retained callsites from both `render` and `inspect` with repeatable
  `--calls-in`, `--calls-from`, and `--calls-to` path prefixes. JSON reports now
  include pre-filter counts and the normalized filters.
- Add source-to-target direction legends to SVG and PDF output.

### Changed

- Compress interactive HTML data losslessly and reconstruct hierarchical spline
  samples in the browser, substantially reducing self-contained viewer size.
- Let `--fixed-call-opacity` optionally accept its own opacity value while
  preserving the separate `--call-opacity` behavior when no value is supplied.
- Keep ordinary call graphs at full pigment strength and reserve exposure
  reduction for genuinely dense graphs.
- Render architectural PDF borders as translucent pencil passes using multiply
  blending in light mode and screen blending in dark mode.
- Skip the expensive Git-history walk unless `commits` is the selected treemap
  metric.

### Fixed

- Keep long viewer tooltip paths readable and prevent file-search interactions
  from leaking into map gestures.

## [0.1.0] - 2026-08-16

### Added

- Exact weighted repository treemaps with line-addressed file parcels.
- Individual, unaggregated cross-file callsite splines with hierarchical edge
  bundling and direction color.
- Elixir, Rust, JavaScript, TypeScript, and Gleam analysis, with explicit
  coverage reporting and conservative fallbacks where semantic indexes are not
  available.
- Architectural light and dark themes for PNG, SVG, hybrid PDF, and a
  self-contained interactive HTML viewer.
- Software and WebGPU optical-density renderers with density-aware pigment.
- High-DPI, resumable, bounded-memory PDF call-layer rendering.

[Unreleased]: https://github.com/tessi/code-atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tessi/code-atlas/releases/tag/v0.1.0
