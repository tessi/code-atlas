# Code Atlas

[![CI](https://github.com/tessi/code-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/tessi/code-atlas/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/code-atlas.svg)](https://crates.io/crates/code-atlas)
[![docs.rs](https://img.shields.io/docsrs/code-atlas)](https://docs.rs/code-atlas)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

Code Atlas renders an architectural map of a code repository. Files and
directories form a data-driven treemap, overlaid with a hierarchically bundled
call graph of method and function callsites.

| light | dark |
| --- | --- |
| ![Wasmex calls in the light theme](https://raw.githubusercontent.com/tessi/code-atlas/main/docs/images/wasmex-light.webp) | ![Wasmex calls in the dark theme](https://raw.githubusercontent.com/tessi/code-atlas/main/docs/images/wasmex-night.webp) |

## Reading the map

- Treemap parcel area represents lines of code, byte size, or commit count.
- Every visible route is one callsite. A route begins at the source line's position inside its file, follows the directory hierarchy, and ends at the target line.
- Color shows direction: red at the source, blue at the target. Call routes get more intense in color when they overlap. This allows to see main communication routes in a project.

## Install

Code Atlas supports Linux and macOS and requires Git at runtime. Installing from
source requires Rust 1.87 or newer.

```sh
cargo install code-atlas --locked
```

Checksummed native archives are also attached to tagged
[GitHub releases](https://github.com/tessi/code-atlas/releases).

## Try it

Create a self-contained interactive viewer from any local checkout:

```sh
code-atlas render \
  --repo /path/to/checkout \
  --output atlas.html \
  --width 1920 --height 1080 \
  --exclude assets --exclude docs --exclude guides
```

Open `atlas.html` directly or put that single file on any static host. It embeds
the map, calls, light and dark themes, filters, and viewer code; it makes no
runtime network requests. The architectural payload is losslessly compressed,
and a modern browser reconstructs the derived spline samples when opening it.

Create an A2 landscape poster with a higher-resolution call layer:

```sh
code-atlas render \
  --repo /path/to/checkout \
  --output atlas.pdf \
  --calls-in apps/billing \
  --width 7016 --height 4961 \
  --theme night --backend wgpu \
  --pdf-dpi 300 --pdf-call-dpi 600 \
  --resume
```

The output extension selects PNG, SVG, PDF, or HTML. SVG and PDF keep parcels,
boundaries, and labels as vectors. PDF calls are rendered in bounded tiles;
`--resume` verifies and reuses completed tiles after interruption.

Useful controls:

| Option | Meaning |
| --- | --- |
| `--metric loc\|bytes\|commits` | Choose the file-area metric. |
| `--theme architect\|night\|ink\|solarized-dark\|solarized-light` | Choose the static-output appearance. HTML always embeds light and dark. |
| `--backend software\|wgpu` | Choose the call renderer. Unsupported GPUs fall back to software. |
| `--exclude PATH` | Exclude a repository-relative prefix; repeat as needed. |
| `--calls-in PATH` | Keep calls with either endpoint below a repository-relative prefix; repeat to match any prefix. |
| `--calls-from PATH` | Keep calls whose source is below a repository-relative prefix; repeat to match any prefix. |
| `--calls-to PATH` | Keep calls whose target is below a repository-relative prefix; repeat to match any prefix. |
| `--include-tests` | Include conventional test paths, excluded by default. |
| `--include-hidden` | Include dotfiles and dot-directories, excluded by default. |

Run `code-atlas render --help` for every layout, pigment, and print option.

## Explore calls in the browser

Pan and zoom like a map, select a file or individual spline, and filter calls
without collapsing them. The query language is case-insensitive:

```text
in:*delivery*
out:*attempts* in:*delivery*
path:*clickhouse* -in:*deprecated*
in:*delivery* | in:*authoring*
```

`in:`/`to:` match target paths, `out:`/`from:` source paths, and `path:` either
endpoint. `callee:` and `analyzer:` match call metadata. Spaces mean AND, `|`
means OR, and a leading `-` excludes a term. The viewer includes examples and
specific syntax-error help.

Selection, query, and camera position are stored in the URL hash for sharing.
Appearance follows the system by default and can be saved as Light or Dark.
WebGPU is used when available, with a Canvas2D fallback; pigment automatically
strengthens when filters leave only a few visible calls.

Source and target lines can link back to a hosted repository:

```sh
code-atlas render \
  --repo /path/to/checkout \
  --output atlas.html \
  --source-url-template \
    'https://github.com/OWNER/REPO/blob/{revision}/{path}#L{line}'
```

## Analyzer coverage

| Language | Preferred analysis | Fallback |
| --- | --- | --- |
| Elixir (`.ex`, `.exs`, `.eex`) | compiled BEAM debug data | `mix xref trace` plus compiler-backed parsing for scripts and EEx templates |
| Erlang (`.erl`) | static in-repository remote calls, function references, and imports | - |
| Rust | rust-analyzer SCIP | uniquely resolved syntax calls |
| TypeScript / JavaScript | scip-typescript | relative-import calls |
| Gleam | explicit-import resolution | - |

Use `CODE_ATLAS_RUST_ANALYZER` or `CODE_ATLAS_SCIP_TYPESCRIPT` to override the
semantic indexer binaries. Code Atlas never downloads analyzer tools while
rendering. Inspect coverage and warnings without producing an image:

```sh
code-atlas inspect --repo /path/to/checkout
```

"Every call" means every cross-file callsite resolved by the active analyzers.
Dynamic dispatch, macros, reflection, and generated code prevent any static tool
from seeing every runtime call. Same-file calls are intentionally excluded. The
JSON report records analyzer coverage, unresolved calls, exclusions, the Git
revision, dirty-worktree state, renderer, and timing information.

Semantic analysis is cached under `.git/code-atlas/`; endpoint filters reuse it,
so experimenting with sub-domain posters does not repeat parsing.

## Security and privacy

Only analyze repositories and language toolchains you trust: semantic analysis
may run project-local tools such as `mix compile`. HTML exports contain no source
text, but they do contain paths, revision identifiers, metrics, call metadata,
and geometry. Treat private-repository exports as sensitive project metadata.
See [SECURITY.md](SECURITY.md) for the full boundary and reporting process.

## Development

The CLI is the supported interface. The Rust library remains experimental during
the `0.x` series. See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and
[RELEASING.md](RELEASING.md) for the release procedure.

This project was built through LLMs.

## License

Code Atlas is released under the [MIT License](LICENSE-MIT).
