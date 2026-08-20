# Contributing to Code Atlas

Thank you for helping improve Code Atlas. Bug reports, visual experiments,
analyzer improvements, documentation, and focused performance work are welcome.

## Development setup

Install Rust 1.87 or newer and Git, then run:

```sh
cargo build --locked
cargo test --locked
```

Elixir/Erlang, `rust-analyzer`, and `scip-typescript` are optional for the unit
tests. They are useful when manually validating analyzer coverage against real
local repositories. Do not commit third-party repository checkouts, generated
atlas outputs, or analysis caches.

Before submitting a change, run the same core checks as CI:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --locked
cargo package --locked
```

## Design invariants

Please preserve these properties unless a proposal explicitly changes them:

- one retained cross-file callsite produces one spline;
- calls are never aggregated or sampled;
- source and target anchors retain their line information;
- same-file calls, tests, hidden paths, and configured paths are excluded before
  layout;
- endpoint filters run after reusable semantic analysis and never aggregate the
  retained callsites;
- software optical-density rendering remains the deterministic reference;
- light and dark modes use the same physical pigment model;
- generated HTML remains self-contained and makes no runtime network requests.

Changes to analysis or rendering should include focused regression tests. Visual
changes should include before/after output from a public repository and explain
the intended data meaning, not only the aesthetic difference.

## Pull requests

Keep pull requests focused, explain the observable behavior, and mention the
commands used to verify it. By contributing, you agree that your contribution is
licensed under the project's MIT license.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security reports belong
in the private process described in [SECURITY.md](SECURITY.md), not a public
issue.
