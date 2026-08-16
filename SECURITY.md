# Security policy

## Supported versions

Until Code Atlas reaches 1.0, security fixes are provided for the latest
published minor release only. Users should upgrade to the newest release before
reporting an issue that may already be fixed.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting](https://github.com/tessi/code-atlas/security/advisories/new)
to send a description, affected versions, reproduction steps, and impact.

You should receive an acknowledgement within seven days. The maintainer will
coordinate validation, a fix, release timing, and credit with the reporter.

## Security boundary

Code Atlas analyzes local Git checkouts and may invoke locally installed
language tooling such as `mix`, `elixir`, `rust-analyzer`, or `scip-typescript`.
Only run it against repositories and toolchains you trust. Generated HTML is
self-contained and contains no source text, but it does contain repository
paths, revision identifiers, metrics, call metadata, and curve geometry; treat
those files as potentially sensitive project metadata.
