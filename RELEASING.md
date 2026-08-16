# Releasing Code Atlas

Releases publish the Rust crate to crates.io and attach checksummed native
binaries to a GitHub Release. Publishing is intentionally tag-driven; normal CI
never publishes anything.

## Release checklist

1. Start from a clean, up-to-date `main` branch and confirm CI is green.
2. Update `version` in `Cargo.toml`, then run `cargo check` so `Cargo.lock`
   records the same package version.
3. Move noteworthy entries from `Unreleased` into that version's changelog
   section and replace `Unreleased` with the UTC release date.
4. Run:

   ```sh
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features --locked -- -D warnings
   cargo test --all-features --locked
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --locked
   cargo package --locked
   ```

5. Inspect the package with `cargo package --list --locked`. Commit the version
   and changelog changes.
6. Create and push an annotated tag matching the manifest version:

   ```sh
   git tag -a v0.1.0 -m "Code Atlas 0.1.0"
   git push origin v0.1.0
   ```

The release workflow checks that the tag and manifest versions match, runs the
test and package gates again, publishes to crates.io, creates the GitHub
Release, and uploads archives plus SHA-256 files for Linux and macOS.
GitHub build-provenance attestations are generated for every archive.

## Verification

After the workflow completes:

1. Confirm every expected archive, checksum, and attestation is present.
2. Verify one archive with its `.sha256` file and run `code-atlas --version`.
3. Install independently with `cargo install code-atlas --version VERSION --locked`.
4. Confirm the crates.io README, docs.rs build, GitHub release notes, and
   changelog all describe the same version.
