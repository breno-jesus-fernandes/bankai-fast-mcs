# Contributing

Contributions are welcome through GitHub issues and pull requests. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- Use the [bug report](https://github.com/breno-jesus-fernandes/bankai-fast-mcs/issues/new?template=bug_report.yml) or [feature request](https://github.com/breno-jesus-fernandes/bankai-fast-mcs/issues/new?template=feature_request.yml) form for public reports.
- For security vulnerabilities, follow the private reporting instructions in [SECURITY.md](.github/SECURITY.md); do not open a public issue.
- For substantial changes, open an issue first to discuss the approach.
- Discuss public API changes in an issue before implementing them.

## Development setup

Install Git, Python 3.11 or newer, a stable Rust toolchain, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). On Windows, install the Rust MSVC build tools; on macOS, install the Xcode command-line tools.

Fork the repository, clone your fork, and configure the upstream remote:

```bash
git clone https://github.com/<your-user>/bankai-fast-mcs.git
cd bankai-fast-mcs
git remote add upstream https://github.com/breno-jesus-fernandes/bankai-fast-mcs.git
uv python install 3.11
uv sync --locked --extra dev --no-install-project
uv run maturin develop --release --locked
```

Create a focused branch from the current `main` branch:

```bash
git fetch upstream
git switch -c descriptive-change upstream/main
```

## Checks

Run both test suites from the repository root before opening a pull request:

```bash
uv run pytest -q
uv run cargo test --locked
```

CI also tests and builds the native extension across Linux, Windows, and macOS. A local pass does not replace the required CI checks.

## Dependencies and lockfiles

Keep `uv.lock` and `Cargo.lock` reproducible. When dependencies change, update the relevant manifest and lockfile intentionally, explain the reason in the pull request, and run the checks above. Do not allow installs or tooling to update lockfiles implicitly. Dependency changes are reviewed by CI's Dependency Review, Python dependency audit, and Rust dependency audit. Read the [dependency security policy](.github/SECURITY.md) before changing scanner configuration or adding an exception.

## Pull requests

Open a pull request against `main`, describe the user-visible impact, link related issues, and include tests and documentation updates where relevant. Keep changes focused and respond to review feedback. Maintainers may request revisions or additional checks before merging.

There is no additional Contributor License Agreement (CLA) or Developer Certificate of Origin (DCO) requirement. Contributions are covered by the project's [MIT license](LICENSE).
