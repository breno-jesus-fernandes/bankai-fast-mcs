# Security policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues. Use GitHub's private vulnerability reporting for this repository. Include the affected version or commit, steps to reproduce, and any relevant impact. You will receive a response through the private report.

## Dependency security

`Cargo.lock` and `uv.lock` are the reproducible dependency inputs. CI uses locked dependency resolution. Dependency Review checks pull requests for vulnerable dependencies, `cargo-deny` audits Rust advisories, licenses, duplicate versions, and dependency sources, and `pip-audit` scans the locked Python dependencies.

Exceptions must be narrowly scoped and documented beside the scanner configuration with the advisory or crate, rationale, owner, and review date. Rust advisory exceptions belong in `.cargo/deny.toml`. Do not suppress unresolved Python vulnerabilities; update the lockfile or track an exception in an issue and document its identifier and review date in the security workflow.

## Repository settings

Enable GitHub's dependency graph, Dependabot alerts, and private vulnerability reporting. Require pull requests and these checks on `main`:

- `Test (Linux x86_64, Python 3.11)`
- `Test (Linux x86_64, Python 3.14)`
- `Test (Linux ARM64, Python 3.11)`
- `Test (Linux ARM64, Python 3.14)`
- `Test (Windows x86_64, Python 3.11)`
- `Test (Windows x86_64, Python 3.14)`
- `Test (Windows ARM64, Python 3.11)`
- `Test (Windows ARM64, Python 3.14)`
- `Test (macOS x86_64, Python 3.11)`
- `Test (macOS x86_64, Python 3.14)`
- `Test (macOS ARM64, Python 3.11)`
- `Test (macOS ARM64, Python 3.14)`
- `Rust dependency audit`
- `Python dependency audit`
- `Dependency Review`

GitHub repository rules and security settings must be enabled in the repository settings; workflow files cannot turn on branch protection or vulnerability reporting by themselves.

For a PyPI release, configure Trusted Publishing for this GitHub repository, the `CI and release artifacts` workflow, and the `pypi` environment. The workflow publishes only when a `v*` tag matches the version in `pyproject.toml`, the version is not already on PyPI, CI succeeds, and artifact provenance has been generated. The first release tag should be `v0.1.0a1`; PyPI will display it as version `0.1.0a1` and mark it as a pre-release.
