# Releasing

Releases are created from version tags on `main`. A release tag must match the Python package version in `pyproject.toml`. The workflow builds wheels for Linux, Windows, and macOS on x86_64 and ARM64, plus a source distribution; it then publishes to PyPI and creates a GitHub Release with the distributions and SBOMs.

## One-time account setup

The PyPI project does not exist yet, so create a **pending publisher** from your PyPI account's Publishing page. Use these exact values:

| PyPI field | Value |
| --- | --- |
| Project name | `bankai-fast-mcs` |
| Owner | `breno-jesus-fernandes` |
| Repository | `bankai-fast-mcs` |
| Workflow filename | `ci.yml` |
| Environment | `pypi` |

PyPI uses the workflow filename, which is `.github/workflows/ci.yml` in this repository. The publisher uses GitHub's OIDC identity; no PyPI API token or GitHub secret is needed.

In GitHub repository settings, create the `pypi` Actions environment. Restrict deployments to tags matching `v*`; optionally require a trusted reviewer for each deployment. Create a tag ruleset for `v*` to limit who can create, update, or delete release tags. The workflow grants `id-token: write` only to the PyPI publishing job and `contents: write` only to the GitHub Release job.

## Publish a release

1. Update `project.version` in `pyproject.toml` and the crate version in `Cargo.toml`; update `uv.lock` and `Cargo.lock` as needed.
2. Merge the release commit into `main` and wait for CI to pass.
3. Create and push a tag matching the Python version. For the first alpha release:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a v0.1.0a1 -m "Bankai Fast MCS 0.1.0a1"
   git push origin v0.1.0a1
   ```

The workflow verifies that the tag points to `main`, matches `pyproject.toml`, and is not already on PyPI. It runs the full test and build matrix before publishing. Alpha, beta, and release-candidate versions are marked as pre-releases on GitHub. The GitHub Release step can be safely rerun after a partial release: it reuses an existing release and replaces its attached build artifacts.
