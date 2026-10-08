# GitHub checks and container releases

The repository includes CI checks, a container release workflow, and weekly Dependabot updates. Checks use synthetic data and mocked providers; no Discord or Gemini credentials are needed on GitHub. Publishing an image does not deploy it to a Raspberry Pi or restart any service.

## Enable the checks

1. Commit and push `.github/`, the updated Docker/Compose files, the CI scripts/tests, and this documentation to your GitHub repository. Normal pushes and pull requests run **CI**. No workflow needs your `.env`, private `config.yaml`, database, or API keys.
2. Open the repository's **Actions** tab and inspect the first CI run. If Actions are disabled, enable them under **Settings → Actions → General**. The repository must allow the GitHub and Docker actions used here. The release job requests `packages: write` itself; ordinary jobs have read-only repository access.
3. Put these files on the repository's default branch to activate Dependabot and make the manual **Run workflow** button available. If your work is on another branch, merge it into the default branch when ready.
4. Once the first run succeeds, optionally add a branch ruleset under **Settings → Rules → Rulesets** requiring the Python, dashboard browser, repository security, and both container checks before merging.

### What CI checks

| Check | Coverage |
| --- | --- |
| Python 3.12 and 3.13 | The offline suite, including database migration preservation, lore isolation, authentication, provider handling, and token accounting. |
| Dashboard browser | Real browser interactions against a disposable HTTPS dashboard with mocked Discord. Failure screenshots and fixture logs expire after seven days. |
| Repository security | Tracked private/runtime paths, SQLite files even when renamed, and a redacted Gitleaks scan of the checked-out Git history. |
| Container AMD64 and ARM64 | Native builds on GitHub-hosted PC and ARM runners, followed by image asset checks, Pillow/JavaScript execution, configuration loading, database migration, and dashboard startup with external networking disabled. |

Third-party actions are pinned to complete commit IDs. Gitleaks is installed from a fixed version and its release checksum is verified. Dependabot opens weekly update PRs for pip requirements, the Docker base image, and GitHub Actions; it does not automatically merge them. The Gitleaks CLI version is maintained explicitly in `ci.yml`.

Security scanning is an additional guard, not a guarantee that every possible credential format is detectable. If it flags a real credential already in Git history, revoke/rotate it and remove it from history before publishing; deleting only the latest file does not remove older copies. False positives should get a narrow, reviewed exception rather than disabling the scan.

## Publish a version

After committing and pushing a version you want to distribute, tag that exact commit. For example, using an unused version:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The **Container release** workflow runs all CI checks on that tag before publishing a combined `linux/amd64` and `linux/arm64` image. Both architectures are tested natively; the publishing build reuses their caches. Images are tagged by version and full commit ID:

```text
ghcr.io/<owner>/<repository>:v0.1.0
ghcr.io/<owner>/<repository>:sha-<full-commit-id>
```

There is deliberately no floating `latest` tag. Use a version, or an image digest for an immutable deployment. Publishing uses GitHub's automatically supplied `GITHUB_TOKEN`, with package-write permission only in the release job; you do not need a Docker Hub account or an extra publishing token. If organization policy prevents package publishing, the repository/organization owner needs to allow it. Keep the package private or explicitly choose public visibility in its package settings.

## Use a published image on a Pi

Keep `docker-compose.yaml`, a private `.env`, your private `config.yaml`, and the persistent `data/` directory together. The image contains application code, JavaScript, and sample cards. Compose mounts only your configuration and data. `.env` supplies runtime environment variables and is never copied into the image.

Add the published image name to `.env`:

```dotenv
LLMCORD_IMAGE=ghcr.io/<owner>/<repository>:v0.1.0
```

If the package is private, sign in to GHCR on the Pi using a personal access token (classic) with `read:packages` and access to that package. Supply it through `docker login ghcr.io --username <owner> --password-stdin`. Do not save it in this repository. Public packages can be pulled anonymously.

For a first installation, follow [Raspberry Pi deployment](deployment-raspberry-pi.md) for credentials, Discord OAuth, persistent data, and Tailscale, then run:

```bash
docker compose pull
docker compose up --no-build -d
docker compose ps
```

For upgrades, first stop the bot and web writers and make a private database backup as described in that guide, then pull/start the chosen version. To build locally instead, remove `LLMCORD_IMAGE` from `.env` and use `docker compose up --build -d`.

The existing native systemd deployment remains a separate hosting option. Before switching the same Pi to Compose, stop the native bot and web services so there is one bot instance and one service listening on port 8080. An image alone does not restore your characters/lore database or configure Discord OAuth/Tailscale. No automatic deployment or billable live-model test is configured by these workflows.

## Run checks locally

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_repository.py
python -m unittest discover -s tests -v
python -m playwright install chromium
LLMCORD_BROWSER_TESTS=1 python -m unittest discover -s tests -p test_dashboard_browser.py -v
docker build -t llmcord:ci .
docker run --rm --interactive --network none --entrypoint python llmcord:ci - < scripts/smoke_container.py
```

For an installed Chromium on a Pi, set `LLMCORD_BROWSER_EXECUTABLE=/usr/bin/chromium` alongside `LLMCORD_BROWSER_TESTS=1`. The container smoke test must run from the image's application directory without production mounts; it rejects images containing `.env`, private config, or data.

References: [Playwright CI](https://playwright.dev/python/docs/ci), [Docker multi-platform Actions builds](https://docs.docker.com/build/ci/github-actions/multi-platform/), [GHCR authentication and access](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), and [Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
