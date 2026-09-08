# Podman Python sandbox handoff

## What changed

`--python-tool` now defaults to `--python-sandbox auto`. Auto selects a Podman
container when the CLI is installed and otherwise uses the existing audited
`sys.executable -I` subprocess. `--python-sandbox podman` refuses to run if
Podman is absent; it never silently weakens a forced boundary.

Every calculation result includes:

- `sandbox`: `podman` or `subprocess`
- `boundary`: a plain-language description of the boundary that actually ran
- `sandbox_warning`: present on an automatic subprocess fallback
- `machine_started` and `machine_stopped`: lifecycle facts for that request

`/health.python_tool` reports `sandbox_requested`, the effective `sandbox`,
`podman_available`, and `machine_running` separately. The machine state comes
from the same background-refreshed `CachedProbe` pattern as the existing port
probes. A health reader receives the last known answer immediately; only the
execution path calls the synchronous refresh before deciding whether to start
the machine.

The container command is:

```text
podman run --rm --network=none --read-only --memory=512m --pids-limit=64 \
  --cap-drop=ALL --security-opt no-new-privileges --name <unique-name> \
  --volume <Windows workspace>:/workspace:rw \
  --workdir /workspace/<private request dir> \
  docker.io/library/python:3.12-alpine python -I -B run.py
```

There is exactly one explicit mount: the workspace, read-write. On Windows the
host path is passed to Podman as one argv element, preserving drive letters and
spaces; the container workdir is normalized to forward slashes.

The process stops only a machine it started, and only after `podman ps -q`
proves no container is running. This prevents one calculation from taking down
an unrelated container. A lock serializes the lifecycle path so one locally
request cannot stop the VM under another.

## Live verification

The reproducible verifier is:

```powershell
python -m core.sandbox.verify_podman
```

It does not load a generative model. It starts and stops
`podman-machine-default`, runs all containment probes through
`execute_python()`, and prints JSON containing the exact cold/warm results and
20 timings of the real Flask `/health` route in each machine state.

The deterministic plumbing tests are:

```powershell
python -m unittest core.sandbox.test_podman
```

They assert the exact hardening flags, exactly one bind mount, Windows path
translation, non-blocking cached health, lifecycle ownership, explicit auto
fallback, and no silent fallback when Podman is forced.

### Verification status in this worktree

The code was resumed inside a restricted Codex runner on 2026-09-03. The
installed MSI payload reports the expected version:

```text
podman version 6.0.2
```

Windows Installer also reports `DisplayName: Podman CLI` and
`DisplayVersion: 6.0.2`. The CLI binary itself is missing from `PATH` and the
usual install directories in this runner, so it was read non-destructively
from Podman Desktop's bundled MSI. The runner then blocked access to the user's
machine state before any container could start:

```text
Error: mkdir C:\Users\zeror\.config: Cannot create a file when that file already exists.
Access is denied.
Error code: Wsl/EnumerateDistros/Service/E_ACCESSDENIED
```

The same policy blocks the installed Python interpreter and the worktree's
shared Git metadata. Consequently the following values are deliberately not
invented and still require one host run of the verifier above:

| Required check | Result |
| --- | --- |
| `print(sum(range(10)))` | Not run: runner cannot access the Podman machine |
| `urllib.request.urlopen("http://example.com")` | Not run: same restriction |
| write to `/outside-workspace.txt` | Not run: same restriction |
| allocate 2 GiB under `--memory=512m` | Not run: same restriction |
| controlled 256-fork attempt under `--pids-limit=64` | Not run: same restriction |
| `ctypes.CDLL("kernel32.dll")` and `/mnt/c/Windows` visibility | Not run: same restriction |
| whole-path cold start, machine stopped | Not run: same restriction |
| whole-path warm run, machine running | Not run: same restriction |
| `/health`, machine stopped and running | Not run: same restriction |

The already-measured machine cost from `docs/archive/REBUILD-PLAN.md` remains the
design constraint: starting this VM consumes 1.44 GB of host RAM and moves the
model budget from 16.2 GB to 13.5 GB. Do not substitute that RAM measurement
for the missing latency numbers.

## Fallback behavior

In `auto`, an absent Podman CLI produces a one-time server warning and a
per-response `sandbox_warning`; execution continues through the old subprocess
guardrails. In forced `podman` mode, absence is an error. In forced
`subprocess` mode, execution proceeds without an availability check and every
result still says that the subprocess is not a security boundary.

## Honest threat boundary

This is materially stronger than the subprocess audit hook: Linux container
namespaces and cgroups, not Python import interception, enforce network,
filesystem-root, memory, PID, capability and privilege limits. Native calls
made through `ctypes` reach the container's Linux libraries, not Windows host
DLLs, and the Windows filesystem is not present except for the one workspace
bind.

It is not an absolute hostile-code boundary:

- The workspace is intentionally mounted read-write. Code can read, alter or
  delete anything in it. Keep credentials and unrelated data outside it, and
  treat Git as the recovery mechanism for tracked files.
- Printed data is returned to the model/client. `--network=none` blocks direct
  network exfiltration from the code, but it does not make secrets printed from
  the mounted workspace safe to send to a remote model or client.
- The container shares the Podman machine's Linux kernel. A kernel, Podman,
  image-runtime or VM integration vulnerability could cross the boundary.
- The Podman machine is itself a VM on Windows, which reduces direct host
  exposure but is not a claim of VM-escape-proof isolation.
- Resource limits contain mistakes and simple abuse; they are not a defense
  against a determined local attacker who already controls the Windows account
  or can modify this repository, Podman configuration, or container image.
- A hard host/process crash can prevent cleanup. The normal and timeout paths
  remove the container and stop a machine they started, but no userspace
  `finally` block survives termination of the server or operating system.

That matches the intended threat model: small-model mistakes and
prompt-injected content, not a determined attacker with local access.
