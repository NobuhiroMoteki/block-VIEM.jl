# Installation Guide — block-VIEM.jl

This document describes how to set up a working block-VIEM.jl
environment from scratch on:

- **Windows + WSL2 (Ubuntu)** — recommended for Windows users.
- **Native Linux (Ubuntu 22.04 LTS or 24.04 LTS)** — server / workstation.

The same procedure works on both targets after WSL2 is installed,
because WSL2 *is* an Ubuntu environment.  macOS is **not officially
supported**: the project pins Julia 1.12.5 with a Linux Manifest, and
Gmsh.jl's macOS binary occasionally lags the Linux one.

The end state is:

- Julia **1.12.5** installed via `juliaup`, set as the override for
  this project directory (other projects on the same machine keep
  their own Julia versions).
- `BlockVIEM` package instantiated from the checked-in
  `Project.toml` / `Manifest.toml` (114 dependencies, fully
  reproducible).
- `viz/` sub-environment instantiated for figure generation.
- The full test suite (14 422 tests across Mie, CAS-v2, AIM,
  block-Krylov, aggregate-mesh, GRE) passing.

---

## 1. Prerequisites

### 1.1 Hardware
- CPU: any x86-64 with AVX2.  Multi-core is strongly recommended
  (the solver scales linearly to ≈ 10 threads on AIM-MVP and
  near-field assembly).
- RAM: **16 GB minimum** for benchmark / test runs; **64 GB
  recommended** for paper-production sweeps (peak RSS at GRE
  `r_v = 0.4` μm with `m_p = n20` reaches ~ 30 GB; see
  [CLAUDE.md §2](CLAUDE.md) for sizing).
- Disk: ~ 5 GB for Julia, dependencies, and FFTW plans;
  add ~ 10 GB if you intend to run the full HDF5 paper-production
  sweeps.

### 1.2 Operating system
- **Windows 10 (build ≥ 19044) or Windows 11** with WSL2 enabled, or
- **Ubuntu 22.04 LTS / 24.04 LTS** native.

WSL1 is **not supported** — Gmsh.jl's bundled OpenCASCADE binary
relies on a real Linux kernel.

---

## 2. WSL2 setup (Windows users only — skip if on native Linux)

In an **Administrator PowerShell** prompt:

```powershell
wsl --install -d Ubuntu-24.04
```

This installs WSL2 + the Ubuntu-24.04 distro.  Reboot when prompted.
On first launch the Ubuntu console asks for a Unix username and
password — pick anything; this becomes your Linux user inside WSL.

Verify:

```powershell
wsl --list --verbose
```

The `Ubuntu-24.04` entry should show `VERSION 2`.

From now on, run all commands inside the **Ubuntu shell** (start menu
→ "Ubuntu-24.04"), **not** in PowerShell.

### Recommended WSL2 tweaks

Create or edit `C:\Users\<you>\.wslconfig` to give WSL enough memory
for the solver:

```ini
[wsl2]
memory=24GB         ; raise to fit your machine; leave ≥ 8 GB for Windows
processors=8        ; raise to your physical core count
swap=16GB
localhostForwarding=true
```

Restart WSL afterwards: `wsl --shutdown` in PowerShell, then re-open
the Ubuntu shell.

---

## 3. System packages (run inside Ubuntu / WSL2 Ubuntu)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libfreetype6
```

Notes:
- `build-essential` is needed if any Julia dependency falls back to
  source builds (rare, but harmless to install).
- The `libX*` / `libfontconfig` / `libfreetype` packages are
  required by **Gmsh.jl** at first launch — even though our use is
  fully head-less (`General.Terminal=0`), the bundled gmsh binary
  still loads these shared libraries on `gmsh.initialize()`.

The `viz/` figure-generation environment uses **CairoMakie** which
is pure-software and needs no GPU drivers.  WSLg / X server is
**not** required.

---

## 4. Install Julia 1.12.5 via juliaup

`juliaup` is the official Julia version manager.  We pin Julia
1.12.5 because the project's `Manifest.toml` is resolved against
that version (see [CLAUDE.md §7](CLAUDE.md) — "do not unpin without
re-running the validation suite").

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes
```

Open a new shell so `~/.juliaup/bin` is on `PATH`, then install
1.12.5 and make it available:

```bash
juliaup add 1.12.5
juliaup status        # confirm 1.12.5 is listed
```

You may keep other channels (`release`, `lts`, etc.) installed for
other projects.  We will pin 1.12.5 to *this* project below.

---

## 5. Clone the repository

```bash
mkdir -p ~/Julia && cd ~/Julia
git clone https://github.com/NobuhiroMoteki/block-VIEM.jl.git
cd block-VIEM.jl
```

(If you cloned to a different parent directory, adjust paths in §6 –
§9 accordingly.)

---

## 6. Pin Julia 1.12.5 to this project

```bash
juliaup override set -p . 1.12.5
juliaup override status      # the entry for this project should read 1.12.5
```

After this, every `julia --project=.` invocation from
`~/Julia/block-VIEM.jl/` (and below) launches **Julia 1.12.5**
regardless of the system default.

**Verify:**

```bash
julia --project=. -e 'using InteractiveUtils; versioninfo()' | head -3
```

Output must include `Version 1.12.5`.

---

## 7. Instantiate the main environment

This step downloads and precompiles the 114 dependencies recorded in
`Manifest.toml` (FFTW, Gmsh, HDF5, Krylov, IterativeSolvers,
SpecialFunctions, StaticArrays, …).  Allow ~5–10 minutes on first
run depending on network bandwidth and CPU.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Smoke-test the package loads cleanly:

```bash
julia --project=. -e 'using BlockVIEM; @info "BlockVIEM loaded"'
```

Expect a single `[ Info: BlockVIEM loaded` line and no warnings
(modulo a benign `Gmsh: ...` startup banner from the bundled OCC
kernel).

---

## 8. Run the test suite

The full test set (14 422 tests on Julia 1.12.5) covers Mie sphere,
CAS-v2 doublet, AIM vs dense, block-Krylov, aggregate-mesh / GRE
mesh, and surface-integral validations.  Use multiple threads to
keep wall time around 5–10 minutes:

```bash
JULIA_NUM_THREADS=auto julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests should pass.  If any fail at this stage **stop and
investigate** before running paper-production sweeps — the validation
ensures the solver matches Mie / MSTM / DDA references to the
documented tolerances.

---

## 9. Set up the `viz/` figure-generation environment (optional)

`viz/` is an independent Julia environment that adds **CairoMakie**
+ **GeometryBasics** for the wireframe galleries documented in
[docs/descriptions_particle_shape_model.md](docs/descriptions_particle_shape_model.md).
The main `Project.toml` deliberately avoids these heavy plotting
dependencies so headless production runs stay lean.

```bash
julia --project=viz -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Generate the example galleries:

```bash
julia --project=viz viz/visualize_gre.jl         # writes viz/figs/gre_*.png
julia --project=viz viz/visualize_aggregate.jl   # writes viz/figs/agg_*.png
```

---

## 10. Recommended editor / shell setup

### VSCode + Julia extension
1. Install **VS Code** for Windows (or Linux native).
2. On Windows, install the **WSL** extension — this lets VS Code run
   inside WSL with the Ubuntu file system.
3. Install the **Julia** extension (`julialang.language-julia`).
4. Open `~/Julia/block-VIEM.jl` (`Ctrl+Shift+P` → "WSL: Open Folder
   in WSL…" on Windows, or just `code .` on native Linux).
5. The Julia extension will auto-detect the project and the
   1.12.5 override.

### Shell convenience aliases (optional)
Add to `~/.bashrc`:

```bash
alias jvi='julia --project=. -t auto'           # main project, all threads
alias jviz='julia --project=viz -t auto'        # viz environment
alias bvi='cd ~/Julia/block-VIEM.jl'
```

---

## 11. Daily workflow

```bash
cd ~/Julia/block-VIEM.jl

# Run the production sweep (block-DDA_Py-compatible HDF5 output)
JULIA_NUM_THREADS=auto julia --project=. viem_results/run_viem.jl

# Inspect results
julia --project=. viem_results/check_h5.jl

# Re-render a figure gallery after editing visualize_*.jl
julia --project=viz viz/visualize_aggregate.jl
```

See [README.md](README.md) for the full sweep / benchmark workflow
and [CLAUDE.md](CLAUDE.md) §5 for the pre-run cost-estimation
checklist (memory, wall-time).

---

## 12. Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| `juliaup: command not found` | Open a new shell so `~/.juliaup/bin` is on `PATH`, or `source ~/.bashrc`. |
| `ERROR: LoadError: could not load library "libGL.so.1"` from `using Gmsh` | Install the X11 / fontconfig dependencies in §3 (`libxrender1 libxext6 libfontconfig1 libfreetype6`). |
| `Pkg.instantiate()` resolves a different Julia version | Re-run `juliaup override set -p . 1.12.5` from inside the project directory. |
| WSL: `Cannot allocate memory` mid-solve | Raise the `memory=` setting in `~/.wslconfig` (Windows side) and `wsl --shutdown` to apply. |
| `Pkg.test()` failures only on `test_aggregate_mesh.jl` | Likely Gmsh-binary mismatch.  Re-instantiate (`Pkg.instantiate()`) to refresh the artifact cache. |
| Tests run single-threaded despite `-t auto` | `Pkg.test()` does not inherit `-t auto`; set the env var instead: `JULIA_NUM_THREADS=auto julia ...`. |
| `viz/visualize_*.jl` segfaults on first launch | First-time CairoMakie precompile; just re-run.  If it persists, `rm -rf ~/.julia/compiled/v1.12/CairoMakie` and try again. |

---

## 13. Updating to a new release

When a new release is tagged on GitHub:

```bash
cd ~/Julia/block-VIEM.jl
git pull --ff-only origin main

# Re-instantiate in case Manifest.toml changed
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Re-run tests to confirm nothing regressed in your environment
JULIA_NUM_THREADS=auto julia --project=. -e 'using Pkg; Pkg.test()'
```

`Manifest.toml` is intentionally tracked in this repository (see
`.gitignore` comment) so an upstream pull gives you the **exact
same dependency versions** the maintainers tested against.  Avoid
running `Pkg.update()` against a release-tagged checkout.

---

## 14. Uninstalling

```bash
# Remove the project clone
rm -rf ~/Julia/block-VIEM.jl

# (optional) remove the project-pinned Julia override
juliaup override unset -p ~/Julia/block-VIEM.jl

# (optional) remove Julia 1.12.5 if no other project needs it
juliaup remove 1.12.5

# (optional) remove juliaup itself
rm -rf ~/.juliaup ~/.julia
```

WSL2 itself can be uninstalled from Windows
**Settings → Apps → Apps & features → "Ubuntu-24.04" → Uninstall**.
