# BlockVIEM.jl

A Julia implementation of the **Volume Integral Equation Method (VIEM)** for
electromagnetic scattering by arbitrarily shaped, high-contrast dielectric
particles (e.g., iron-oxide aggregates, rough gold nanoparticles).

The package targets the same observables as
[`block-DDA_Py`](https://github.com/NobuhiroMoteki/block-DDA_Py) (CAS-v2
amplitudes), with higher accuracy and efficiency for high-refractive-index
inhomogeneous targets via tetrahedral discretization, AIM/FFT acceleration,
and block-Krylov multi-orientation batching.

## Status

**Phase 0 — Project skeleton.** Numerical kernels are not yet implemented.

| Phase | Module | Status |
|-------|--------|--------|
| 0 | Project basis (Project.toml, CI, I/O spec) | in progress |
| 1 | Mesh & SWG basis | not started |
| 2 | Duffy-transform singular integration | not started |
| 3 | AIM (FFT-MVP) | not started |
| 4 | Block-Krylov solver | not started |
| 5 | PostProcess (CAS-v2 observables) | not started |

## Design references

- `.claude/technical_note.md` — formulation (SWG, EFVIE, weakening, Duffy)
- `docs/io_spec.md` — `block-DDA_Py` compatible I/O specification
- `.claude/reference/` — primary literature (SWG 1984, Volakis-Sertel,
  Sheng-Song, Mousavi-Sukumar 2010)

## Installation (development)

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> Pkg.test()
```

## License

MIT (see `LICENSE`).

## Author

Nobuhiro Moteki ([@NobuhiroMoteki](https://github.com/NobuhiroMoteki),
`nobuhiro.moteki@gmail.com`)
