"""
    BlockVIEM

Volume Integral Equation Method (VIEM) solver for electromagnetic scattering by
arbitrarily shaped, high-contrast dielectric particles, written in Julia.

The package implements:
- SWG (Schaubert-Wilton-Glisson) basis on tetrahedral meshes
- Duffy-transform-based singular/near-singular volume integration
- AIM (Adaptive Integral Method) FFT acceleration
- Block-Krylov iterative solvers for multi-orientation batch problems
- CAS-v2 compatible scattering observables (mirroring block-DDA_Py)

See `.claude/technical_note.md` for the formulation reference.
"""
module BlockVIEM

# Phase 0 skeleton: submodules will be populated in subsequent phases.
# Phase 1: Mesh / SWG basis
# Phase 2: Singular volume integration (Duffy)
# Phase 3: AIM (FFT-MVP)
# Phase 4: Block-Krylov solver
# Phase 5: PostProcess (CAS-v2 observables)

end # module BlockVIEM
