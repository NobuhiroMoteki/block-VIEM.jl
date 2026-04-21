# Pilot HDF5: 2-sphere doublet, n=1.5+0.01i, a_eq=0.05 μm only.
# Smoke-test for the doublet pipeline (mesh build, run_viem dispatch,
# rhs-scaling diagnostic).
# Run:  julia --project=. viem_results/paper/pilot_doublet.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "pilot_doublet.hdf5");
                m_p        = N_LOW,
                a_eq_list  = [0.05],
                bc_ratio   = 1.0,
                ab_ratio   = 1.0,
                gre_beta   = 0.0,
                shape_kind = "doublet",
                light_source = "(pilot) λ=0.638 μm, n=1.5+0.01i, doublet a_eq=0.05")
