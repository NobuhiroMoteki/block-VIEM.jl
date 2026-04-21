# 2-sphere doublet, low-index (n = 1.5 + 0.01i) @ λ = 0.638 μm.
# Touching equal-sphere doublet aligned along particle z; monomer
# radius R = a_eq / 2^(1/3); gap = 0.1 R (CLAUDE.md §2 / README §Benchmark).
# Run:  julia --project=. viem_results/paper/doublet_n15.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "doublet_n15.hdf5");
                m_p        = N_LOW,
                a_eq_list  = A_EQ_FULL,
                bc_ratio   = 1.0,
                ab_ratio   = 1.0,
                gre_beta   = 0.0,
                shape_kind = "doublet",
                light_source = "(paper) λ=0.638 μm, n=1.5+0.01i, doublet (gap=0.1R)")
