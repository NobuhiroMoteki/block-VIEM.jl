# GRE (a/b = b/c = 1, β_gre = 0.2), paper "high" index
# (n = 2.0 + 0.0i) @ λ = 0.638 μm.
# Swapped from n317 at v0.7.5 — gre_n317 slot 4 blew through 215 GB RSS
# and 3 h wall time in pilot; n20 drops that to tractable levels.
# Run:  julia --project=. viem_results/paper/gre_n20.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "gre_n20.hdf5");
                m_p       = N_20,
                a_eq_list = A_EQ_FULL,
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.2,
                light_source = "(paper) λ=0.638 μm, n=2.0+0.0i, GRE β=0.2")
