# Oblate spheroid (b/c = 3, a/b = 1), high-index (n = 3.17 + 0.16i) @ λ = 0.638 μm.
# Run:  julia --project=. viem_results/paper/oblate_n317.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "oblate_n317.hdf5");
                m_p       = N_HIGH,
                a_eq_list = A_EQ_FULL,
                bc_ratio  = 3.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(paper) λ=0.638 μm, n=3.17+0.16i, oblate b/c=3")
