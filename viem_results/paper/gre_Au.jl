# GRE (a/b = b/c = 1, β_gre = 0.2), Au (J&C 1972) @ λ = 0.638 μm.
# Au sweep is truncated to a_eq ≤ 0.2 μm.
# Run:  julia --project=. viem_results/paper/gre_Au.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "gre_Au.hdf5");
                m_p       = N_AU,
                a_eq_list = A_EQ_AU,
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.2,
                light_source = "(paper) λ=0.638 μm, Au (J&C 1972), GRE β=0.2")
