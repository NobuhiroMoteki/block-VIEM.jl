# Single sphere, Au (J&C 1972: n = 0.18 + 3.48i) @ λ = 0.638 μm.
# Au sweep is truncated to a_eq ≤ 0.2 μm (CLAUDE.md §2).
# Run:  julia --project=. viem_results/paper/sphere_Au.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "sphere_Au.hdf5");
                m_p       = N_AU,
                a_eq_list = A_EQ_AU,
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(paper) λ=0.638 μm, Au (J&C 1972), sphere")
