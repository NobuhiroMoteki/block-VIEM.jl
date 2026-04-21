# Single sphere, low-index dielectric (n = 1.5 + 0.01i) @ λ = 0.638 μm.
# Run:  julia --project=. viem_results/paper/sphere_n15.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "sphere_n15.hdf5");
                m_p       = N_LOW,
                a_eq_list = A_EQ_FULL,
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(paper) λ=0.638 μm, n=1.5+0.01i, sphere")
