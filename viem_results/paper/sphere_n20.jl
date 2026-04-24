# Single sphere, paper "high" index (n = 2.0 + 0.0i) @ λ = 0.638 μm.
# Swapped from n317 (3.17+0.16i) at v0.7.5 to keep VIEM wall time per
# slot manageable (n317 × r_v=0.4 on DDA spiked to ~215 GB RSS / 3.3 h).
# Run:  julia --project=. viem_results/paper/sphere_n20.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "sphere_n20.hdf5");
                m_p       = N_20,
                a_eq_list = A_EQ_FULL,
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(paper) λ=0.638 μm, n=2.0+0.0i, sphere")
