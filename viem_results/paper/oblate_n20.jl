# Oblate spheroid (b/c = 3, a/b = 1), paper "high" index
# (n = 2.0 + 0.0i) @ λ = 0.638 μm.
# Swapped from n317 at v0.7.5 (see sphere_n20.jl for rationale).
# Run:  julia --project=. viem_results/paper/oblate_n20.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "oblate_n20.hdf5");
                m_p       = N_20,
                a_eq_list = A_EQ_FULL,
                bc_ratio  = 3.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(paper) λ=0.638 μm, n=2.0+0.0i, oblate b/c=3")
