# Pilot HDF5: single sphere, n=2.0+0.0i (paper "high" since v0.7.5), a_eq=0.1 μm only.
# Used to validate the estimate → run_viem → check_h5 small-loop before
# launching the full paper sweep.
# Run:  julia --project=. viem_results/paper/pilot_sphere_n20.jl

include(joinpath(@__DIR__, "_common.jl"))

create_paper_h5(joinpath(@__DIR__, "pilot_sphere_n20.hdf5");
                m_p       = N_20,
                a_eq_list = [0.1],
                bc_ratio  = 1.0,
                ab_ratio  = 1.0,
                gre_beta  = 0.0,
                light_source = "(pilot) λ=0.638 μm, n=2.0+0.0i, sphere a_eq=0.1")
