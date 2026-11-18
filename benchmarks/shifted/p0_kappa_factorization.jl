# Phase 0, items 1 and 2: does kappa factor out of the radiation operator?
#
#   Z(eps_p) = (1/eps_p) M - (kappa/eps_bg) K0,  with K0 independent of eps_p.
#
# Item 1 (dense): assemble Z for several isotropic eps_p on the coarse unit
#   sphere of test/meshes (half-SWG boundary DOFs included) and compare the K0
#   recovered from each. A second check does not use the assembled M at all:
#   M and K0 are solved from two Z matrices and must reproduce a third one.
# Item 2 (AIM): same statement for the AIM operator acting on a random block.
#
# Run: julia --project=. -t 8 benchmarks/shifted/p0_kappa_factorization.jl

include(joinpath(@__DIR__, "_common.jl"))

const EPS_LIST = ComplexF64[2.25 + 0.03im, 9.0 + 1.0im, -12.1 + 1.22im]
const K0_TEST = 2.0     # background wavenumber for the unit sphere (x = 2)

function dense_item(basis, eps_bg, symmetrize)
    M = Matrix(assemble_mass_matrix(basis))                    # shape: (N, N)
    Zs = [assemble_impedance_matrix(basis; k0 = K0_TEST, eps_p = e,
                                    eps_bg = eps_bg, symmetrize = symmetrize)
          for e in EPS_LIST]
    K0s = [(eps_bg / kappa_of(e, eps_bg)) .* (M ./ e .- Z)
           for (e, Z) in zip(EPS_LIST, Zs)]
    d12 = relerr(K0s[1], K0s[2])
    d13 = relerr(K0s[1], K0s[3])

    # M-free check: Z_i = a_i M - c_i K0 with a = 1/eps_p, c = kappa/eps_bg.
    a = [1 / e for e in EPS_LIST]
    c = [kappa_of(e, eps_bg) / eps_bg for e in EPS_LIST]
    det = -a[1] * c[2] + a[2] * c[1]
    Mfit  = (-c[2] .* Zs[1] .+ c[1] .* Zs[2]) ./ det
    K0fit = (-a[2] .* Zs[1] .+ a[1] .* Zs[2]) ./ det
    d3 = relerr(a[3] .* Mfit .- c[3] .* K0fit, Zs[3])
    dM = relerr(Mfit, M)
    return (eps_bg = real(eps_bg), symmetrize = symmetrize,
            K0_diff_12 = d12, K0_diff_13 = d13,
            Z3_pred_from_Z1_Z2 = d3, M_fit_vs_assembled = dM), K0s[1]
end

function aim_item(basis, eps_bg, K0_dense)
    N = n_basis(basis)
    pitch = 0.5 * mean_edge_length(basis.mesh)
    grid = aim_grid(basis.mesh; pitch = pitch, padding = 4)
    proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass = assemble_mass_matrix(basis)
    ops = [build_aim_operator(basis; k0 = K0_TEST, eps_p = e, eps_bg = eps_bg,
                              projection = proj, mass = mass) for e in EPS_LIST]
    X = randn(MersenneTwister(7), ComplexF64, N, 4)            # shape: (N, 4)
    Ya = [k0_mvp(op, X) for op in ops]
    Yb = [k0_mvp_by_difference(op, X) for op in ops]
    return (eps_bg = real(eps_bg),
            K0X_diff_12 = relerr(Ya[1], Ya[2]),
            K0X_diff_13 = relerr(Ya[1], Ya[3]),
            method_a_vs_b = [relerr(Ya[i], Yb[i]) for i in eachindex(ops)],
            aim_vs_dense_K0X = relerr(Ya[1], K0_dense * X))
end

function main()
    mesh = read_msh(joinpath(@__DIR__, "..", "..", "test", "meshes", "sphere.msh"))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    n_bnd = count(basis.is_boundary)
    @printf("unit sphere: N_tet = %d, N_DOF = %d (boundary half-SWG: %d), k0 = %.1f\n",
            n_tets(mesh), N, n_bnd, K0_TEST)
    println("eps_p list: ", EPS_LIST)

    dense = NamedTuple[]
    aim = NamedTuple[]
    for eps_bg in (ComplexF64(1.0), ComplexF64(1.33^2))
        local K0d
        for symmetrize in (false, true)
            r, K0d = dense_item(basis, eps_bg, symmetrize)
            push!(dense, r)
            @printf("dense eps_bg=%.4f sym=%-5s  K0 diff (1,2) %.2e  (1,3) %.2e  | Z3 predicted %.2e  | M fit %.2e\n",
                    r.eps_bg, r.symmetrize, r.K0_diff_12, r.K0_diff_13,
                    r.Z3_pred_from_Z1_Z2, r.M_fit_vs_assembled)
        end
        r = aim_item(basis, eps_bg, K0d)
        push!(aim, r)
        @printf("AIM   eps_bg=%.4f            K0 X diff (1,2) %.2e  (1,3) %.2e  | a vs b %s  | AIM vs dense %.2e\n",
                r.eps_bg, r.K0X_diff_12, r.K0X_diff_13,
                join([@sprintf("%.1e", v) for v in r.method_a_vs_b], " "),
                r.aim_vs_dense_K0X)
    end
    worst_dense = maximum(max(r.K0_diff_12, r.K0_diff_13, r.Z3_pred_from_Z1_Z2) for r in dense)
    worst_aim = maximum(max(r.K0X_diff_12, r.K0X_diff_13) for r in aim)
    @printf("\nworst dense = %.2e, worst AIM = %.2e (criterion 1e-10): %s\n",
            worst_dense, worst_aim,
            (worst_dense <= 1e-10 && worst_aim <= 1e-10) ? "PASS" : "FAIL")
    save_json("p0_kappa_factorization.json",
              (n_tet = n_tets(mesh), n_dof = N, n_boundary = n_bnd, k0 = K0_TEST,
               eps_list = [(real(e), imag(e)) for e in EPS_LIST],
               dense = dense, aim = aim,
               worst_dense = worst_dense, worst_aim = worst_aim))
end

main()
