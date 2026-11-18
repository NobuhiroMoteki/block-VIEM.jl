# Refractive-index sweep of an isotropic, homogeneous particle with one block
# Arnoldi process (shifted block GMRES).
#
# With kappa = (eps_p − eps_bg)/eps_p and tau = eps_bg/(eps_p − eps_bg), the
# impedance operator factors as
#
#     Z(eps_p) = (1/eps_p) M − (kappa/eps_bg) K0 = (kappa/eps_bg) (tau M − K0),
#
# where the mass matrix M (real, sparse, SPD) and the radiation operator K0
# do not depend on eps_p. Right preconditioning with M^{-1} gives
#
#     (tau I − K0 M^{-1}) Yhat = B,      X = (eps_bg/kappa) M^{-1} Yhat,
#
# which is a shift of the fixed operator C = K0 M^{-1}. Since
# B − Z X = B − (tau I − C) Yhat, the residual that GMRES minimises is the
# residual of the original system, `‖B − Z X‖_F / ‖B‖_F`.
#
# The Cholesky factor of M and the block solve with it are in solver.jl.

# C = K0 M^{-1}, the operator handed to `block_gmres_shifted`.
struct _K0MinvOp{F}
    op::AIMOperator
    minv::_MassCholesky{F}
end
_K0MinvOp(op::AIMOperator) = _K0MinvOp(op, _MassCholesky(op.mass))

Base.:*(A::_K0MinvOp, X::AbstractMatrix) = aim_K0_mvp(A.op, _mass_solve(A.minv, X))

# Shift and contrast of an isotropic permittivity.
_shift_tau(eps_p::Number, eps_bg::Number) =
    ComplexF64(eps_bg) / (ComplexF64(eps_p) - ComplexF64(eps_bg))
_shift_kappa(eps_p::Number, eps_bg::Number) =
    (ComplexF64(eps_p) - ComplexF64(eps_bg)) / ComplexF64(eps_p)

# CAS-v2 observables of every column of the solution block D, shape (N, L).
function _cas_observables_block(basis::AbstractDivBasis, D::Matrix{ComplexF64},
                                orientations::Vector{CASOrientation},
                                k0::ComplexF64, eps_p::ComplexF64, eps_bg::ComplexF64,
                                rule::TetQuadRule)
    L = length(orientations)
    out = Vector{CASv2Result}(undef, L)
    Threads.@threads for i in 1:L
        out[i] = compute_cas_observables(basis, view(D, :, i);
                                         orientation = orientations[i],
                                         k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                         rule = rule)
    end
    return out
end

"""
    solve_cas_v2_orientations_shifted(basis::AbstractDivBasis,
                                      euler_angles::AbstractVector;
                                      wl_0::Real, m_m::Real,
                                      m_p_list::AbstractVector,
                                      duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                      outer_rule::TetQuadRule = TET_QUAD_5PT,
                                      ff_rule::TetQuadRule = TET_QUAD_5PT,
                                      pitch::Union{Float64,Nothing} = nothing,
                                      padding::Integer = 4,
                                      tol::Float64 = 1e-5,
                                      maxiter::Integer = 100,
                                      verify::Symbol = :hardest,
                                      return_D::Bool = false,
                                      projection = nothing, mass = nothing,
                                      verbose::Bool = false)
        -> (results::Matrix{CASv2Result}, solve_info::NamedTuple)

Solve the VIEM system of one mesh for `N_m` isotropic refractive indices and
`L` orientations with a single block Arnoldi process, and return the CAS-v2
observables as a matrix of shape `(N_m, L)`.

The counterpart for one refractive index is
[`solve_cas_v2_orientations`](@ref) with `method = :aim_gmres`. Inputs
follow its block-DDA_Py-compatible form: `wl_0` is the vacuum wavelength in
the length unit of the mesh, `m_m` the real refractive index of the medium,
and `m_p_list` the complex particle refractive indices.

# Method

For an isotropic, homogeneous particle

    Z(m_p) = (kappa/eps_bg) (tau M − K0),    tau = 1/((m_p/m_m)^2 − 1),

where the mass matrix `M` and the radiation operator `K0` do not depend on
`m_p`. With right preconditioning by `M⁻¹` every refractive index becomes a
shift of the same operator, `(tau I − K0 M⁻¹) Y = B`, and
[`block_gmres_shifted`](@ref) solves all of them in the block Krylov space of
`K0 M⁻¹` from `B`. The AIM operator is built once, for the index of largest
`|kappa|` in the list. `M⁻¹` is applied with a sparse Cholesky factor. The
convergence criterion of each index is that of the existing solver,
`‖B − Z D‖_F / ‖B‖_F < tol`.

The number of block iterations is set by the slowest index of the list.
Indices of similar contrast should therefore be solved together. The mesh
and the AIM grid are shared, so the mesh must resolve the largest `|m_p|`.

# Keywords

- `verify`: recompute the residual `‖B − Z D‖_F / ‖B‖_F` of a solution from
  `M` and `K0` after the solve (one block MVP per verified index).
  `:hardest` (default) checks the index with the most iterations, `:all`
  checks every index, `:none` skips the check.
- `return_D`: also return the solution blocks, a vector of `N_m` matrices of
  shape `(N, L)`. The return value is then `(results, D, solve_info)`.
- `projection`, `mass`: pre-built AIM projection and mass matrix, as in
  `solve_cas_v2_orientations`.

# Returned `solve_info`

- `iterations::Vector{Int}`, `residual_norm::Vector{Float64}`,
  `converged::Vector{Bool}`: per refractive index.
- `residual_history::Matrix{Float64}`: shape `(n_block_iterations, N_m)`,
  `NaN` after an index has converged.
- `true_residual::Vector{Float64}`: recomputed residual, `NaN` where not
  verified.
- `n_block_iterations::Int`: block iterations of the Arnoldi process, equal
  to the number of block MVPs of the solve.
- `reference_index::Int`: position in `m_p_list` of the index the AIM
  operator was built for.
- `timing`: wall times in seconds, `(rhs, operator, factor, krylov, verify,
  post)`.

Results of an index with `converged == false` are computed from its
minimum-residual iterate at `maxiter`.

Throws `ArgumentError` if `m_p_list` is empty, contains a non-scalar
(anisotropic) entry or an entry equal to `m_m`.
"""
function solve_cas_v2_orientations_shifted(basis::AbstractDivBasis,
        euler_angles::AbstractVector;
        wl_0::Real, m_m::Real, m_p_list::AbstractVector,
        duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
        outer_rule::TetQuadRule = TET_QUAD_5PT,
        ff_rule::TetQuadRule = TET_QUAD_5PT,
        pitch::Union{Float64,Nothing} = nothing,
        padding::Integer = 4,
        tol::Float64 = 1e-5,
        maxiter::Integer = 100,
        verify::Symbol = :hardest,
        return_D::Bool = false,
        projection::Union{AIMProjection,Nothing} = nothing,
        mass::Union{SparseMatrixCSC{Float64,Int},Nothing} = nothing,
        verbose::Bool = false)
    isempty(m_p_list) && throw(ArgumentError("m_p_list is empty"))
    all(m -> m isa Number, m_p_list) || throw(ArgumentError(
        "m_p_list must hold scalar (isotropic) refractive indices; the shifted " *
        "sweep does not apply to anisotropic particles"))
    verify in (:none, :hardest, :all) || throw(ArgumentError(
        "unknown verify: $verify (expected :none, :hardest or :all)"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be ≥ 1, got $maxiter"))
    isempty(euler_angles) && throw(ArgumentError("at least one orientation is required"))

    k0_c = ComplexF64(2π * m_m / wl_0)                # wavenumber in the medium
    eps_bg_c = ComplexF64(m_m)^2
    eps_list = ComplexF64[ComplexF64(m)^2 for m in m_p_list]
    any(==(eps_bg_c), eps_list) && throw(ArgumentError(
        "m_p_list contains m_m (kappa = 0): no scattering problem to solve"))
    S = length(eps_list)
    kappas = [_shift_kappa(e, eps_bg_c) for e in eps_list]
    taus = [_shift_tau(e, eps_bg_c) for e in eps_list]

    orientations = CASOrientation[cas_orientation(ea[1], ea[2], ea[3])
                                  for ea in euler_angles]
    L = length(orientations)
    N = n_basis(basis)

    # ── Right-hand sides, shape (N, L): independent of m_p ──
    B = Matrix{ComplexF64}(undef, N, L)
    t_rhs = @elapsed begin
        tasks = Task[]
        for (i, ori) in enumerate(orientations)
            push!(tasks, Threads.@spawn _project_rhs_column!(
                B, i, basis, ori.u_inc, ori.e0_inc, k0_c, outer_rule))
        end
        foreach(wait, tasks)
    end

    # ── One AIM operator for the whole list ──
    if pitch === nothing && projection === nothing
        pitch = 0.5 * mean_edge_length(basis.mesh)
    end
    s_ref = argmax(abs.(kappas))
    t_op = @elapsed op = build_aim_operator(basis; k0 = k0_c, eps_p = eps_list[s_ref],
                                            eps_bg = eps_bg_c,
                                            pitch = pitch, padding = padding,
                                            outer_rule = outer_rule,
                                            duffy_rule = duffy_rule,
                                            projection = projection, mass = mass)
    t_fact = @elapsed C = _K0MinvOp(op)

    # ── Shifted block GMRES on C = K0 M⁻¹ ──
    t_kry = @elapsed kry = _block_gmres_shifted_core(C, B, taus, tol, Int(maxiter),
                                                     verbose)

    # ── Solutions and observables, one refractive index at a time ──
    check = falses(S)
    if verify === :all
        fill!(check, true)
    elseif verify === :hardest
        check[argmax(kry.iterations)] = true
    end
    Bnorm = norm(B)
    true_res = fill(NaN, S)
    results = Matrix{CASv2Result}(undef, S, L)
    D_all = Matrix{ComplexF64}[]
    t_verify = 0.0
    t_post = 0.0
    for s in 1:S
        # D = (eps_bg/kappa) M⁻¹ Yhat
        D = _mass_solve(C.minv, _shifted_solution(kry, s))
        D .*= eps_bg_c / kappas[s]
        if check[s]
            t_verify += @elapsed begin
                ZD = op.mass * D                      # shape: (N, L)
                ZD .*= 1 / eps_list[s]
                ZD .-= (kappas[s] / eps_bg_c) .* aim_K0_mvp(op, D)
                ZD .-= B
                true_res[s] = Bnorm == 0 ? 0.0 : norm(ZD) / Bnorm
            end
        end
        t_post += @elapsed begin
            @views results[s, :] .= _cas_observables_block(
                basis, D, orientations, k0_c, eps_list[s], eps_bg_c, ff_rule)
        end
        return_D && push!(D_all, D)
    end

    solve_info = (iterations = kry.iterations,
                  residual_norm = kry.residual_norm,
                  converged = kry.converged,
                  residual_history = kry.residual_history,
                  true_residual = true_res,
                  n_block_iterations = size(kry.residual_history, 1),
                  reference_index = s_ref,
                  timing = (rhs = t_rhs, operator = t_op, factor = t_fact,
                            krylov = t_kry, verify = t_verify, post = t_post))
    verbose && @info "solve_cas_v2_orientations_shifted" n_m = S L = L iterations =
        kry.iterations converged = count(kry.converged)
    return return_D ? (results, D_all, solve_info) : (results, solve_info)
end
