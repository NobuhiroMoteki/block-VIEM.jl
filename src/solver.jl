# VIEM solver: wraps the AIM operator or dense Z matrix with a Krylov
# iterative solver (BiCGSTAB from Krylov.jl) or a direct dense solver
# for small problems.
#
# Phase 4 of BlockVIEM.jl.

import LinearAlgebra
using LinearAlgebra: norm
using Krylov

"""
    SolveResult

Container for the solution and solver metadata.

# Fields
- `D_coeffs::Vector{ComplexF64}` — SWG expansion coefficients of D(r)
- `converged::Bool`              — solver convergence flag
- `residual_norm::Float64`       — final relative residual ‖Ax-b‖/‖b‖
- `iterations::Int`              — number of Krylov iterations (0 for direct)
"""
struct SolveResult
    D_coeffs::Vector{ComplexF64}
    converged::Bool
    residual_norm::Float64
    iterations::Int
end

"""
    solve_direct(basis::AbstractDivBasis;
                 k0::Number,
                 eps_p,
                 eps_bg::Number = 1,
                 k_hat::Vec3,
                 E0::SVector{3,ComplexF64}) -> SolveResult

Solve the VIEM system using a direct dense factorization (LU). This is
`O(N³)` and only practical for small meshes (N ≲ a few hundred). Intended
as a validation reference for the AIM-accelerated iterative solver.

**Convention.** `k0` is the wavenumber **in the background medium**,
i.e. `k0 = 2π·m_m/λ₀` with `m_m` the (real) background refractive index,
matching block-DDA_Py's `self.k`. `eps_p`, `eps_bg` are absolute
permittivities (so for an absorbing particle in air, pass
`k0 = 2π/λ₀`, `eps_p = m_p^2` with `Im(m_p) > 0`, `eps_bg = 1`).
"""
function solve_direct(basis::AbstractDivBasis;
                      k0::Number,
                      eps_p,
                      eps_bg::Number = 1,
                      k_hat::Vec3,
                      E0::SVector{3,ComplexF64})
    # `k0` is the wavenumber in the background medium
    # (k0 = 2π·m_m/λ₀, not the vacuum wavenumber).
    k_bg = ComplexF64(k0)
    b = project_plane_wave(basis;
                           k_hat = k_hat, E0 = E0, k_bg = k_bg)
    Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                  symmetrize = true)
    x = Z \ b
    res = norm(Z * x - b) / norm(b)
    return SolveResult(x, res < 1e-6, res, 0)
end

"""
    solve_iterative(basis::AbstractDivBasis;
                    k0::Number,
                    eps_p,
                    eps_bg::Number = 1,
                    k_hat::Vec3,
                    E0::SVector{3,ComplexF64},
                    pitch::Float64,
                    padding::Integer = 3,
                    tol::Float64 = 1e-6,
                    maxiter::Integer = 200) -> SolveResult

Solve the VIEM system using the AIM-accelerated BiCGSTAB iteration
(via Krylov.jl). The AIM operator is wrapped as a closure that computes
`aim_mvp(op, x)`.

**Convention.** Same as [`solve_direct`](@ref): `k0` is the
background-medium wavenumber (`2π·m_m/λ₀`), and `eps_p`, `eps_bg` are
absolute permittivities.
"""
function solve_iterative(basis::AbstractDivBasis;
                         k0::Number,
                         eps_p,
                         eps_bg::Number = 1,
                         k_hat::Vec3,
                         E0::SVector{3,ComplexF64},
                         pitch::Float64,
                         padding::Integer = 3,
                         tol::Float64 = 1e-6,
                         maxiter::Integer = 200)
    # `k0` is the wavenumber in the background medium
    # (k0 = 2π·m_m/λ₀, not the vacuum wavenumber).
    k_bg = ComplexF64(k0)
    b = project_plane_wave(basis;
                           k_hat = k_hat, E0 = E0, k_bg = k_bg)
    op = build_aim_operator(basis; k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                            pitch = pitch, padding = padding)

    N = n_basis(basis)

    # Krylov.jl's bicgstab accepts a function-based operator via the
    # `opM` interface: any callable A(x) → y or an AbstractMatrix.
    # We use a simple wrapper struct.
    A = _AIMLinOp(op, N)

    x, stats = Krylov.bicgstab(A, b; atol = tol * norm(b), rtol = 0.0,
                                itmax = maxiter)
    res = norm(A * x - b) / norm(b)
    return SolveResult(x, stats.solved, res, stats.niter)
end

# Minimal wrapper to make AIMOperator usable as an AbstractLinearOperator
# by Krylov.jl.
struct _AIMLinOp
    op::AIMOperator
    n::Int
end

Base.size(A::_AIMLinOp) = (A.n, A.n)
Base.size(A::_AIMLinOp, d::Integer) = d ∈ (1, 2) ? A.n : 1
Base.eltype(::_AIMLinOp) = ComplexF64
Base.:*(A::_AIMLinOp, x::AbstractVector) = aim_mvp(A.op, x)
Base.:*(A::_AIMLinOp, X::AbstractMatrix) = aim_mvp(A.op, X)

function LinearAlgebra.mul!(y::AbstractVector, A::_AIMLinOp, x::AbstractVector)
    y .= aim_mvp(A.op, x)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::_AIMLinOp, X::AbstractMatrix)
    Y .= aim_mvp(A.op, X)
    return Y
end

# ---------------------------------------------------------------------------
# Block multi-RHS iterative solve (Phase 5.5)
# ---------------------------------------------------------------------------

"""
    solve_iterative_block(basis::AbstractDivBasis;
                          k0, eps_p, eps_bg = 1,
                          k_hat_list, E0_list,
                          pitch::Float64, padding::Integer = 3,
                          tol::Float64 = 1e-6, maxiter::Integer = 200,
                          method::Symbol = :bicgstab,
                          verbose::Bool = false)
        -> (D_block::Matrix{ComplexF64}, converged::Bool,
            residual::Float64, iterations::Int)

AIM-accelerated block Krylov solve of the VIEM system for `L` right-hand
sides at once, one per `(k_hat, E0)` pair in the provided lists. The
impedance operator is built once, and a single block Krylov iteration
solves `Z * X = B` where each column of `B` is the projection of one
incident plane wave.

`method` selects the block solver:
- `:bicgstab` — Block BiCGSTAB (Tadano et al. 2009). Default.
- `:gmres`    — Unrestarted Block GMRES (Simoncini & Szyld 1996).

**Convention.** Same as [`solve_direct`](@ref): `k0` is the
background-medium wavenumber (`2π·m_m/λ₀`), and `eps_p`, `eps_bg` are
absolute permittivities.
"""
function solve_iterative_block(basis::AbstractDivBasis;
                               k0::Number,
                               eps_p,
                               eps_bg::Number = 1,
                               k_hat_list::AbstractVector,
                               E0_list::AbstractVector,
                               pitch::Float64,
                               padding::Integer = 3,
                               tol::Float64 = 1e-6,
                               maxiter::Integer = 200,
                               method::Symbol = :bicgstab,
                               verbose::Bool = false)
    length(k_hat_list) == length(E0_list) ||
        throw(ArgumentError("k_hat_list and E0_list must have equal length"))
    L = length(k_hat_list)
    L == 0 && throw(ArgumentError("at least one RHS required"))

    # `k0` is the wavenumber in the background medium
    # (k0 = 2π·m_m/λ₀, not the vacuum wavenumber).
    k_bg = ComplexF64(k0)
    N = n_basis(basis)
    B = Matrix{ComplexF64}(undef, N, L)
    for j in 1:L
        B[:, j] = project_plane_wave(basis;
                                     k_hat = k_hat_list[j],
                                     E0 = E0_list[j], k_bg = k_bg)
    end

    op = build_aim_operator(basis; k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                            pitch = pitch, padding = padding)
    A = _AIMLinOp(op, N)

    res = _block_solve(A, B, method; tol = tol, maxiter = maxiter,
                       verbose = verbose)
    return res.X, res.converged, res.residual_norm, res.iterations
end

function _block_solve(A, B, method::Symbol; tol, maxiter, verbose)
    if method === :bicgstab
        return block_bicgstab(A, B; tol = tol, maxiter = maxiter,
                              verbose = verbose)
    elseif method === :gmres
        return block_gmres(A, B; tol = tol, maxiter = maxiter,
                           verbose = verbose)
    else
        throw(ArgumentError("unknown block Krylov method: $method " *
                            "(expected :bicgstab or :gmres)"))
    end
end
