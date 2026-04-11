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
                 eps_p::Number,
                 eps_bg::Number = 1,
                 k_hat::Vec3,
                 E0::SVector{3,ComplexF64}) -> SolveResult

Solve the VIEM system using a direct dense factorization (LU). This is
`O(N³)` and only practical for small meshes (N ≲ a few hundred). Intended
as a validation reference for the AIM-accelerated iterative solver.
"""
function solve_direct(basis::AbstractDivBasis;
                      k0::Number,
                      eps_p::Number,
                      eps_bg::Number = 1,
                      k_hat::Vec3,
                      E0::SVector{3,ComplexF64})
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
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
                    eps_p::Number,
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
"""
function solve_iterative(basis::AbstractDivBasis;
                         k0::Number,
                         eps_p::Number,
                         eps_bg::Number = 1,
                         k_hat::Vec3,
                         E0::SVector{3,ComplexF64},
                         pitch::Float64,
                         padding::Integer = 3,
                         tol::Float64 = 1e-6,
                         maxiter::Integer = 200)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
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

function LinearAlgebra.mul!(y::AbstractVector, A::_AIMLinOp, x::AbstractVector)
    y .= aim_mvp(A.op, x)
    return y
end
