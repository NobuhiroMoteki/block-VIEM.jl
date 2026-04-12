# Block Krylov solvers for multi-RHS linear systems A * X = B, where
# `X`, `B` are (N × L) complex matrices and `A` is any linear operator
# supporting the `*(A, X::AbstractMatrix)` interface (e.g. `AIMOperator`
# wrapped in `_AIMLinOp`).
#
# Implemented:
#
#   block_bicgstab  — Block BiCGSTAB(1) following
#                     Tadano, Sakurai & Kuramashi (JSIAM Lett. 2009).
#                     Direct port of block-DDA_Py's
#                     `bl_bicgstab_jacobi_mvp_fft` (bl_krylov/bl_krylov.py)
#                     without the Jacobi preconditioner — the VIEM
#                     operator already produces well-scaled systems
#                     through the mass-matrix normalization.
#
#   block_gmres     — unrestarted Block GMRES via block-Arnoldi with thin
#                     QR of each new block Krylov vector. Follows
#                     Simoncini & Szyld (Num. Lin. Alg. 1996). Useful as
#                     a fallback when BiCGSTAB breakdowns or stagnates.
#
# Both solvers only need `A * X` (block MVP) — no transpose and no
# preconditioner. The calling convention matches
# `Krylov.bicgstab` in spirit: returns a `BlockSolveResult` containing
# the solution block, convergence flag, relative residual, and iteration
# count.

using LinearAlgebra
using LinearAlgebra: norm, mul!, tr

"""
    BlockSolveResult

Return type for [`block_bicgstab`](@ref) and [`block_gmres`](@ref).

# Fields
- `X::Matrix{ComplexF64}`     — solution block, shape `(N, L)`
- `converged::Bool`           — whether the solver reached `tol`
- `residual_norm::Float64`    — final relative residual `‖B − A X‖_F / ‖B‖_F`
- `iterations::Int`           — number of outer iterations taken
"""
struct BlockSolveResult
    X::Matrix{ComplexF64}
    converged::Bool
    residual_norm::Float64
    iterations::Int
end

"""
    block_bicgstab(A, B::AbstractMatrix; tol = 1e-6, maxiter = 200,
                   verbose = false) -> BlockSolveResult

Block BiCGSTAB for `A X = B`, with `X`, `B` of shape `(N, L)`. `A` must
support `A * P::AbstractMatrix` returning an `(N, L)` block.

Direct port of block-DDA_Py's `bl_bicgstab_jacobi_mvp_fft`
(Tadano, Sakurai, Kuramashi 2009, JSIAM Lett.) without the Jacobi
preconditioner. The Jacobi preconditioner in the DDA code is cheap
because the dipole self-term is trivially diagonal; for VIEM the AIM
operator's diagonal costs `O(N)` FFT evaluations to extract, so we run
the unpreconditioned variant — the mass matrix already normalizes the
system.

The columns of `B` should be linearly independent; if two columns are
parallel, Block BiCGSTAB can break down in the `L×L` solve on `RV` — use
[`block_gmres`](@ref) in that case.
"""
function block_bicgstab(A, B::AbstractMatrix; tol::Real = 1e-6,
                        maxiter::Integer = 200, verbose::Bool = false)
    N, L = size(B)
    Bc = Matrix{ComplexF64}(B)
    Bnorm = norm(Bc)
    Bnorm == 0 && return BlockSolveResult(zeros(ComplexF64, N, L), true, 0.0, 0)

    X = zeros(ComplexF64, N, L)
    R = copy(Bc)
    P = copy(R)
    R0 = copy(R)           # shadow residual

    err = Inf
    iter = 0
    for k in 1:maxiter
        iter = k
        V = A * P                     # N × L
        RV = R0' * V                  # L × L
        alpha = RV \ (R0' * R)        # L × L
        T = R - V * alpha             # N × L

        Zblk = A * T                  # N × L
        ZHT = tr(Zblk' * T)
        ZHZ = tr(Zblk' * Zblk)
        qsi = ZHT / ZHZ               # complex scalar

        X .= X .+ P * alpha .+ qsi .* T
        R .= T .- qsi .* Zblk

        err = norm(R) / Bnorm
        verbose && @info "block_bicgstab" iter = k err = err
        if err < tol
            return BlockSolveResult(X, true, err, k)
        end

        beta = RV \ (-(R0' * Zblk))   # L × L
        P .= R .+ (P .- qsi .* V) * beta
    end
    return BlockSolveResult(X, err < tol, err, iter)
end

"""
    block_gmres(A, B::AbstractMatrix; tol = 1e-6, maxiter = 200,
                verbose = false) -> BlockSolveResult

Unrestarted Block GMRES for `A X = B`, with `X`, `B` of shape `(N, L)`.
Builds a block Krylov basis `V_1, V_2, …` via block-Arnoldi with thin
QR factorization of each new block MVP result (Simoncini & Szyld,
NLAA 1996). Each outer iteration performs exactly one block MVP
`W = A * V_k`.

Memory cost is `O(maxiter · N · L)` — the full block Krylov basis is
retained. For moderate L and ~a few hundred iterations this is fine;
for very long runs, use [`block_bicgstab`](@ref) instead.

Intended as a robust fallback when BiCGSTAB breakdowns / stagnates on
rank-deficient RHS blocks.
"""
function block_gmres(A, B::AbstractMatrix; tol::Real = 1e-6,
                     maxiter::Integer = 200, verbose::Bool = false)
    N, L = size(B)
    Bc = Matrix{ComplexF64}(B)
    Bnorm = norm(Bc)
    Bnorm == 0 && return BlockSolveResult(zeros(ComplexF64, N, L), true, 0.0, 0)

    # Initial residual (X0 = 0) and its thin QR: R0 = V1 * Λ.
    Qfac = qr(Bc)
    V1 = Matrix{ComplexF64}(Qfac.Q)[:, 1:L]  # N × L
    Λ  = Matrix{ComplexF64}(Qfac.R)          # L × L (upper triangular)

    Vblocks = Vector{Matrix{ComplexF64}}()
    push!(Vblocks, V1)
    # H is stored as a dense (maxiter+1)·L × maxiter·L upper block-Hessenberg
    # matrix, grown column-block by column-block.
    Hfull = zeros(ComplexF64, (maxiter + 1) * L, maxiter * L)

    err = Inf
    iter = 0
    for k in 1:maxiter
        iter = k
        W = A * Vblocks[k]               # N × L
        # Block Gram–Schmidt against prior blocks
        for i in 1:k
            Hik = Vblocks[i]' * W        # L × L
            Hfull[(i - 1) * L + 1:i * L, (k - 1) * L + 1:k * L] .= Hik
            W .-= Vblocks[i] * Hik
        end
        # Thin QR of the remaining block
        Qfk = qr(W)
        Vk1 = Matrix{ComplexF64}(Qfk.Q)[:, 1:L]
        Hk1k = Matrix{ComplexF64}(Qfk.R)
        Hfull[k * L + 1:(k + 1) * L, (k - 1) * L + 1:k * L] .= Hk1k
        push!(Vblocks, Vk1)

        # Solve least-squares  min ‖ H_k · Y − E1·Λ ‖_F  where
        # H_k is the current (k+1)L × kL sub-matrix and the RHS top block
        # is Λ (others are zero). We solve it each iteration — cheap for
        # small k.
        m = (k + 1) * L
        n = k * L
        Hk_view = @view Hfull[1:m, 1:n]
        rhs = zeros(ComplexF64, m, L)
        rhs[1:L, :] .= Λ
        Y = Hk_view \ rhs
        resblk = Hk_view * Y - rhs
        err = norm(resblk) / Bnorm
        verbose && @info "block_gmres" iter = k err = err
        if err < tol || k == maxiter
            X = zeros(ComplexF64, N, L)
            for j in 1:k
                X .+= Vblocks[j] * Y[(j - 1) * L + 1:j * L, :]
            end
            return BlockSolveResult(X, err < tol, err, k)
        end
    end
    # Unreachable — the loop returns on the final iteration.
    return BlockSolveResult(zeros(ComplexF64, N, L), false, err, iter)
end
