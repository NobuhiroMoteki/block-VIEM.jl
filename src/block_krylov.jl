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
    block_bicgstab(A, B::AbstractMatrix; tol = 1e-5, maxiter = 100,
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
function block_bicgstab(A, B::AbstractMatrix; tol::Real = 1e-5,
                        maxiter::Integer = 100, verbose::Bool = false)
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
    block_gmres(A, B::AbstractMatrix; tol = 1e-5, maxiter = 100,
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
function block_gmres(A, B::AbstractMatrix; tol::Real = 1e-5,
                     maxiter::Integer = 100, verbose::Bool = false)
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

    # Incremental block-Givens QR of the block-Hessenberg H.
    #
    # Previous implementation rebuilt and re-solved an (k+1)L × kL
    # least-squares problem every iteration (O((kL)^3) each), which
    # dominated the per-iter cost for large L and large k in
    # paper-production runs.  We instead maintain:
    #
    #   R       — kL × kL upper-triangular factor of the block-Hessenberg
    #             H after rolling block-Givens triangularisation
    #   b_hat   — (maxiter+1)L × L transformed RHS; starts as [Λ; 0; …]
    #             and rotations are applied as each new block column of H
    #             is generated.  The residual Frobenius norm at iteration
    #             k equals ‖b_hat[kL+1:(k+1)L, :]‖.
    #   Qstore  — one 2L × 2L orthogonal Q per iteration k, obtained from
    #             QR of the 2L × L "super-block" [H_{kk}; H_{k+1,k}].
    #             Applied from the left to future H-columns to propagate
    #             the triangularisation, and to the corresponding row
    #             range of b_hat.
    #
    # Per-iter cost drops from O((kL)^3) for the LS solve to O(kL^2) for
    # rotation propagation + one 2L × L block QR (O(L^3)).
    R_tri   = zeros(ComplexF64, maxiter * L, maxiter * L)
    b_hat   = zeros(ComplexF64, (maxiter + 1) * L, L)
    @views b_hat[1:L, :] .= Λ
    Qstore  = Vector{Matrix{ComplexF64}}(undef, maxiter)

    err = Inf
    iter = 0
    for k in 1:maxiter
        iter = k
        W = A * Vblocks[k]               # N × L

        # ── Block Gram–Schmidt ──────────────────────────────────────
        H_col = zeros(ComplexF64, (k + 1) * L, L)
        for i in 1:k
            Hik = Vblocks[i]' * W        # L × L
            @views H_col[(i - 1) * L + 1 : i * L, :] .= Hik
            W .-= Vblocks[i] * Hik
        end
        # Thin QR of the remaining block → V_{k+1}, H_{k+1,k}
        Qfk  = qr(W)
        Vk1  = Matrix{ComplexF64}(Qfk.Q)[:, 1:L]
        Hk1k = Matrix{ComplexF64}(Qfk.R)
        @views H_col[k * L + 1 : (k + 1) * L, :] .= Hk1k
        push!(Vblocks, Vk1)

        # ── Apply previously stored block-Givens rotations ──────────
        # Rotation j acts on rows (j-1)L+1 : (j+1)L of any H-column.
        for j in 1:k-1
            rng = (j - 1) * L + 1 : (j + 1) * L
            @views H_col[rng, :] = Qstore[j]' * H_col[rng, :]
        end

        # ── New block-Givens to zero the sub-diagonal L×L block ────
        # QR the 2L × L super-block [H_{kk}; H_{k+1,k}].  Q is 2L × 2L
        # orthogonal; R's top L×L is the new diagonal block of R_tri.
        # Multiplying `Qs.Q` by a `(2L × 2L)` identity forces materialisation
        # of the full Q — `Matrix(Qs.Q)` on modern Julia returns the thin
        # form by default, which would break the subsequent rotations of
        # b_hat (off-diagonal block-rows must be rotated, not discarded).
        super_rng = (k - 1) * L + 1 : (k + 1) * L
        Qs        = qr(@view H_col[super_rng, :])
        Qmat      = Matrix{ComplexF64}(Qs.Q * I(2 * L))  # 2L × 2L full Q
        Rmat      = Matrix{ComplexF64}(Qs.R)             # L × L (thin R)
        Qstore[k] = Qmat
        @views H_col[(k - 1) * L + 1 : k * L, :] .= Rmat
        @views H_col[k * L + 1 : (k + 1) * L, :] .= 0

        # Store k-th block column of R_tri (only first k block-rows non-zero).
        @views R_tri[1 : k * L, (k - 1) * L + 1 : k * L] .= H_col[1 : k * L, :]

        # Apply new rotation to b_hat on the same row range.
        @views b_hat[super_rng, :] = Qmat' * b_hat[super_rng, :]

        # Residual Frobenius norm: rows kL+1:(k+1)L of b_hat.
        err = norm(@view b_hat[k * L + 1 : (k + 1) * L, :]) / Bnorm
        verbose && @info "block_gmres" iter = k err = err

        if err < tol || k == maxiter
            # Upper-triangular solve R_k · Y = b_hat[1:kL, :]
            R_top = UpperTriangular(@view R_tri[1 : k * L, 1 : k * L])
            Y_sol = R_top \ @view b_hat[1 : k * L, :]
            # Reconstruct X = Σ_j V_j * Y_j
            X = zeros(ComplexF64, N, L)
            for j in 1:k
                @views X .+= Vblocks[j] * Y_sol[(j - 1) * L + 1 : j * L, :]
            end
            return BlockSolveResult(X, err < tol, err, k)
        end
    end
    # Unreachable — the loop returns on the final iteration.
    return BlockSolveResult(zeros(ComplexF64, N, L), false, err, iter)
end
