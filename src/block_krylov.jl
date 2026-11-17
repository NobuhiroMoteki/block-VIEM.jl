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
- `X::Matrix{ComplexF64}`               — solution block, shape `(N, L)`
- `converged::Bool`                     — whether the solver reached `tol`
- `residual_norm::Float64`              — final relative residual `‖B − A X‖_F / ‖B‖_F`
- `iterations::Int`                     — number of outer iterations taken
- `residual_history::Vector{Float64}`   — relative residual after each
  outer iteration (length `iterations`).  Enables convergence-profile
  figures; symmetric with `block-DDA_Py`'s `err_history` returned by
  `bl_bicgstab_mvp_fft` / `bl_gmres_mvp_fft` (paper-production parity,
  v0.7.5+).
"""
struct BlockSolveResult
    X::Matrix{ComplexF64}
    converged::Bool
    residual_norm::Float64
    iterations::Int
    residual_history::Vector{Float64}
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
    Bnorm == 0 && return BlockSolveResult(
        zeros(ComplexF64, N, L), true, 0.0, 0, Float64[])

    X = zeros(ComplexF64, N, L)
    R = copy(Bc)
    P = copy(R)
    R0 = copy(R)           # shadow residual

    err = Inf
    iter = 0
    residual_history = Float64[]
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
        push!(residual_history, err)
        verbose && @info "block_bicgstab" iter = k err = err
        if err < tol
            return BlockSolveResult(X, true, err, k, residual_history)
        end

        beta = RV \ (-(R0' * Zblk))   # L × L
        P .= R .+ (P .- qsi .* V) * beta
    end
    return BlockSolveResult(X, err < tol, err, iter, residual_history)
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
    Bnorm == 0 && return BlockSolveResult(
        zeros(ComplexF64, N, L), true, 0.0, 0, Float64[])

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
    residual_history = Float64[]
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
        push!(residual_history, err)
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
            return BlockSolveResult(X, err < tol, err, k, residual_history)
        end
    end
    # Unreachable — the loop returns on the final iteration.
    return BlockSolveResult(zeros(ComplexF64, N, L), false, err, iter, residual_history)
end

# ---------------------------------------------------------------------------
# Shifted block GMRES: one block Arnoldi process for a family of shifts
# ---------------------------------------------------------------------------

"""
    ShiftedBlockSolveResult

Return type of [`block_gmres_shifted`](@ref). `S` is the number of shifts.

# Fields
- `X::Vector{Matrix{ComplexF64}}`     — `X[s]` solves `(tau_s I − C) X = B`, shape `(N, L)`
- `converged::Vector{Bool}`           — whether shift `s` reached `tol`
- `residual_norm::Vector{Float64}`    — final `‖B − (tau_s I − C) X[s]‖_F / ‖B‖_F`
- `iterations::Vector{Int}`           — block iterations used by shift `s`
- `residual_history::Matrix{Float64}` — shape `(k_fin, S)`; row `k` holds the
  relative residual of every shift after block iteration `k`. Entries after
  the convergence of a shift are `NaN`.
"""
struct ShiftedBlockSolveResult
    X::Vector{Matrix{ComplexF64}}
    converged::Vector{Bool}
    residual_norm::Vector{Float64}
    iterations::Vector{Int}
    residual_history::Matrix{Float64}
end

# Outcome of the shared block Arnoldi process before the solutions are formed.
# `V[j]` is the j-th orthonormal basis block, shape (N, L). `Y[s]` holds the
# coefficients of shift `s`, shape (iterations[s] * L, L), so that
# X_s = sum_j V[j] * Y[s][(j-1)L+1 : jL, :]. Keeping this form lets a caller
# build one solution at a time instead of holding all S of them.
struct _ShiftedKrylov
    V::Vector{Matrix{ComplexF64}}
    Y::Vector{Matrix{ComplexF64}}
    converged::Vector{Bool}
    residual_norm::Vector{Float64}
    iterations::Vector{Int}
    residual_history::Matrix{Float64}
end

# col[1:(j+1)L, :] = j-th block column of (tau * Ibar − Hbar).
function _shifted_column!(col::Matrix{ComplexF64}, Hc::Matrix{ComplexF64},
                          tau::ComplexF64, j::Int, L::Int)
    n = (j + 1) * L
    @inbounds for c in 1:L
        for r in 1:n
            col[r, c] = -Hc[r, c]
        end
        col[(j - 1) * L + c, c] += tau
    end
    return col
end

# Apply the stored block-Givens rotations 1..nrot to a block column.
# Rotation j acts on rows (j-1)L+1 : (j+1)L. `tmp` has shape (2L, L).
function _apply_block_rotations!(col::Matrix{ComplexF64},
                                 Q::Vector{Matrix{ComplexF64}}, nrot::Int, L::Int,
                                 tmp::Matrix{ComplexF64})
    for j in 1:nrot
        rows = (j - 1) * L + 1 : (j + 1) * L
        blk = view(col, rows, :)
        mul!(tmp, Q[j]', blk)
        blk .= tmp
    end
    return col
end

# Coefficients of one shift after k block iterations. The triangular factor
# is not kept during the iteration (it would cost (kL)^2 per shift); it is
# rebuilt here from the shared Hessenberg columns and the rotations of the shift.
function _shifted_coefficients(Hcols::Vector{Matrix{ComplexF64}},
                               Q::Vector{Matrix{ComplexF64}},
                               gtop::Vector{Matrix{ComplexF64}},
                               tau::ComplexF64, k::Int, L::Int)
    n = k * L
    R = zeros(ComplexF64, n, n)                       # shape: (kL, kL)
    col = Matrix{ComplexF64}(undef, (k + 1) * L, L)
    tmp = Matrix{ComplexF64}(undef, 2 * L, L)
    G = Matrix{ComplexF64}(undef, n, L)               # shape: (kL, L)
    for j in 1:k
        _shifted_column!(col, Hcols[j], tau, j, L)
        _apply_block_rotations!(col, Q, j, L, tmp)
        @views R[1 : j * L, (j - 1) * L + 1 : j * L] .= col[1 : j * L, :]
        @views G[(j - 1) * L + 1 : j * L, :] .= gtop[j]
    end
    ldiv!(UpperTriangular(R), G)
    return G
end

# Solution block of shift `s`: X_s = sum_j V[j] * Y[s]_j, shape (N, L).
function _shifted_solution(kry::_ShiftedKrylov, s::Integer)
    V1 = kry.V[1]
    L = size(V1, 2)
    X = zeros(ComplexF64, size(V1))
    Ys = kry.Y[s]
    for j in 1:kry.iterations[s]
        @views mul!(X, kry.V[j], Ys[(j - 1) * L + 1 : j * L, :], one(ComplexF64), one(ComplexF64))
    end
    return X
end

@noinline function _log_shifted_gmres(k::Int, n_active::Int, err_max::Float64)
    @info "block_gmres_shifted" iter = k active = n_active err_max = err_max
    return nothing
end

function _block_gmres_shifted_core(C, B::Matrix{ComplexF64}, taus::Vector{ComplexF64},
                                   tol::Float64, maxiter::Int, verbose::Bool)
    N, L = size(B)
    S = length(taus)
    Bnorm = norm(B)
    converged = fill(false, S)
    resnorm = fill(NaN, S)
    iters = zeros(Int, S)
    Ycoef = [zeros(ComplexF64, 0, L) for _ in 1:S]
    if Bnorm == 0 || S == 0
        fill!(converged, true)
        fill!(resnorm, 0.0)
        return _ShiftedKrylov([zeros(ComplexF64, N, L)], Ycoef, converged, resnorm,
                              iters, zeros(Float64, 0, S))
    end

    # B = V_1 Λ. The Krylov space of C from B is shared by every shift because
    # (tau I − C) V_k = V_{k+1} (tau Ibar_k − Hbar_k).
    Qfac = qr(B)
    Λ = Matrix{ComplexF64}(Qfac.R)                    # shape: (L, L)
    V = Matrix{ComplexF64}[Matrix{ComplexF64}(Qfac.Q)[:, 1:L]]
    Hcols = Matrix{ComplexF64}[]                      # Hcols[k]: ((k+1)L, L)

    # Per-shift state of the incremental block-Givens QR of (tau Ibar − Hbar):
    # the rotations, the finished blocks of the rotated right-hand side and its
    # last block, whose Frobenius norm is the residual norm.
    Qrot = [Matrix{ComplexF64}[] for _ in 1:S]        # Qrot[s][j]: (2L, 2L)
    gtop = [Matrix{ComplexF64}[] for _ in 1:S]        # gtop[s][j]: (L, L)
    glast = [copy(Λ) for _ in 1:S]                    # glast[s]:   (L, L)

    tmp = Matrix{ComplexF64}(undef, 2 * L, L)
    g2 = Matrix{ComplexF64}(undef, 2 * L, L)
    sup = Matrix{ComplexF64}(undef, 2 * L, L)
    active = collect(1:S)
    history = Vector{Float64}[]

    for k in 1:maxiter
        W = convert(Matrix{ComplexF64}, C * V[k])     # shape: (N, L)
        size(W) == (N, L) || throw(DimensionMismatch(
            "C * V returned $(size(W)), expected $((N, L))"))

        # ── Block Gram–Schmidt and thin QR (same steps as block_gmres) ──
        Hc = zeros(ComplexF64, (k + 1) * L, L)
        for i in 1:k
            Hik = V[i]' * W                           # shape: (L, L)
            @views Hc[(i - 1) * L + 1 : i * L, :] .= Hik
            mul!(W, V[i], Hik, -one(ComplexF64), one(ComplexF64))
        end
        Qfk = qr!(W)
        push!(V, Matrix{ComplexF64}(Qfk.Q)[:, 1:L])
        @views Hc[k * L + 1 : (k + 1) * L, :] .= Qfk.R
        push!(Hcols, Hc)

        # ── Advance the small QR of every unconverged shift ──
        errs = fill(NaN, S)
        col = Matrix{ComplexF64}(undef, (k + 1) * L, L)
        still_active = Int[]
        for s in active
            tau = taus[s]
            _shifted_column!(col, Hc, tau, k, L)
            _apply_block_rotations!(col, Qrot[s], k - 1, L, tmp)

            # New rotation from the QR of the (2L, L) block [H_kk; H_{k+1,k}].
            @views sup .= col[(k - 1) * L + 1 : (k + 1) * L, :]
            F = qr!(sup)
            Qm = zeros(ComplexF64, 2 * L, 2 * L)
            @inbounds for d in 1:2*L
                Qm[d, d] = 1
            end
            lmul!(F.Q, Qm)                            # full (2L, 2L) Q
            push!(Qrot[s], Qm)

            @views g2[1:L, :] .= glast[s]
            @views g2[L+1:2*L, :] .= 0
            mul!(tmp, Qm', g2)
            push!(gtop[s], tmp[1:L, :])
            @views glast[s] .= tmp[L+1:2*L, :]

            err = norm(glast[s]) / Bnorm
            errs[s] = err
            if err < tol || k == maxiter
                Ycoef[s] = _shifted_coefficients(Hcols, Qrot[s], gtop[s], tau, k, L)
                iters[s] = k
                resnorm[s] = err
                converged[s] = err < tol
                empty!(Qrot[s])
                empty!(gtop[s])
            else
                push!(still_active, s)
            end
        end
        push!(history, errs)
        verbose && _log_shifted_gmres(k, length(still_active),
                                      maximum(errs[s] for s in active))
        active = still_active
        isempty(active) && break
    end

    k_fin = length(history)
    hist = Matrix{Float64}(undef, k_fin, S)           # shape: (k_fin, S)
    for k in 1:k_fin
        @views hist[k, :] .= history[k]
    end
    return _ShiftedKrylov(V, Ycoef, converged, resnorm, iters, hist)
end

"""
    block_gmres_shifted(C, B::AbstractMatrix, taus::AbstractVector{<:Number};
                        tol = 1e-5, maxiter = 100, verbose = false)
        -> ShiftedBlockSolveResult

Unrestarted block GMRES for the family of shifted systems

    (tau_s I − C) X_s = B,        s = 1, …, S,

with `B` of shape `(N, L)`. `C` is any operator that supports
`C * V::AbstractMatrix` and returns an `(N, L)` block. One block Arnoldi
process on `C` serves all shifts, because the block Krylov space of
`tau I − C` from `B` does not depend on `tau`. Each block iteration costs
one product `C * V`. The Gram–Schmidt step and the thin QR are the ones of
[`block_gmres`](@ref).

Every shift minimises its own residual over the shared space. A shift is
frozen at the first iteration where `‖B − (tau_s I − C) X_s‖_F / ‖B‖_F < tol`.
The process stops when all shifts are frozen or after `maxiter` block
iterations. Shifts that have not reached `tol` by then are returned with
`converged[s] == false` and their current minimum-residual iterate.

For a single shift the result agrees with `block_gmres(tau*I − C, B)` to
rounding error and uses the same number of iterations.

Memory: the basis takes `16 N L (k+1)` bytes as in `block_gmres`. The
shared block Hessenberg matrix takes about `8 (kL)^2` bytes. Each shift
keeps `k` rotations of shape `(2L, 2L)`. The triangular factor of a shift
is formed only when that shift is frozen (`16 (kL)^2` bytes, transient).
"""
function block_gmres_shifted(C, B::AbstractMatrix, taus::AbstractVector{<:Number};
                             tol::Real = 1e-5, maxiter::Integer = 100,
                             verbose::Bool = false)
    maxiter >= 1 || throw(ArgumentError("maxiter must be ≥ 1, got $maxiter"))
    size(B, 1) >= size(B, 2) || throw(ArgumentError(
        "B must have at least as many rows as columns, got $(size(B))"))
    kry = _block_gmres_shifted_core(C, Matrix{ComplexF64}(B), Vector{ComplexF64}(taus),
                                    Float64(tol), Int(maxiter), verbose)
    X = [_shifted_solution(kry, s) for s in eachindex(kry.Y)]
    return ShiftedBlockSolveResult(X, kry.converged, kry.residual_norm,
                                   kry.iterations, kry.residual_history)
end
