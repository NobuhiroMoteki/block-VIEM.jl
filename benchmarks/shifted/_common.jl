# Shared helpers for the Phase 0 feasibility scripts of the shifted-Krylov
# refractive-index sweep (see .claude/shifted_krylov/instructions/).
#
# Nothing here is part of the package API. The helpers reach into BlockVIEM
# internals on purpose so that Phase 0 can be judged before anything is added
# to src/.
#
# Notation (isotropic, homogeneous particle):
#
#     Z(eps_p) = (1/eps_p) M - (kappa/eps_bg) K0,   kappa = (eps_p - eps_bg)/eps_p
#     tau      = eps_bg / (eps_p - eps_bg) = 1 / (m_r^2 - 1)
#     Z        = (kappa/eps_bg) (tau M - K0)
#
# Lengths are in micrometres. k0 is the wavenumber in the background medium.

using BlockVIEM
using LinearAlgebra
using SparseArrays
using Random
using Printf
using StaticArrays
using JSON3

const WL_0 = 0.638        # vacuum wavelength [um]
const M_M  = 1.0          # background refractive index
const K_BG = 2π * M_M / WL_0
const EPS_BG = ComplexF64(M_M^2)
const RESULTS_DIR = joinpath(@__DIR__, "results")

relerr(a, b) = norm(a .- b) / norm(b)

kappa_of(eps_p, eps_bg = EPS_BG) = (ComplexF64(eps_p) - eps_bg) / ComplexF64(eps_p)
tau_of(eps_p, eps_bg = EPS_BG)   = eps_bg / (ComplexF64(eps_p) - eps_bg)

function save_json(name::AbstractString, obj)
    mkpath(RESULTS_DIR)
    path = joinpath(RESULTS_DIR, name)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    println("wrote ", path)
    return path
end

# ── Particle and right-hand sides ────────────────────────────────────────────

"""
    build_shape(shape, a_eq; m_p_max, lc = nothing) -> (mesh, basis)

`shape` is `:sphere` or `:oblate` (a:b:c = 3:3:1). `a_eq` is the
volume-equivalent radius [um]. With `lc === nothing` the mesh size follows
`adaptive_lc` for `m_p_max` (the paper's default discretization).
"""
function build_shape(shape::Symbol, a_eq::Real; m_p_max::Real = NaN,
                     lc::Union{Float64,Nothing} = nothing)
    p = shape === :sphere ? GREParams(a_eq, 1.0, 1.0, 0.0) :
        shape === :oblate ? GREParams(a_eq, 3.0, 1.0, 0.0) :
        throw(ArgumentError("unknown shape $shape"))
    rng = MersenneTwister(20260417)
    mesh, _ = lc === nothing ?
        gre_mesh(p, rng; wl_0 = WL_0, m_p_max = m_p_max) :
        gre_mesh(p, rng; lc = lc)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    return mesh, basis
end

"Euler angles (alpha, beta, gamma) [rad] of the paper's 5-point cos(beta) grid."
euler_cosbeta5() = [(0.0, acos(u), 0.0) for u in (-0.8, -0.4, 0.0, 0.4, 0.8)]

"`L` uniformly random orientations, nested in `L` for a fixed seed."
function euler_random(L::Integer; seed::Integer = 12345)
    rng = MersenneTwister(seed)
    out = Vector{NTuple{3,Float64}}(undef, L)
    for i in 1:L
        out[i] = (2π * rand(rng), acos(1 - 2 * rand(rng)), 2π * rand(rng))
    end
    out[1] = (0.0, 0.0, 0.0)
    return out
end

"RHS block, shape (N, L): one projected CAS-v2 plane wave per orientation."
function rhs_block(basis, eulers; k_bg = K_BG)
    N = n_basis(basis)
    L = length(eulers)
    B = Matrix{ComplexF64}(undef, N, L)   # shape: (N, L)
    Threads.@threads for i in 1:L
        ori = cas_orientation(eulers[i]...)
        B[:, i] = project_plane_wave(basis; k_hat = ori.u_inc,
                                     E0 = ori.e0_inc, k_bg = k_bg)
    end
    return B
end

# ── K0 action from an existing AIM operator (plan §2.3) ──────────────────────

function _require_isotropic(op::AIMOperator)
    kv = op.kappa_v
    (kv[1] == kv[2] == kv[3]) ||
        throw(ArgumentError("shift structure needs an isotropic operator"))
    iszero(kv[1]) && throw(ArgumentError("kappa = 0 (m_p == m_m)"))
    return kv[1]
end

"""
    k0_mvp(op, X) -> Matrix{ComplexF64}

Method a: `K0 X = (1/kappa) [far-field AIM + near-field precorrection] X`.
Both stored pieces already carry the factor `kappa`, so no cancellation occurs.
`X` has shape (N, L).
"""
function k0_mvp(op::AIMOperator, X::AbstractMatrix)
    kappa = _require_isotropic(op)
    N, L = size(X)
    Y = Matrix{ComplexF64}(undef, N, L)   # shape: (N, L)
    Threads.@threads for j in 1:L
        Y[:, j] = BlockVIEM.aim_far_mvp_weighted(op.projection, op.G_hat, op.k0,
                                                 op.kappa_v, op.kappa_avg,
                                                 view(X, :, j))
    end
    BlockVIEM._sym_upper_axpy!(Y, op.precorrection, op.diag_precorrection, X,
                               one(ComplexF64))
    Y ./= kappa
    return Y
end

"Method b (check only): `K0 X = (eps_bg/kappa) [(1/eps_p) M X - Z X]`."
function k0_mvp_by_difference(op::AIMOperator, X::AbstractMatrix)
    kappa = _require_isotropic(op)
    inv_eps = op.inv_eps_v[1]
    return (op.eps_bg / kappa) .* (inv_eps .* (op.mass * X) .- aim_mvp(op, X))
end

# ── M^{-1} actions ───────────────────────────────────────────────────────────

"Sparse Cholesky of the real SPD mass matrix; applies to complex blocks."
struct MinvChol{F}
    fac::F
end
MinvChol(M::SparseMatrixCSC{Float64,Int}) = MinvChol(cholesky(Symmetric(M)))

function (Mi::MinvChol)(X::AbstractMatrix{ComplexF64})
    return complex.(Mi.fac \ real(X), Mi.fac \ imag(X))
end

"""
    pcg_jacobi!(X, M, B; rtol = 1e-13, maxiter = 500) -> Vector{Int}

Column-wise Jacobi-preconditioned CG for `M X = B` with real SPD `M` and a
complex block `B` of shape (N, L). Columns run on separate threads.
Returns the iteration count of each column.
"""
function pcg_jacobi!(X::Matrix{ComplexF64}, M::SparseMatrixCSC{Float64,Int},
                     B::AbstractMatrix{ComplexF64};
                     rtol::Float64 = 1e-13, maxiter::Int = 500)
    N, L = size(B)
    dinv = 1 ./ Vector(diag(M))
    iters = zeros(Int, L)
    Threads.@threads for j in 1:L
        b = view(B, :, j)
        x = view(X, :, j)
        fill!(x, 0)
        r = Vector{ComplexF64}(b)
        z = dinv .* r
        p = copy(z)
        q = similar(r)
        rz = dot(r, z)
        bn = norm(b)
        k = 0
        while k < maxiter && norm(r) > rtol * bn
            k += 1
            mul!(q, M, p)
            alpha = rz / dot(p, q)
            @. x += alpha * p
            @. r -= alpha * q
            @. z = dinv * r
            rz_new = dot(r, z)
            beta = rz_new / rz
            rz = rz_new
            @. p = z + beta * p
        end
        iters[j] = k
    end
    return iters
end

struct MinvPCG
    M::SparseMatrixCSC{Float64,Int}
    rtol::Float64
end
function (Mi::MinvPCG)(X::AbstractMatrix{ComplexF64})
    Y = Matrix{ComplexF64}(undef, size(X))
    pcg_jacobi!(Y, Mi.M, X; rtol = Mi.rtol)
    return Y
end

# ── Operators handed to block_gmres (need only `A * X`) ──────────────────────

"Current solver: Z itself."
struct ZOp
    op::AIMOperator
end
Base.:*(A::ZOp, X::AbstractMatrix) = aim_mvp(A.op, X)

"Left preconditioning: M^{-1} Z, to be solved with the right-hand side M^{-1} B."
struct LeftOp{T}
    op::AIMOperator
    minv::T
end
Base.:*(A::LeftOp, X::AbstractMatrix) = A.minv(aim_mvp(A.op, X))

"Right preconditioning: Z M^{-1} Y = B, then X = M^{-1} Y."
struct RightOp{T}
    op::AIMOperator
    minv::T
end
Base.:*(A::RightOp, X::AbstractMatrix) = aim_mvp(A.op, A.minv(X))

"Iteration at which a residual history first drops below `thr` (0 = never)."
function iters_to(hist::AbstractVector{<:Real}, thr::Real)
    i = findfirst(<(thr), hist)
    return i === nothing ? 0 : i
end

# ── Shifted block GMRES prototype, right preconditioning ─────────────────────

"""
    shifted_block_gmres_right(cmul, B, taus; tol, maxiter)
        -> (Y::Vector{Matrix{ComplexF64}}, iters::Vector{Int}, res::Vector{Float64},
            k_total::Int, t_small::Float64)

Solve `(tau_s I - C) Y_s = B` for every shift with one block Arnoldi process on
`C` (`cmul(V)` returns `C * V`). `B` has shape (N, L). The least-squares problem
of each unconverged shift is solved from scratch at every iteration (naive, as in
the DDA prototype). `res[s]` is `||B - (tau_s I - C) Y_s||_F / ||B||_F`, which for
`C = K0 M^{-1}` equals the true residual of the original system `Z_s X_s = B`.
"""
function shifted_block_gmres_right(cmul, B::AbstractMatrix{ComplexF64},
                                   taus::AbstractVector{ComplexF64};
                                   tol::Float64 = 1e-5, maxiter::Int = 200)
    N, L = size(B)
    S = length(taus)
    Bn = norm(B)
    Qf = qr(B)
    V = [Matrix{ComplexF64}(Qf.Q)[:, 1:L]]
    Lam = Matrix{ComplexF64}(Qf.R)                       # shape: (L, L)
    H = zeros(ComplexF64, (maxiter + 1) * L, maxiter * L)
    active = collect(1:S)
    Ysmall = Vector{Matrix{ComplexF64}}(undef, S)
    iters = zeros(Int, S)
    res = fill(NaN, S)
    t_small = 0.0
    k_total = 0
    for k in 1:maxiter
        k_total = k
        W = cmul(V[k])
        for i in 1:k
            Hik = V[i]' * W
            H[(i-1)*L+1:i*L, (k-1)*L+1:k*L] .= Hik
            W .-= V[i] * Hik
        end
        Qk = qr(W)
        push!(V, Matrix{ComplexF64}(Qk.Q)[:, 1:L])
        H[k*L+1:(k+1)*L, (k-1)*L+1:k*L] .= Matrix{ComplexF64}(Qk.R)

        t_small += @elapsed begin
            rhs = zeros(ComplexF64, (k + 1) * L, L)
            rhs[1:L, :] .= Lam
            Hk = view(H, 1:(k+1)*L, 1:k*L)
            for s in copy(active)
                Hs = -Matrix(Hk)
                @inbounds for d in 1:k*L
                    Hs[d, d] += taus[s]
                end
                Fs = qr(Hs)
                c = Fs.Q' * rhs                          # shape: ((k+1)L, L)
                err = norm(view(c, k*L+1:(k+1)*L, :)) / Bn
                if err < tol || k == maxiter
                    Ysmall[s] = UpperTriangular(Fs.R) \ c[1:k*L, :]
                    iters[s] = k
                    res[s] = err
                    filter!(!=(s), active)
                end
            end
        end
        isempty(active) && break
    end
    Y = Vector{Matrix{ComplexF64}}(undef, S)
    for s in 1:S
        ks = iters[s]
        Ys = zeros(ComplexF64, N, L)
        for j in 1:ks
            Ys .+= V[j] * view(Ysmall[s], (j-1)*L+1:j*L, :)
        end
        Y[s] = Ys
    end
    return Y, iters, res, k_total, t_small
end
