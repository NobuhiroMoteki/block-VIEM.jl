# AIM operator: combines the mass matrix, far-field FFT-MVP, and
# near-field precorrection into a single matrix-vector product.
#
# The full Z-matrix action on a vector x (SWG expansion coefficients) is
#
#   y = Z x = (1/ε_p) M x  −  (κ/ε_bg) K x
#
# where K is the radiation kernel. AIM splits K into far + near:
#
#   K x ≈ K_AIM(x)  +  K_near x
#
# K_AIM is computed via FFT convolution (O(N log N)):
#   K_AIM(x) = k0² Σ_α Wα   G_conv(Wα^T x)
#              − (Wdiv − Wsurf)   G_conv((Wdiv − Wsurf)^T x)
# where the (Wdiv − Wsurf) effective channel folds the bulk scalar
# divergence with the boundary surface density σ = f·n̂ so that the
# full kernel K^A + K^B + K^C + K^D is evaluated through a single
# scalar FFT pass (Phase A / theory_note.tex §5.5; Anastassiu 1998).
# For interior-only bases, Wsurf = 0 and this reduces to Phase-1a.
#
# K_near is a sparse precorrection matrix:
#   K_near[m,n] = K_direct[m,n] − K_AIM_element[m,n]  for near pairs
#   K_near[m,n] = 0                                     for far pairs
#
# Near pairs: basis pairs (m, n) that share at least one tetrahedron
# (i.e., the same pairs that contribute to the mass matrix), plus pairs
# whose centroid separation is within `near_radius`.
#
# Phase 3.5–3.8 of BlockVIEM.jl; Phase A (2026-04-19) adds the
# surface-moment channel to cover K^B + K^C + K^D.

using SparseArrays
using LinearAlgebra
using LinearAlgebra: dot, mul!

# ---------------------------------------------------------------------------
# Sparse mass matrix assembly (Phase 3.6)
# ---------------------------------------------------------------------------

"""
    assemble_mass_matrix(basis::AbstractDivBasis; rule::TetQuadRule = TET_QUAD_5PT)
        -> SparseMatrixCSC{Float64,Int}

Assemble the mass matrix `M_mn = ∫ f_m · f_n dV` as a sparse matrix.
`M` is non-zero only when `m` and `n` share at least one tetrahedron
(i.e. the supports overlap).

Works for any `AbstractDivBasis` (SWGBasis, RT1Basis, etc.).
"""
function assemble_mass_matrix(basis::AbstractDivBasis; rule::TetQuadRule = TET_QUAD_5PT)
    N = n_basis(basis)
    mesh = basis.mesh

    # Build tet → basis index map
    tet_to_basis = build_tet_to_dofs(basis)

    Is = Int[]
    Js = Int[]
    Vs = Float64[]
    for (tet, blist) in tet_to_basis
        verts = _tet_vertices(mesh, tet)
        V = tet_volume(verts...)
        for m in blist, n in blist
            s = 0.0
            @inbounds for i in 1:rule.n
                r = bary_to_point(rule.bary[i], verts)
                fm = evaluate(basis, m, r, tet)
                fn = evaluate(basis, n, r, tet)
                s += rule.weights[i] * dot(fm, fn)
            end
            push!(Is, m)
            push!(Js, n)
            push!(Vs, s * V)
        end
    end
    # SparseArrays sums duplicates (same m, n from different shared tets).
    return sparse(Is, Js, Vs, N, N)
end

# ---------------------------------------------------------------------------
# Near-pair enumeration (Phase 3.7 helper)
# ---------------------------------------------------------------------------

"""
    near_pairs(basis::AbstractDivBasis) -> Vector{Tuple{Int,Int}}

Return all `(m, n)` pairs of basis functions whose supports overlap
(share at least one tetrahedron). Includes diagonal `(m, m)` and
both orderings `(m, n)` and `(n, m)`.

This is a fallback helper; for AIM precorrection use
[`near_pairs_by_distance`](@ref) to get a geometry-aware near-field
set that also includes non-overlapping but close pairs.
"""
function near_pairs(basis::AbstractDivBasis)
    tet_to_basis = build_tet_to_dofs(basis)
    pair_set = Set{Tuple{Int,Int}}()
    for blist in values(tet_to_basis)
        for m in blist, n in blist
            push!(pair_set, (m, n))
        end
    end
    return collect(pair_set)
end

"""
    near_pairs_by_distance(basis::AbstractDivBasis; radius::Float64)
        -> Vector{Tuple{Int,Int}}

Return all `(m, n)` basis pairs whose volume-weighted centroids lie
within Euclidean distance `radius`. This is the geometry-aware near-
field set used for AIM precorrection: any pair whose centroid
separation is smaller than `radius` is considered "near" and the
AIM multipole approximation for that pair is replaced by direct
quadrature via the precorrection matrix.

The canonical choice for `radius` is a small multiple of the AIM
stencil extent, `radius = α * (stencil - 1) * pitch`, with `α ∈ [2, 4]`
giving precorrection cost proportional to `N * (α * stencil)^3` per
unit volume.

Uses a uniform bucket grid for O(N) average-case enumeration; the
bucket pitch equals `radius` so each basis centroid only needs to
check its bucket plus the 26 neighbors.
"""
function near_pairs_by_distance(basis::AbstractDivBasis; radius::Float64)
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))
    N = n_basis(basis)
    # Precompute all centroids once.
    centroids = Vector{Vec3}(undef, N)
    @inbounds for n in 1:N
        centroids[n] = basis_centroid(basis, n)
    end

    # Build a uniform bucket grid (cell size = radius). Each centroid
    # falls in exactly one bucket; neighbors are the 27 buckets within
    # (bi±1, bj±1, bk±1).
    r2 = radius^2
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    @inbounds for n in 1:N
        c = centroids[n]
        bi = floor(Int, c[1] / radius)
        bj = floor(Int, c[2] / radius)
        bk = floor(Int, c[3] / radius)
        push!(get!(Vector{Int}, buckets, (bi, bj, bk)), n)
    end

    pair_set = Set{Tuple{Int,Int}}()
    @inbounds for ((bi, bj, bk), me) in buckets
        for di in -1:1, dj in -1:1, dk in -1:1
            key = (bi + di, bj + dj, bk + dk)
            haskey(buckets, key) || continue
            you = buckets[key]
            for m in me, n in you
                d = centroids[m] - centroids[n]
                if d[1]^2 + d[2]^2 + d[3]^2 <= r2
                    push!(pair_set, (m, n))
                end
            end
        end
    end
    return collect(pair_set)
end

# ---------------------------------------------------------------------------
# AIM far-field MVP  K_AIM(x) via FFT convolution (Phase 3.5)
# ---------------------------------------------------------------------------

"""
    aim_far_mvp(proj::AIMProjection, G_hat::Array{ComplexF64,3},
                k0::Number, x::AbstractVector) -> Vector{ComplexF64}

Compute the far-field radiation kernel action `K_AIM x` using AIM:

```
K_AIM(x) = k0² Σ_α Wα G_conv(Wα^T x)
         − (Wdiv − Wsurf) G_conv((Wdiv − Wsurf)^T x)
```

where `Σ_α` runs over {x, y, z} Cartesian components, `G_conv` is the
Toeplitz convolution via FFT, and `x` is a length-`N_basis` complex
vector. The effective scalar channel `Wdiv − Wsurf` folds the bulk
divergence with the boundary surface density σ = f·n̂ so that the
full radiation kernel `K^A + K^B + K^C + K^D` is covered by a single
FFT pass (Phase A, theory_note.tex §5.5). For interior-only bases,
`Wsurf = 0` and this reduces to the Phase-1a formula.

`G_hat` is the pre-computed FFT of the Toeplitz kernel from
[`precompute_green_fft`](@ref).
"""
function aim_far_mvp(proj::AIMProjection, G_hat::Array{ComplexF64,3},
                     k0::Number, x::AbstractVector)
    Nx, Ny, Nz = proj.grid.dims
    k0_sq = ComplexF64(k0)^2
    N = size(proj.Wx, 1)
    y = zeros(ComplexF64, N)

    # Vector channels: k0² Σ_α Wα G_conv(Wα^T x)
    # W is (N_basis × N_grid). Projection basis→grid: q = W^T x (N_grid).
    # Back-projection grid→basis: result = W conv (N_basis).
    for W in (proj.Wx, proj.Wy, proj.Wz)
        q = W' * x                                       # N_grid vector
        q_3d = reshape(Vector{ComplexF64}(q), Nx, Ny, Nz)
        conv_3d = fft_convolve(G_hat, q_3d)
        y .+= k0_sq .* (W * vec(conv_3d))
    end

    # Effective scalar channel: − (Wdiv − Wsurf) G_conv((Wdiv − Wsurf)^T x).
    # Realised as two sparse matvecs in projection and interpolation; the
    # FFT between them is shared across both contributions.
    q_scalar = (proj.Wdiv' * x) .- (proj.Wsurf' * x)
    q_scalar_3d = reshape(Vector{ComplexF64}(q_scalar), Nx, Ny, Nz)
    conv_scalar_3d = fft_convolve(G_hat, q_scalar_3d)
    conv_scalar_vec = vec(conv_scalar_3d)
    y .-= proj.Wdiv * conv_scalar_vec
    y .+= proj.Wsurf * conv_scalar_vec

    return y
end

"""
    aim_far_mvp_weighted(proj, G_hat, k0, kappa_v, kappa_avg, x) -> Vector

Anisotropic AIM far-field MVP: applies per-component κ_α to vector
channels and κ_avg to the effective scalar channel.

    y = k0² Σ_α κ_α Wα G_conv(Wα^T x)
      − κ_avg (Wdiv − Wsurf) G_conv((Wdiv − Wsurf)^T x)

For isotropic `κ_α = κ`, this equals `κ × aim_far_mvp(...)`. The
`Wsurf` surface-density contribution (Phase A) is folded into the
scalar channel to cover K^B + K^C + K^D.
"""
function aim_far_mvp_weighted(proj::AIMProjection, G_hat::Array{ComplexF64,3},
                               k0::Number,
                               kappa_v::SVector{3,ComplexF64},
                               kappa_avg::ComplexF64,
                               x::AbstractVector)
    Nx, Ny, Nz = proj.grid.dims
    k0_sq = ComplexF64(k0)^2
    N = size(proj.Wx, 1)
    y = zeros(ComplexF64, N)

    for (W, kap) in zip((proj.Wx, proj.Wy, proj.Wz), kappa_v)
        q = W' * x
        q_3d = reshape(Vector{ComplexF64}(q), Nx, Ny, Nz)
        conv_3d = fft_convolve(G_hat, q_3d)
        y .+= (k0_sq * kap) .* (W * vec(conv_3d))
    end

    q_scalar = (proj.Wdiv' * x) .- (proj.Wsurf' * x)
    q_scalar_3d = reshape(Vector{ComplexF64}(q_scalar), Nx, Ny, Nz)
    conv_scalar_3d = fft_convolve(G_hat, q_scalar_3d)
    conv_scalar_vec = vec(conv_scalar_3d)
    y .-= kappa_avg .* (proj.Wdiv * conv_scalar_vec)
    y .+= kappa_avg .* (proj.Wsurf * conv_scalar_vec)

    return y
end

# ---------------------------------------------------------------------------
# AIM element evaluation for precorrection (Phase 3.7)
# ---------------------------------------------------------------------------

"""
    aim_radiation_element(proj::AIMProjection, G_hat::Array{ComplexF64,3},
                          k0::Number, m::Int, n::Int) -> ComplexF64

Evaluate the AIM approximation to the radiation kernel element `K_mn` by
extracting the (m, n) entry of the implicit AIM matrix. This is equivalent
to `(aim_far_mvp(proj, G_hat, k0, e_n))[m]` where `e_n` is the n-th
unit vector, but avoids building the full MVP. Used for computing the
precorrection matrix.
"""
function aim_radiation_element(proj::AIMProjection, G_hat::Array{ComplexF64,3},
                               k0::Number, m::Int, n::Int)
    Nx, Ny, Nz = proj.grid.dims
    k0_sq = ComplexF64(k0)^2
    s = zero(ComplexF64)

    for W in (proj.Wx, proj.Wy, proj.Wz)
        wm = W[m, :]   # sparse row
        wn = W[n, :]
        # The (m,n) entry of Wα^T G_conv Wα is:
        # Σ_{g,g'} wm[g] G[g-g'] wn[g']
        # = dot(wm, G_conv(wn_on_grid))
        q = zeros(ComplexF64, Nx, Ny, Nz)
        for (j, v) in zip(findnz(wn)...)
            ci = CartesianIndices((Nx, Ny, Nz))[j]
            q[ci] = v
        end
        conv = fft_convolve(G_hat, q)
        for (j, v) in zip(findnz(wm)...)
            ci = CartesianIndices((Nx, Ny, Nz))[j]
            s += k0_sq * v * conv[ci]
        end
    end

    # Effective scalar channel: −(Wdiv − Wsurf) G (Wdiv − Wsurf)^T.
    # Wsurf has zero rows for interior DOFs, so for interior (m, n) this
    # degenerates to the Phase-1a bulk divergence contribution.
    wm_eff = proj.Wdiv[m, :] - proj.Wsurf[m, :]
    wn_eff = proj.Wdiv[n, :] - proj.Wsurf[n, :]
    q_scalar = zeros(ComplexF64, Nx, Ny, Nz)
    for (j, v) in zip(findnz(wn_eff)...)
        ci = CartesianIndices((Nx, Ny, Nz))[j]
        q_scalar[ci] = v
    end
    conv_scalar = fft_convolve(G_hat, q_scalar)
    for (j, v) in zip(findnz(wm_eff)...)
        ci = CartesianIndices((Nx, Ny, Nz))[j]
        s -= v * conv_scalar[ci]
    end
    return s
end

# ---------------------------------------------------------------------------
# Precorrection (Phase 3.7)
# ---------------------------------------------------------------------------

"""
    assemble_precorrection(basis::SWGBasis, proj::AIMProjection,
                           G_hat::Array{ComplexF64,3};
                           k0::Number,
                           outer_rule::TetQuadRule = TET_QUAD_5PT,
                           duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
        -> SparseMatrixCSC{ComplexF64,Int}

Build the sparse precorrection matrix `K_near` for the radiation kernel:

```
K x ≈ K_AIM(x) + K_near x
```

For every near pair `(m, n)` (shared-tet pairs) the entry is

```
K_near[m,n] = K_direct[m,n] − K_AIM[m,n]
```

where `K_direct` is the radiation kernel from direct quadrature and
`K_AIM` is the AIM approximation via grid convolution. Far pairs have
`K_near[m,n] = 0` by definition.

The full impedance MVP then combines mass, far-field AIM, and this
precorrection:

```
Z x = (1/ε_p) M x  −  (κ/ε_bg) [ K_AIM(x) + K_near x ]
```
"""
function assemble_precorrection(basis::AbstractDivBasis, proj::AIMProjection,
                                G_hat::Array{ComplexF64,3};
                                k0::Number,
                                eps_p = 1,
                                eps_bg::Number = 1,
                                near_radius::Union{Float64,Nothing} = nothing,
                                outer_rule::TetQuadRule = TET_QUAD_5PT,
                                duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                tri_rule::TriQuadRule = tri_collapsed_rule(4),
                                tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6))
    # Distance-based near-field. Default = 2 * stencil_extent, empirically
    # 0.5% AIM-vs-dense error on sphere meshes at P=2, M=3, pitch=0.5·h̄.
    #
    # Optimization: for each near pair (m,n), compute K_AIM[m,n] directly
    # by summing over the (≤27)×(≤27) stencil pairs and looking up the
    # real-space Green's function G_toep at the folded displacement. This
    # gives ~2× speedup over the FFT-per-column path at N~2600 and much
    # more as N grows, while matching the FFT path to machine precision.
    #
    # Phase A: the scalar channel uses the effective weights
    # (v_div − v_surf) so K_AIM[m,n] = K^A_AIM[m,n] + K^{B+C+D}_AIM[m,n].
    # The K_direct side is correspondingly augmented with the boundary
    # kernels K^B+K^C+K^D for pairs where at least one of (m,n) is a
    # boundary half-SWG DOF; for interior-only pairs Wsurf = 0 and this
    # reduces to the Phase-1a bulk-only correction.
    #
    # The `G_hat` positional arg is ignored here (we need the real-space
    # G_toep instead) but kept for API compatibility with build_aim_operator.
    r_near = near_radius === nothing ?
             2.0 * (proj.stencil - 1) * proj.grid.pitch :
             Float64(near_radius)
    pairs = near_pairs_by_distance(basis; radius = r_near)
    for p in near_pairs(basis)
        push!(pairs, p)
    end
    unique!(pairs)
    # Z is complex-symmetric under reciprocity (isotropic + diagonal-aniso ε).
    # Store only upper-triangle (m ≤ n) entries; the MVP reconstructs the
    # lower triangle as (K + Kᵀ − D) x.  Halves sparse-matrix memory and
    # the stencil inner-sum work; K_direct is computed for both orderings
    # and averaged so the result matches `symmetrize=true` on the dense path.
    filter!(p -> p[1] <= p[2], pairs)
    N = n_basis(basis)

    col_to_rows = Dict{Int,Vector{Int}}()
    for (m, n) in pairs
        push!(get!(Vector{Int}, col_to_rows, n), m)
    end
    near_cols = sort!(collect(keys(col_to_rows)))

    grid = proj.grid
    Nx, Ny, Nz = grid.dims
    N2x = 2Nx; N2y = 2Ny; N2z = 2Nz
    k0_c = ComplexF64(k0)
    k0_sq = k0_c * k0_c
    (_, _, _, kappa_v, kappa_avg) = _aniso_params(eps_p, eps_bg)
    # Real-space Toeplitz Green kernel (circulant-folded on 2N grid).
    G_toep = build_green_toeplitz(grid, k0_c)

    # Cache per-basis stencil: Cartesian indices + channel values.
    # All five W matrices share the same nonzero pattern (built in one
    # pass in build_aim_projection), so we walk Wx_T and look up the
    # other channels at the same grid positions.  v_div_eff stores the
    # charge-neutral combination (v_div − v_surf) so the inner loop uses
    # a single scalar weight per (stencil, basis) entry.
    Wx, Wy, Wz, Wdiv, Wsurf = proj.Wx, proj.Wy, proj.Wz, proj.Wdiv, proj.Wsurf
    Wx_T = copy(transpose(Wx))

    stencil_max = proj.stencil^3
    n_per = Vector{Int}(undef, N)
    ci_all = CartesianIndices((Nx, Ny, Nz))
    idx_i     = Matrix{Int}(undef, stencil_max, N)
    idx_j     = Matrix{Int}(undef, stencil_max, N)
    idx_k     = Matrix{Int}(undef, stencil_max, N)
    v_x       = Matrix{ComplexF64}(undef, stencil_max, N)
    v_y       = Matrix{ComplexF64}(undef, stencil_max, N)
    v_z       = Matrix{ComplexF64}(undef, stencil_max, N)
    v_div_eff = Matrix{ComplexF64}(undef, stencil_max, N)

    for b in 1:N
        col_ptr_x = Wx_T.colptr
        count = col_ptr_x[b+1] - col_ptr_x[b]
        n_per[b] = count
        k = 0
        for p in col_ptr_x[b]:(col_ptr_x[b+1]-1)
            k += 1
            lin = Wx_T.rowval[p]
            ci = ci_all[lin]
            idx_i[k, b] = ci[1]
            idx_j[k, b] = ci[2]
            idx_k[k, b] = ci[3]
            v_x[k, b] = Wx_T.nzval[p]
            v_y[k, b] = Wy[b, lin]
            v_z[k, b] = Wz[b, lin]
            v_div_eff[k, b] = Wdiv[b, lin] - Wsurf[b, lin]
        end
    end

    # Parallel over near_cols.  Each task owns its own (Is, Js, Vs)
    # triplet arrays and reads the shared stencil caches (idx_*, v_*)
    # plus the Green tensor — all read-only after init.  We pass the
    # buffers as explicit function arguments so the spawned closure
    # cannot accidentally share state across tasks (the same pattern as
    # `_projection_chunk!` in src/aim_projection.jl).  Interleave
    # columns across chunks so that high-connectivity columns (which
    # dominate the inner work) are roughly balanced across tasks.
    nthr_def  = max(1, Threads.nthreads())
    n_chunks  = min(nthr_def, length(near_cols))
    col_chunks = [near_cols[k:n_chunks:end] for k in 1:n_chunks]
    est_per    = cld(length(pairs), max(1, n_chunks))
    Is_t = [sizehint!(Int[],         est_per) for _ in 1:n_chunks]
    Js_t = [sizehint!(Int[],         est_per) for _ in 1:n_chunks]
    Vs_t = [sizehint!(ComplexF64[],  est_per) for _ in 1:n_chunks]

    prev_blas_nthr = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        tasks = Task[]
        for k in 1:n_chunks
            isempty(col_chunks[k]) && continue
            t = Threads.@spawn _precorrection_chunk!(
                basis, k0_c, kappa_v, kappa_avg,
                outer_rule, duffy_rule, tri_rule, tri_duffy_rule,
                col_chunks[k], col_to_rows,
                n_per, idx_i, idx_j, idx_k,
                v_x, v_y, v_z, v_div_eff,
                G_toep, k0_sq, N2x, N2y, N2z,
                Is_t[k], Js_t[k], Vs_t[k])
            push!(tasks, t)
        end
        foreach(wait, tasks)
    finally
        LinearAlgebra.BLAS.set_num_threads(prev_blas_nthr)
    end

    Is = reduce(vcat, Is_t)
    Js = reduce(vcat, Js_t)
    Vs = reduce(vcat, Vs_t)
    return sparse(Is, Js, Vs, N, N)
end

# Worker for the parallel `assemble_precorrection` loop. The arguments
# are all either per-task output buffers (Is, Js, Vs) or read-only
# shared data (basis, quadrature rules, stencil caches, Green tensor).
function _precorrection_chunk!(basis::AbstractDivBasis,
                               k0_c::ComplexF64,
                               kappa_v::SVector{3,ComplexF64},
                               kappa_avg::ComplexF64,
                               outer_rule::TetQuadRule,
                               duffy_rule::DuffyQuadRule,
                               tri_rule::TriQuadRule,
                               tri_duffy_rule::TriDuffyRule,
                               cols::AbstractVector{Int},
                               col_to_rows::Dict{Int,Vector{Int}},
                               n_per::Vector{Int},
                               idx_i::Matrix{Int}, idx_j::Matrix{Int}, idx_k::Matrix{Int},
                               v_x::Matrix{ComplexF64}, v_y::Matrix{ComplexF64},
                               v_z::Matrix{ComplexF64}, v_div_eff::Matrix{ComplexF64},
                               G_toep::Array{ComplexF64,3},
                               k0_sq::ComplexF64,
                               N2x::Int, N2y::Int, N2z::Int,
                               Is::Vector{Int}, Js::Vector{Int},
                               Vs::Vector{ComplexF64})
    for n in cols
        nn = n_per[n]
        n_is_bnd = _is_boundary_dof(basis, n)
        for m in col_to_rows[n]
            mm = n_per[m]
            m_is_bnd = _is_boundary_dof(basis, m)
            s = zero(ComplexF64)
            @inbounds for a in 1:mm
                im_ = idx_i[a, m]; jm_ = idx_j[a, m]; km_ = idx_k[a, m]
                vx_m = v_x[a, m]; vy_m = v_y[a, m]; vz_m = v_z[a, m]
                ve_m = v_div_eff[a, m]
                for b in 1:nn
                    in_ = idx_i[b, n]; jn_ = idx_j[b, n]; kn_ = idx_k[b, n]
                    di = im_ - in_
                    dj = jm_ - jn_
                    dk = km_ - kn_
                    a_idx = di >= 0 ? di : di + N2x
                    b_idx = dj >= 0 ? dj : dj + N2y
                    c_idx = dk >= 0 ? dk : dk + N2z
                    G = G_toep[a_idx + 1, b_idx + 1, c_idx + 1]
                    vx_n = v_x[b, n]; vy_n = v_y[b, n]; vz_n = v_z[b, n]
                    ve_n = v_div_eff[b, n]
                    dot_w = kappa_v[1] * vx_m * vx_n +
                            kappa_v[2] * vy_m * vy_n +
                            kappa_v[3] * vz_m * vz_n
                    s += (k0_sq * dot_w - kappa_avg * ve_m * ve_n) * G
                end
            end
            K_direct_mn = _bulk_radiation_kernel_weighted(
                              basis, m, n, k0_c, kappa_v, kappa_avg,
                              outer_rule, duffy_rule)
            K_direct_avg = if m == n
                K_direct_mn
            else
                K_direct_nm = _bulk_radiation_kernel_weighted(
                                  basis, n, m, k0_c, kappa_v, kappa_avg,
                                  outer_rule, duffy_rule)
                (K_direct_mn + K_direct_nm) / 2
            end
            if m_is_bnd || n_is_bnd
                Ks_mn = _half_swg_surface_kernel(basis, m, n, k0_c,
                                                  outer_rule, duffy_rule,
                                                  tri_rule, tri_duffy_rule)
                Ks_avg = if m == n
                    Ks_mn
                else
                    Ks_nm = _half_swg_surface_kernel(basis, n, m, k0_c,
                                                      outer_rule, duffy_rule,
                                                      tri_rule, tri_duffy_rule)
                    (Ks_mn + Ks_nm) / 2
                end
                K_direct_avg += kappa_avg * Ks_avg
            end
            push!(Is, m)
            push!(Js, n)
            push!(Vs, K_direct_avg - s)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Full AIM operator (Phase 3.8)
# ---------------------------------------------------------------------------

"""
    AIMOperator

Pre-computed data for the AIM matrix-vector product. Construct via
[`build_aim_operator`](@ref) and apply via [`aim_mvp`](@ref).

`precorrection` and `half_swg_extra` store only the **upper triangle**
(entries (m, n) with m ≤ n). The MVP reconstructs the full symmetric
action as `(K + Kᵀ − Diagonal(diag_K)) * x`.  Z is complex-symmetric
under reciprocity (isotropic and diagonal-anisotropic ε), so this is
algebraically exact up to quadrature tolerance.  Halves sparse-matrix
memory vs the previous both-triangles storage.
"""
struct AIMOperator
    projection::AIMProjection
    G_hat::Array{ComplexF64,3}
    mass::SparseMatrixCSC{Float64,Int}
    # Upper-triangle precorrection (K_direct − K_AIM, averaged over both
    # orderings to match the dense `symmetrize=true` convention).
    precorrection::SparseMatrixCSC{ComplexF64,Int}
    diag_precorrection::Vector{ComplexF64}
    # Stage 2.7: half-SWG surface-correction terms (K^B + K^C + K^D),
    # already scaled by -(κ_avg/ε_bg). Upper triangle only; zero sparse
    # matrix for interior-only bases. See §9 of `.claude/technical_note.md`.
    half_swg_extra::SparseMatrixCSC{ComplexF64,Int}
    diag_half_swg_extra::Vector{ComplexF64}
    k0::ComplexF64
    eps_bg::ComplexF64
    # Anisotropic parameters (3-component vectors; equal for isotropic)
    kappa_v::SVector{3,ComplexF64}
    kappa_avg::ComplexF64
    inv_eps_v::SVector{3,ComplexF64}
end

# Apply y += scale * (A + Aᵀ − Diagonal(diag_A)) * x for upper-triangle A.
# Used for both precorrection and half_swg_extra, in vector and block forms.
@inline function _sym_upper_axpy!(y::AbstractVector, A::SparseMatrixCSC,
                                  diag_A::AbstractVector, x::AbstractVector,
                                  scale::Number)
    mul!(y, A, x, scale, one(eltype(y)))
    mul!(y, transpose(A), x, scale, one(eltype(y)))
    @inbounds for i in eachindex(diag_A)
        y[i] -= scale * diag_A[i] * x[i]
    end
    return y
end

@inline function _sym_upper_axpy!(Y::AbstractMatrix, A::SparseMatrixCSC,
                                  diag_A::AbstractVector, X::AbstractMatrix,
                                  scale::Number)
    mul!(Y, A, X, scale, one(eltype(Y)))
    mul!(Y, transpose(A), X, scale, one(eltype(Y)))
    N, L = size(X)
    @inbounds for j in 1:L
        for i in 1:N
            Y[i, j] -= scale * diag_A[i] * X[i, j]
        end
    end
    return Y
end

"""
    build_aim_operator(basis::SWGBasis;
                       k0::Number,
                       eps_p::Number = 1,
                       eps_bg::Number = 1,
                       pitch::Float64,
                       padding::Integer = 3,
                       poly_order::Integer = 2,
                       stencil_size::Integer = 3,
                       outer_rule::TetQuadRule = TET_QUAD_5PT,
                       duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                       projection::Union{AIMProjection,Nothing} = nothing,
                       mass::Union{SparseMatrixCSC,Nothing} = nothing)
        -> AIMOperator

One-shot constructor for the complete AIM operator: builds the grid,
projection matrices, Green FFT, mass matrix, and precorrection.

**Reuse across parameter sweeps.**  The projection `W` matrices and the
mass matrix depend only on `(basis, grid, poly_order, stencil_size)`
and on `outer_rule`, *not* on `k0` or `ε_p`.  For wavelength or
material sweeps, build them once and pass them back in through the
`projection` and `mass` keyword arguments — the constructor will skip
the corresponding setup (`build_aim_projection`, `assemble_mass_matrix`)
and only rebuild the `k0`-dependent pieces (Green FFT and
precorrection).  When `projection` is supplied the `pitch`, `padding`,
`poly_order`, `stencil_size`, and `tri_rule` keywords are ignored
because they are already baked into the provided object.
"""
function build_aim_operator(basis::AbstractDivBasis;
                            k0::Number,
                            eps_p = 1,
                            eps_bg::Number = 1,
                            pitch::Union{Float64,Nothing} = nothing,
                            padding::Integer = 3,
                            poly_order::Integer = 2,
                            stencil_size::Integer = 3,
                            near_radius::Union{Float64,Nothing} = nothing,
                            outer_rule::TetQuadRule = TET_QUAD_5PT,
                            duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                            tri_rule::TriQuadRule = tri_collapsed_rule(4),
                            tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6),
                            projection::Union{AIMProjection,Nothing} = nothing,
                            mass::Union{SparseMatrixCSC{Float64,Int},Nothing} = nothing)
    if projection === nothing
        pitch === nothing && throw(ArgumentError(
            "build_aim_operator: `pitch` is required when `projection` is not supplied"))
        grid = aim_grid(basis.mesh; pitch = pitch, padding = padding)
        proj = build_aim_projection(basis, grid;
                                    poly_order = poly_order,
                                    stencil = stencil_size,
                                    rule = outer_rule,
                                    tri_rule = tri_rule)
    else
        proj = projection
    end
    G_toep = build_green_toeplitz(proj.grid, k0)
    G_hat = precompute_green_fft(G_toep)
    mass_m = mass === nothing ?
        assemble_mass_matrix(basis; rule = outer_rule) : mass
    # Phase A: precorrection now folds in K^B + K^C + K^D direct for
    # near pairs involving a boundary DOF, and the AIM far-field covers
    # the same kernels via the (Wdiv − Wsurf) effective scalar channel.
    # `half_swg_extra` therefore becomes an empty sparse placeholder on
    # the AIM path (kept in the struct for backward compatibility so
    # that external callers reading the field keep working).
    precorr = assemble_precorrection(basis, proj, G_hat;
                                     k0 = k0,
                                     eps_p = eps_p, eps_bg = eps_bg,
                                     near_radius = near_radius,
                                     outer_rule = outer_rule,
                                     duffy_rule = duffy_rule,
                                     tri_rule = tri_rule,
                                     tri_duffy_rule = tri_duffy_rule)
    _, eps_bg_c, inv_eps_v, kappa_v, kappa_avg = _aniso_params(eps_p, eps_bg)
    N = n_basis(basis)
    diag_pre = Vector{ComplexF64}(diag(precorr))
    length(diag_pre) == N || error("diag_precorrection length mismatch")
    half_extra = spzeros(ComplexF64, N, N)
    diag_half = zeros(ComplexF64, N)
    return AIMOperator(proj, G_hat, mass_m,
                       precorr, diag_pre,
                       half_extra, diag_half,
                       ComplexF64(k0), eps_bg_c, kappa_v, kappa_avg, inv_eps_v)
end

"""
    aim_mvp(op::AIMOperator, x::AbstractVector) -> Vector{ComplexF64}

Apply the AIM-accelerated impedance operator to vector `x`:

```
y = (1/ε_p) M x  −  (κ/ε_bg) [ K_AIM(x) + K_near x ]
```
"""
function aim_mvp(op::AIMOperator, x::AbstractVector)
    # Mass term: Σ_α (1/ε_α) M_α x  ≈ inv_eps_avg * M*x for isotropic.
    # For SWG, M is the unweighted mass; anisotropic weighting enters here.
    # When inv_eps components are all equal this is exactly (1/ε_p) M*x.
    inv_avg = (op.inv_eps_v[1] + op.inv_eps_v[2] + op.inv_eps_v[3]) / 3
    y = ComplexF64.(inv_avg .* (op.mass * x))
    if !all(iszero, op.kappa_v)
        K_aim_x = aim_far_mvp_weighted(op.projection, op.G_hat, op.k0,
                                        op.kappa_v, op.kappa_avg, x)
        y .-= (1 / op.eps_bg) .* K_aim_x
        # Near-field precorrection: upper-triangle storage, symmetric action.
        _sym_upper_axpy!(y, op.precorrection, op.diag_precorrection, x,
                         -(1 / op.eps_bg))
    end
    # Half-SWG surface correction: already includes the -(κ_avg/ε_bg) scale.
    if nnz(op.half_swg_extra) > 0
        _sym_upper_axpy!(y, op.half_swg_extra, op.diag_half_swg_extra, x,
                         one(ComplexF64))
    end
    return y
end

"""
    aim_mvp(op::AIMOperator, X::AbstractMatrix) -> Matrix{ComplexF64}

Block (multi-RHS) form of [`aim_mvp`](@ref). For `X` of shape `(N, L)`,
returns `Y = Z * X` of shape `(N, L)`, where `Z` is the AIM-accelerated
impedance operator. Used by [`block_bicgstab`](@ref) for multi-orientation
batch solves. The mass and precorrection terms are applied as sparse
matrix–matrix products; the far-field FFT kernel is applied column-wise
and the columns are dispatched across `Threads.nthreads()` Julia
threads.

!!! note "Thread balance for block MVP"
    The inner FFT (see [`fft_convolve`](@ref)) honours the global FFTW
    thread count set by [`set_fft_threads`](@ref). For maximum
    throughput when `L > 1`, reduce FFTW threads so that
    `julia_threads × fftw_threads ≈ physical_cores`. A good rule:
    `set_fft_threads(max(1, cld(Threads.nthreads(), L)))`.
"""
function aim_mvp(op::AIMOperator, X::AbstractMatrix)
    N, L = size(X)
    Y = Matrix{ComplexF64}(undef, N, L)
    inv_avg = (op.inv_eps_v[1] + op.inv_eps_v[2] + op.inv_eps_v[3]) / 3
    mul!(Y, op.mass, X)
    Y .*= inv_avg
    if !all(iszero, op.kappa_v)
        inv_eb = 1 / op.eps_bg
        # aim_far_mvp_weighted allocates its own scratch and only reads
        # shared data (proj.W*, G_hat, kappa_*). FFTW plan execution is
        # thread-safe for disjoint input buffers, which is what we have
        # here — each iteration reshapes its own `q` into a fresh 3-D
        # buffer inside fft_convolve.
        Threads.@threads for j in 1:L
            xj = @view X[:, j]
            Kf = aim_far_mvp_weighted(op.projection, op.G_hat, op.k0,
                                       op.kappa_v, op.kappa_avg, xj)
            @views Y[:, j] .-= inv_eb .* Kf
        end
        _sym_upper_axpy!(Y, op.precorrection, op.diag_precorrection, X,
                         -inv_eb)
    end
    if nnz(op.half_swg_extra) > 0
        _sym_upper_axpy!(Y, op.half_swg_extra, op.diag_half_swg_extra, X,
                         one(ComplexF64))
    end
    return Y
end
