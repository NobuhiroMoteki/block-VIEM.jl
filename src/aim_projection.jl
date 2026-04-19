# AIM projection from SWG basis functions to grid currents via polynomial
# moment matching.
#
# For each SWG basis function f_n with support T_n^± and centroid c_n, we
# choose a stencil of M^3 grid points around c_n and find weights
# {w_n,j^α} (one set per Cartesian component α ∈ {x,y,z}) such that
#
#   Σ_j w_n,j^α (r_j - c_n)^{abc} = ∫ f_n^α(r) (r - c_n)^{abc} dV
#
# for every multi-index (a,b,c) with a+b+c ≤ P (P = polynomial order). Once
# this moment-matching condition holds, the discrete grid sources reproduce
# the field of f_n to order P in the multipole expansion of the Helmholtz
# Green's function (Bleszynski et al. 1996; Sheng & Song Ch. 8).
#
# Reference: technical_note.md §5 / aim plan and CLAUDE.md Phase 3.

using SparseArrays
using LinearAlgebra: dot, BLAS

"""
    AIMProjection

SWG-to-grid projection matrices for the Adaptive Integral Method.

# Fields
- `grid::AIMGrid`               — auxiliary Cartesian grid
- `Wx, Wy, Wz::SparseMatrixCSC` — sparse `N_basis × N_grid` matrices for
  the three Cartesian components of the SWG vector basis
- `Wdiv::SparseMatrixCSC`       — sparse `N_basis × N_grid` matrix for the
  bulk scalar divergence `∇·f_n` (needed by the `-(∇·f_m)(∇'·f_n')G` term)
- `Wsurf::SparseMatrixCSC`      — sparse `N_basis × N_grid` matrix for the
  scalar boundary-face surface density `σ_n = f_n·n̂` on half-SWG basis
  functions (Phase A, theory_note.tex §5.5). Zero rows for non-boundary
  DOFs. Shares the same grid stencil and expansion centre `c_n` as
  `Wdiv`, so the charge-neutral effective scalar projection is simply
  `Wdiv - Wsurf` (Anastassiu 1998 thin-sheet-constraint + Q_vol+Q_surf=0
  identity)
- `poly_order::Int`             — polynomial order `P` actually used
- `stencil::Int`                — stencil size `M` per axis
"""
struct AIMProjection
    grid::AIMGrid
    Wx::SparseMatrixCSC{Float64,Int}
    Wy::SparseMatrixCSC{Float64,Int}
    Wz::SparseMatrixCSC{Float64,Int}
    Wdiv::SparseMatrixCSC{Float64,Int}
    Wsurf::SparseMatrixCSC{Float64,Int}
    poly_order::Int
    stencil::Int
end

"""
    n_moments(P::Integer) -> Int

Number of monomials of total degree ≤ `P` in three variables, i.e.
`binomial(P + 3, 3)`.
"""
@inline n_moments(P::Integer) = binomial(P + 3, 3)

"""
    multi_indices(P::Integer) -> Vector{NTuple{3,Int}}

All multi-indices `(a, b, c)` with `0 ≤ a + b + c ≤ P`, ordered by total
degree first and then lexicographically.
"""
function multi_indices(P::Integer)
    P >= 0 || throw(ArgumentError("P must be >= 0, got $P"))
    out = NTuple{3,Int}[]
    sizehint!(out, n_moments(P))
    for total in 0:P
        for a in 0:total, b in 0:(total - a)
            c = total - a - b
            push!(out, (a, b, c))
        end
    end
    return out
end

@inline function _monomial(r::Vec3, c::Vec3, abc::NTuple{3,Int})
    return (r[1] - c[1])^abc[1] *
           (r[2] - c[2])^abc[2] *
           (r[3] - c[3])^abc[3]
end

"""
    basis_centroid(basis::AbstractDivBasis, n::Integer) -> Vec3

Volume-weighted centroid of the supporting tets. This is the expansion
centre used by [`AIMProjection`](@ref).
"""
function basis_centroid(basis::AbstractDivBasis, n::Integer)
    mesh = basis.mesh
    tets = support_tets(basis, Int(n))
    t1 = tets[1]
    t2 = tets[2]
    if t2 == 0
        return mesh.tet_centroids[t1]
    end
    V1 = mesh.tet_volumes[t1]
    V2 = mesh.tet_volumes[t2]
    return (V1 * mesh.tet_centroids[t1] + V2 * mesh.tet_centroids[t2]) /
           (V1 + V2)
end

"""
    basis_moments(basis, n, c, indices, rule) -> Matrix{Float64}

Target moments for SWG basis function `n` expanded around `c`. Returns a
`(n_moments × 3)` matrix whose column `α` holds
`∫ f_n^α(r) (r - c)^{abc} dV` for every `(a, b, c)` in `indices`. The
volume integral is evaluated with `rule` on each support tet (the integrand
is a polynomial, so a sufficiently high-degree tet rule is exact).
"""
function basis_moments(basis::AbstractDivBasis, n::Integer, c::Vec3,
                       indices::Vector{NTuple{3,Int}}, rule::TetQuadRule)
    nmom = length(indices)
    M = zeros(Float64, nmom, 3)
    @inbounds for tet in support_tets(basis, Int(n))
        tet == 0 && continue
        verts = _tet_vertices(basis.mesh, tet)
        V = tet_volume(verts...)
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            wt = rule.weights[i] * V
            f = evaluate(basis, Int(n), r, tet)
            fx, fy, fz = f[1], f[2], f[3]
            for k in eachindex(indices)
                m = _monomial(r, c, indices[k])
                M[k, 1] += wt * fx * m
                M[k, 2] += wt * fy * m
                M[k, 3] += wt * fz * m
            end
        end
    end
    return M
end

"""
    divergence_moments(basis, n, c, indices, rule) -> Vector{Float64}

Target moments for the divergence `∇·f_n` of SWG basis function `n` expanded
around `c`. Returns a vector of length `n_moments` holding
`∫ (∇·f_n)(r) (r - c)^{abc} dV` for every `(a, b, c)` in `indices`.
"""
function divergence_moments(basis::AbstractDivBasis, n::Integer, c::Vec3,
                            indices::Vector{NTuple{3,Int}}, rule::TetQuadRule)
    nmom = length(indices)
    M = zeros(Float64, nmom)
    @inbounds for tet in support_tets(basis, Int(n))
        tet == 0 && continue
        verts = _tet_vertices(basis.mesh, tet)
        V = tet_volume(verts...)
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            wt = rule.weights[i] * V
            div_val = divergence(basis, Int(n), r, tet)
            for k in eachindex(indices)
                M[k] += wt * div_val * _monomial(r, c, indices[k])
            end
        end
    end
    return M
end

"""
    surface_moments(basis, n, c, indices, tri_rule) -> Vector{Float64}

Target surface moments for the scalar density `σ_n = f_n·n̂` on boundary
face `S_n`, expanded around centre `c`. Returns a vector of length
`n_moments` holding

    μ_n^{α,S} = ∫_{S_n} σ_n(r)(r - c)^{abc} dS,    (a,b,c) ∈ indices.

By the half-SWG unit-flux property `σ_n(r) = 1` on `S_n` (zero
elsewhere), so the integral reduces to a pure geometric monomial
integral over the triangle, evaluated with `tri_rule`. Returns the zero
vector when `n` is not a boundary SWG DOF (no surface contribution).

Phase A / theory_note.tex §5.5.
"""
function surface_moments(basis::AbstractDivBasis, n::Integer, c::Vec3,
                         indices::Vector{NTuple{3,Int}},
                         tri_rule::TriQuadRule)
    nmom = length(indices)
    M = zeros(Float64, nmom)
    _is_boundary_dof(basis, Int(n)) || return M
    verts = boundary_face_vertices(basis, Int(n))
    a = triangle_area(verts...)
    @inbounds for i in 1:tri_rule.n
        r = tri_bary_to_point(tri_rule.bary[i], verts)
        wt = tri_rule.weights[i] * a
        for k in eachindex(indices)
            M[k] += wt * _monomial(r, c, indices[k])
        end
    end
    return M
end

"""
    build_aim_projection(basis::AbstractDivBasis, grid::AIMGrid;
                         poly_order::Integer = 2,
                         stencil::Integer = 3,
                         rule::TetQuadRule = TET_QUAD_5PT,
                         tri_rule::TriQuadRule = tri_collapsed_rule(4))
        -> AIMProjection

Construct the AIM projection from `basis` onto `grid` using polynomial
moment matching of order `poly_order` with an `stencil^3` grid stencil
around each basis centroid. The default `(P, M) = (2, 3)` gives 10
moments and 27 stencil points, which is the canonical AIM choice.

For every basis function `n` the routine
1. computes the volume-weighted centroid `c_n`,
2. evaluates the bulk target moments via [`basis_moments`](@ref) and
   [`divergence_moments`](@ref),
3. for boundary half-SWG DOFs (Phase A), also evaluates the surface
   target moments via [`surface_moments`](@ref),
4. selects the centred `M^3` stencil through [`grid_stencil`](@ref),
5. assembles the moment matrix `Φ` (rows = moments, cols = stencil pts),
6. solves the underdetermined system `Φ w = M_target` in the
   minimum-norm sense via Julia's `\\` (QR-based) for each channel, and
7. scatters the resulting weights into the sparse rows `Wx[n,:]`,
   `Wy[n,:]`, `Wz[n,:]`, `Wdiv[n,:]`, and (boundary only) `Wsurf[n,:]`.

All five projection matrices share the same sparsity pattern so the
effective charge-neutral scalar projection `Wdiv − Wsurf`
(theory_note.tex §5.5, Anastassiu 1998 thin-sheet constraint) preserves
the same pattern without any extra allocation.

The grid `padding` must be large enough that every basis stencil falls
inside the grid; otherwise the moment system becomes deficient and the
constructor throws.
"""
function _projection_chunk!(basis, grid, rng::AbstractUnitRange{Int},
                            indices, nmom::Int, M_sten::Int,
                            rule::TetQuadRule, tri_rule::TriQuadRule,
                            zero_mom::Vector{Float64},
                            Is::Vector{Int}, Js::Vector{Int},
                            Vx::Vector{Float64}, Vy::Vector{Float64},
                            Vz::Vector{Float64}, Vdiv::Vector{Float64},
                            Vsurf::Vector{Float64})
    Phi = Matrix{Float64}(undef, nmom, M_sten^3)
    for n in rng
        c = basis_centroid(basis, n)
        M_target = basis_moments(basis, n, c, indices, rule)
        M_div_target = divergence_moments(basis, n, c, indices, rule)
        M_surf_target = _is_boundary_dof(basis, n) ?
            surface_moments(basis, n, c, indices, tri_rule) : zero_mom

        sten = grid_stencil(grid, c, M_sten)
        n_sten = length(sten)
        n_sten == M_sten^3 || throw(ErrorException(
            "basis $n stencil truncated ($n_sten of $(M_sten^3)); increase grid padding"))

        Phi_view = view(Phi, :, 1:n_sten)
        @inbounds for j in 1:n_sten
            r_j = grid_point_at_linear(grid, sten[j])
            for kk in 1:nmom
                Phi_view[kk, j] = _monomial(r_j, c, indices[kk])
            end
        end

        w_vec  = Phi_view \ M_target        # (n_sten × 3)
        w_div  = Phi_view \ M_div_target    # (n_sten,)
        w_surf = _is_boundary_dof(basis, n) ?
            Phi_view \ M_surf_target : zeros(Float64, n_sten)

        @inbounds for j in 1:n_sten
            push!(Is, n)
            push!(Js, sten[j])
            push!(Vx, w_vec[j, 1])
            push!(Vy, w_vec[j, 2])
            push!(Vz, w_vec[j, 3])
            push!(Vdiv, w_div[j])
            push!(Vsurf, w_surf[j])
        end
    end
    return nothing
end

function build_aim_projection(basis::AbstractDivBasis, grid::AIMGrid;
                              poly_order::Integer = 2,
                              stencil::Integer = 3,
                              rule::TetQuadRule = TET_QUAD_5PT,
                              tri_rule::TriQuadRule = tri_collapsed_rule(4))
    P = Int(poly_order)
    M_sten = Int(stencil)
    M_sten^3 >= n_moments(P) ||
        throw(ArgumentError("stencil^3 = $(M_sten^3) is smaller than the number of moments $(n_moments(P)) for poly_order=$P"))
    indices = multi_indices(P)
    nmom = length(indices)
    N = n_basis(basis)
    n_grid_total = n_grid_points(grid)

    # Parallel COO assembly.  SparseMatrixCSC push is not thread-safe,
    # and `@threads :static` + `threadid()` indexing is fragile under
    # Julia 1.12 (LAPACK yield points can migrate tasks even in :static
    # mode), so we partition `1:N` into contiguous chunks and spawn one
    # task per chunk via an explicit worker function — passing each
    # task's buffers as arguments is the only way to guarantee the
    # closure does not share state across tasks.
    zero_mom = zeros(Float64, nmom)
    nthr_def  = max(1, Threads.nthreads())
    n_chunks  = min(nthr_def, N)
    chunk_sz  = cld(N, n_chunks)
    chunks    = [((k-1)*chunk_sz + 1):min(k*chunk_sz, N) for k in 1:n_chunks]
    nz_est_per = cld(N * M_sten^3, n_chunks)
    Is_t    = [sizehint!(Int[],     nz_est_per) for _ in 1:n_chunks]
    Js_t    = [sizehint!(Int[],     nz_est_per) for _ in 1:n_chunks]
    Vx_t    = [sizehint!(Float64[], nz_est_per) for _ in 1:n_chunks]
    Vy_t    = [sizehint!(Float64[], nz_est_per) for _ in 1:n_chunks]
    Vz_t    = [sizehint!(Float64[], nz_est_per) for _ in 1:n_chunks]
    Vdiv_t  = [sizehint!(Float64[], nz_est_per) for _ in 1:n_chunks]
    Vsurf_t = [sizehint!(Float64[], nz_est_per) for _ in 1:n_chunks]

    # Force LAPACK `\` to run single-threaded inside the parallel region
    # to avoid nthreads × BLAS_threads oversubscription.
    prev_blas_nthr = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        tasks = Task[]
        for k in 1:n_chunks
            rng = chunks[k]
            isempty(rng) && continue
            t = Threads.@spawn _projection_chunk!(
                basis, grid, rng, indices, nmom, M_sten, rule, tri_rule,
                zero_mom,
                Is_t[k], Js_t[k], Vx_t[k], Vy_t[k], Vz_t[k],
                Vdiv_t[k], Vsurf_t[k])
            push!(tasks, t)
        end
        foreach(wait, tasks)
    finally
        BLAS.set_num_threads(prev_blas_nthr)
    end

    Is    = reduce(vcat, Is_t)
    Js    = reduce(vcat, Js_t)
    Vx    = reduce(vcat, Vx_t)
    Vy    = reduce(vcat, Vy_t)
    Vz    = reduce(vcat, Vz_t)
    Vdiv  = reduce(vcat, Vdiv_t)
    Vsurf = reduce(vcat, Vsurf_t)

    Wx = sparse(Is, Js, Vx, N, n_grid_total)
    Wy = sparse(Is, Js, Vy, N, n_grid_total)
    Wz = sparse(Is, Js, Vz, N, n_grid_total)
    Wdiv = sparse(Is, Js, Vdiv, N, n_grid_total)
    Wsurf = sparse(Is, Js, Vsurf, N, n_grid_total)
    return AIMProjection(grid, Wx, Wy, Wz, Wdiv, Wsurf, P, M_sten)
end
