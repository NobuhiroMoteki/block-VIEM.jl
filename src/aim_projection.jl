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
using LinearAlgebra: dot

"""
    AIMProjection

SWG-to-grid projection matrices for the Adaptive Integral Method.

# Fields
- `grid::AIMGrid`               — auxiliary Cartesian grid
- `Wx, Wy, Wz::SparseMatrixCSC` — sparse `N_basis × N_grid` matrices for
  the three Cartesian components of the SWG vector basis
- `Wdiv::SparseMatrixCSC`       — sparse `N_basis × N_grid` matrix for the
  scalar divergence `∇·f_n` (needed by the `-(∇·f_m)(∇'·f_n')G` term)
- `poly_order::Int`             — polynomial order `P` actually used
- `stencil::Int`                — stencil size `M` per axis
"""
struct AIMProjection
    grid::AIMGrid
    Wx::SparseMatrixCSC{Float64,Int}
    Wy::SparseMatrixCSC{Float64,Int}
    Wz::SparseMatrixCSC{Float64,Int}
    Wdiv::SparseMatrixCSC{Float64,Int}
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
    basis_centroid(basis::SWGBasis, n::Integer) -> Vec3

Volume-weighted centroid of the union of `T_n^+` and `T_n^-`. This is the
expansion centre used by [`AIMProjection`](@ref).
"""
function basis_centroid(basis::SWGBasis, n::Integer)
    mesh = basis.mesh
    tp = basis.tet_plus[n]
    tm = basis.tet_minus[n]
    Vp = mesh.tet_volumes[tp]
    Vm = mesh.tet_volumes[tm]
    return (Vp * mesh.tet_centroids[tp] + Vm * mesh.tet_centroids[tm]) /
           (Vp + Vm)
end

"""
    basis_moments(basis, n, c, indices, rule) -> Matrix{Float64}

Target moments for SWG basis function `n` expanded around `c`. Returns a
`(n_moments × 3)` matrix whose column `α` holds
`∫ f_n^α(r) (r - c)^{abc} dV` for every `(a, b, c)` in `indices`. The
volume integral is evaluated with `rule` on each support tet (the integrand
is a polynomial, so a sufficiently high-degree tet rule is exact).
"""
function basis_moments(basis::SWGBasis, n::Integer, c::Vec3,
                       indices::Vector{NTuple{3,Int}}, rule::TetQuadRule)
    nmom = length(indices)
    M = zeros(Float64, nmom, 3)
    @inbounds for tet in (basis.tet_plus[n], basis.tet_minus[n])
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
function divergence_moments(basis::SWGBasis, n::Integer, c::Vec3,
                            indices::Vector{NTuple{3,Int}}, rule::TetQuadRule)
    nmom = length(indices)
    M = zeros(Float64, nmom)
    @inbounds for tet in (basis.tet_plus[n], basis.tet_minus[n])
        verts = _tet_vertices(basis.mesh, tet)
        V = tet_volume(verts...)
        div_val = divergence(basis, Int(n), tet)
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            wt = rule.weights[i] * V
            for k in eachindex(indices)
                M[k] += wt * div_val * _monomial(r, c, indices[k])
            end
        end
    end
    return M
end

"""
    build_aim_projection(basis::SWGBasis, grid::AIMGrid;
                         poly_order::Integer = 2,
                         stencil::Integer = 3,
                         rule::TetQuadRule = TET_QUAD_5PT)
        -> AIMProjection

Construct the AIM projection from `basis` onto `grid` using polynomial
moment matching of order `poly_order` with an `stencil^3` grid stencil
around each basis centroid. The default `(P, M) = (2, 3)` gives 10
moments and 27 stencil points, which is the canonical AIM choice.

For every basis function `n` the routine
1. computes the volume-weighted centroid `c_n`,
2. evaluates the target moments via [`basis_moments`](@ref),
3. selects the centred `M^3` stencil through [`grid_stencil`](@ref),
4. assembles the moment matrix `Φ` (rows = moments, cols = stencil pts),
5. solves the underdetermined system `Φ w = M_target` in the
   minimum-norm sense via Julia's `\\` (QR-based) for each Cartesian
   component, and
6. scatters the resulting weights into the sparse rows `Wx[n,:]`,
   `Wy[n,:]`, `Wz[n,:]`.

The grid `padding` must be large enough that every basis stencil falls
inside the grid; otherwise the moment system becomes deficient and the
constructor throws.
"""
function build_aim_projection(basis::SWGBasis, grid::AIMGrid;
                              poly_order::Integer = 2,
                              stencil::Integer = 3,
                              rule::TetQuadRule = TET_QUAD_5PT)
    P = Int(poly_order)
    M_sten = Int(stencil)
    M_sten^3 >= n_moments(P) ||
        throw(ArgumentError("stencil^3 = $(M_sten^3) is smaller than the number of moments $(n_moments(P)) for poly_order=$P"))
    indices = multi_indices(P)
    nmom = length(indices)
    N = n_basis(basis)
    n_grid_total = n_grid_points(grid)

    Is = Int[]
    Js = Int[]
    Vx = Float64[]
    Vy = Float64[]
    Vz = Float64[]
    Vdiv = Float64[]
    nz_est = N * M_sten^3
    sizehint!(Is, nz_est)
    sizehint!(Js, nz_est)
    sizehint!(Vx, nz_est)
    sizehint!(Vy, nz_est)
    sizehint!(Vz, nz_est)
    sizehint!(Vdiv, nz_est)

    Phi = Matrix{Float64}(undef, nmom, M_sten^3)

    for n in 1:N
        c = basis_centroid(basis, n)
        M_target = basis_moments(basis, n, c, indices, rule)
        M_div_target = divergence_moments(basis, n, c, indices, rule)

        sten = grid_stencil(grid, c, M_sten)
        n_sten = length(sten)
        n_sten == M_sten^3 || throw(ErrorException(
            "basis $n stencil truncated ($n_sten of $(M_sten^3)); increase grid padding"))

        Phi_view = view(Phi, :, 1:n_sten)
        @inbounds for j in 1:n_sten
            r_j = grid_point_at_linear(grid, sten[j])
            for k in 1:nmom
                Phi_view[k, j] = _monomial(r_j, c, indices[k])
            end
        end

        # Solve Φ w = M_target for w (n_sten × 3) in the minimum-norm sense.
        w_vec = Phi_view \ M_target        # (n_sten × 3)
        w_div = Phi_view \ M_div_target    # (n_sten,)

        @inbounds for j in 1:n_sten
            push!(Is, n)
            push!(Js, sten[j])
            push!(Vx, w_vec[j, 1])
            push!(Vy, w_vec[j, 2])
            push!(Vz, w_vec[j, 3])
            push!(Vdiv, w_div[j])
        end
    end

    Wx = sparse(Is, Js, Vx, N, n_grid_total)
    Wy = sparse(Is, Js, Vy, N, n_grid_total)
    Wz = sparse(Is, Js, Vz, N, n_grid_total)
    Wdiv = sparse(Is, Js, Vdiv, N, n_grid_total)
    return AIMProjection(grid, Wx, Wy, Wz, Wdiv, P, M_sten)
end
