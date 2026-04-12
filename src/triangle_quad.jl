# Triangle (2D) quadrature rules for surface-integral terms in the half-SWG
# VIEM formulation. Mirrors the structure of `quadrature.jl` / `duffy.jl` for
# tetrahedra but operates on triangular faces living in 3-space.
#
# Needed by the Stage-2 boundary-face correction terms K^B, K^C, K^D derived
# in `.claude/technical_note.md` §9.
#
# Provides:
#   - `TriQuadRule`                regular positive-weight rules on a triangle
#   - `TRI_QUAD_1PT`, `TRI_QUAD_3PT`  fixed low-order rules
#   - `tri_collapsed_rule(n)`      n^2-point rule of degree 2n-2 via the
#                                  triangle collapsed-coordinate map
#   - `TriDuffyRule`               vertex-Duffy rule on a reference triangle,
#                                  the u1 Jacobian cancelling a 1/R singularity
#                                  at the singular vertex
#   - `tri_duffy_reference_rule`, `tri_duffy_quadrature`,
#     `tri_duffy_quadrature_around`, `subdivide_triangle_around`
#
# Convention: like `TetQuadRule`, `TriQuadRule.weights` sum to 1 over the
# reference triangle, and `integrate_tri` multiplies by the physical area.
# `TriDuffyRule.weights` sum to 1 as well.

# ---------------------------------------------------------------------------
# Regular triangle Gauss rules
# ---------------------------------------------------------------------------

"""
    TriQuadRule

Triangle quadrature rule in barycentric coordinates. A rule with `n` points
has `bary[i]` summing to 1 per point and `weights` summing to 1. To
integrate `f` over a physical triangle with area `a_T` use

```julia
I ≈ a_T * sum(rule.weights[i] * f(tri_bary_to_point(rule.bary[i], verts)) for i in 1:rule.n)
```

The `degree` field is the algebraic degree of polynomial precision on the
reference triangle.
"""
struct TriQuadRule
    n::Int
    bary::Vector{NTuple{3,Float64}}
    weights::Vector{Float64}
    degree::Int
end

"1-point centroid rule, degree 1."
const TRI_QUAD_1PT = TriQuadRule(
    1,
    [(1/3, 1/3, 1/3)],
    [1.0],
    1,
)

"""
3-point symmetric rule, degree 2 (positive weights).

Barycentric points `(2/3, 1/6, 1/6)` and permutations, each with weight `1/3`.
Exact for polynomials up to total degree 2.
"""
const TRI_QUAD_3PT = TriQuadRule(
    3,
    [(2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)],
    [1/3, 1/3, 1/3],
    2,
)

"""
    tri_collapsed_rule(n::Int) -> TriQuadRule

Build a triangle rule with `n^2` points and algebraic degree `2n - 2` from the
collapsed-coordinate map `(u1, u2) ∈ [0,1]^2 → (λ1, λ2, λ3)`:

    λ1 = 1 - u1
    λ2 = u1 (1 - u2)
    λ3 = u1 u2

The Jacobian on the reference triangle is `u1` (normalised so reference area
= 1/2). Multiplying by `2` makes the rule weights sum to `1` over the unit-
area reference triangle so that `integrate_tri` can simply scale by the
physical area.

All weights are positive. Rule order → 1D Gauss-Legendre points per axis.
- `n = 2` → 4 points, degree 2
- `n = 3` → 9 points, degree 4
- `n = 4` → 16 points, degree 6
"""
function tri_collapsed_rule(n::Int)
    nodes, weights = gauss_legendre_unit(n)
    npts = n^2
    bary = Vector{NTuple{3,Float64}}(undef, npts)
    wts  = Vector{Float64}(undef, npts)
    idx = 0
    @inbounds for i in 1:n
        u1 = nodes[i]
        w1 = weights[i]
        for j in 1:n
            u2 = nodes[j]
            w2 = weights[j]
            idx += 1
            λ1 = 1 - u1
            λ2 = u1 * (1 - u2)
            λ3 = u1 * u2
            bary[idx] = (λ1, λ2, λ3)
            # Jacobian u1 × factor 2 so weights sum to 1.
            wts[idx] = w1 * w2 * 2 * u1
        end
    end
    return TriQuadRule(npts, bary, wts, max(2n - 2, 0))
end

"""
    tri_bary_to_point(λ, vertices) -> Vec3

Convert a barycentric tuple `λ = (λ1, λ2, λ3)` into a Cartesian point inside
the triangle with vertices `vertices = (v1, v2, v3)`.
"""
@inline function tri_bary_to_point(λ::NTuple{3,Float64}, vertices::NTuple{3,Vec3})
    return λ[1] * vertices[1] + λ[2] * vertices[2] + λ[3] * vertices[3]
end

"""
    integrate_tri(rule::TriQuadRule, vertices::NTuple{3,Vec3}, f) -> Number

Approximate `∫_S f(r) dS` over a flat physical triangle `S` using `rule`.
The integrand `f` may return any numeric type that supports addition and
scalar multiplication.
"""
function integrate_tri(rule::TriQuadRule, vertices::NTuple{3,Vec3}, f)
    a = triangle_area(vertices...)
    s = rule.weights[1] * f(tri_bary_to_point(rule.bary[1], vertices))
    @inbounds for i in 2:rule.n
        s += rule.weights[i] * f(tri_bary_to_point(rule.bary[i], vertices))
    end
    return a * s
end

# ---------------------------------------------------------------------------
# Vertex-Duffy rule on triangles (for 1/R self-integrals and near-singular
# observation points coincident with a triangle vertex)
# ---------------------------------------------------------------------------

"""
    TriDuffyRule

Pre-computed nodes and weights of a vertex-Duffy quadrature on a *reference*
triangle. The first barycentric coordinate is always the singular one. Use
[`tri_duffy_quadrature`](@ref) to apply the rule to a physical triangle with
a specified singular vertex.

Weights sum to `1` (physical area multiplied externally). The `u1` factor in
the Jacobian cancels the `1/R` singularity at the singular vertex.
"""
struct TriDuffyRule
    bary::Vector{NTuple{3,Float64}}
    weights::Vector{Float64}
    n_per_axis::Int
end

"""
    tri_duffy_reference_rule(n::Int) -> TriDuffyRule

Build a vertex-Duffy quadrature on the reference triangle using `n`-point
Gauss-Legendre per parametric axis. Total points: `n^2`.
"""
function tri_duffy_reference_rule(n::Int)
    nodes, weights = gauss_legendre_unit(n)
    npts = n^2
    bary = Vector{NTuple{3,Float64}}(undef, npts)
    wts  = Vector{Float64}(undef, npts)
    idx = 0
    @inbounds for i in 1:n, j in 1:n
        u1 = nodes[i]
        u2 = nodes[j]
        λs = 1 - u1
        λa = u1 * (1 - u2)
        λb = u1 * u2
        idx += 1
        bary[idx] = (λs, λa, λb)
        # Duffy Jacobian u1 × normalisation factor 2.
        wts[idx] = weights[i] * weights[j] * 2 * u1
    end
    return TriDuffyRule(bary, wts, n)
end

@inline function _tri_duffy_permutation(s::Int)
    if s == 1
        return (1, 2, 3)
    elseif s == 2
        return (2, 1, 3)
    elseif s == 3
        return (3, 1, 2)
    else
        throw(ArgumentError("singular_local must be in 1:3, got $s"))
    end
end

"""
    tri_duffy_quadrature(vertices::NTuple{3,Vec3},
                         singular_local::Integer,
                         rule::TriDuffyRule) -> (Vector{Vec3}, Vector{Float64})

Apply the precomputed reference Duffy `rule` to a physical triangle with
the singular vertex at local index `singular_local ∈ 1:3`. Returns
`(points, weights)` such that

```julia
∫_S g(r) dS ≈ sum(weights .* g.(points))
```

The weights already include the physical-area Jacobian, so callers should
NOT multiply by `triangle_area(vertices...)` themselves.
"""
function tri_duffy_quadrature(vertices::NTuple{3,Vec3},
                              singular_local::Integer,
                              rule::TriDuffyRule)
    perm = _tri_duffy_permutation(Int(singular_local))
    v = (vertices[perm[1]], vertices[perm[2]], vertices[perm[3]])
    a = triangle_area(vertices...)
    npts = length(rule.bary)
    pts = Vector{Vec3}(undef, npts)
    wts = Vector{Float64}(undef, npts)
    @inbounds for i in 1:npts
        λ = rule.bary[i]
        pts[i] = tri_bary_to_point(λ, v)
        wts[i] = a * rule.weights[i]
    end
    return pts, wts
end

"""
    tri_duffy_quadrature(vertices, singular_local, n::Integer)

Convenience overload that builds a fresh reference rule of order `n`.
"""
function tri_duffy_quadrature(vertices::NTuple{3,Vec3},
                              singular_local::Integer,
                              n::Integer)
    return tri_duffy_quadrature(vertices, singular_local,
                                 tri_duffy_reference_rule(Int(n)))
end

"""
    subdivide_triangle_around(vertices::NTuple{3,Vec3}, p::Vec3)
        -> NTuple{3,NTuple{3,Vec3}}

Split a triangle `S = (v1, v2, v3)` into three sub-triangles each having
`p` as its first vertex. The `k`-th sub-triangle pairs `p` with the edge
of `S` opposite vertex `k`:

```text
sub[1] = (p, v2, v3)
sub[2] = (p, v3, v1)
sub[3] = (p, v1, v2)
```

If `p` lies in the strict interior of `S`, all three sub-triangles are
non-degenerate and their areas sum to `a_S`. If `p` lies on an edge or
vertex of `S`, the sub-triangles corresponding to features that contain
`p` collapse to zero area while the rest still sum to `a_S`.
"""
@inline function subdivide_triangle_around(vertices::NTuple{3,Vec3}, p::Vec3)
    v1, v2, v3 = vertices
    return (
        (p, v2, v3),
        (p, v3, v1),
        (p, v1, v2),
    )
end

"""
    tri_duffy_quadrature_around(vertices::NTuple{3,Vec3}, p::Vec3,
                                rule::TriDuffyRule;
                                zero_area_tol=eps(Float64))
        -> (Vector{Vec3}, Vector{Float64})

Vertex-Duffy quadrature for an integrand with a (possibly singular) feature
at point `p` lying in the closure of triangle `vertices`. Subdivides the
triangle via [`subdivide_triangle_around`](@ref), applies Duffy with `p` as
the singular vertex on every non-degenerate sub-triangle, and concatenates
nodes/weights. Degenerate sub-triangles (area ≤ `zero_area_tol × a_total`)
are skipped.

The returned weights include the physical-area Jacobian, so
```julia
∫_S g(r) dS ≈ sum(weights .* g.(points))
```
"""
function tri_duffy_quadrature_around(vertices::NTuple{3,Vec3},
                                     p::Vec3,
                                     rule::TriDuffyRule;
                                     zero_area_tol::Float64 = eps(Float64))
    sub_tris = subdivide_triangle_around(vertices, p)
    n_per_sub = length(rule.bary)
    pts = Vector{Vec3}(undef, 3 * n_per_sub)
    wts = Vector{Float64}(undef, 3 * n_per_sub)
    a_total = triangle_area(vertices...)
    skip_thresh = zero_area_tol * a_total
    npts = 0
    for sub in sub_tris
        a_sub = triangle_area(sub...)
        a_sub <= skip_thresh && continue
        spts, swts = tri_duffy_quadrature(sub, 1, rule)
        @inbounds for i in eachindex(spts)
            npts += 1
            pts[npts] = spts[i]
            wts[npts] = swts[i]
        end
    end
    resize!(pts, npts)
    resize!(wts, npts)
    return pts, wts
end

"""
    tri_duffy_quadrature_around(vertices, p, n::Integer)

Convenience overload that builds a fresh reference rule of order `n`.
"""
function tri_duffy_quadrature_around(vertices::NTuple{3,Vec3},
                                     p::Vec3, n::Integer)
    return tri_duffy_quadrature_around(vertices, p,
                                        tri_duffy_reference_rule(Int(n)))
end
