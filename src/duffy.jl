# Vertex Duffy transformation for integrating 1/R-singular integrands over a
# tetrahedron. Reference: Mousavi & Sukumar (2010), Eq. (1) with β = 1.
# See `.claude/technical_note.md` §6 for the explicit mapping and Jacobian.
#
# Given the singular vertex `v_s` of a tetrahedron T = (v_s, v_a, v_b, v_c),
# the parametric map  u = (u1, u2, u3) ∈ [0, 1]^3  ->  λ = (λ_s, λ_a, λ_b, λ_c)
#
#   λ_s = 1 - u1
#   λ_a = u1 (1 - u2)
#   λ_b = u1 u2 (1 - u3)
#   λ_c = u1 u2 u3
#
# sends `u1 = 0` to `v_s` (the singular vertex collapses to the face). The
# Jacobian of (u1, u2, u3) -> physical (x, y, z) is
#
#   J_D(u) = 6 * V_T * u1^2 * u2
#
# The factor `u1^2` cancels the 1/R singularity at `v_s` because R ~ u1 there.

"""
    DuffyQuadRule

Pre-computed nodes and weights of a vertex-Duffy quadrature applied to a
*reference* tetrahedron. The rule is independent of the physical tetrahedron;
to integrate over a specific tetrahedron call [`duffy_quadrature`](@ref) which
applies the affine map.

# Fields
- `bary::Vector{NTuple{4,Float64}}` — barycentric coordinates of the points,
  ordered so that `bary[i][1]` is always the singular barycentric `λ_s`.
- `weights::Vector{Float64}`        — barycentric weights (sum to `1/6`,
  i.e. the volume of the standard simplex). Multiply by `6 V_T` to obtain
  physical-volume weights.
- `n_per_axis::Int`                  — 1D Gauss-Legendre order used.
"""
struct DuffyQuadRule
    bary::Vector{NTuple{4,Float64}}
    weights::Vector{Float64}
    n_per_axis::Int
end

"""
    duffy_reference_rule(n::Int) -> DuffyQuadRule

Build a vertex-Duffy quadrature on the *reference* tetrahedron using `n`-point
Gauss-Legendre quadrature in each parametric direction. The total number of
points is `n^3`.

The first barycentric coordinate of every point is `λ_s = 1 - u1`, i.e. the
coordinate of the singular vertex. Use [`duffy_quadrature`](@ref) to apply
the rule to a physical tetrahedron after permuting the singular vertex into
the first slot.
"""
function duffy_reference_rule(n::Int)
    nodes, weights = gauss_legendre_unit(n)
    npts = n^3
    bary = Vector{NTuple{4,Float64}}(undef, npts)
    wts = Vector{Float64}(undef, npts)
    idx = 0
    @inbounds for i in 1:n, j in 1:n, k in 1:n
        u1 = nodes[i]
        u2 = nodes[j]
        u3 = nodes[k]
        λs = 1 - u1
        λa = u1 * (1 - u2)
        λb = u1 * u2 * (1 - u3)
        λc = u1 * u2 * u3
        # Jacobian on the *reference* simplex (volume = 1/6) is u1^2 * u2,
        # so the barycentric weights here sum to 1/6. Physical-volume
        # weights are obtained by multiplying by `6 V_T` later.
        idx += 1
        bary[idx] = (λs, λa, λb, λc)
        wts[idx] = weights[i] * weights[j] * weights[k] * (u1^2) * u2
    end
    return DuffyQuadRule(bary, wts, n)
end

"""
    duffy_quadrature(vertices::NTuple{4,Vec3}, singular_local::Int, rule::DuffyQuadRule)
        -> (Vector{Vec3}, Vector{Float64})

Apply the precomputed reference Duffy `rule` to a physical tetrahedron with
the singular vertex at local index `singular_local ∈ 1:4`. Returns
`(points, weights)` such that

```julia
∫_T g(r) dV ≈ sum(weights .* g.(points))
```

The weights already include the `6 V_T` factor that converts barycentric
weights into physical-volume weights, so callers should NOT multiply by
the tetrahedron volume themselves.
"""
function duffy_quadrature(vertices::NTuple{4,Vec3},
                          singular_local::Integer,
                          rule::DuffyQuadRule)
    perm = _duffy_permutation(Int(singular_local))
    v = (vertices[perm[1]], vertices[perm[2]],
         vertices[perm[3]], vertices[perm[4]])
    V = tet_volume(vertices...)
    npts = length(rule.bary)
    pts = Vector{Vec3}(undef, npts)
    wts = Vector{Float64}(undef, npts)
    six_V = 6 * V
    @inbounds for i in 1:npts
        λ = rule.bary[i]
        pts[i] = bary_to_point(λ, v)
        wts[i] = six_V * rule.weights[i]
    end
    return pts, wts
end

"""
    duffy_quadrature(vertices, singular_local, n::Int)

Convenience overload that builds a fresh reference rule of order `n`. For
repeated calls (e.g. inside Z-matrix assembly) prefer to cache a single
[`duffy_reference_rule`](@ref) and pass it to the rule-based overload above.
"""
function duffy_quadrature(vertices::NTuple{4,Vec3},
                          singular_local::Integer,
                          n::Integer)
    return duffy_quadrature(vertices, singular_local, duffy_reference_rule(Int(n)))
end

"""
    _duffy_permutation(s) -> NTuple{4,Int}

Local-vertex permutation that places vertex `s` first; the relative order of
the remaining three vertices is preserved.
"""
@inline function _duffy_permutation(s::Int)
    if s == 1
        return (1, 2, 3, 4)
    elseif s == 2
        return (2, 1, 3, 4)
    elseif s == 3
        return (3, 1, 2, 4)
    elseif s == 4
        return (4, 1, 2, 3)
    else
        throw(ArgumentError("singular_local must be in 1:4, got $s"))
    end
end

# ---------------------------------------------------------------------------
# Subdivision around an observation point
# ---------------------------------------------------------------------------

"""
    subdivide_around(vertices::NTuple{4,Vec3}, p::Vec3) -> NTuple{4,NTuple{4,Vec3}}

Split a tetrahedron `T = (v1, v2, v3, v4)` into four sub-tetrahedra each
having `p` as its first vertex. The `k`-th sub-tet pairs `p` with the face
of `T` opposite vertex `k`:

```text
sub[1] = (p, v2, v3, v4)
sub[2] = (p, v1, v3, v4)
sub[3] = (p, v1, v2, v4)
sub[4] = (p, v1, v2, v3)
```

If `p` lies in the strict interior of `T`, all four sub-tets are
non-degenerate and their volumes sum to `V_T`. If `p` lies on a face,
edge, or vertex of `T`, the sub-tets corresponding to features that
contain `p` collapse to zero volume, and the remaining ones still sum to
`V_T`.

Combined with [`duffy_quadrature`](@ref) at `singular_local = 1`, the
output of this function is the natural building block for self-term and
near-singular integration.
"""
@inline function subdivide_around(vertices::NTuple{4,Vec3}, p::Vec3)
    v1, v2, v3, v4 = vertices
    return (
        (p, v2, v3, v4),
        (p, v1, v3, v4),
        (p, v1, v2, v4),
        (p, v1, v2, v3),
    )
end

"""
    duffy_quadrature_around(vertices::NTuple{4,Vec3}, p::Vec3, rule::DuffyQuadRule;
                            zero_volume_tol=eps(Float64)) -> (Vector{Vec3}, Vector{Float64})

Vertex-Duffy quadrature for an integrand with a (possibly singular) feature
at `p`, where `p` lies in the closure of the tetrahedron `vertices`. The
tetrahedron is split via [`subdivide_around`](@ref), Duffy is applied to
every non-degenerate sub-tet with `p` as its singular vertex, and the
resulting nodes/weights are concatenated.

Sub-tets with volume below `zero_volume_tol * V_T` are discarded; with the
default tolerance of `eps(Float64)` this only filters sub-tets that are
exactly degenerate (e.g. when `p` lies on a face of `T`).

The returned weights already include the physical-volume Jacobian, so
```julia
∫_T g(r) dV ≈ sum(weights .* g.(points))
```
"""
function duffy_quadrature_around(vertices::NTuple{4,Vec3},
                                 p::Vec3,
                                 rule::DuffyQuadRule;
                                 zero_volume_tol::Float64 = eps(Float64))
    sub_tets = subdivide_around(vertices, p)
    n_per_sub = length(rule.bary)
    pts = Vector{Vec3}(undef, 4 * n_per_sub)
    wts = Vector{Float64}(undef, 4 * n_per_sub)
    V_total = tet_volume(vertices...)
    skip_thresh = zero_volume_tol * V_total
    npts = 0
    for sub in sub_tets
        Vsub = tet_volume(sub...)
        Vsub <= skip_thresh && continue
        spts, swts = duffy_quadrature(sub, 1, rule)
        @inbounds for i in 1:n_per_sub
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
    duffy_quadrature_around(vertices, p, n::Integer)

Convenience overload that builds a fresh reference rule of order `n`.
"""
function duffy_quadrature_around(vertices::NTuple{4,Vec3}, p::Vec3, n::Integer)
    return duffy_quadrature_around(vertices, p, duffy_reference_rule(Int(n)))
end
