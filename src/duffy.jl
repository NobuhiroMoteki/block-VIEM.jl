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
