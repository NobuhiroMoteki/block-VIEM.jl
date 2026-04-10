# Quadrature rules used by the Phase 2 singular integration engine.
#
# - Standard tetrahedron Gauss rules (degree 1, 2, 3) for non-singular
#   "far" integration.
# - 1D Gauss-Legendre nodes/weights on [0, 1] (via Golub-Welsch on the
#   symmetric tridiagonal Jacobi matrix). Used as a building block by the
#   tensor-product quadrature on the Duffy reference cube.

using LinearAlgebra: SymTridiagonal, eigen

# ---------------------------------------------------------------------------
# Tetrahedron Gauss rules
# ---------------------------------------------------------------------------

"""
    TetQuadRule

Tetrahedron quadrature rule expressed in barycentric coordinates.

A rule with `n` points has `bary[i]` summing to 1 for every `i`, and the
weights are normalized so `sum(weights) == 1`. To integrate `f` over a
physical tetrahedron `T` with volume `V_T`, use

```julia
I ≈ V_T * sum(rule.weights[i] * f(point_at(rule.bary[i], vertices)) for i in 1:rule.n)
```

The `degree` field is the algebraic degree of polynomial precision.
"""
struct TetQuadRule
    n::Int
    bary::Vector{NTuple{4,Float64}}
    weights::Vector{Float64}
    degree::Int
end

"1-point centroid rule, degree 1."
const TET_QUAD_1PT = TetQuadRule(
    1,
    [(0.25, 0.25, 0.25, 0.25)],
    [1.0],
    1,
)

"4-point symmetric rule, degree 2 (positive weights)."
const TET_QUAD_4PT = let
    a = (5 + 3 * sqrt(5)) / 20
    b = (5 - sqrt(5)) / 20
    TetQuadRule(
        4,
        [(a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)],
        fill(0.25, 4),
        2,
    )
end

"""
5-point rule, degree 3.

The centroid carries a negative weight `-4/5`; the four face-centred points
carry `+9/20`. Despite the negative centroid weight, this rule is widely
used and exact for cubic polynomials.
"""
const TET_QUAD_5PT = TetQuadRule(
    5,
    [
        (0.25, 0.25, 0.25, 0.25),
        (0.5, 1 / 6, 1 / 6, 1 / 6),
        (1 / 6, 0.5, 1 / 6, 1 / 6),
        (1 / 6, 1 / 6, 0.5, 1 / 6),
        (1 / 6, 1 / 6, 1 / 6, 0.5),
    ],
    [-4 / 5, 9 / 20, 9 / 20, 9 / 20, 9 / 20],
    3,
)

"""
11-point rule, degree 4 (Keast 1986, Rule index 6).

Uses 4 orbits: 1 centroid + two sets of 4 vertex-orbit + one edge orbit of 6.
All weights normalized so that `sum(weights) == 1` (reference simplex volume
is factored out and applied in [`integrate`](@ref)).
"""

"""
    bary_to_point(λ, vertices) -> Vec3

Convert a barycentric tuple `λ = (λ1, λ2, λ3, λ4)` into a Cartesian point
inside the tetrahedron with vertices `vertices = (v1, v2, v3, v4)`.
"""
@inline function bary_to_point(λ::NTuple{4,Float64}, vertices::NTuple{4,Vec3})
    return λ[1] * vertices[1] + λ[2] * vertices[2] +
           λ[3] * vertices[3] + λ[4] * vertices[4]
end

"""
    integrate(rule::TetQuadRule, vertices::NTuple{4,Vec3}, f) -> Number

Approximate `∫_T f(r) dV` using `rule`. The integrand `f` may return any
numeric type that supports addition and scalar multiplication.
"""
function integrate(rule::TetQuadRule, vertices::NTuple{4,Vec3}, f)
    V = tet_volume(vertices...)
    s = rule.weights[1] * f(bary_to_point(rule.bary[1], vertices))
    @inbounds for i in 2:rule.n
        s += rule.weights[i] * f(bary_to_point(rule.bary[i], vertices))
    end
    return V * s
end

# ---------------------------------------------------------------------------
# 1D Gauss-Legendre on [0, 1] (Golub-Welsch)
# ---------------------------------------------------------------------------

"""
    gauss_legendre_unit(n::Int) -> (Vector{Float64}, Vector{Float64})

`n`-point Gauss-Legendre nodes and weights on the unit interval `[0, 1]`,
computed via the Golub-Welsch algorithm applied to the symmetric tridiagonal
Jacobi matrix of the Legendre polynomials. Both arrays have length `n`.
The rule integrates polynomials of degree up to `2n - 1` exactly.
"""
function gauss_legendre_unit(n::Int)
    n < 1 && throw(ArgumentError("Gauss-Legendre order must be >= 1, got $n"))
    if n == 1
        return [0.5], [1.0]
    end
    β = Vector{Float64}(undef, n - 1)
    @inbounds for k in 1:(n - 1)
        β[k] = k / sqrt(4 * k^2 - 1)
    end
    T = SymTridiagonal(zeros(n), β)
    F = eigen(T)
    nodes_m11 = F.values                      # nodes on [-1, 1]
    weights_m11 = 2 .* (F.vectors[1, :]) .^ 2 # weights for ∫_{-1}^{1}
    # Affine map [-1, 1] -> [0, 1]: x = (t + 1)/2, dx = dt/2.
    nodes = (nodes_m11 .+ 1) ./ 2
    weights = weights_m11 ./ 2
    return nodes, weights
end
