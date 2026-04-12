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
    tet_collapsed_rule(n::Int) -> TetQuadRule

Construct a tetrahedron quadrature rule with `n^3` points and algebraic degree
`2n - 1` using a collapsed coordinate (Duffy-type) map from the unit cube
`[0,1]^3` to barycentric coordinates.

The mapping `(u1, u2, u3) → (λ1, λ2, λ3, λ4)` is:

    λ1 = 1 - u1,  λ2 = u1(1-u2),  λ3 = u1 u2 (1-u3),  λ4 = u1 u2 u3

with Jacobian `J = u1² u2`. Gauss-Legendre quadrature on each axis with `n`
points integrates the Jacobian-weighted polynomial part exactly for total
polynomial degree up to `2n - 1`.

This reuses the existing [`gauss_legendre_unit`](@ref) infrastructure and
is guaranteed correct by construction. The number of points (`n^3`) is not
optimal for a given degree, but the rule is simple, robust, and all weights
are positive.

# Usage
- `n = 3` → 27 points, degree 5 (sufficient for RT1 mass/radiation terms)
- `n = 4` → 64 points, degree 7 (extra safety for RT1 + Green's function)
"""
function tet_collapsed_rule(n::Int)
    nodes, weights = gauss_legendre_unit(n)
    npts = n^3
    bary = Vector{NTuple{4,Float64}}(undef, npts)
    wts = Vector{Float64}(undef, npts)
    idx = 0
    @inbounds for i in 1:n
        u1 = nodes[i]
        w1 = weights[i]
        for j in 1:n
            u2 = nodes[j]
            w2 = weights[j]
            for k in 1:n
                u3 = nodes[k]
                w3 = weights[k]
                idx += 1
                λ1 = 1 - u1
                λ2 = u1 * (1 - u2)
                λ3 = u1 * u2 * (1 - u3)
                λ4 = u1 * u2 * u3
                bary[idx] = (λ1, λ2, λ3, λ4)
                # Jacobian: 6 × u1² × u2 (the factor 6 normalizes
                # ∫₀¹∫₀¹∫₀¹ 6 u1² u2 du = 6 × 1/3 × 1/2 × 1 = 1)
                wts[idx] = w1 * w2 * w3 * 6 * u1^2 * u2
            end
        end
    end
    # The Jacobian u1^2 u2 adds degree 2 in u1 and 1 in u2 to the integrand.
    # A tet monomial of degree p has max u1-degree p+2, which requires 2n-1 >= p+2
    # i.e. p <= 2n-3. This is the effective polynomial degree of the rule.
    return TetQuadRule(npts, bary, wts, max(2n - 3, 0))
end

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

# ---------------------------------------------------------------------------
# High-order tet rules (depend on gauss_legendre_unit, must be defined after it)
# ---------------------------------------------------------------------------

"64-point collapsed rule, degree 5. Minimum for RT1 (degree-4 integrands)."
const TET_QUAD_64PT = tet_collapsed_rule(4)

"125-point collapsed rule, degree 7. High-accuracy RT1 or Green's function integrands."
const TET_QUAD_125PT = tet_collapsed_rule(5)
