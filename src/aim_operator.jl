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
#   K_AIM(x) = k0² Σ_α Wα^T G_conv(Wα x)  −  Wdiv^T G_conv(Wdiv x)
#
# K_near is a sparse precorrection matrix:
#   K_near[m,n] = K_direct[m,n] − K_AIM_element[m,n]  for near pairs
#   K_near[m,n] = 0                                     for far pairs
#
# Near pairs: basis pairs (m, n) that share at least one tetrahedron
# (i.e., the same pairs that contribute to the mass matrix).
#
# Phase 3.5–3.8 of BlockVIEM.jl.

using SparseArrays
using LinearAlgebra: dot

# ---------------------------------------------------------------------------
# Sparse mass matrix assembly (Phase 3.6)
# ---------------------------------------------------------------------------

"""
    assemble_mass_matrix(basis::SWGBasis; rule::TetQuadRule = TET_QUAD_5PT)
        -> SparseMatrixCSC{Float64,Int}

Assemble the SWG mass matrix `M_mn = ∫ f_m · f_n dV` as a sparse matrix.
`M` is non-zero only when `m` and `n` share at least one tetrahedron
(i.e. the supports overlap). The integrand is degree 2, so the default
5-point rule (degree 3) is exact.

The returned matrix does **not** include the `1/ε` prefactor; the caller
multiplies by `1/ε_p`.
"""
function assemble_mass_matrix(basis::SWGBasis; rule::TetQuadRule = TET_QUAD_5PT)
    N = n_basis(basis)
    mesh = basis.mesh

    # Build tet → basis index map
    tet_to_basis = Dict{Int,Vector{Int}}()
    for n in 1:N
        for tet in (basis.tet_plus[n], basis.tet_minus[n])
            push!(get!(Vector{Int}, tet_to_basis, tet), n)
        end
    end

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
    near_pairs(basis::SWGBasis) -> Vector{Tuple{Int,Int}}

Return all `(m, n)` pairs of basis functions whose supports overlap
(share at least one tetrahedron). Includes diagonal `(m, m)` and
both orderings `(m, n)` and `(n, m)`.
"""
function near_pairs(basis::SWGBasis)
    N = n_basis(basis)
    tet_to_basis = Dict{Int,Vector{Int}}()
    for n in 1:N
        for tet in (basis.tet_plus[n], basis.tet_minus[n])
            push!(get!(Vector{Int}, tet_to_basis, tet), n)
        end
    end
    pair_set = Set{Tuple{Int,Int}}()
    for blist in values(tet_to_basis)
        for m in blist, n in blist
            push!(pair_set, (m, n))
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
K_AIM(x) = k0² Σ_α Wα^T G_conv(Wα x)  −  Wdiv^T G_conv(Wdiv x)
```

where `Σ_α` runs over {x, y, z} Cartesian components, `G_conv` is the
Toeplitz convolution via FFT, and `x` is a length-`N_basis` complex vector.

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

    # Divergence channel: − Wdiv G_conv(Wdiv^T x)
    q_div = proj.Wdiv' * x
    q_div_3d = reshape(Vector{ComplexF64}(q_div), Nx, Ny, Nz)
    conv_div_3d = fft_convolve(G_hat, q_div_3d)
    y .-= proj.Wdiv * vec(conv_div_3d)

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

    # Divergence channel
    wm_div = proj.Wdiv[m, :]
    wn_div = proj.Wdiv[n, :]
    q_div = zeros(ComplexF64, Nx, Ny, Nz)
    for (j, v) in zip(findnz(wn_div)...)
        ci = CartesianIndices((Nx, Ny, Nz))[j]
        q_div[ci] = v
    end
    conv_div = fft_convolve(G_hat, q_div)
    for (j, v) in zip(findnz(wm_div)...)
        ci = CartesianIndices((Nx, Ny, Nz))[j]
        s -= v * conv_div[ci]
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
function assemble_precorrection(basis::SWGBasis, proj::AIMProjection,
                                G_hat::Array{ComplexF64,3};
                                k0::Number,
                                outer_rule::TetQuadRule = TET_QUAD_5PT,
                                duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
    pairs = near_pairs(basis)
    N = n_basis(basis)
    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    sizehint!(Is, length(pairs))
    sizehint!(Js, length(pairs))
    sizehint!(Vs, length(pairs))

    for (m, n) in pairs
        # Direct radiation kernel K_direct[m,n] (without mass and prefactors).
        K_direct_mn = _radiation_kernel(basis, Int(m), Int(n),
                                        ComplexF64(k0),
                                        outer_rule, duffy_rule)
        # AIM approximation of the same radiation kernel.
        K_aim_mn = aim_radiation_element(proj, G_hat, k0, Int(m), Int(n))

        push!(Is, m)
        push!(Js, n)
        push!(Vs, K_direct_mn - K_aim_mn)
    end
    return sparse(Is, Js, Vs, N, N)
end

# ---------------------------------------------------------------------------
# Full AIM operator (Phase 3.8)
# ---------------------------------------------------------------------------

"""
    AIMOperator

Pre-computed data for the AIM matrix-vector product. Construct via
[`build_aim_operator`](@ref) and apply via [`aim_mvp`](@ref).
"""
struct AIMOperator
    projection::AIMProjection
    G_hat::Array{ComplexF64,3}
    mass::SparseMatrixCSC{Float64,Int}
    precorrection::SparseMatrixCSC{ComplexF64,Int}
    k0::ComplexF64
    eps_p::ComplexF64
    eps_bg::ComplexF64
    kappa::ComplexF64
    inv_eps::ComplexF64
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
                       duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
        -> AIMOperator

One-shot constructor for the complete AIM operator: builds the grid,
projection matrices, Green FFT, mass matrix, and precorrection.
"""
function build_aim_operator(basis::SWGBasis;
                            k0::Number,
                            eps_p::Number = 1,
                            eps_bg::Number = 1,
                            pitch::Float64,
                            padding::Integer = 3,
                            poly_order::Integer = 2,
                            stencil_size::Integer = 3,
                            outer_rule::TetQuadRule = TET_QUAD_5PT,
                            duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
    grid = aim_grid(basis.mesh; pitch = pitch, padding = padding)
    proj = build_aim_projection(basis, grid;
                                poly_order = poly_order,
                                stencil = stencil_size,
                                rule = outer_rule)
    G_toep = build_green_toeplitz(grid, k0)
    G_hat = precompute_green_fft(G_toep)
    mass = assemble_mass_matrix(basis; rule = outer_rule)
    precorr = assemble_precorrection(basis, proj, G_hat;
                                     k0 = k0,
                                     outer_rule = outer_rule, duffy_rule = duffy_rule)
    eps_p_c = ComplexF64(eps_p)
    eps_bg_c = ComplexF64(eps_bg)
    kappa = (eps_p_c - eps_bg_c) / eps_p_c
    inv_eps = 1 / eps_p_c
    return AIMOperator(proj, G_hat, mass, precorr,
                       ComplexF64(k0), eps_p_c, eps_bg_c, kappa, inv_eps)
end

"""
    aim_mvp(op::AIMOperator, x::AbstractVector) -> Vector{ComplexF64}

Apply the AIM-accelerated impedance operator to vector `x`:

```
y = (1/ε_p) M x  −  (κ/ε_bg) [ K_AIM(x) + K_near x ]
```
"""
function aim_mvp(op::AIMOperator, x::AbstractVector)
    y = op.inv_eps .* (op.mass * x)
    if !iszero(op.kappa)
        K_aim_x = aim_far_mvp(op.projection, op.G_hat, op.k0, x)
        K_near_x = op.precorrection * x
        y .-= (op.kappa / op.eps_bg) .* (K_aim_x .+ K_near_x)
    end
    return y
end
