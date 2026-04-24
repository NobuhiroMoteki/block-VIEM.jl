# Gaussian Random Ellipsoid (GRE) shape model and Gmsh mesh generation.
#
# Ports the surface generation algorithm from
#   block-DDA_Py/shape_model/gaussian_ellipsoid.py
# (Muinonen & Pieniluoma 2011 JQSRT) and produces a tetrahedral mesh via
# Gmsh for VIEM computation.
#
# Strategy for β > 0:
#   1. Create an ellipsoid via Gmsh OCC and generate the full 3D tet-mesh.
#   2. Sample the Gaussian random deformation field on a (θ, φ) grid.
#   3. Deform every mesh node *in-place* inside Gmsh (radially scaled so that
#      the deformation is zero at the centre and full at the surface).
#   4. Write the deformed mesh.
# This avoids STL I/O and the fragile classifySurfaces/createGeometry pipeline.

using Random: AbstractRNG, randn
using LinearAlgebra: cholesky, Symmetric

import Gmsh: gmsh

# ──────────────────────────────────────────────────────────────────────────────
#  Data types
# ──────────────────────────────────────────────────────────────────────────────

"""
    GREParams(r_v_base, bc_ratio, ab_ratio, beta)

Parameters for a Gaussian Random Ellipsoid shape
(Muinonen & Pieniluoma 2011, JQSRT).

# Fields
- `r_v_base::Float64` — volume-equivalent radius of base ellipsoid [length unit]
- `bc_ratio::Float64`  — b/c semi-axis ratio (≥ 1.0, typical range [1, 7])
- `ab_ratio::Float64`  — a/b semi-axis ratio (≥ 1.0, typical range [1, 2])
- `beta::Float64`      — std-dev of log-normal surface deformation [0, 0.3]

Semi-axis convention: `c ≤ b ≤ a`, with `c` along the z-axis.
"""
struct GREParams
    r_v_base::Float64
    bc_ratio::Float64
    ab_ratio::Float64
    beta::Float64

    function GREParams(r_v_base::Real, bc_ratio::Real, ab_ratio::Real, beta::Real)
        r_v_base > 0   || throw(ArgumentError("r_v_base must be positive, got $r_v_base"))
        bc_ratio >= 1   || throw(ArgumentError("bc_ratio must be ≥ 1, got $bc_ratio"))
        ab_ratio >= 1   || throw(ArgumentError("ab_ratio must be ≥ 1, got $ab_ratio"))
        beta >= 0        || throw(ArgumentError("beta must be ≥ 0, got $beta"))
        new(Float64(r_v_base), Float64(bc_ratio), Float64(ab_ratio), Float64(beta))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Semi-axes and adaptive mesh size
# ──────────────────────────────────────────────────────────────────────────────

"""
    gre_semi_axes(p::GREParams) -> (a, b, c)

Semi-axes of the base ellipsoid (before Gaussian deformation).
Convention: `a ≥ b ≥ c`, with `c` along the z-axis.
Volume = (4π/3)abc = (4π/3)r_v_base³.
"""
function gre_semi_axes(p::GREParams)
    c = cbrt(p.r_v_base^3 / (p.ab_ratio * p.bc_ratio^2))
    b = c * p.bc_ratio
    a = b * p.ab_ratio
    return (a, b, c)
end

"""
    adaptive_lc(p::GREParams; wl_0=NaN, m_p_max=NaN, N_pw=10) -> Float64

Compute an appropriate characteristic mesh size `lc` based on:
1. Wavelength inside particle: `λ₀ / (max|m_p| · N_pw)`
2. Smallest semi-axis:         `c / 3`
3. GRE correlation length:     `0.3c / 3`  (only when β > 0)

Returns the minimum of applicable constraints.  At least the geometric
constraint is always available; the wavelength constraint requires both
`wl_0` and `m_p_max` to be provided.
"""
function adaptive_lc(p::GREParams; wl_0::Real=NaN, m_p_max::Real=NaN, N_pw::Int=10)
    _, _, c = gre_semi_axes(p)
    candidates = Float64[]

    # wavelength-based
    if !isnan(wl_0) && !isnan(m_p_max) && m_p_max > 0
        push!(candidates, Float64(wl_0) / (Float64(m_p_max) * N_pw))
    end

    # geometry-based
    push!(candidates, c / 3.0)   # c is the smallest semi-axis

    # surface-deformation correlation length
    if p.beta > 0
        lc_corr = 0.3 * c
        push!(candidates, lc_corr / 3.0)
    end

    return minimum(candidates)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Gaussian random field sampling
# ──────────────────────────────────────────────────────────────────────────────

"""
    _sample_gaussian_field(rng, px, py, pz, beta, lc_corr) -> Vector{Float64}

Sample a zero-mean Gaussian random field with covariance
  C(i,j) = β² exp(-‖rᵢ - rⱼ‖² / (2 lc²))
at the points `(px[k], py[k], pz[k])` via Cholesky decomposition.
"""
function _sample_gaussian_field(rng::AbstractRNG,
                                px::Vector{Float64}, py::Vector{Float64},
                                pz::Vector{Float64},
                                beta::Float64, lc_corr::Float64)
    N = length(px)
    beta2 = beta^2
    inv2lc2 = 1.0 / (2.0 * lc_corr^2)

    cov = Matrix{Float64}(undef, N, N)
    nugget = 1e-10 * beta2   # regularise for near-duplicate pole points
    @inbounds for j in 1:N
        cov[j, j] = beta2 + nugget
        for i in (j + 1):N
            d2 = (px[i] - px[j])^2 + (py[i] - py[j])^2 + (pz[i] - pz[j])^2
            v = beta2 * exp(-d2 * inv2lc2)
            cov[i, j] = v
            cov[j, i] = v
        end
    end

    L = cholesky(Symmetric(cov)).L
    z = randn(rng, N)
    return L * z
end

# ──────────────────────────────────────────────────────────────────────────────
#  Bilinear interpolation on a (θ, φ) grid (φ periodic)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _interp_bilinear_periodic(data, θ_c, φ_c, θ_f, φ_f) -> Matrix

Bilinear interpolation of `data[i_θ, j_φ]` from a coarse (θ_c, φ_c) grid
to a fine (θ_f, φ_f) grid.  `φ` is treated as periodic with period 2π;
`θ` is clamped at boundaries.
"""
function _interp_bilinear_periodic(data::Matrix{Float64},
                                   θ_c::AbstractVector{Float64},
                                   φ_c::AbstractVector{Float64},
                                   θ_f::AbstractVector{Float64},
                                   φ_f::AbstractVector{Float64})
    Nt_c = length(θ_c);  Np_c = length(φ_c)
    dθ_c = θ_c[2] - θ_c[1]
    dφ_c = φ_c[2] - φ_c[1]
    Nt_f = length(θ_f);  Np_f = length(φ_f)
    result = Matrix{Float64}(undef, Nt_f, Np_f)

    @inbounds for j in 1:Np_f
        # φ index (periodic: wraps from last to first)
        fp = (φ_f[j] - φ_c[1]) / dφ_c + 1.0
        j0 = clamp(floor(Int, fp), 1, Np_c)
        j1 = j0 < Np_c ? j0 + 1 : 1          # periodic wrap
        tp = clamp(fp - j0, 0.0, 1.0)

        for i in 1:Nt_f
            ft = (θ_f[i] - θ_c[1]) / dθ_c + 1.0
            i0 = clamp(floor(Int, ft), 1, Nt_c)
            i1 = min(i0 + 1, Nt_c)
            tt = clamp(ft - i0, 0.0, 1.0)

            result[i, j] = ((1 - tt) * (1 - tp) * data[i0, j0] +
                            tt       * (1 - tp) * data[i1, j0] +
                            (1 - tt) * tp       * data[i0, j1] +
                            tt       * tp       * data[i1, j1])
        end
    end
    return result
end

"""
    _interp_s_point(s_fine, θ_grid, φ_grid, θ, φ) -> Float64

Bilinear interpolation of the deformation field `s_fine[i_θ, j_φ]` at an
arbitrary point `(θ, φ)`.  `φ` is periodic; `θ` is clamped to `[0, π]`.
"""
function _interp_s_point(s_fine::Matrix{Float64},
                         θ_grid::Vector{Float64}, φ_grid::Vector{Float64},
                         θ::Float64, φ::Float64)
    Nt = length(θ_grid);  Np = length(φ_grid)
    dθ = θ_grid[2] - θ_grid[1]
    dφ = φ_grid[2] - φ_grid[1]

    ft = (clamp(θ, 0.0, π) - θ_grid[1]) / dθ + 1.0
    i0 = clamp(floor(Int, ft), 1, Nt)
    i1 = min(i0 + 1, Nt)
    tt = clamp(ft - i0, 0.0, 1.0)

    # normalise φ to [0, 2π)
    φn = mod(φ, 2π)
    fp = (φn - φ_grid[1]) / dφ + 1.0
    j0 = clamp(floor(Int, fp), 1, Np)
    j1 = j0 < Np ? j0 + 1 : 1
    tp = clamp(fp - j0, 0.0, 1.0)

    return ((1 - tt) * (1 - tp) * s_fine[i0, j0] +
            tt       * (1 - tp) * s_fine[i1, j0] +
            (1 - tt) * tp       * s_fine[i0, j1] +
            tt       * tp       * s_fine[i1, j1])
end

# ──────────────────────────────────────────────────────────────────────────────
#  GRE deformation field generation (separated from surface point cloud)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _generate_gre_field(p, rng; N_theta_coarse, N_phi_coarse, interp_factor)
        -> (s_fine, θ_grid, φ_grid)

Sample and interpolate the Gaussian random deformation field for a GRE.
Returns the field on a fine `(θ, φ)` grid (θ includes poles, φ periodic).
"""
function _generate_gre_field(p::GREParams, rng::AbstractRNG;
                             N_theta_coarse::Int=25, N_phi_coarse::Int=100,
                             interp_factor::Union{Int,Nothing}=nothing)
    a, b, c = gre_semi_axes(p)
    lc_corr = 0.3 * c

    if interp_factor === nothing
        interp_factor = max(1, round(Int, 4 * cbrt(p.bc_ratio * p.ab_ratio)))
    end

    # coarse grid (φ periodic, no duplicate endpoint)
    θ_c = collect(range(0.0, π, length=N_theta_coarse))
    φ_c = collect(range(0.0, 2π * (N_phi_coarse - 1) / N_phi_coarse,
                        length=N_phi_coarse))

    # ellipsoid positions on coarse grid (for covariance distance)
    N_coarse = N_theta_coarse * N_phi_coarse
    px = Vector{Float64}(undef, N_coarse)
    py = Vector{Float64}(undef, N_coarse)
    pz = Vector{Float64}(undef, N_coarse)
    idx = 0
    for i in 1:N_theta_coarse
        st, ct = sincos(θ_c[i])
        for j in 1:N_phi_coarse
            sp, cp = sincos(φ_c[j])
            idx += 1
            px[idx] = a * st * cp
            py[idx] = b * st * sp
            pz[idx] = c * ct
        end
    end

    s_flat = _sample_gaussian_field(rng, px, py, pz, p.beta, lc_corr)
    s_coarse = Matrix{Float64}(undef, N_theta_coarse, N_phi_coarse)
    idx = 0
    for i in 1:N_theta_coarse, j in 1:N_phi_coarse
        idx += 1
        s_coarse[i, j] = s_flat[idx]
    end

    # fine grid
    N_theta = interp_factor * N_theta_coarse
    N_phi   = interp_factor * N_phi_coarse
    θ_f = collect(range(0.0, π, length=N_theta))
    φ_f = collect(range(0.0, 2π * (N_phi - 1) / N_phi, length=N_phi))

    if interp_factor > 1
        s_fine = _interp_bilinear_periodic(s_coarse, θ_c, φ_c, θ_f, φ_f)
    else
        s_fine = s_coarse
    end

    return s_fine, θ_f, φ_f
end

# ──────────────────────────────────────────────────────────────────────────────
#  GRE surface point generation (for visualisation / testing)
# ──────────────────────────────────────────────────────────────────────────────

"""
    gre_surface(p::GREParams, rng::AbstractRNG;
                N_theta_coarse=25, N_phi_coarse=100,
                interp_factor=nothing)
        -> (pts::Matrix{Float64}, N_theta::Int, N_phi::Int)

Generate surface points of a GRE on a structured (θ, φ) grid.

The returned `pts` is `3 × (N_theta * N_phi)`, column-major with θ as the
outer index: column index `k = (i-1)*N_phi + j`.

- `θ` runs from `0` to `π` (inclusive, `N_theta` points — includes poles).
- `φ` runs from `0` to `2π(N_phi-1)/N_phi` (**excludes** the duplicate 2π
  endpoint), so the grid wraps cleanly in φ.
"""
function gre_surface(p::GREParams, rng::AbstractRNG;
                     N_theta_coarse::Int=25, N_phi_coarse::Int=100,
                     interp_factor::Union{Int,Nothing}=nothing)
    a, b, c = gre_semi_axes(p)
    h0 = c^2 / a

    if p.beta > 0
        s_fine, θ_f, φ_f = _generate_gre_field(p, rng;
                                N_theta_coarse=N_theta_coarse,
                                N_phi_coarse=N_phi_coarse,
                                interp_factor=interp_factor)
    else
        if interp_factor === nothing
            interp_factor = max(1, round(Int, 4 * cbrt(p.bc_ratio * p.ab_ratio)))
        end
        N_theta = interp_factor * N_theta_coarse
        N_phi   = interp_factor * N_phi_coarse
        θ_f = collect(range(0.0, π, length=N_theta))
        φ_f = collect(range(0.0, 2π * (N_phi - 1) / N_phi, length=N_phi))
        s_fine = zeros(N_theta, N_phi)
    end

    N_theta = length(θ_f)
    N_phi   = length(φ_f)
    Npts = N_theta * N_phi
    pts = Matrix{Float64}(undef, 3, Npts)

    idx = 0
    for i in 1:N_theta
        st, ct = sincos(θ_f[i])
        for j in 1:N_phi
            sp, cp = sincos(φ_f[j])
            idx += 1

            x0 = a * st * cp
            y0 = b * st * sp
            z0 = c * ct

            if p.beta > 0
                nx = x0 / a^2;  ny = y0 / b^2;  nz = z0 / c^2
                nn = sqrt(nx^2 + ny^2 + nz^2)
                if nn > 0;  nx /= nn;  ny /= nn;  nz /= nn  end
                deform = h0 * (exp(s_fine[i, j]) - 0.5 * p.beta^2 - 1.0)
                pts[1, idx] = x0 + deform * nx
                pts[2, idx] = y0 + deform * ny
                pts[3, idx] = z0 + deform * nz
            else
                pts[1, idx] = x0
                pts[2, idx] = y0
                pts[3, idx] = z0
            end
        end
    end

    return pts, N_theta, N_phi
end

# ──────────────────────────────────────────────────────────────────────────────
#  Gmsh mesh generation
# ──────────────────────────────────────────────────────────────────────────────

"""
Create an ellipsoid volume tet-mesh via Gmsh OpenCASCADE kernel.
Returns (node_tags, coords_flat) before finalize for optional deformation.
When `finalize=true`, writes to `mesh_path` and finalizes Gmsh.
"""
function _gmsh_mesh_ellipsoid(a::Float64, b::Float64, c::Float64,
                              lc::Float64, mesh_path::AbstractString)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gre_ellipsoid")
        sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, sph)], 0.0, 0.0, 0.0, a, b, c)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [sph], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_path)
    finally
        gmsh.finalize()
    end
    return nothing
end

"""
Create an ellipsoid mesh and deform all nodes by the GRE field.

The deformation is radially scaled: nodes at the centre are undeformed,
nodes on the ellipsoid surface get the full GRE deformation `δ(θ, φ)`.
For a node at position `(x, y, z)` inside the ellipsoid with normalised
ellipsoidal radius `r̃ = √((x/a)² + (y/b)² + (z/c)²)`:

    new_pos = old_pos + r̃ · δ(θ, φ) · n̂

where `n̂` is the outward unit normal of the ellipsoid and
`δ = h₀ (eˢ − ½β² − 1)`.
"""
function _gmsh_mesh_gre_deformed(a::Float64, b::Float64, c::Float64,
                                 h0::Float64, beta::Float64,
                                 s_fine::Matrix{Float64},
                                 θ_grid::Vector{Float64},
                                 φ_grid::Vector{Float64},
                                 lc::Float64, mesh_path::AbstractString)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gre_deformed")
        sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, sph)], 0.0, 0.0, 0.0, a, b, c)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [sph], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)

        # ── deform every node ────────────────────────────────────────
        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        N_nodes = length(node_tags)
        inv_a2 = 1.0 / a^2;  inv_b2 = 1.0 / b^2;  inv_c2 = 1.0 / c^2
        beta2_half = 0.5 * beta^2

        for k in 1:N_nodes
            ci = 3 * (k - 1)
            x = coords_flat[ci + 1]
            y = coords_flat[ci + 2]
            z = coords_flat[ci + 3]

            # normalised ellipsoidal radius ∈ [0, 1]
            r_ell = sqrt(x^2 * inv_a2 + y^2 * inv_b2 + z^2 * inv_c2)
            if r_ell < 1e-15;  continue  end

            # parametric angles on the ellipsoid
            θ = atan(sqrt(x^2 * inv_a2 + y^2 * inv_b2), z * sqrt(inv_c2))
            φ = atan(y * sqrt(inv_b2), x * sqrt(inv_a2))
            if φ < 0;  φ += 2π  end

            # interpolate deformation field
            s_val = _interp_s_point(s_fine, θ_grid, φ_grid, θ, φ)
            deform = h0 * (exp(s_val) - beta2_half - 1.0)

            # outward unit normal of the ellipsoid at this direction
            nx = x * inv_a2;  ny = y * inv_b2;  nz = z * inv_c2
            nn = sqrt(nx^2 + ny^2 + nz^2)
            if nn > 0;  nx /= nn;  ny /= nn;  nz /= nn  end

            # apply radially-scaled deformation
            d = r_ell * deform
            coords_flat[ci + 1] = x + d * nx
            coords_flat[ci + 2] = y + d * ny
            coords_flat[ci + 3] = z + d * nz
        end

        # write deformed coordinates back (entity -1 = all entities)
        # setNode works per-node; for bulk update, rebuild from arrays
        for k in 1:N_nodes
            ci = 3 * (k - 1)
            gmsh.model.mesh.setNode(Int(node_tags[k]),
                [coords_flat[ci + 1], coords_flat[ci + 2], coords_flat[ci + 3]],
                Float64[])
        end

        gmsh.write(mesh_path)
    finally
        gmsh.finalize()
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────────────

"""
    gre_mesh(p::GREParams, rng::AbstractRNG;
             lc=nothing, wl_0=NaN, m_p_max=NaN, N_pw=10,
             mesh_path=nothing,
             N_theta_coarse=25, N_phi_coarse=100,
             interp_factor=nothing)
        -> (mesh::TetMesh, r_ve::Float64)

Generate a tetrahedral mesh for a Gaussian Random Ellipsoid particle.

# Strategy
- **β = 0**: creates a smooth ellipsoid via the Gmsh OpenCASCADE kernel.
- **β > 0**: creates the base ellipsoid mesh, then deforms every mesh node
  according to the sampled Gaussian random field.  Interior nodes are
  deformed proportionally to their normalised ellipsoidal radius (zero at
  centre, full at surface) to preserve mesh quality.

# Mesh-size determination

If `lc` is not supplied it is computed via [`adaptive_lc`](@ref):
the minimum of a wavelength-based constraint (`λ₀ / (max|m_p| · N_pw)`),
a geometric constraint (`c / 3`), and a surface-correlation constraint
(`0.1 c`, only when `β > 0`).  Providing `wl_0` and `m_p_max` enables the
wavelength constraint (recommended).

# Returns
- `mesh::TetMesh` — the tetrahedral mesh
- `r_ve::Float64` — volume-equivalent radius computed from the mesh volume
"""
function gre_mesh(p::GREParams, rng::AbstractRNG;
                  lc::Union{Float64,Nothing}=nothing,
                  wl_0::Real=NaN, m_p_max::Real=NaN, N_pw::Int=10,
                  mesh_path::Union{AbstractString,Nothing}=nothing,
                  N_theta_coarse::Int=25, N_phi_coarse::Int=100,
                  interp_factor::Union{Int,Nothing}=nothing)

    a, b, c = gre_semi_axes(p)

    # resolve mesh size
    if lc === nothing
        lc = adaptive_lc(p; wl_0=wl_0, m_p_max=m_p_max, N_pw=N_pw)
    end

    # resolve output path
    if mesh_path === nothing
        mesh_path = tempname() * ".msh"
    end

    if p.beta == 0
        # ── clean ellipsoid: use Gmsh OCC directly ──────────────────
        _gmsh_mesh_ellipsoid(a, b, c, lc, mesh_path)
    else
        # ── deformed GRE: mesh ellipsoid, then deform nodes ─────────
        h0 = c^2 / a
        s_fine, θ_grid, φ_grid = _generate_gre_field(p, rng;
                                    N_theta_coarse=N_theta_coarse,
                                    N_phi_coarse=N_phi_coarse,
                                    interp_factor=interp_factor)
        _gmsh_mesh_gre_deformed(a, b, c, h0, p.beta,
                                s_fine, θ_grid, φ_grid, lc, mesh_path)
    end

    mesh = read_msh(mesh_path)
    V = total_volume(mesh)
    r_ve = cbrt(3V / (4π))

    # ── Volume-preserving rescale (block-DDA_Py parity, v0.7.7+) ──────
    # Scale node coordinates so the discretised mesh volume equals the
    # target (4π/3)·r_v_base³ exactly.  DDA's Target.__init__ enforces
    # the same invariant via lattice spacing adjustment, so paper 1:1
    # cross-solver correlation plots are not biased by mesh-volume loss.
    s = p.r_v_base / r_ve
    apply_scale!(mesh, s)
    V = total_volume(mesh)             # now exact: (4π/3)·r_v_base³
    r_ve = cbrt(3V / (4π))             # now == p.r_v_base to machine precision

    return mesh, r_ve
end

"""
    gre_mesh_with_field(p::GREParams, s_field, θ_grid, φ_grid;
                        lc, mesh_path=nothing) -> (TetMesh, Float64)

Generate a tetrahedral mesh for a GRE particle using a **pre-computed**
deformation field `s_field[i_θ, j_φ]` on the grid `(θ_grid, φ_grid)`.

This is useful for cross-validation against block-DDA_Py: the same
Gaussian random field can be generated in Python, saved, and loaded
here to ensure both codes solve the identical shape.

The `θ_grid` and `φ_grid` do **not** need to match the Julia convention
(periodic φ without the 2π endpoint); the bilinear interpolator handles
arbitrary grids.

`p.beta` must be > 0 (the field is meaningless for a smooth ellipsoid).
"""
function gre_mesh_with_field(p::GREParams,
                             s_field::Matrix{Float64},
                             θ_grid::Vector{Float64},
                             φ_grid::Vector{Float64};
                             lc::Float64,
                             mesh_path::Union{AbstractString,Nothing}=nothing)
    p.beta > 0 || throw(ArgumentError("beta must be > 0 for gre_mesh_with_field"))
    a, b, c = gre_semi_axes(p)
    h0 = c^2 / a

    if mesh_path === nothing
        mesh_path = tempname() * ".msh"
    end

    _gmsh_mesh_gre_deformed(a, b, c, h0, p.beta,
                            s_field, θ_grid, φ_grid, lc, mesh_path)

    mesh = read_msh(mesh_path)
    V = total_volume(mesh)
    r_ve = cbrt(3V / (4π))

    # Volume-preserving rescale (see gre_mesh; v0.7.7+).
    s = p.r_v_base / r_ve
    apply_scale!(mesh, s)
    V = total_volume(mesh)
    r_ve = cbrt(3V / (4π))             # now == p.r_v_base to machine precision

    return mesh, r_ve
end
