# Sphere-aggregate shape model and Gmsh tetrahedral meshing.
#
# Supports two classes of aggregate targets for VIEM scattering computations:
#   1. Random fractal-like aggregates imported from the PTSA HDF5 files
#      produced by `aggregate_generator_PTSA` (monomer centers + radii).
#   2. Regular (deterministic) aggregates generated in-package:
#        - linear chains
#        - planar arrays (square or triangular/hexagonal 2-D lattice)
#        - close-packed compact clusters (FCC, BCC, HCP) bounded by a sphere.
#
# Neck / partial-sinter model
# ---------------------------
# Point contacts between spheres are both physically degenerate (singular
# electric field in VIEM) and numerically fragile in Gmsh's OpenCASCADE
# boolean fuse.  We therefore enforce a small overlap by inflating every
# monomer radius by a factor (1 + overlap_factor).  For two equal spheres
# with centres at the original touching distance (d = r_i + r_j), the
# resulting neck radius is
#
#     a_neck / r = sqrt((1 + overlap_factor)^2 - 1)
#                ≈ sqrt(2 * overlap_factor)   for small overlap_factor.
#
# The helper `neck_ratio_to_overlap` inverts this relation.

using Random: AbstractRNG, MersenneTwister
using HDF5
using LinearAlgebra: norm
using Printf: @sprintf

import Gmsh: gmsh

# ──────────────────────────────────────────────────────────────────────────────
#  Data type
# ──────────────────────────────────────────────────────────────────────────────

"""
    SphereAggregate(centers, radii; metadata=Dict{String,Any}())

Container for a sphere-aggregate geometric specification.

# Fields
- `centers::Matrix{Float64}` — monomer centers; shape `(3, N)`, column-major
  (each column is one monomer).  All lengths use the same physical unit
  (typically μm to match PTSA input).
- `radii::Vector{Float64}`   — monomer radii, length `N`, same unit as
  `centers`.
- `metadata::Dict{String,Any}` — free-form provenance dictionary
  (e.g. PTSA sweep parameters, generator kind).

# Invariants
- `size(centers, 1) == 3`
- `size(centers, 2) == length(radii)`
- `all(radii .> 0)`
"""
struct SphereAggregate
    centers::Matrix{Float64}
    radii::Vector{Float64}
    metadata::Dict{String,Any}

    function SphereAggregate(centers::AbstractMatrix{<:Real},
                             radii::AbstractVector{<:Real};
                             metadata::Dict{String,Any}=Dict{String,Any}())
        size(centers, 1) == 3 ||
            throw(ArgumentError("centers must have shape (3, N); got $(size(centers))"))
        size(centers, 2) == length(radii) ||
            throw(ArgumentError("centers has $(size(centers,2)) columns but radii has $(length(radii)) entries"))
        all(r -> r > 0, radii) ||
            throw(ArgumentError("all monomer radii must be positive"))
        new(Matrix{Float64}(centers), Vector{Float64}(radii), metadata)
    end
end

"""
    n_monomers(agg::SphereAggregate) -> Int

Number of monomer spheres in the aggregate.
"""
n_monomers(agg::SphereAggregate) = length(agg.radii)

"""
    monomer_volume_sum(agg::SphereAggregate) -> Float64

Sum of monomer volumes (ignoring overlaps); upper bound for the aggregate
volume when spheres may overlap.
"""
monomer_volume_sum(agg::SphereAggregate) = sum((4π / 3) .* agg.radii .^ 3)

"""
    aggregate_bounding_radius(agg::SphereAggregate) -> Float64

Maximum distance from the origin to any point on any monomer surface.
Useful for setting simulation domain sizes.
"""
function aggregate_bounding_radius(agg::SphereAggregate)
    rmax = 0.0
    @inbounds for j in 1:n_monomers(agg)
        d = sqrt(agg.centers[1, j]^2 + agg.centers[2, j]^2 + agg.centers[3, j]^2)
        rmax = max(rmax, d + agg.radii[j])
    end
    return rmax
end

"""
    aggregate_centroid(agg::SphereAggregate) -> NTuple{3,Float64}

Volume-weighted centroid of the aggregate (each monomer weighted by its
own volume, overlaps ignored).
"""
function aggregate_centroid(agg::SphereAggregate)
    N = n_monomers(agg)
    wsum = 0.0
    cx = 0.0;  cy = 0.0;  cz = 0.0
    @inbounds for j in 1:N
        w = agg.radii[j]^3
        wsum += w
        cx += w * agg.centers[1, j]
        cy += w * agg.centers[2, j]
        cz += w * agg.centers[3, j]
    end
    return (cx / wsum, cy / wsum, cz / wsum)
end

"""
    recenter!(agg::SphereAggregate) -> SphereAggregate

Translate the aggregate so its volume-weighted centroid is at the origin.
Mutates `agg.centers` in place and returns `agg`.
"""
function recenter!(agg::SphereAggregate)
    cx, cy, cz = aggregate_centroid(agg)
    @views agg.centers[1, :] .-= cx
    @views agg.centers[2, :] .-= cy
    @views agg.centers[3, :] .-= cz
    return agg
end

# ──────────────────────────────────────────────────────────────────────────────
#  Neck-radius parameterization
# ──────────────────────────────────────────────────────────────────────────────

"""
    neck_ratio_to_overlap(neck_ratio) -> Float64

Inverse of the neck-radius formula for two equal touching monomers:

    a_neck / r = sqrt((1 + overlap_factor)^2 - 1)

so `overlap_factor = sqrt(1 + neck_ratio^2) - 1`.

`neck_ratio` is the desired neck radius divided by the monomer radius
(must be in `[0, 1)` — a neck ratio of 1 would merge the two monomer
centres completely).
"""
function neck_ratio_to_overlap(neck_ratio::Real)
    0.0 <= neck_ratio < 1.0 ||
        throw(ArgumentError("neck_ratio must be in [0, 1); got $neck_ratio"))
    return sqrt(1.0 + Float64(neck_ratio)^2) - 1.0
end

"""
    overlap_to_neck_ratio(overlap_factor) -> Float64

Forward evaluation: for two equal monomers originally touching, the neck
radius that results from inflating each radius by `(1 + overlap_factor)`.
"""
function overlap_to_neck_ratio(overlap_factor::Real)
    overlap_factor >= 0 ||
        throw(ArgumentError("overlap_factor must be non-negative"))
    return sqrt((1.0 + Float64(overlap_factor))^2 - 1.0)
end

"""
    pair_neck_radius(r_i, r_j, d_ij; overlap_factor) -> Float64

Exact neck (intersection-circle) radius produced when two monomers of
radii `r_i`, `r_j` with centre distance `d_ij` are both inflated by a
factor `(1 + overlap_factor)` before union.  Handles the general
polydisperse case correctly (equal-radius is the special case used by
[`overlap_to_neck_ratio`](@ref)).

Returns `0.0` when the inflated spheres do not overlap, and `min(R_i, R_j)`
when one sphere entirely contains the other.

# Derivation
With `s = 1 + overlap_factor`, `R_i = s·r_i`, `R_j = s·r_j`,
the plane of intersection is at signed distance
    d_i = (d_ij² + R_i² − R_j²) / (2 d_ij)
from centre i, and the neck radius is
    a_neck = √(R_i² − d_i²).
"""
function pair_neck_radius(r_i::Real, r_j::Real, d_ij::Real;
                          overlap_factor::Real=0.0)
    r_i > 0 || throw(ArgumentError("r_i must be positive"))
    r_j > 0 || throw(ArgumentError("r_j must be positive"))
    d_ij >= 0 || throw(ArgumentError("d_ij must be non-negative"))
    overlap_factor >= 0 ||
        throw(ArgumentError("overlap_factor must be non-negative"))

    s = 1.0 + Float64(overlap_factor)
    Ri = s * Float64(r_i)
    Rj = s * Float64(r_j)
    d  = Float64(d_ij)

    # disjoint
    d >= Ri + Rj && return 0.0
    # one contained in the other
    d + min(Ri, Rj) <= max(Ri, Rj) && return min(Ri, Rj)

    di = (d^2 + Ri^2 - Rj^2) / (2 * d)
    val = Ri^2 - di^2
    return val > 0 ? sqrt(val) : 0.0
end

# ──────────────────────────────────────────────────────────────────────────────
#  PTSA HDF5 I/O
# ──────────────────────────────────────────────────────────────────────────────

"""
    ptsa_h5_key(mean_rp, rel_std_rp, k, Df, Np, agg_num) -> String

HDF5 group path for one aggregate in the `aggregate_generator_PTSA`
HDF5 layout.  Mirrors the Python `make_h5key` helper:
`{mean_rp:.4f}/{rel_std_rp:.2f}/{k:.3f}/{Df:.2f}/{Np:05d}/{agg_num}`.
"""
function ptsa_h5_key(mean_rp::Real, rel_std_rp::Real, k::Real,
                     Df::Real, Np::Integer, agg_num::Integer)
    # Match Python's fixed-width format: %.4f / %.2f / %.3f / %.2f / %05d / %d
    return string(
        @sprintf("%.4f", Float64(mean_rp)),    "/",
        @sprintf("%.2f", Float64(rel_std_rp)), "/",
        @sprintf("%.3f", Float64(k)),          "/",
        @sprintf("%.2f", Float64(Df)),         "/",
        lpad(string(Int(Np)), 5, '0'),         "/",
        string(Int(agg_num)),
    )
end

"""
    load_ptsa_h5(h5_path; mean_rp, rel_std_rp, k, Df, Np, agg_num)
        -> SphereAggregate

Load one aggregate from a PTSA HDF5 file.  Returns a `SphereAggregate`
with the PTSA lookup parameters stored in `metadata` and the centroid
translated to the origin.

# Arguments
- `h5_path`    : path to an `aggregates_YYYYMMDD_NN.h5` file
- `mean_rp`    : mean monomer radius [μm]       (key component, .4f)
- `rel_std_rp` : relative std of monomer radius (key component, .2f)
- `k`          : fractal prefactor              (key component, .3f)
- `Df`         : fractal dimension              (key component, .2f)
- `Np`         : number of monomers (nominal)   (key component, %05d)
- `agg_num`    : aggregate index within the sweep

The HDF5 datasets `xp` (shape `(Np_actual, 3)`) and `rp` (shape
`(Np_actual,)`) are read and transposed to the column-major
`(3, Np_actual)` layout used by `SphereAggregate`.
"""
function load_ptsa_h5(h5_path::AbstractString;
                      mean_rp::Real, rel_std_rp::Real, k::Real,
                      Df::Real, Np::Integer, agg_num::Integer)
    isfile(h5_path) || throw(ArgumentError("PTSA HDF5 file not found: $h5_path"))
    key = ptsa_h5_key(mean_rp, rel_std_rp, k, Df, Np, agg_num)

    local xp, rp
    h5open(h5_path, "r") do h5f
        haskey(h5f, key) ||
            throw(KeyError("Aggregate key not found in $h5_path: $key"))
        g = h5f[key]
        xp = read(g["xp"])   # on-disk (Np, 3); HDF5.jl returns it as (3, Np)
        rp = read(g["rp"])   # (Np_actual,)
    end

    # HDF5.jl maps Python's row-major (Np, 3) to Julia column-major (3, Np).
    size(xp, 1) == 3 ||
        throw(ErrorException("unexpected xp shape from $key: $(size(xp))"))

    centers = Matrix{Float64}(xp)   # already (3, Np_actual)
    radii   = Vector{Float64}(rp)

    metadata = Dict{String,Any}(
        "source"     => "PTSA_HDF5",
        "h5_path"    => String(h5_path),
        "h5_key"     => key,
        "mean_rp"    => Float64(mean_rp),
        "rel_std_rp" => Float64(rel_std_rp),
        "k"          => Float64(k),
        "Df"         => Float64(Df),
        "Np"         => Int(Np),
        "agg_num"    => Int(agg_num),
    )
    agg = SphereAggregate(centers, radii; metadata=metadata)
    return recenter!(agg)
end

"""
    list_ptsa_keys(h5_path; limit=nothing) -> Vector{String}

Walk the HDF5 file and return all group paths that contain an `xp`
dataset (i.e. one aggregate per key).  Useful for discovering available
(mean_rp, rel_std_rp, k, Df, Np, agg_num) tuples.
"""
function list_ptsa_keys(h5_path::AbstractString;
                        limit::Union{Int,Nothing}=nothing)
    isfile(h5_path) || throw(ArgumentError("PTSA HDF5 file not found: $h5_path"))
    keys_found = String[]
    h5open(h5_path, "r") do h5f
        _collect_ptsa_keys!(keys_found, h5f, ""; limit=limit)
    end
    return keys_found
end

function _collect_ptsa_keys!(acc::Vector{String}, g, prefix::AbstractString;
                             limit::Union{Int,Nothing}=nothing)
    for name in keys(g)
        limit !== nothing && length(acc) >= limit && return
        child = g[name]
        path = isempty(prefix) ? name : string(prefix, "/", name)
        if isa(child, HDF5.Group)
            if haskey(child, "xp")
                push!(acc, path)
            else
                _collect_ptsa_keys!(acc, child, path; limit=limit)
            end
        end
    end
    return
end

# ──────────────────────────────────────────────────────────────────────────────
#  Regular aggregate generators (deterministic geometries)
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_linear_chain(N, radius; gap=0.0) -> SphereAggregate

Straight chain of `N` equal monomers along the x-axis.  Adjacent monomer
centres are separated by `2·radius + gap` (i.e. `gap = 0` produces
tangent contacts).  The chain is centred at the origin.
"""
function make_linear_chain(N::Integer, radius::Real; gap::Real=0.0)
    N >= 1 || throw(ArgumentError("N must be >= 1"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))

    step = 2.0 * radius + Float64(gap)
    centers = zeros(Float64, 3, N)
    offset = (N - 1) / 2.0
    @inbounds for i in 1:N
        centers[1, i] = ((i - 1) - offset) * step
    end
    radii = fill(Float64(radius), N)
    metadata = Dict{String,Any}(
        "source"    => "make_linear_chain",
        "N"         => Int(N),
        "radius"    => Float64(radius),
        "gap"       => Float64(gap),
    )
    return SphereAggregate(centers, radii; metadata=metadata)
end

"""
    make_planar_array(nx, ny, radius; lattice=:square, gap=0.0) -> SphereAggregate

Planar (z = 0) array of equal monomers on a 2-D lattice.

# Arguments
- `nx, ny` : number of monomers along the two in-plane directions
- `radius` : monomer radius
- `lattice` : `:square` (simple 2-D square lattice) or `:triangular`
  (2-D hexagonal close packing of discs, a.k.a. hex-close-packed layer)
- `gap`    : extra spacing added to the tangent separation `2·radius`
"""
function make_planar_array(nx::Integer, ny::Integer, radius::Real;
                           lattice::Symbol=:square, gap::Real=0.0)
    nx >= 1 && ny >= 1 || throw(ArgumentError("nx and ny must be >= 1"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))

    a = 2.0 * radius + Float64(gap)  # nearest-neighbour spacing
    N = nx * ny
    centers = zeros(Float64, 3, N)

    if lattice === :square
        @inbounds for iy in 1:ny, ix in 1:nx
            j = (iy - 1) * nx + ix
            centers[1, j] = (ix - 1) * a
            centers[2, j] = (iy - 1) * a
        end
    elseif lattice === :triangular
        # Hex-close-packed discs in the plane: each odd row is shifted by a/2
        # along x, row spacing is a·√3/2.
        row_dy = a * sqrt(3.0) / 2.0
        @inbounds for iy in 1:ny, ix in 1:nx
            j = (iy - 1) * nx + ix
            x_shift = iseven(iy - 1) ? 0.0 : 0.5 * a
            centers[1, j] = (ix - 1) * a + x_shift
            centers[2, j] = (iy - 1) * row_dy
        end
    else
        throw(ArgumentError("lattice must be :square or :triangular; got :$lattice"))
    end

    radii = fill(Float64(radius), N)
    agg = SphereAggregate(centers, radii;
            metadata=Dict{String,Any}(
                "source"  => "make_planar_array",
                "nx"      => Int(nx),
                "ny"      => Int(ny),
                "radius"  => Float64(radius),
                "lattice" => String(lattice),
                "gap"     => Float64(gap),
            ))
    return recenter!(agg)
end

"""
    make_fcc_cluster(radius; cluster_radius, gap=0.0) -> SphereAggregate

Compact cluster of equal monomers on an FCC lattice (face-centred cubic,
close packing) whose centres lie within a sphere of radius
`cluster_radius` from the origin.

Nearest-neighbour distance is `2·radius + gap`; the FCC cubic lattice
constant is therefore `a_fcc = (2·radius + gap) · √2`.

Typical usage: `make_fcc_cluster(0.02; cluster_radius=0.08)` produces a
compact cluster about 4 monomer-diameters across.
"""
function make_fcc_cluster(radius::Real; cluster_radius::Real, gap::Real=0.0)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    cluster_radius > 0 || throw(ArgumentError("cluster_radius must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))

    nn = 2.0 * radius + Float64(gap)
    a_fcc = nn * sqrt(2.0)
    basis = ((0.0, 0.0, 0.0),
             (0.5, 0.5, 0.0),
             (0.5, 0.0, 0.5),
             (0.0, 0.5, 0.5))
    return _fill_lattice_sphere(radius, cluster_radius, a_fcc, basis;
            source="make_fcc_cluster", gap=Float64(gap))
end

"""
    make_bcc_cluster(radius; cluster_radius, gap=0.0) -> SphereAggregate

Compact cluster on a BCC lattice.  The nearest-neighbour bond is along
the body diagonal, so the cubic lattice constant for tangent monomers is
`a_bcc = (2·radius + gap) · 2/√3`.  BCC is *not* a close packing; use
this for body-centred sphere arrays rather than densest packings.
"""
function make_bcc_cluster(radius::Real; cluster_radius::Real, gap::Real=0.0)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    cluster_radius > 0 || throw(ArgumentError("cluster_radius must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))

    nn = 2.0 * radius + Float64(gap)
    a_bcc = nn * 2.0 / sqrt(3.0)
    basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    return _fill_lattice_sphere(radius, cluster_radius, a_bcc, basis;
            source="make_bcc_cluster", gap=Float64(gap))
end

"""
    make_hcp_cluster(radius; cluster_radius, gap=0.0) -> SphereAggregate

Compact cluster on an HCP (hexagonal close-packed) lattice.  Lattice
constants: `a_hcp = 2·radius + gap` (in-plane), `c_hcp = a_hcp · √(8/3)`
(ideal HCP).  The two-atom basis of conventional HCP in orthorhombic
coordinates is:
- atom 1: (0, 0, 0)
- atom 2: (a/2, a·√3/6, c/2)

Centers within `cluster_radius` from the origin are kept.
"""
function make_hcp_cluster(radius::Real; cluster_radius::Real, gap::Real=0.0)
    radius > 0 || throw(ArgumentError("radius must be positive"))
    cluster_radius > 0 || throw(ArgumentError("cluster_radius must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))

    a = 2.0 * radius + Float64(gap)
    c = a * sqrt(8.0 / 3.0)
    # Orthorhombic lattice vectors: (a, 0, 0), (a/2, a√3/2, 0), (0, 0, c)
    # Rather than using generic lattice-vector machinery, enumerate directly.
    rmax2 = Float64(cluster_radius)^2
    # Generous bounding-box extent in unit-cells along each direction.
    nmax = Int(ceil(Float64(cluster_radius) / min(a, c) + 2))

    pts = Tuple{Float64,Float64,Float64}[]
    for i in -nmax:nmax, j in -nmax:nmax, k in -nmax:nmax
        ox = i * a + j * (a / 2)
        oy = j * (a * sqrt(3.0) / 2.0)
        oz = k * c
        # basis atom 1
        if ox^2 + oy^2 + oz^2 <= rmax2
            push!(pts, (ox, oy, oz))
        end
        # basis atom 2
        bx = ox + a / 2
        by = oy + a * sqrt(3.0) / 6.0
        bz = oz + c / 2
        if bx^2 + by^2 + bz^2 <= rmax2
            push!(pts, (bx, by, bz))
        end
    end

    # remove duplicates (can arise at the cell boundaries with the two-atom basis)
    sort!(pts, by = t -> (t[1], t[2], t[3]))
    uniq = Tuple{Float64,Float64,Float64}[]
    tol = 1e-9 * a
    for p in pts
        if isempty(uniq) ||
           abs(p[1] - uniq[end][1]) > tol ||
           abs(p[2] - uniq[end][2]) > tol ||
           abs(p[3] - uniq[end][3]) > tol
            push!(uniq, p)
        end
    end

    N = length(uniq)
    centers = Matrix{Float64}(undef, 3, N)
    for (j, p) in enumerate(uniq)
        centers[1, j] = p[1];  centers[2, j] = p[2];  centers[3, j] = p[3]
    end
    agg = SphereAggregate(centers, fill(Float64(radius), N);
            metadata=Dict{String,Any}(
                "source"         => "make_hcp_cluster",
                "radius"         => Float64(radius),
                "cluster_radius" => Float64(cluster_radius),
                "gap"            => Float64(gap),
            ))
    return recenter!(agg)
end

# Internal helper for simple cubic-cell lattices (FCC, BCC, simple cubic).
function _fill_lattice_sphere(radius::Real, cluster_radius::Real,
                              a_cell::Float64,
                              basis::Tuple{Vararg{NTuple{3,Float64}}};
                              source::String, gap::Float64)
    rmax2 = Float64(cluster_radius)^2
    nmax = Int(ceil(Float64(cluster_radius) / a_cell + 2))
    pts = Tuple{Float64,Float64,Float64}[]
    for k in -nmax:nmax, j in -nmax:nmax, i in -nmax:nmax
        for (bx, by, bz) in basis
            x = (i + bx) * a_cell
            y = (j + by) * a_cell
            z = (k + bz) * a_cell
            if x*x + y*y + z*z <= rmax2
                push!(pts, (x, y, z))
            end
        end
    end
    N = length(pts)
    centers = Matrix{Float64}(undef, 3, N)
    for (jj, p) in enumerate(pts)
        centers[1, jj] = p[1];  centers[2, jj] = p[2];  centers[3, jj] = p[3]
    end
    agg = SphereAggregate(centers, fill(Float64(radius), N);
            metadata=Dict{String,Any}(
                "source"         => source,
                "radius"         => Float64(radius),
                "cluster_radius" => Float64(cluster_radius),
                "gap"            => gap,
                "a_cell"         => a_cell,
            ))
    return recenter!(agg)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Mesh-size heuristic
# ──────────────────────────────────────────────────────────────────────────────

"""
    adaptive_lc_aggregate(agg::SphereAggregate;
                          wl_0=NaN, m_p_max=NaN, N_pw=10,
                          N_per_radius=3) -> Float64

Characteristic mesh size for a sphere-aggregate tet-mesh, taken as the
minimum of:
1. Wavelength constraint `λ₀ / (|m_p|_max · N_pw)` (if `wl_0`, `m_p_max` given).
2. Geometric constraint `r_min / N_per_radius` based on the smallest
   monomer radius.

Defaults are consistent with the GRE mesher (`N_pw=10`, `r / 3`).
"""
function adaptive_lc_aggregate(agg::SphereAggregate;
                               wl_0::Real=NaN, m_p_max::Real=NaN,
                               N_pw::Int=10, N_per_radius::Int=3)
    N_per_radius >= 1 || throw(ArgumentError("N_per_radius must be >= 1"))
    r_min = minimum(agg.radii)
    candidates = Float64[]
    if !isnan(wl_0) && !isnan(m_p_max) && m_p_max > 0
        push!(candidates, Float64(wl_0) / (Float64(m_p_max) * N_pw))
    end
    push!(candidates, r_min / N_per_radius)
    return minimum(candidates)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Gmsh OCC tetrahedral meshing of a sphere aggregate
# ──────────────────────────────────────────────────────────────────────────────

"""
    mesh_sphere_aggregate(agg::SphereAggregate;
                          overlap_factor=0.02,
                          neck_ratio=nothing,
                          lc=nothing,
                          wl_0=NaN, m_p_max=NaN, N_pw=10, N_per_radius=3,
                          mesh_path=nothing,
                          verbosity=0)
        -> (mesh::TetMesh, mesh_path::String)

Generate a tetrahedral mesh of a sphere aggregate via Gmsh OpenCASCADE.

Every monomer radius is inflated by a factor `(1 + overlap_factor)` prior
to meshing, producing an explicit neck at every pair of originally
touching monomers.  `overlap_factor = 0.02` (the default) yields a
neck-to-radius ratio of about 0.20 for equal monomers.

If `neck_ratio` is provided it overrides `overlap_factor` via
[`neck_ratio_to_overlap`](@ref).

# Arguments
- `overlap_factor` : uniform radius inflation ε ≥ 0.  Default `0.02`.
- `neck_ratio`     : target neck-to-radius ratio (equal-monomer convention);
  if given, overrides `overlap_factor`.
- `lc`             : characteristic mesh size; defaults to
  [`adaptive_lc_aggregate`](@ref).
- `wl_0`, `m_p_max`, `N_pw`, `N_per_radius` : forwarded to
  `adaptive_lc_aggregate` when `lc === nothing`.
- `mesh_path`      : `.msh` output path; defaults to a tempfile.
- `verbosity`      : Gmsh `General.Terminal` level (0 = silent).

# Returns
- `mesh::TetMesh`     : the tet mesh parsed via [`read_msh`](@ref)
- `mesh_path::String` : the `.msh` file actually written (useful when a
  temp path is auto-generated)
"""
function mesh_sphere_aggregate(agg::SphereAggregate;
                               overlap_factor::Real=0.02,
                               neck_ratio::Union{Real,Nothing}=nothing,
                               lc::Union{Real,Nothing}=nothing,
                               wl_0::Real=NaN, m_p_max::Real=NaN,
                               N_pw::Int=10, N_per_radius::Int=3,
                               mesh_path::Union{AbstractString,Nothing}=nothing,
                               verbosity::Int=0,
                               rescale_to_target_volume::Bool=true)
    eps_eff = neck_ratio === nothing ?
        Float64(overlap_factor) :
        neck_ratio_to_overlap(neck_ratio)
    eps_eff >= 0 || throw(ArgumentError("overlap_factor must be >= 0"))

    lc_eff = lc === nothing ?
        adaptive_lc_aggregate(agg;
            wl_0=wl_0, m_p_max=m_p_max,
            N_pw=N_pw, N_per_radius=N_per_radius) :
        Float64(lc)

    path = mesh_path === nothing ? (tempname() * ".msh") : String(mesh_path)

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.model.add("sphere_aggregate")

        N = n_monomers(agg)
        sphere_tags = Vector{Int}(undef, N)
        @inbounds for j in 1:N
            r_inflated = agg.radii[j] * (1.0 + eps_eff)
            sphere_tags[j] = gmsh.model.occ.addSphere(
                agg.centers[1, j], agg.centers[2, j], agg.centers[3, j],
                r_inflated,
            )
        end

        # Fuse all spheres.  For overlapping monomers OCC returns a single
        # merged volume; for disjoint monomers (gap > 0 with ε = 0) it
        # returns multiple volumes — we collect them all.
        local volume_tags::Vector{Int}
        if N > 1
            objects = [(3, sphere_tags[1])]
            tools   = [(3, t) for t in sphere_tags[2:end]]
            outDimTags, _ = gmsh.model.occ.fuse(objects, tools)
            isempty(outDimTags) &&
                throw(ErrorException("Gmsh OCC fuse produced no output volumes"))
            volume_tags = Int[dt[2] for dt in outDimTags if dt[1] == 3]
        else
            volume_tags = Int[sphere_tags[1]]
        end
        gmsh.model.occ.synchronize()

        # Tag every resulting volume as physical group 1 (matches GRE mesher).
        gmsh.model.addPhysicalGroup(3, volume_tags, 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_eff)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_eff)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end

    mesh = read_msh(path)

    # ── Volume-preserving rescale (block-DDA_Py parity, v0.7.7+) ──────
    # Target volume = sum of ideal monomer volumes (= (4π/3)·r_v_total³
    # for an equal-radius cluster).  Scale node coordinates so the
    # discretised mesh hits this exactly, matching DDA's Target spacing
    # adjustment and removing the discretisation-volume bias from paper
    # 1:1 cross-solver plots.  Inter-monomer geometry (gap, axis) is
    # preserved by the uniform scaling.
    #
    # Disabled (`rescale_to_target_volume=false`) when callers want the
    # gmsh-faceted overlap geometry preserved — e.g. analytical
    # overlap-volume validation tests where V_mesh should match
    # `V_inflated_spheres − V_lens` rather than the no-overlap
    # `monomer_volume_sum`.
    if rescale_to_target_volume
        V_target = monomer_volume_sum(agg)
        V_mesh   = total_volume(mesh)
        s = (V_target / V_mesh) ^ (1/3)
        apply_scale!(mesh, s)
    end

    return mesh, path
end
