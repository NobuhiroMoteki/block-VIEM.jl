# Wireframe visualisation of a discretised GRE target for block-VIEM.jl.
#
# Specify GRE parameters (r_v_base, bc_ratio, ab_ratio, beta) and an Euler
# angle triple (alpha, beta, gamma) in degrees, build the tetrahedral mesh
# via `gre_mesh`, and render the surface wireframe + optional internal
# tet centroids in the laboratory frame, together with x/y/z axis arrows.
# The z-axis coincides with the incident beam direction.
#
# Mirrors block-DDA_Py/run_gaussian_ellipsoid.ipynb in spirit, but uses
# the VIEM tetrahedral discretisation.
#
# Usage:
#     julia --project=viz viz/visualize_gre.jl          # runs the example sweep
#     include("viz/visualize_gre.jl"); visualize_gre(...)  # interactive use

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using Printf
using Random
using CairoMakie
using GeometryBasics: Point3f, Vec3f, TriangleFace
import GeometryBasics

# Hidden-edge removal helpers (opaque-surface rendering with proper
# back-face occlusion), Euler/camera rotation utilities are shared
# with `visualize_aggregate.jl`.
include(joinpath(@__DIR__, "render_helpers.jl"))

# ──────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

"""
    visualize_gre(params, euler_deg; output_path, kwargs...) -> (mesh, r_ve)

Build the tetrahedral mesh of a GRE particle, rotate it to the given
laboratory-frame orientation, and render a wireframe figure.

# Arguments
- `params::GREParams` — GRE shape parameters
- `euler_deg::NTuple{3,Real}` — (α, β, γ) ZYZ Euler angles in **degrees**

# Keywords
- `output_path`     — PNG path to save the figure (required if `save_fig=true`)
- `lc`              — explicit Gmsh characteristic length [μm]. If `nothing`,
                       uses `adaptive_lc(params; wl_0, m_p_max, N_pw)`. For
                       readable visualisation a coarser mesh is typically
                       preferred (e.g. `lc = c/2` where `c` is the smallest
                       semi-axis); the default adaptive value is tuned for
                       physics accuracy, not visualisation.
- `wl_0`            — vacuum wavelength [μm] used to size the mesh (default 0.638)
- `m_p_max`         — maximum |m_p| used to size the mesh (default 1.5)
- `N_pw`            — elements per wavelength inside the particle (default 10)
- `rng_seed`        — seed for the GRE deformation field (default 42)
- `show_centroids`  — overlay tet centroids as small dots (default `false`)
- `edge_linewidth`  — linewidth for wireframe edges (default 0.5)
- `surface_color`   — fill colour of the opaque boundary surface
                      (default `:lightsteelblue`)
- `draw_wireframe`  — overlay hidden-line-removed mesh edges (default `true`)
- `azimuth`, `elevation` — Axis3 camera angles (default `1.275π`, `π/8`)
- `figsize`         — figure size in points (default (760, 760))
- `save_fig`        — write PNG to `output_path` (default `true`)
"""
function visualize_gre(params::GREParams, euler_deg::NTuple{3,<:Real};
                       output_path::Union{AbstractString,Nothing}=nothing,
                       lc::Union{Float64,Nothing}=nothing,
                       wl_0::Real=0.638, m_p_max::Real=1.5, N_pw::Int=10,
                       rng_seed::Int=42,
                       show_centroids::Bool=false,
                       edge_linewidth::Real=0.5,
                       surface_color=:lightsteelblue,
                       draw_wireframe::Bool=true,
                       azimuth::Real=1.275π,
                       elevation::Real=π/8,
                       figsize::NTuple{2,Int}=(760, 760),
                       save_fig::Bool=true)

    rng = MersenneTwister(rng_seed)
    mesh, r_ve = gre_mesh(params, rng;
                          lc=lc, wl_0=wl_0, m_p_max=m_p_max, N_pw=N_pw)

    α_rad = deg2rad(euler_deg[1])
    β_rad = deg2rad(euler_deg[2])
    γ_rad = deg2rad(euler_deg[3])
    R = euler_zyz_matrix(α_rad, β_rad, γ_rad)

    # Rotate all nodes and centroids into the lab frame
    nodes_L     = [R * n for n in mesh.nodes]
    centroids_L = [R * c for c in mesh.tet_centroids]

    # Outward-oriented boundary triangles in the lab frame so the
    # cross product `(p2-p1) × (p3-p1)` points outward — needed for
    # backface culling and the Möller–Trumbore self-occlusion test.
    tris = boundary_triangles_oriented(mesh)

    # Axis range covers the whole rotated target plus a small margin
    axis_range = maximum(maximum(abs, n) for n in nodes_L) * 1.25
    axis_range = max(axis_range, 1e-6)

    fig = Figure(size=figsize)
    title = @sprintf("r_v_base=%.2fμm  bc=%.1f  ab=%.1f  β=%.2f  Euler=(%d,%d,%d)°  N_tet=%d  r_ve=%.3fμm",
                     params.r_v_base, params.bc_ratio, params.ab_ratio,
                     params.beta,
                     Int(round(euler_deg[1])), Int(round(euler_deg[2])),
                     Int(round(euler_deg[3])),
                     n_tets(mesh), r_ve)
    az = Float64(azimuth)
    el = Float64(elevation)
    ax = Axis3(fig[1, 1];
               aspect    = :data,
               xlabel    = "x [μm]",
               ylabel    = "y [μm]",
               zlabel    = "z [μm]  (incident)",
               title     = title,
               titlesize = 11,
               limits    = (-axis_range, axis_range,
                            -axis_range, axis_range,
                            -axis_range, axis_range),
               azimuth   = az,
               elevation = el)

    # ── Opaque solid surface of the boundary mesh ────────────────────
    # Render the boundary as a fully opaque solid with directional
    # shading; the back hemisphere is then hidden by the surface
    # itself.  Used in tandem with the hidden-line-removed wireframe
    # below — without this, transparent wireframes would leak both
    # back-face and self-occluded front-face edges (especially for
    # GREs with strong `β` deformation that can develop concavities).
    surf_pts   = [Point3f(n[1], n[2], n[3]) for n in nodes_L]
    surf_faces = [GeometryBasics.TriangleFace{Int}(t[1], t[2], t[3]) for t in tris]
    surf_mesh  = GeometryBasics.Mesh(surf_pts, surf_faces)
    mesh!(ax, surf_mesh;
          color   = surface_color,
          shading = true)

    # ── Hidden-line wireframe overlay ────────────────────────────────
    if draw_wireframe
        view_dir = camera_direction(az, el)
        vis_edges = visible_edges(tris, nodes_L, view_dir)
        seg_pts = Point3f[]
        sizehint!(seg_pts, 2 * length(vis_edges))
        for (i, j) in vis_edges
            n1 = nodes_L[i]; n2 = nodes_L[j]
            push!(seg_pts, Point3f(n1[1], n1[2], n1[3]))
            push!(seg_pts, Point3f(n2[1], n2[2], n2[3]))
        end
        linesegments!(ax, seg_pts;
                      color     = :black,
                      linewidth = edge_linewidth)
    end

    # ── Tet centroids (optional, analogue of DDA dipoles) ──────────────
    if show_centroids
        num = n_tets(mesh)
        marker_size  = clamp(6 / (log(max(num, 2)) + 1.0), 1.2, 4.0)
        marker_alpha = clamp(18 / (log(max(num, 2)) + 4.0), 0.15, 0.55)
        scatter!(ax,
                 [Point3f(c[1], c[2], c[3]) for c in centroids_L];
                 markersize = marker_size,
                 color      = (:red, marker_alpha))
    end

    # ── Lab-frame axis arrows ──────────────────────────────────────────
    #   z (incident beam): solid black, bold; x: red; y: blue
    r_axis = axis_range
    tiplen = 0.09 * r_axis
    tiprad = 0.04 * r_axis
    shaft_z = 0.010 * r_axis
    shaft_xy = 0.007 * r_axis
    origin = [Point3f(0.0, 0.0, 0.0)]

    arrows3d!(ax, origin, [Vec3f(0, 0, r_axis)];
              color = :black, shaftradius = shaft_z,
              tipradius = tiprad, tiplength = tiplen)
    arrows3d!(ax, origin, [Vec3f(r_axis, 0, 0)];
              color = :red,   shaftradius = shaft_xy,
              tipradius = tiprad, tiplength = tiplen)
    arrows3d!(ax, origin, [Vec3f(0, r_axis, 0)];
              color = :blue,  shaftradius = shaft_xy,
              tipradius = tiprad, tiplength = tiplen)

    if save_fig && output_path !== nothing
        save(output_path, fig; px_per_unit=2)
    end

    return mesh, r_ve, fig
end

# ──────────────────────────────────────────────────────────────────────────────
#  Example sweep (when run as a script)
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_example_gallery(; figs_dir)

Render a handful of representative GRE shapes × orientations and save
PNGs under `figs_dir` (default: `<this-file>/figs/`).
"""
function run_example_gallery(; figs_dir::AbstractString = joinpath(@__DIR__, "figs"),
                               wl_0::Real=0.638, m_p_max::Real=1.5, N_pw::Int=10)

    isdir(figs_dir) || mkpath(figs_dir)

    # For visualisation we pick `lc` coarser than the physics value so
    # the wireframe is readable. `c` is the smallest semi-axis.
    # β > 0 GREs need a finer mesh than the smooth ellipsoids so the
    # surface deformation is well-resolved (ℓ_corr = 0.3·c, paper
    # value is ℓ_c ~ 0.3·c/3 = 0.1·c; we use 0.2·c here).
    lc_for(p::GREParams) = p.beta > 0 ?
        gre_semi_axes(p)[3] / 5 :
        gre_semi_axes(p)[3] / 2.5

    examples = [
        (name   = "sphere",
         params = GREParams(0.30, 1.0, 1.0, 0.00),
         euler  = (0, 0, 0),
         seed   = 42),
        (name   = "oblate_bc3",
         params = GREParams(0.30, 3.0, 1.0, 0.00),
         euler  = (0, 60, 0),
         seed   = 42),
        (name   = "triaxial_bc2_ab15",
         params = GREParams(0.30, 2.0, 1.5, 0.00),
         euler  = (30, 45, 0),
         seed   = 42),
        (name   = "beta020_bc1_ab1",
         params = GREParams(0.30, 1.0, 1.0, 0.20),
         euler  = (0, 0, 0),
         seed   = 7),
        (name   = "beta020_bc15_ab15",
         params = GREParams(0.30, 1.5, 1.5, 0.20),
         euler  = (20, 60, 10),
         seed   = 3),
    ]

    for ex in examples
        path = joinpath(figs_dir, "gre_" * ex.name * ".png")
        @info "Rendering" name=ex.name output=path
        visualize_gre(ex.params, ex.euler;
                      output_path = path,
                      lc          = lc_for(ex.params),
                      wl_0        = wl_0,
                      m_p_max     = m_p_max,
                      N_pw        = N_pw,
                      rng_seed    = ex.seed)
    end
    @info "Gallery complete" dir=figs_dir n=length(examples)
    return nothing
end

# If run as a script, produce the gallery
if abspath(PROGRAM_FILE) == @__FILE__
    run_example_gallery()
end
