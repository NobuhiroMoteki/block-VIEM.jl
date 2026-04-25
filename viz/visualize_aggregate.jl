# Wireframe visualisation of a discretised sphere-aggregate target for
# block-VIEM.jl.
#
# Builds a `SphereAggregate` (linear chain, planar array, FCC/BCC/HCP
# cluster, or arbitrary monomer-center/radius list), meshes it via
# `mesh_sphere_aggregate` with a configurable `neck_ratio`, and renders
# the surface wireframe in the laboratory frame together with x/y/z
# axis arrows.  The z-axis coincides with the incident beam direction.
#
# Mirrors the structure of `visualize_gre.jl` and shares the same
# boundary-extraction / Euler-rotation helpers.
#
# Usage:
#     julia --project=viz viz/visualize_aggregate.jl                # full gallery
#     include("viz/visualize_aggregate.jl"); visualize_aggregate(...)  # interactive

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using Printf
using Random
using CairoMakie
using GeometryBasics: Point3f, Vec3f, TriangleFace
import GeometryBasics

# Hidden-edge removal helpers, opaque-surface rendering helpers, and
# Euler/camera rotation utilities are shared with `visualize_gre.jl`.
include(joinpath(@__DIR__, "render_helpers.jl"))

# ──────────────────────────────────────────────────────────────────────────────
#  Aggregate visualisation
# ──────────────────────────────────────────────────────────────────────────────

"""
    visualize_aggregate(agg, euler_deg; output_path, neck_ratio,
                        lc, wl_0, m_p_max, N_pw, N_per_radius,
                        show_centroids, edge_alpha, edge_linewidth,
                        figsize, title, save_fig)
        -> (mesh, fig)

Build a tetrahedral mesh of a `SphereAggregate` via `mesh_sphere_aggregate`,
rotate the lab-frame nodes by ZYZ Euler angles `(α, β, γ)` in degrees,
and render the boundary-face wireframe.

Keyword arguments mirror `visualize_gre`; in addition:

- `neck_ratio` — interpretable contact-circle radius / monomer radius.
  Default `0.20` matches the equal-monomer convention used throughout
  the sphere-aggregate API (corresponds to `overlap_factor ≈ 0.02`).
- `N_per_radius` — geometric mesh resolution forwarded to
  `adaptive_lc_aggregate` when `lc === nothing`.
- `title` — figure title; an auto-generated string is used when
  `nothing` (default).
- `surface_color` — fill colour of the opaque boundary surface
  (default `:lightsteelblue`).  The surface is rendered fully opaque
  so that back-face mesh edges are correctly occluded by the merged
  geometry; otherwise a pure wireframe lets back-face edges show
  through and visually mimics two distinct complete spheres
  regardless of `neck_ratio`.
- `draw_wireframe` — overlay mesh-edge wireframe on the solid
  surface (default `true`).  Only front-facing edges are visible,
  depth-tested against the opaque surface.
"""
function visualize_aggregate(agg::SphereAggregate, euler_deg::NTuple{3,<:Real};
                             output_path::Union{AbstractString,Nothing}=nothing,
                             neck_ratio::Real=0.20,
                             lc::Union{Float64,Nothing}=nothing,
                             wl_0::Real=0.638, m_p_max::Real=1.5,
                             N_pw::Int=10, N_per_radius::Int=3,
                             show_centroids::Bool=false,
                             edge_linewidth::Real=0.5,
                             surface_color=:lightsteelblue,
                             draw_wireframe::Bool=true,
                             azimuth::Real=1.275π,
                             elevation::Real=π/8,
                             figsize::NTuple{2,Int}=(820, 760),
                             title::Union{AbstractString,Nothing}=nothing,
                             save_fig::Bool=true)

    mesh, _ = mesh_sphere_aggregate(agg;
                neck_ratio   = neck_ratio,
                lc           = lc,
                wl_0         = wl_0,
                m_p_max      = m_p_max,
                N_pw         = N_pw,
                N_per_radius = N_per_radius)

    α_rad = deg2rad(euler_deg[1])
    β_rad = deg2rad(euler_deg[2])
    γ_rad = deg2rad(euler_deg[3])
    R = euler_zyz_matrix(α_rad, β_rad, γ_rad)

    nodes_L     = [R * n for n in mesh.nodes]
    centroids_L = [R * c for c in mesh.tet_centroids]

    # Outward-oriented boundary triangles in the lab frame so the
    # cross product `(p2-p1) × (p3-p1)` points outward — needed for
    # the manual hidden-line removal below.
    tris_particle = boundary_triangles_oriented(mesh)
    tris = tris_particle  # node indices are unchanged by rotation

    axis_range = maximum(maximum(abs, n) for n in nodes_L) * 1.20
    axis_range = max(axis_range, 1e-6)

    fig = Figure(size=figsize)
    auto_title = @sprintf("N_mono=%d  η=%.2f  N_tet=%d  Euler=(%d,%d,%d)°",
                          n_monomers(agg), neck_ratio, n_tets(mesh),
                          Int(round(euler_deg[1])),
                          Int(round(euler_deg[2])),
                          Int(round(euler_deg[3])))
    az = Float64(azimuth)
    el = Float64(elevation)
    ax = Axis3(fig[1, 1];
               aspect    = :data,
               xlabel    = "x [μm]",
               ylabel    = "y [μm]",
               zlabel    = "z [μm]  (incident)",
               title     = title === nothing ? auto_title : String(title),
               titlesize = 11,
               limits    = (-axis_range, axis_range,
                            -axis_range, axis_range,
                            -axis_range, axis_range),
               azimuth   = az,
               elevation = el)

    # ── Opaque solid surface of the boundary mesh ────────────────────
    # OCC fuse produces a single merged volume for overlapping
    # monomers, so the boundary surface IS a peanut/cluster shape.
    # Render it fully opaque with directional shading so the back
    # hemisphere is hidden by the surface itself.
    surf_pts   = [Point3f(n[1], n[2], n[3]) for n in nodes_L]
    surf_faces = [GeometryBasics.TriangleFace{Int}(t[1], t[2], t[3]) for t in tris]
    surf_mesh  = GeometryBasics.Mesh(surf_pts, surf_faces)
    mesh!(ax, surf_mesh;
          color   = surface_color,
          shading = true)

    # ── Hidden-line wireframe overlay ────────────────────────────────
    # CairoMakie has no per-pixel depth test for line primitives, so
    # `wireframe!` would draw back-face edges through the opaque
    # surface and — for non-convex bodies (doublets, chains, fused
    # clusters) — also draw front-facing edges that are physically
    # blocked by other parts of the same surface.  `visible_edges`
    # does both backface culling AND a Möller–Trumbore self-occlusion
    # test against the boundary mesh, so only edges actually visible
    # from the camera (silhouette included) are drawn.
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

    if show_centroids
        num = n_tets(mesh)
        marker_size  = clamp(6 / (log(max(num, 2)) + 1.0), 1.2, 4.0)
        marker_alpha = clamp(18 / (log(max(num, 2)) + 4.0), 0.15, 0.55)
        scatter!(ax,
                 [Point3f(c[1], c[2], c[3]) for c in centroids_L];
                 markersize = marker_size,
                 color      = (:red, marker_alpha))
    end

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

    return mesh, fig
end

# ──────────────────────────────────────────────────────────────────────────────
#  Example gallery
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_aggregate_gallery(; figs_dir)

Render representative sphere-aggregate shapes used in the paper-production
calculation matrix (see `docs/descriptions_particle_shape_model.md`) and
save PNGs under `figs_dir` (default: `<this-file>/figs/`).
"""
function run_aggregate_gallery(; figs_dir::AbstractString = joinpath(@__DIR__, "figs"))

    isdir(figs_dir) || mkpath(figs_dir)

    R = 0.10            # monomer radius [μm], typical of paper doublet a_eq ≈ 0.126 μm
    lc_doublet = R / 5  # fine enough to resolve the neck waist for η ≥ 0.20

    # Doublet axis is along x in the particle frame; keep (α,β,γ)=0 so
    # the lab-frame axis is also x.  Viewing from the +y direction
    # (azimuth ≈ π/2 + π/16) at small elevation puts the doublet in
    # broadside profile — the merged peanut neck waist is then on the
    # silhouette and immediately recognisable.
    doublet_euler = (0, 0, 0)
    doublet_az    = π/2 + π/16   # ~101°, mostly +y with slight +x bias
    doublet_el    = π/9          # ~20° above the xy-plane

    # ── 1. Doublet, η = 0.20 (touching, lightly sintered — narrow neck) ──
    agg_doublet_t = make_linear_chain(2, R; gap=0.0)
    visualize_aggregate(agg_doublet_t, doublet_euler;
        output_path = joinpath(figs_dir, "agg_doublet_eta020.png"),
        neck_ratio  = 0.20,
        lc          = lc_doublet,
        azimuth     = doublet_az,
        elevation   = doublet_el,
        title       = "Doublet  R=0.10 μm  η=0.20  (touching, sintered)")

    # ── 2. Doublet, η = 0.50 (heavy sintering, neck ~ R/2) ──
    visualize_aggregate(agg_doublet_t, doublet_euler;
        output_path = joinpath(figs_dir, "agg_doublet_eta050.png"),
        neck_ratio  = 0.50,
        lc          = lc_doublet,
        azimuth     = doublet_az,
        elevation   = doublet_el,
        title       = "Doublet  R=0.10 μm  η=0.50  (heavy sintering)")

    # ── 3. Doublet with gap = 0.1·R (paper convention; disjoint monomers) ──
    agg_doublet_gap = make_linear_chain(2, R; gap=0.1*R)
    visualize_aggregate(agg_doublet_gap, doublet_euler;
        output_path = joinpath(figs_dir, "agg_doublet_gap010R.png"),
        neck_ratio  = 0.0,
        lc          = lc_doublet,
        azimuth     = doublet_az,
        elevation   = doublet_el,
        title       = "Doublet  R=0.10 μm  gap=0.1R  η=0  (paper convention)")

    # ── 4. Linear chain, N=5, sintered ──
    agg_chain5 = make_linear_chain(5, R; gap=0.0)
    visualize_aggregate(agg_chain5, (0, 30, 20);
        output_path = joinpath(figs_dir, "agg_chain5_eta020.png"),
        neck_ratio  = 0.20,
        lc          = R / 3.5,
        title       = "Linear chain  N=5  R=0.10 μm  η=0.20")

    # ── 5. FCC cluster (compact close-packed), small ──
    agg_fcc = make_fcc_cluster(0.05; cluster_radius=0.13)
    visualize_aggregate(agg_fcc, (20, 35, 0);
        output_path = joinpath(figs_dir, "agg_fcc_cluster.png"),
        neck_ratio  = 0.20,
        lc          = 0.05 / 3,
        title       = @sprintf("FCC cluster  N=%d  R=0.05 μm  η=0.20",
                                n_monomers(agg_fcc)))

    # ── 6. HCP cluster ──
    agg_hcp = make_hcp_cluster(0.05; cluster_radius=0.13)
    visualize_aggregate(agg_hcp, (10, 50, 0);
        output_path = joinpath(figs_dir, "agg_hcp_cluster.png"),
        neck_ratio  = 0.20,
        lc          = 0.05 / 3,
        title       = @sprintf("HCP cluster  N=%d  R=0.05 μm  η=0.20",
                                n_monomers(agg_hcp)))

    @info "Aggregate gallery complete" dir=figs_dir
    return nothing
end

# If run as a script, produce the gallery
if abspath(PROGRAM_FILE) == @__FILE__
    run_aggregate_gallery()
end
