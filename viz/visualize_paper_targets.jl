# Combined 1x3 panel rendering of the three non-spherical paper targets
# (oblate spheroid, sphere doublet, Gaussian random ellipsoid) used in
# Sec. 4.2 / 4.3 of the block-DDA_Py / block-VIEM.jl benchmark paper.
#
# Each panel shows the tetrahedral surface mesh in the laboratory frame
# at the convergence-test orientation (α,β,γ) = (0,0,0), together with
# the lab-frame x/y/z axis arrows; the +z axis (black, bold) is the
# fixed CAS-v2 incident propagation direction.
#
# Output: a single PDF written to the path supplied on the command
# line (or fig_paper_targets.pdf in this directory by default).
#
# Usage:
#     julia --project=viz viz/visualize_paper_targets.jl /abs/path/to/out.pdf

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using Printf
using Random
using CairoMakie
using GeometryBasics: Point3f, Vec3f
import GeometryBasics

include(joinpath(@__DIR__, "render_helpers.jl"))

# ──────────────────────────────────────────────────────────────────────
#  Per-panel rendering into a supplied Axis3 (no Figure of its own)
# ──────────────────────────────────────────────────────────────────────

function _render_panel!(ax::Axis3, mesh, euler_deg::NTuple{3,<:Real},
                        axis_range::Real;
                        edge_linewidth::Real = 0.4,
                        surface_color = :lightsteelblue,
                        azimuth::Real = 1.275π,
                        elevation::Real = π/8)
    α_rad = deg2rad(euler_deg[1])
    β_rad = deg2rad(euler_deg[2])
    γ_rad = deg2rad(euler_deg[3])
    R = euler_zyz_matrix(α_rad, β_rad, γ_rad)

    nodes_L = [R * n for n in mesh.nodes]
    tris    = boundary_triangles_oriented(mesh)

    # Opaque solid surface — rasterised so PDF viewers render the
    # mesh as a single raster image instead of thousands of small
    # vector triangles (much faster opening / scrolling in
    # Acrobat / Foxit / Preview).
    surf_pts   = [Point3f(n[1], n[2], n[3]) for n in nodes_L]
    surf_faces = [GeometryBasics.TriangleFace{Int}(t[1], t[2], t[3]) for t in tris]
    surf_mesh  = GeometryBasics.Mesh(surf_pts, surf_faces)
    mesh!(ax, surf_mesh; color = surface_color, shading = true,
          rasterize = 4)

    # Hidden-line wireframe (front-facing edges only) — also
    # rasterised, since the boundary triangulation can have several
    # hundred edges that bog down vector renderers.
    view_dir = camera_direction(Float64(azimuth), Float64(elevation))
    vis_edges = visible_edges(tris, nodes_L, view_dir)
    seg_pts = Point3f[]
    sizehint!(seg_pts, 2 * length(vis_edges))
    for (i, j) in vis_edges
        n1 = nodes_L[i]; n2 = nodes_L[j]
        push!(seg_pts, Point3f(n1[1], n1[2], n1[3]))
        push!(seg_pts, Point3f(n2[1], n2[2], n2[3]))
    end
    linesegments!(ax, seg_pts; color = :black, linewidth = edge_linewidth,
                  rasterize = 4)

    # Lab-frame axis arrows from the origin, extending beyond the
    # particle bounding box so the tips are always unobstructed by
    # the geometry.  z (black, bold) is the CAS-v2 incident
    # propagation direction; x (red) and y (blue) are the auxiliary
    # lab-frame axes.
    r_axis     = axis_range
    arrow_len  = 0.85 * r_axis   # stays inside the axis bbox
    cone_len   = 0.15 * arrow_len   # ~2.5:1 cone aspect (chunkier
    cone_rad   = 0.060 * arrow_len  # arrowhead, less needle-like)
    shaft_z    = 2.5
    shaft_xy   = 2.0

    # Lab-frame axes: thick line shaft (lines!) + cone tip (mesh!)
    # built explicitly from triangle facets oriented along the axis
    # direction.  This combo gives a visible pointed arrowhead in
    # every panel (Makie's `arrows3d!` produced invisible cone tips
    # in this multi-panel Axis3 layout).
    function _cone_mesh(tip::Point3f, axis::Vec3f, base_radius::Float64,
                        len::Float64; nseg::Int = 24)
        L = sqrt(axis[1]^2 + axis[2]^2 + axis[3]^2)
        ax_unit = Vec3f(axis[1]/L, axis[2]/L, axis[3]/L)
        base_center = Point3f(tip[1] - ax_unit[1]*len,
                              tip[2] - ax_unit[2]*len,
                              tip[3] - ax_unit[3]*len)
        # Build two unit vectors orthogonal to ax_unit
        ref = abs(ax_unit[3]) < 0.9 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0)
        u = Vec3f(ax_unit[2]*ref[3] - ax_unit[3]*ref[2],
                  ax_unit[3]*ref[1] - ax_unit[1]*ref[3],
                  ax_unit[1]*ref[2] - ax_unit[2]*ref[1])
        un = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
        u = Vec3f(u[1]/un, u[2]/un, u[3]/un)
        v = Vec3f(ax_unit[2]*u[3] - ax_unit[3]*u[2],
                  ax_unit[3]*u[1] - ax_unit[1]*u[3],
                  ax_unit[1]*u[2] - ax_unit[2]*u[1])
        pts = Point3f[tip]                             # vertex 1: apex
        for k in 0:(nseg-1)
            θ = 2π * k / nseg
            cu = cos(θ); cv = sin(θ)
            push!(pts, Point3f(
                base_center[1] + base_radius * (cu * u[1] + cv * v[1]),
                base_center[2] + base_radius * (cu * u[2] + cv * v[2]),
                base_center[3] + base_radius * (cu * u[3] + cv * v[3]),
            ))
        end
        push!(pts, base_center)                         # vertex N+2: base center
        faces = GeometryBasics.TriangleFace{Int}[]
        for k in 1:nseg
            kn = k == nseg ? 1 : k + 1
            push!(faces, GeometryBasics.TriangleFace{Int}(1, k+1, kn+1))   # side
            push!(faces, GeometryBasics.TriangleFace{Int}(nseg+2, kn+1, k+1)) # base
        end
        return GeometryBasics.Mesh(pts, faces)
    end

    # x and y: thin line shaft + 3D cone tip (the x/y axes are viewed
    # mostly side-on so the cone reads as a clean arrowhead).
    function _draw_cone_axis!(direction::Vec3f, color, lw)
        tip = Point3f(direction[1], direction[2], direction[3])
        lines!(ax, [Point3f(0, 0, 0), tip];
               color = color, linewidth = lw)
        cone = _cone_mesh(tip, direction, Float64(cone_rad), Float64(cone_len))
        mesh!(ax, cone; color = color, shading = true)
    end
    _draw_cone_axis!(Vec3f(arrow_len, 0, 0), :red,  shaft_xy)
    _draw_cone_axis!(Vec3f(0, arrow_len, 0), :blue, shaft_xy)

    # z: plain line segment only (no arrowhead).  The arrowhead will
    # be added by hand to the saved figure file outside this script.
    lines!(ax, [Point3f(0, 0, 0), Point3f(0, 0, arrow_len)];
           color = :black, linewidth = shaft_z)

    # Axis identification is given in the figure caption (z = incident,
    # x = red, y = blue); no in-panel text labels are needed.

    ax.limits = (-axis_range, axis_range,
                 -axis_range, axis_range,
                 -axis_range, axis_range)
    # aspect = (1, 1, 1) forces equal data ratios on all three axes,
    # independent of camera angle, so the three panels render at
    # consistent screen size.
    ax.aspect    = (1, 1, 1)
    ax.azimuth   = Float64(azimuth)
    ax.elevation = Float64(elevation)
    return ax
end

# ──────────────────────────────────────────────────────────────────────
#  Build the three meshes at paper convergence-test parameters
# ──────────────────────────────────────────────────────────────────────

function build_paper_meshes(; r_ve::Float64 = 0.10)
    rng = MersenneTwister(42)

    # Mesh element size is intentionally coarse here — these are
    # visualisation meshes, NOT the production-sweep meshes.  Coarse
    # meshes keep the embedded PDF light enough for typical viewers
    # (Adobe Acrobat, Foxit, etc.) without sacrificing the qualitative
    # shape impression.

    # Oblate spheroid: 3:3:1 → bc_ratio=3, ab_ratio=1, β=0
    p_oblate = GREParams(r_ve, 3.0, 1.0, 0.0)
    c_oblate = gre_semi_axes(p_oblate)[3]
    mesh_oblate, _ = gre_mesh(p_oblate, rng;
                              lc = c_oblate / 1.5,
                              wl_0 = 0.638, m_p_max = 1.5, N_pw = 10)

    # Doublet: monomer radius R = r_ve / 2^(1/3), gap = 0.1R, no sintering
    R_mon      = r_ve / cbrt(2.0)
    agg_doub   = make_linear_chain(2, R_mon; gap = 0.1 * R_mon)
    mesh_doub, _ = mesh_sphere_aggregate(agg_doub;
                              neck_ratio   = 0.0,
                              lc           = R_mon / 2,
                              wl_0         = 0.638,
                              m_p_max      = 1.5,
                              N_pw         = 10,
                              N_per_radius = 3)

    # GRE: bc_ratio=1, ab_ratio=1, β=0.2 — kept noticeably finer
    # than the smooth shapes so the random surface deformation
    # remains visually expressive (the surface is rasterised
    # downstream, so a denser mesh does not bloat the PDF).
    p_gre = GREParams(r_ve, 1.0, 1.0, 0.2)
    c_gre = gre_semi_axes(p_gre)[3]
    mesh_gre, _ = gre_mesh(p_gre, MersenneTwister(7);
                           lc = c_gre / 4.5,
                           wl_0 = 0.638, m_p_max = 1.5, N_pw = 10)

    return (oblate = mesh_oblate, doublet = mesh_doub, gre = mesh_gre)
end

# ──────────────────────────────────────────────────────────────────────
#  Combined 1x3 panel figure
# ──────────────────────────────────────────────────────────────────────

function render_combined(out_path::AbstractString;
                         r_ve::Float64 = 0.10,
                         figsize::NTuple{2,Int} = (1280, 600))
    meshes = build_paper_meshes(; r_ve = r_ve)

    # Common axis range across all panels: largest extent over all
    # three meshes, with a 30 % margin so the corner triad is
    # visible.  Same view angle for all panels.
    function _max_extent(mesh)
        return maximum(maximum(abs, n) for n in mesh.nodes)
    end
    common_range = max(_max_extent(meshes.oblate),
                       _max_extent(meshes.doublet),
                       _max_extent(meshes.gre)) * 1.30

    # figure_padding = (left, right, bottom, top) in pt — extra
    # horizontal padding so the leftmost ticks of (a) and rightmost
    # of (c) don't run into the LaTeX page text margins after
    # \includegraphics scales the figure to \linewidth.
    fig = Figure(size = figsize, figure_padding = (140, 140, 25, 25))

    # Per-panel view angle: oblate and GRE share the default
    # (azimuth = 1.275π, elevation = π/8); the doublet uses the
    # broadside view (azimuth = π/2 + π/16, elevation = π/9) so the
    # axis-direction surface gap between the two monomers is visible.
    # Per-panel labels are just (a)/(b)/(c); shape identifications
    # (oblate 3:3:1, doublet, GRS) are described in the figure caption
    # in the paper.
    panels = (
        (mesh = meshes.oblate,  euler = (0, 0, 0),
         az = 1.275π,         el = π/8,         label = "(a)"),
        (mesh = meshes.doublet, euler = (0, 0, 0),
         az = π/2 + π/16,     el = π/9,         label = "(b)"),
        (mesh = meshes.gre,     euler = (0, 0, 0),
         az = 1.275π,         el = π/8,         label = "(c)"),
    )

    # Layout: panel | gap | panel | gap | panel.  Outer page-margin
    # padding is provided by `figure_padding` on the Figure itself.
    for (i, p) in enumerate(panels)
        ax = Axis3(fig[1, 2*i - 1];
                   xlabel = "x [μm]",
                   ylabel = "y [μm]",
                   zlabel = "z [μm]",
                   title  = p.label,
                   titlesize = 24,                 # large (a)/(b)/(c) label
                   titlefont = :bold,
                   xlabelsize = 12, ylabelsize = 12, zlabelsize = 12,
                   xticks = [-0.2, -0.1, 0.0, 0.1, 0.2],
                   yticks = [-0.2, -0.1, 0.0, 0.1, 0.2],
                   zticks = [-0.2, -0.1, 0.0, 0.1, 0.2])
        _render_panel!(ax, p.mesh, p.euler, common_range;
                       azimuth = p.az, elevation = p.el)
    end
    # Total panel + gap budget must fit within the layout area
    # (figsize.x − 2·padding.x = 1280 − 280 = 1000 px) so no axis
    # label gets clipped at the left/right edges.
    panel_w_px  = 290   # slightly narrower panels so total fits
    gap_ab_px   = 10    # narrow gap (a)–(b): doublet sits leftward
    gap_bc_px   = 60    # wider gap (b)–(c) gives the +z labels space
    # 3·290 + 10 + 60 = 940 px (60 px slack within the 1000 px area)
    colsize!(fig.layout, 1, Fixed(panel_w_px))
    colsize!(fig.layout, 2, Fixed(gap_ab_px))
    colsize!(fig.layout, 3, Fixed(panel_w_px))
    colsize!(fig.layout, 4, Fixed(gap_bc_px))
    colsize!(fig.layout, 5, Fixed(panel_w_px))

    save(out_path, fig; px_per_unit = 2)
    # Also write a PNG sibling so consumers that prefer raster
    # (slide decks, README previews) don't have to convert by hand.
    out_png = replace(String(out_path), r"\.pdf$"i => ".png")
    if out_png != String(out_path)
        save(out_png, fig; px_per_unit = 2)
        @info "Wrote target-geometry figure" pdf=out_path png=out_png range=common_range
    else
        @info "Wrote target-geometry figure" path=out_path range=common_range
    end
    return fig
end

# ──────────────────────────────────────────────────────────────────────
#  Script entry point
# ──────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] :
          joinpath(@__DIR__, "figs", "fig_paper_targets.pdf")
    isdir(dirname(out)) || mkpath(dirname(out))
    render_combined(out)
end
