# AIM auxiliary Cartesian grid.
#
# The grid is uniform in all three axes with pitch `h`, anchored at
# `origin = (x0, y0, z0)` and extending to `(x0 + (Nx-1)h, ..., z0 + (Nz-1)h)`.
# Indexing is 1-based; linear indexing uses Julia's column-major convention
# `LinearIndices((Nx, Ny, Nz))`.
#
# Phase 3.1 of BlockVIEM.jl. Phase 3.2 (projection matrix W) builds on this.

"""
    AIMGrid

Uniform Cartesian auxiliary grid for the Adaptive Integral Method.

# Fields
- `origin::Vec3`            — physical position of grid point `(1, 1, 1)`
- `pitch::Float64`          — grid spacing (uniform across all axes)
- `dims::NTuple{3,Int}`     — `(Nx, Ny, Nz)`
"""
struct AIMGrid
    origin::Vec3
    pitch::Float64
    dims::NTuple{3,Int}
end

"""
    n_grid_points(grid::AIMGrid) -> Int
"""
@inline n_grid_points(grid::AIMGrid) = prod(grid.dims)

"""
    grid_point(grid::AIMGrid, i::Integer, j::Integer, k::Integer) -> Vec3

Physical position of grid point `(i, j, k)` (1-based).
"""
@inline function grid_point(grid::AIMGrid, i::Integer, j::Integer, k::Integer)
    h = grid.pitch
    return grid.origin + Vec3(h * (i - 1), h * (j - 1), h * (k - 1))
end

"""
    aim_grid(mesh::TetMesh; pitch::Float64, padding::Integer = 2) -> AIMGrid

Build a Cartesian grid that comfortably contains the mesh bounding box,
with `padding` empty cells of margin on every face. The grid spacing
`pitch` is supplied by the caller; pick it on the order of the mesh edge
length (e.g. 0.5 to 1.0 times the mean tet diameter).
"""
function aim_grid(mesh::TetMesh; pitch::Float64, padding::Integer = 2)
    pitch > 0 || throw(ArgumentError("pitch must be positive, got $pitch"))
    padding >= 0 || throw(ArgumentError("padding must be non-negative, got $padding"))

    nodes = mesh.nodes
    isempty(nodes) && throw(ArgumentError("mesh has no nodes"))

    first_node = first(nodes)
    xmin, ymin, zmin = first_node[1], first_node[2], first_node[3]
    xmax, ymax, zmax = xmin, ymin, zmin
    @inbounds for n in Iterators.drop(nodes, 1)
        xmin = min(xmin, n[1]); xmax = max(xmax, n[1])
        ymin = min(ymin, n[2]); ymax = max(ymax, n[2])
        zmin = min(zmin, n[3]); zmax = max(zmax, n[3])
    end

    pad = padding * pitch
    origin = Vec3(xmin - pad, ymin - pad, zmin - pad)
    extent_x = (xmax - xmin) + 2pad
    extent_y = (ymax - ymin) + 2pad
    extent_z = (zmax - zmin) + 2pad
    Nx = ceil(Int, extent_x / pitch) + 1
    Ny = ceil(Int, extent_y / pitch) + 1
    Nz = ceil(Int, extent_z / pitch) + 1
    return AIMGrid(origin, pitch, (Nx, Ny, Nz))
end

"""
    grid_stencil(grid::AIMGrid, c::Vec3, M::Integer) -> Vector{Int}

Return the linear indices (column-major) of the `M^3` grid points forming a
cube around `c`. For odd `M` the cube is centred on the grid point nearest
`c`; for even `M` the cube spans the cell `[i₀, i₀+M-1] × ...` whose
lower-left corner is the largest grid point `≤ c`.

Out-of-bounds entries are silently dropped, so the returned vector may
contain fewer than `M^3` indices when `c` lies near the grid boundary.
This is the responsibility of the caller (use enough `padding` in
[`aim_grid`](@ref) to ensure interior basis functions get a full stencil).
"""
function grid_stencil(grid::AIMGrid, c::Vec3, M::Integer)
    M >= 1 || throw(ArgumentError("M must be >= 1, got $M"))
    h = grid.pitch
    Nx, Ny, Nz = grid.dims
    rel_x = (c[1] - grid.origin[1]) / h
    rel_y = (c[2] - grid.origin[2]) / h
    rel_z = (c[3] - grid.origin[3]) / h
    if isodd(M)
        i0 = round(Int, rel_x) + 1
        j0 = round(Int, rel_y) + 1
        k0 = round(Int, rel_z) + 1
        half = M ÷ 2
        offsets = -half:half
    else
        i0 = floor(Int, rel_x) + 1
        j0 = floor(Int, rel_y) + 1
        k0 = floor(Int, rel_z) + 1
        offsets = 0:(M - 1)
    end
    L = LinearIndices((Nx, Ny, Nz))
    out = Int[]
    sizehint!(out, M^3)
    @inbounds for di in offsets, dj in offsets, dk in offsets
        i = i0 + di
        j = j0 + dj
        k = k0 + dk
        if 1 <= i <= Nx && 1 <= j <= Ny && 1 <= k <= Nz
            push!(out, L[i, j, k])
        end
    end
    return out
end

"""
    grid_point_at_linear(grid::AIMGrid, lin::Integer) -> Vec3

Convert a linear index back into a physical grid point.
"""
@inline function grid_point_at_linear(grid::AIMGrid, lin::Integer)
    Nx, Ny, Nz = grid.dims
    ci = CartesianIndices((Nx, Ny, Nz))[lin]
    return grid_point(grid, ci[1], ci[2], ci[3])
end
