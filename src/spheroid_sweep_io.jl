# HDF5 output of CAS-v2 spheroid parameter-sweep results in the schema
# expected by `block-DDA_Py`'s downstream consumer
# `PCAS_Bayes_APM_Nonspherical/lut_generation/build_spheroid_lut.py`.
#
# Schema reference: `block-DDA_Py/.claude/spheroid_h5_schema_spec.md`.
#
# # File layout
#
#     /
#     ├── D_ve_grid                  (N_Dve,)            float64  [shared root]
#     ├── RI_real_grid               (N_RI,)             float64
#     ├── log_AR_grid                (N_AR,)             float64
#     ├── cos_theta_o_half_grid      (N_u_half,)         float64
#     ├── phi_o_grid                 (N_ph,)             float64
#     │
#     ├── wl_<value>/                (one group per wavelength)
#     │   ├── attrs:  wl_0  ::Float64    (matched wavelength in μm)
#     │   ├── attrs:  m_m   ::Float64    (medium refractive index)
#     │   ├── S_fw_theta_re   shape (N_Dve, N_RI, N_AR, N_u_half, N_ph) Float64
#     │   ├── S_fw_theta_im   ...
#     │   ├── S_fw_phi_re     ...
#     │   ├── S_fw_phi_im     ...
#     │   └── converged       (N_Dve, N_RI, N_AR)        Bool
#     │
#     └── (root attrs:) m_m, solver_tol, block_viem_version, ...
#
# # Symmetry / α-expansion
#
# block-DDA_Py only solves the system at α = 0 for each
# (D_ve, RI_real, AR, cos_theta_o) and applies the analytical α-expansion
#
#     S_θ(α) = A + B exp(+2jα),     S_φ(α) = A - B exp(+2jα)
#
# where A = (S_θ(0) + S_φ(0))/2 and B = (S_θ(0) - S_φ(0))/2. We follow the
# same convention here so that the resulting HDF5 is bit-compatible with the
# DDA producer for the `phi_o ∈ [0, π]` reduced grid.

using HDF5

"""
    SpheroidSweepGrids

5-D parameter grid for the CAS-v2 spheroid sweep used by
[`write_spheroid_sweep_h5`](@ref).

All axes must be **equidistant** (required by the downstream NdimSpline_JAX
consumer in `build_spheroid_lut.py`).

# Fields
- `D_ve::Vector{Float64}`            — volume-equivalent diameter [μm]
- `RI_real::Vector{Float64}`         — Re(m_p), dimensionless
- `log_AR::Vector{Float64}`          — log10 of bc_ratio (b/c). >0 = oblate
- `cos_theta_o_half::Vector{Float64}` — cos(θ_o) on the half-domain [0, 1]
- `phi_o::Vector{Float64}`           — φ_o on [0, π]  (=α grid for output)
"""
struct SpheroidSweepGrids
    D_ve::Vector{Float64}
    RI_real::Vector{Float64}
    log_AR::Vector{Float64}
    cos_theta_o_half::Vector{Float64}
    phi_o::Vector{Float64}
end

"""
    SpheroidSweepData

CAS-v2 forward scattering amplitudes for one wavelength of a spheroid sweep,
laid out in the schema expected by `block-DDA_Py`'s consumer.

# Fields
- `wl_0::Float64`            — vacuum wavelength [μm]
- `m_m::Float64`             — medium refractive index (background)
- `S_fw_theta::Array{ComplexF64,5}` — shape `(N_Dve, N_RI, N_AR, N_u_half, N_ph)`
- `S_fw_phi::Array{ComplexF64,5}`   — same shape
- `converged::Array{Bool,3}`        — shape `(N_Dve, N_RI, N_AR)`

Non-converged cells must have `NaN + NaN*im` in the four S arrays across
all `(N_u_half, N_ph)` entries.
"""
struct SpheroidSweepData
    wl_0::Float64
    m_m::Float64
    S_fw_theta::Array{ComplexF64,5}
    S_fw_phi::Array{ComplexF64,5}
    converged::Array{Bool,3}
end

@inline _wl_group_name(wl::Real) = "wl_" * replace(string(float(wl)), "." => "p")

function _check_grids(grids::SpheroidSweepGrids)
    function _check_eq(name, v::Vector)
        length(v) >= 2 || return  # 1-element grids trivially equidistant
        d = diff(v)
        d_mean = sum(d) / length(d)
        for x in d
            isapprox(x, d_mean; rtol = 1e-9) ||
                error("$name grid is not equidistant (diffs vary: $d)")
        end
    end
    _check_eq("D_ve",            grids.D_ve)
    _check_eq("RI_real",         grids.RI_real)
    _check_eq("log_AR",          grids.log_AR)
    _check_eq("cos_theta_o_half", grids.cos_theta_o_half)
    _check_eq("phi_o",           grids.phi_o)
    return nothing
end

function _check_data_shape(data::SpheroidSweepData, grids::SpheroidSweepGrids)
    expected = (length(grids.D_ve), length(grids.RI_real), length(grids.log_AR),
                length(grids.cos_theta_o_half), length(grids.phi_o))
    size(data.S_fw_theta) == expected ||
        error("S_fw_theta shape $(size(data.S_fw_theta)) != expected $expected")
    size(data.S_fw_phi) == expected ||
        error("S_fw_phi shape $(size(data.S_fw_phi)) != expected $expected")
    size(data.converged) == expected[1:3] ||
        error("converged shape $(size(data.converged)) != expected $(expected[1:3])")
    return nothing
end

"""
    write_spheroid_sweep_h5(filename::AbstractString,
                            grids::SpheroidSweepGrids,
                            data_per_wl::AbstractVector{SpheroidSweepData};
                            block_viem_version::AbstractString = "0.1.1",
                            solver_tol::Real = 1e-8,
                            extra_root_attrs::Dict = Dict())

Write `data_per_wl` (one entry per wavelength) to `filename` in the HDF5
schema expected by `block-DDA_Py`'s `build_spheroid_lut.py`.

The schema is documented in `block-DDA_Py/.claude/spheroid_h5_schema_spec.md`.
This writer is bit-compatible with that schema for the reduced
`(cos_theta_o_half, phi_o)` grid; the consumer applies the cos-mirror and
phi-period extension.

# Arguments

- `filename` — output `.h5` path. The file will be overwritten if it exists.
- `grids` — equidistant 5-D parameter grid (see [`SpheroidSweepGrids`](@ref)).
- `data_per_wl` — one [`SpheroidSweepData`](@ref) per wavelength group. The
  group name follows the `wl_<value>` convention (e.g. `wl_0p638` for
  `wl_0 = 0.638`).
- `block_viem_version` — provenance attribute on the root group.
- `solver_tol` — provenance attribute on the root group.
- `extra_root_attrs` — additional `(name => value)` pairs to write as root
  attributes.
"""
function write_spheroid_sweep_h5(filename::AbstractString,
                                 grids::SpheroidSweepGrids,
                                 data_per_wl::AbstractVector{SpheroidSweepData};
                                 block_viem_version::AbstractString = "0.1.1",
                                 solver_tol::Real = 1e-8,
                                 extra_root_attrs::Dict = Dict{String,Any}())
    _check_grids(grids)
    for d in data_per_wl
        _check_data_shape(d, grids)
    end

    # Sanity: medium RI consistent across wavelengths (used as a root attr).
    m_ms = unique([d.m_m for d in data_per_wl])
    length(m_ms) == 1 || @warn "Multiple m_m values across wavelengths: $m_ms"

    h5open(filename, "w") do f
        # ---- root attributes (provenance) ----
        attrs(f)["m_m"]                = m_ms[1]
        attrs(f)["solver_tol"]         = float(solver_tol)
        attrs(f)["block_viem_version"] = String(block_viem_version)
        for (k, v) in extra_root_attrs
            attrs(f)[String(k)] = v
        end

        # ---- shared root datasets (grid axes) ----
        write_dataset(f, "D_ve_grid",              grids.D_ve)
        write_dataset(f, "RI_real_grid",           grids.RI_real)
        write_dataset(f, "log_AR_grid",            grids.log_AR)
        write_dataset(f, "cos_theta_o_half_grid",  grids.cos_theta_o_half)
        write_dataset(f, "phi_o_grid",             grids.phi_o)

        # ---- per-wavelength groups ----
        for d in data_per_wl
            grp = create_group(f, _wl_group_name(d.wl_0))
            attrs(grp)["wl_0"] = d.wl_0
            attrs(grp)["m_m"]  = d.m_m

            write_dataset(grp, "S_fw_theta_re", real.(d.S_fw_theta))
            write_dataset(grp, "S_fw_theta_im", imag.(d.S_fw_theta))
            write_dataset(grp, "S_fw_phi_re",   real.(d.S_fw_phi))
            write_dataset(grp, "S_fw_phi_im",   imag.(d.S_fw_phi))
            write_dataset(grp, "converged",     d.converged)
        end
    end
    return filename
end

"""
    read_spheroid_sweep_h5(filename) -> (grids, data_per_wl)

Read a spheroid sweep HDF5 file produced by [`write_spheroid_sweep_h5`](@ref)
(or by `block-DDA_Py`'s `run_dda_spheroid_sweep.py`). Returns the parsed
grids and a `Vector{SpheroidSweepData}`, one entry per wavelength group.
"""
function read_spheroid_sweep_h5(filename::AbstractString)
    h5open(filename, "r") do f
        grids = SpheroidSweepGrids(
            read(f["D_ve_grid"]),
            read(f["RI_real_grid"]),
            read(f["log_AR_grid"]),
            read(f["cos_theta_o_half_grid"]),
            read(f["phi_o_grid"]),
        )
        data_per_wl = SpheroidSweepData[]
        for name in keys(f)
            obj = f[name]
            obj isa HDF5.Group || continue
            startswith(name, "wl_") || continue
            wl_0 = haskey(attrs(obj), "wl_0") ? Float64(attrs(obj)["wl_0"]) :
                   parse(Float64, replace(name[4:end], "p" => "."))
            m_m = haskey(attrs(obj), "m_m") ? Float64(attrs(obj)["m_m"]) : 1.0
            S_re = read(obj["S_fw_theta_re"]); S_im = read(obj["S_fw_theta_im"])
            T_re = read(obj["S_fw_phi_re"]);   T_im = read(obj["S_fw_phi_im"])
            S_fw_theta = ComplexF64.(S_re) .+ im .* S_im
            S_fw_phi   = ComplexF64.(T_re) .+ im .* T_im
            converged  = read(obj["converged"])
            push!(data_per_wl, SpheroidSweepData(wl_0, m_m,
                                                 S_fw_theta, S_fw_phi, converged))
        end
        # Sort by wavelength for deterministic order
        sort!(data_per_wl; by = d -> d.wl_0)
        return grids, data_per_wl
    end
end

"""
    expand_alpha_from_alpha0(S_theta_alpha0::ComplexF64,
                              S_phi_alpha0::ComplexF64,
                              alpha::AbstractVector{<:Real})
        -> (S_theta::Vector{ComplexF64}, S_phi::Vector{ComplexF64})

Apply the analytical α-expansion (spheroid C∞ symmetry about the c-axis):

    S_θ(α) = A + B · exp(+2j·α)
    S_φ(α) = A - B · exp(+2j·α)

with `A = (S_θ(0) + S_φ(0))/2`, `B = (S_θ(0) - S_φ(0))/2`. The `+2jα` sign
matches block-DDA_Py and the physics convention used by VIEM since the
2026-04-13 convention switch.
"""
function expand_alpha_from_alpha0(S_theta_alpha0::Number,
                                  S_phi_alpha0::Number,
                                  alpha::AbstractVector{<:Real})
    A = (ComplexF64(S_theta_alpha0) + ComplexF64(S_phi_alpha0)) / 2
    B = (ComplexF64(S_theta_alpha0) - ComplexF64(S_phi_alpha0)) / 2
    S_theta = Vector{ComplexF64}(undef, length(alpha))
    S_phi   = Vector{ComplexF64}(undef, length(alpha))
    @inbounds for k in eachindex(alpha)
        e2ja = exp(+2im * alpha[k])
        S_theta[k] = A + B * e2ja
        S_phi[k]   = A - B * e2ja
    end
    return S_theta, S_phi
end
