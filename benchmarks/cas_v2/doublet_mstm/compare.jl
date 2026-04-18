# Compare block-VIEM.jl and MSTMforCAS.jl for the 2-sphere doublet
# CAS-v2 benchmark.  Reads the JSON outputs written by `run_viem.jl` and
# `run_mstm.jl` and prints / writes a relative-error report.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/doublet_mstm/compare.jl

using Printf
using JSON3

include(joinpath(@__DIR__, "config.jl"))

function load_records(path::AbstractString)
    open(path, "r") do io
        doc = JSON3.read(io)
        return collect(doc.results)
    end
end

function index_by_key(records)
    out = Dict{String,Any}()
    for r in records
        key = string(r.material, "/beta_", round(Float64(r.beta); digits=6))
        out[key] = r
    end
    return out
end

cplx(r, pre) = Complex(Float64(r[Symbol(pre * "_re")]), Float64(r[Symbol(pre * "_im")]))

"""
Compute absolute, relative, magnitude, and phase errors of complex values.
"""
function complex_errors(z_viem::Complex, z_mstm::Complex)
    abs_err  = abs(z_viem - z_mstm)
    rel_err  = abs_err / abs(z_mstm)
    mag_err  = abs(abs(z_viem) - abs(z_mstm)) / abs(z_mstm)
    phase_err = abs(angle(z_viem) - angle(z_mstm))
    # unwrap phase difference into [-π, π]
    while phase_err > π; phase_err -= 2π; end
    return (abs_err = abs_err, rel_err = rel_err,
            mag_err = mag_err, phase_err = phase_err)
end

function main()
    viem_recs = load_records(VIEM_JSON)
    mstm_recs = load_records(MSTM_JSON)
    vi = index_by_key(viem_recs)
    mi = index_by_key(mstm_recs)

    header = """
# CAS-v2 2-sphere doublet benchmark: block-VIEM.jl vs MSTMforCAS.jl

**Geometry**: two disjoint spheres of radius $(R_MONOMER) μm with centre-to-centre
separation $(D_CENTRE) μm (gap $(GAP) μm along the doublet axis).
Vacuum wavelength $(WL_0) μm; background medium refractive index $(M_MEDIUM).

**Orientation angle** β is the angle between the incidence direction and the
doublet axis.  In block-VIEM the particle is fixed along lab-z and the incidence
is rotated via `cas_orientation(0, β, 0)`; in MSTMforCAS the doublet axis is
rotated to make angle β with the lab-z incidence direction — physically
equivalent configurations.

**Observable**: CAS-v2 forward amplitudes (block-DDA_Py / block-VIEM
convention, RCP incidence):
    S_fw_θ = S11 + i·S12,  S_fw_φ = S22 − i·S21,
    S_fw_mean = (S_fw_θ + S_fw_φ)/2,
where the MI02 scattering amplitudes follow from BH83 via
    S11 = S₂/(-ik),  S12 = S₃/(ik),  S21 = S₄/(ik),  S22 = S₁/(-ik).
Combined:
    S_fw_θ = (S₂ − i·S₃)/(-ik),  S_fw_φ = (S₁ + i·S₄)/(-ik).

"""
    println(header)
    buf = IOBuffer()
    println(buf, header)

    for mat in MATERIALS
        name = mat.name
        section_hdr = "\n## Material: $(name)  (m_p = $(mat.m_p))\n"
        println(section_hdr); println(buf, section_hdr)

        # ── High-level magnitude summary ─────────────────────────────────
        summary_hdr = """
**Summary** (complex S_fw_mean magnitude + phase):

| β [rad] | \\|S_fw_mean\\| VIEM | \\|S_fw_mean\\| MSTM | rel \\|·\\| err | complex rel err | phase err [rad] |
|---------|----------------------|----------------------|-----------------|-----------------|-----------------|
"""
        println(summary_hdr);  println(buf, summary_hdr)
        for β in BETAS
            key = string(name, "/beta_", round(β; digits=6))
            haskey(vi, key) && haskey(mi, key) || continue
            rv = vi[key]; rm_ = mi[key]
            sv = cplx(rv, "S_fw_mean");  sm = cplx(rm_, "S_fw_mean")
            e = complex_errors(sv, sm)
            row = @sprintf("| %.4f | %.4e | %.4e | %.2e | %.2e | %.2e |",
                           β, abs(sv), abs(sm), e.mag_err, e.rel_err, e.phase_err)
            println(row);  println(buf, row)
        end

        # ── Re/Im breakdown for S_fw_θ and S_fw_φ ────────────────────────
        for (comp_sym, comp_label) in (("S_fw_theta", "S_fw_θ"),
                                       ("S_fw_phi",   "S_fw_φ"))
            comp_hdr = """

**$(comp_label)** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|
"""
            println(comp_hdr);  println(buf, comp_hdr)
            for β in BETAS
                key = string(name, "/beta_", round(β; digits=6))
                haskey(vi, key) && haskey(mi, key) || continue
                rv = vi[key]; rm_ = mi[key]
                zv = cplx(rv, comp_sym);  zm = cplx(rm_, comp_sym)
                rel_re = abs(zm) > 0 ?
                    abs(real(zv) - real(zm)) /
                        (abs(real(zm)) > 0 ? abs(real(zm)) : abs(zm)) : NaN
                rel_im = abs(zm) > 0 ?
                    abs(imag(zv) - imag(zm)) /
                        (abs(imag(zm)) > 0 ? abs(imag(zm)) : abs(zm)) : NaN
                row = @sprintf("| %.4f | %+.4e | %+.4e | %.2e | %+.4e | %+.4e | %.2e |",
                               β,
                               real(zv), real(zm), rel_re,
                               imag(zv), imag(zm), rel_im)
                println(row);  println(buf, row)
            end
        end
    end

    open(REPORT_MD, "w") do io
        write(io, String(take!(buf)))
    end
    println("\nWrote ", REPORT_MD)
end

main()
