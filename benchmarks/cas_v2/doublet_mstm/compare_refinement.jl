# Analyse the mesh-refinement sweep: compute VIEM-vs-MSTM relative
# errors at every (lc, β) point.  An exponent `err ~ h^p` is fitted in
# log-log on each observable component only when at least
# MIN_LEVELS_FOR_FIT refinement levels are present.  With the default
# inputs (R/5 and R/6) only the error tables are written: an exponent
# from two levels is not a convergence rate.
#
# Inputs (produced by run_mstm.jl and run_viem_refinement.jl):
#   results_mstm.json           — MSTM reference at N=15
#   results_viem.json           — baseline VIEM at lc = R/5
#   results_viem_refinement.json — refinement VIEM at lc = R/k for k in LC_RATIOS
#
# Outputs:
#   refinement.md   — human-readable error tables (+ fitted exponents
#                     when enough levels are present)
#
# Usage:
#   julia --project=. benchmarks/cas_v2/doublet_mstm/compare_refinement.jl

using Printf
using JSON3
using LinearAlgebra: norm

include(joinpath(@__DIR__, "config.jl"))

function load_records(path::AbstractString)
    open(path, "r") do io
        doc = JSON3.read(io)
        return collect(doc.results)
    end
end

# Smallest number of refinement levels for which an exponent is fitted.
const MIN_LEVELS_FOR_FIT = 3

cplx(r, pre) = Complex(Float64(r[Symbol(pre * "_re")]), Float64(r[Symbol(pre * "_im")]))

# Simple log-log linear fit: log(y) = a + p·log(x).  Returns (p, intercept).
function loglog_fit(xs::Vector{Float64}, ys::Vector{Float64})
    lx = log.(xs);  ly = log.(ys)
    n  = length(xs)
    x̄  = sum(lx) / n;  ȳ = sum(ly) / n
    num = sum((lx .- x̄) .* (ly .- ȳ))
    den = sum((lx .- x̄) .^ 2)
    p   = num / den
    a   = ȳ - p * x̄
    return (p = p, intercept = a)
end

function main()
    viem_baseline  = load_records(VIEM_JSON)                                  # lc = R/5
    viem_refine    = load_records(joinpath(@__DIR__, "results_viem_refinement.json"))
    mstm_recs      = load_records(MSTM_JSON)

    # Index MSTM by (material, beta).
    mstm_by_key = Dict{String,Any}()
    for r in mstm_recs
        mstm_by_key[string(r.material, "/beta_", round(Float64(r.beta); digits=6))] = r
    end

    # Build per-lc Au records: baseline R/5 + refinement.
    # Baseline run_viem.jl did not store `lc_over_R`; we know it's R/5.
    records = Dict{Tuple{Float64,Float64},Any}()
    for r in viem_baseline
        String(r.material) == "Au" || continue
        β  = Float64(r.beta)
        lc = Float64(r.lc)
        records[(round(lc/R_MONOMER; digits=6), β)] = r
    end
    for r in viem_refine
        String(r.material) == "Au" || continue
        β  = Float64(r.beta)
        lc = Float64(r.lc)
        records[(round(lc/R_MONOMER; digits=6), β)] = r
    end

    # ── Emit per-β convergence tables ────────────────────────────────
    buf = IOBuffer()
    # Collect distinct lc/R values actually present in the data (Au)
    present_ratios = sort!(unique!([k[1] for k in keys(records)]); rev = true)
    ratio_str = join(["R/" * string(Int(round(1/r))) for r in present_ratios], ", ")

    hdr = """
# Mesh refinement: 2-sphere Au doublet

Refines the linear-SWG discretisation at a sequence of mesh sizes
`lc / R ∈ {$(ratio_str)}` with the MSTM reference held fixed at
`N_trunc = 15` (converged to rtol ≤ 1e-6).  The relative errors of
the five CAS-v2 complex components (`|S_fw_mean|`, `Re/Im S_fw_θ`,
`Re/Im S_fw_φ`) are tabulated below for each β.

An exponent `p` of `err ~ (lc/R)^p` (log-log least-squares fit) is
given only when at least $(MIN_LEVELS_FOR_FIT) refinement levels are present.
A fitted exponent is a convergence rate only where the error decreases
monotonically with `lc`.  Components for which it does not are marked.
In the four-level study of docs/benchmark_results.md §8 (R/5 to R/8)
the errors below R/6 are not monotone, and no rate is quoted there.
"""
    println(hdr);  println(buf, hdr)

    components = (
        ("abs_S_fw_mean", "\\|S_fw_mean\\|", :complex_abs),
        ("Re S_fw_theta", "Re S_fw_θ",        :re_theta),
        ("Im S_fw_theta", "Im S_fw_θ",        :im_theta),
        ("Re S_fw_phi",   "Re S_fw_φ",        :re_phi),
        ("Im S_fw_phi",   "Im S_fw_φ",        :im_phi),
    )

    # fit each component per β using all available lc ratios
    fit_results = Dict{Tuple{Symbol,Float64}, NamedTuple{(:p, :intercept), Tuple{Float64,Float64}}}()

    for β in BETAS
        key_mstm = string("Au/beta_", round(β; digits=6))
        rm_ = mstm_by_key[key_mstm]
        section = @sprintf("\n## β = %.4f rad (Au)\n\n", β)
        println(section);  print(buf, section)
        tab_hdr = """
| lc / R | \\|S_fw_mean\\| err | Re S_fw_θ err | Im S_fw_θ err | Re S_fw_φ err | Im S_fw_φ err |
|--------|---------------------|---------------|---------------|---------------|---------------|
"""
        println(tab_hdr);  print(buf, tab_hdr)

        lc_keys = sort!([k for k in keys(records) if k[2] ≈ β]; by = first, rev = true)
        if isempty(lc_keys)
            println("  (no records for β=", β, ")");  continue
        end

        # For the convergence fit, collect lc/R and err for each component.
        abs_ratios = Float64[]
        errs = Dict(
            :complex_abs => Float64[],
            :re_theta    => Float64[],
            :im_theta    => Float64[],
            :re_phi      => Float64[],
            :im_phi      => Float64[],
        )

        for (ratio, β_stored) in lc_keys
            rv = records[(ratio, β_stored)]
            sv_fw_mean = cplx(rv, "S_fw_mean");  sm_fw_mean = cplx(rm_, "S_fw_mean")
            sv_th = cplx(rv, "S_fw_theta");      sm_th = cplx(rm_, "S_fw_theta")
            sv_ph = cplx(rv, "S_fw_phi");        sm_ph = cplx(rm_, "S_fw_phi")

            e_abs = abs(abs(sv_fw_mean) - abs(sm_fw_mean)) / abs(sm_fw_mean)
            e_re_th = abs(real(sv_th) - real(sm_th)) / abs(real(sm_th))
            e_im_th = abs(imag(sv_th) - imag(sm_th)) / abs(imag(sm_th))
            e_re_ph = abs(real(sv_ph) - real(sm_ph)) / abs(real(sm_ph))
            e_im_ph = abs(imag(sv_ph) - imag(sm_ph)) / abs(imag(sm_ph))

            push!(abs_ratios, ratio)
            push!(errs[:complex_abs], e_abs)
            push!(errs[:re_theta],    e_re_th)
            push!(errs[:im_theta],    e_im_th)
            push!(errs[:re_phi],      e_re_ph)
            push!(errs[:im_phi],      e_im_ph)

            row = @sprintf("| %.4f | %.2e | %.2e | %.2e | %.2e | %.2e |",
                           ratio, e_abs, e_re_th, e_im_th, e_re_ph, e_im_ph)
            println(row);  print(buf, row, "\n")
        end

        # Fit an exponent on each component.  `abs_ratios` runs from the
        # coarsest to the finest level, so a converging error is sorted in
        # decreasing order.
        if length(abs_ratios) >= MIN_LEVELS_FOR_FIT
            pline = "\nFitted exponent `p` from `err ~ (lc/R)^p`:\n\n"
            pline *= "| component | p | err at smallest lc | monotone |\n|-----------|---|--------------------|----------|\n"
            for (_, label, comp_key) in components
                f = loglog_fit(abs_ratios, errs[comp_key])
                fit_results[(comp_key, β)] = f
                i_min = argmin(abs_ratios)
                err_min = errs[comp_key][i_min]
                mono = issorted(errs[comp_key]; rev = true) ? "yes" : "no (not a rate)"
                pline *= @sprintf("| %s | %.2f | %.2e | %s |\n", label, f.p, err_min, mono)
            end
            println(pline);  print(buf, pline)
        else
            note = @sprintf("\n%d refinement levels: no exponent is fitted (at least %d are needed).\n",
                            length(abs_ratios), MIN_LEVELS_FOR_FIT)
            println(note);  print(buf, note)
        end
    end

    # ── Summary of the fitted exponents (only when a fit was made) ───
    concl = """

## Fitted exponents

"""
    isempty(fit_results) || (println(concl);  print(buf, concl))
    for β in BETAS
        for (_, label, comp_key) in components
            if haskey(fit_results, (comp_key, β))
                f = fit_results[(comp_key, β)]
                line = @sprintf("- β=%.4f, %s: p = %.2f\n", β, label, f.p)
                println(line);  print(buf, line)
            end
        end
    end

    out = joinpath(@__DIR__, "refinement.md")
    open(out, "w") do io
        write(io, String(take!(buf)))
    end
    println("\nWrote ", out)
end

main()
