# Compare the results of two HDF5 files filled by viem_results/run_viem.jl.
#
# Every dataset of /target/simulated_data is compared. For each one the script
# prints the largest absolute difference and the largest difference relative to
# the largest magnitude of the dataset in file A. The cost group is not
# compared (wall times differ between runs); its solver diagnostics are listed
# for both files.
#
# Run: julia --project=. benchmarks/shifted/compare_run_viem_h5.jl A.hdf5 B.hdf5 [RTOL]
#
# Exit status 1 if a dataset differs by more than RTOL (default 0: identical
# numbers) or if the shapes or the NaN patterns differ.

using HDF5
using Printf

function main()
    length(ARGS) >= 2 || error("usage: compare_run_viem_h5.jl A.hdf5 B.hdf5 [RTOL]")
    rtol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
    fa, fb = h5open(ARGS[1], "r"), h5open(ARGS[2], "r")
    sa, sb = fa["target/simulated_data"], fb["target/simulated_data"]
    ok = true
    worst = 0.0
    for d in sort(collect(union(keys(sa), keys(sb))))
        if !(haskey(sa, d) && haskey(sb, d))
            println("  dataset ", d, " missing in one file"); ok = false; continue
        end
        A, B = read(sa[d]), read(sb[d])
        if size(A) != size(B)
            println("  dataset ", d, ": shapes differ"); ok = false; continue
        end
        nanA, nanB = isnan.(A), isnan.(B)
        nanA == nanB || (println("  dataset ", d, ": NaN pattern differs"); ok = false)
        m = .!nanA .& .!nanB
        scale = maximum(abs, A[m]; init = 0.0)
        dabs = maximum(abs, A[m] .- B[m]; init = 0.0)
        drel = scale > 0 ? dabs / scale : 0.0
        worst = max(worst, drel)
        @printf("  %-16s max |A-B| = %.3e   relative to max|A| = %.3e   (%d values, %d NaN)\n",
                d, dabs, drel, length(A), count(nanA))
    end
    for (label, f) in (("A", fa), ("B", fb))
        t = f["target"]
        c = t["cost"]
        mode = haskey(attributes(t), "shifted_ri") && read_attribute(t, "shifted_ri") != 0 ?
               "shifted" : "default"
        @printf("%s (%s mode): iters %s\n   converged %s\n   t_solve_s %s\n   sum of t_total_s = %.1f s\n",
                label, mode, string(vec(read(c["iters"]))), string(vec(read(c["converged"]))),
                string(round.(vec(read(c["t_solve_s"])); digits = 1)), sum(read(c["t_total_s"])))
    end
    @printf("largest relative difference: %.3e (limit %.1e)\n", worst, rtol)
    ok &= worst <= rtol
    println(ok ? "PASS" : "FAIL")
    close(fa); close(fb)
    exit(ok ? 0 : 1)
end

main()
