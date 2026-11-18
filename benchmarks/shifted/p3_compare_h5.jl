# Phase 3 acceptance: compare two spheroid-sweep HDF5 files.
#
# Reports, for every dataset of every group, the largest absolute difference
# and the largest difference relative to the largest amplitude of the group.
# Root and group attributes are compared too. The provenance attributes that
# differ between any two runs (time stamp, git state) are listed apart.
#
# Run: julia --project=. benchmarks/shifted/p3_compare_h5.jl A.h5 B.h5 [RTOL]
#
# Exit status 1 if a dataset differs by more than RTOL (default 0: the files
# must hold identical numbers) or if the shapes or the `converged` flags differ.

using HDF5
using Printf

const RUN_ATTRS = ("created_utc", "git_sha", "git_dirty")

function compare_attrs(a, b, where_::AbstractString)
    ka, kb = Set(keys(attrs(a))), Set(keys(attrs(b)))
    n_diff = 0
    for k in sort(collect(union(ka, kb)))
        k in RUN_ATTRS && continue
        if !(k in ka) || !(k in kb)
            @printf("  attr %s/%s only in %s\n", where_, k, k in ka ? "A" : "B")
            n_diff += 1
        elseif !isequal(attrs(a)[k], attrs(b)[k])
            @printf("  attr %s/%s differs: %s | %s\n", where_, k,
                    repr(attrs(a)[k]), repr(attrs(b)[k]))
            n_diff += 1
        end
    end
    return n_diff
end

function main()
    length(ARGS) >= 2 || error("usage: p3_compare_h5.jl A.h5 B.h5 [RTOL]")
    rtol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
    fa, fb = h5open(ARGS[1], "r"), h5open(ARGS[2], "r")
    ok = true
    n_attr = compare_attrs(fa, fb, "")
    worst = 0.0
    for name in sort(collect(union(keys(fa), keys(fb))))
        if !(haskey(fa, name) && haskey(fb, name))
            println("  object ", name, " missing in one file"); ok = false; continue
        end
        if fa[name] isa HDF5.Dataset
            A, B = read(fa[name]), read(fb[name])
            if size(A) != size(B) || !isequal(A, B)
                println("  root dataset ", name, " differs"); ok = false
            end
            continue
        end
        ga, gb = fa[name], fb[name]
        n_attr += compare_attrs(ga, gb, name)
        # Scale of the group: largest amplitude over the four S datasets of A.
        scale = 0.0
        for d in keys(ga)
            startswith(d, "S_") || continue
            x = read(ga[d])
            scale = max(scale, maximum(abs, filter(!isnan, x); init = 0.0))
        end
        for d in sort(collect(union(keys(ga), keys(gb))))
            if !(haskey(ga, d) && haskey(gb, d))
                println("  dataset ", name, "/", d, " missing in one file"); ok = false; continue
            end
            A, B = read(ga[d]), read(gb[d])
            if size(A) != size(B)
                println("  dataset ", name, "/", d, " shapes differ"); ok = false; continue
            end
            if eltype(A) <: AbstractFloat
                isequal(isnan.(A), isnan.(B)) ||
                    (println("  dataset ", name, "/", d, ": NaN pattern differs"); ok = false)
                m = .!isnan.(A) .& .!isnan.(B)
                dabs = maximum(abs, A[m] .- B[m]; init = 0.0)
                drel = scale > 0 ? dabs / scale : 0.0
                worst = max(worst, drel)
                @printf("  %s/%-14s max |A-B| = %.3e   relative to max|S| = %.3e\n",
                        name, d, dabs, drel)
            else
                same = isequal(A, B)
                @printf("  %s/%-14s %s\n", name, d, same ? "identical" : "DIFFERS")
                ok &= same
            end
        end
    end
    @printf("attributes that differ (run attributes %s excluded): %d\n",
            join(RUN_ATTRS, ", "), n_attr)
    @printf("largest relative difference: %.3e (limit %.1e)\n", worst, rtol)
    ok &= worst <= rtol
    println(ok ? "PASS" : "FAIL")
    close(fa); close(fb)
    exit(ok ? 0 : 1)
end

main()
