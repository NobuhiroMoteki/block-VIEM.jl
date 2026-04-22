# Peak-RSS sampler for per-slot cost accounting in paper-production runs.
#
# Mirrors block-DDA_Py/utils/rss_monitor.py: a daemon task reads
# /proc/self/status:VmRSS every `interval` seconds and keeps the running
# maximum.  `reset!` snaps the baseline to the current RSS so each
# production slot can record its own peak independently.
#
# Julia's built-in `Sys.maxrss()` reports the process-lifetime monotone
# maximum — unsuitable for per-slot measurement, hence the sampler.
#
# Usage:
#   include(joinpath(@__DIR__, "rss_monitor.jl"))
#   using .RSSMonitor
#
#   mon = RSSMonitor.Monitor()
#   RSSMonitor.start!(mon)
#   try
#       for slot in ...
#           RSSMonitor.reset!(mon)
#           # ... work ...
#           peak = RSSMonitor.peak_bytes(mon)
#       end
#   finally
#       RSSMonitor.stop!(mon)
#   end

module RSSMonitor

using Base.Threads

mutable struct Monitor
    interval::Float64
    peak_kb::Int
    stop::Threads.Atomic{Bool}
    task::Union{Task,Nothing}
    lock::ReentrantLock
end

Monitor(interval::Real=0.2) =
    Monitor(Float64(interval), 0, Threads.Atomic{Bool}(false),
            nothing, ReentrantLock())

function _read_rss_kb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Int, split(line)[2])
            end
        end
    catch
        # /proc/self/status not available (non-Linux) — return 0 so the
        # downstream record shows 0 rather than crashing.
    end
    return 0
end

"""
    start!(m::Monitor) -> Monitor

Launch the background sampler.  Safe to call even when multiple threads
are running — `Threads.@spawn` respects the Julia thread pool.
"""
function start!(m::Monitor)
    m.stop[] = false
    m.task = Threads.@spawn begin
        while !m.stop[]
            rss = _read_rss_kb()
            lock(m.lock) do
                if rss > m.peak_kb
                    m.peak_kb = rss
                end
            end
            sleep(m.interval)
        end
    end
    lock(m.lock) do
        m.peak_kb = _read_rss_kb()
    end
    return m
end

"""
    reset!(m::Monitor)

Reset the recorded peak to the current RSS.  Call before each slot to
make `peak_bytes(m)` reflect only that slot's allocations.
"""
function reset!(m::Monitor)
    rss = _read_rss_kb()
    lock(m.lock) do
        m.peak_kb = rss
    end
    return m
end

"""
    peak_bytes(m::Monitor) -> Int

Current recorded peak, in bytes (the sampler reads kilobytes from /proc).
"""
peak_bytes(m::Monitor) = lock(m.lock) do
    m.peak_kb * 1024
end

"""
    stop!(m::Monitor)

Signal the sampler to exit and wait for the task to finish.
"""
function stop!(m::Monitor)
    m.stop[] = true
    if m.task !== nothing
        try
            wait(m.task)
        catch
            # task may have already finished normally
        end
        m.task = nothing
    end
    return m
end

end  # module RSSMonitor
