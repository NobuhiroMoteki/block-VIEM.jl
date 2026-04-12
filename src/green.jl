# Free-space scalar Helmholtz Green's function in 3D.
#
# Time convention: exp(-iωt)  (physics convention, matching block-DDA_Py
# and the Mie reference). With this convention an outgoing wave carries
# the factor exp(+i k0 R), so
#
#     G(R) = exp(+i k0 R) / (4 π R)
#
# `k0` may be real (lossless background) or complex with positive imaginary
# part (lossy background, decaying outgoing wave). `R` must be strictly
# positive — singular cases are handled by the Duffy quadrature, which
# never evaluates the Green function exactly at R = 0.

const _INV_FOUR_PI = 1 / (4π)

"""
    helmholtz_green(R, k0) -> Complex{Float64}

Free-space scalar Helmholtz Green's function `G(R) = exp(-i k0 R) / (4π R)`.

Arguments
- `R::Real`        — distance, must be strictly positive
- `k0::Number`     — wavenumber (real for lossless, complex for lossy media)

Throws an `ArgumentError` if `R <= 0`.
"""
@inline function helmholtz_green(R::Real, k0::Number)
    R > 0 || throw(ArgumentError("helmholtz_green requires R > 0, got R = $R"))
    # PHYSICS CONVENTION (e^{-iωt}): outgoing wave is exp(+ik0 R).
    # (Was exp(-im * k0 * R) — engineering convention. Switching to physics
    # to make the optical-theorem path consistent with the volumetric C_abs
    # formula and with block-DDA_Py / Mie reference.)
    return exp(im * k0 * R) * _INV_FOUR_PI / R
end

"""
    helmholtz_green_static(R) -> Float64

Static-limit (`k0 = 0`) scalar Green's function `G_0(R) = 1 / (4π R)`.
Returned as a real `Float64` to avoid complex-arithmetic overhead in the
electrostatic validation tests of Phase 2.
"""
@inline function helmholtz_green_static(R::Real)
    R > 0 || throw(ArgumentError("helmholtz_green_static requires R > 0, got R = $R"))
    return _INV_FOUR_PI / R
end
