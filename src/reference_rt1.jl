# RT1 (Raviart-Thomas order 1) basis functions on the reference tetrahedron.
#
# Reference tet vertices: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1).
#
# Each of the 15 basis functions is a 3-component vector polynomial of degree 2.
# Each component is expressed as:
#   f(x,y,z) = c[1]x² + c[2]y² + c[3]z² + c[4]xy + c[5]xz + c[6]yz
#            + c[7]x + c[8]y + c[9]z + c[10]
#
# The divergence of each basis is linear:
#   div(f) = d[1]x + d[2]y + d[3]z + d[4]
#
# Coefficient table verified against symfem (degree=1, Raviart-Thomas, tetrahedron).
# DOF ordering: face 0 (z=0): 3 DOFs, face 1 (y=0): 3 DOFs,
#               face 2 (x=0): 3 DOFs, face 3 (x+y+z=1): 3 DOFs,
#               interior: 3 DOFs.

using StaticArrays

# ---------------------------------------------------------------------------
# Coefficient table: 15 basis × 3 components × 10 monomials
# Monomial order: [x², y², z², xy, xz, yz, x, y, z, 1]
# ---------------------------------------------------------------------------

# Each column is a 10-element vector of coefficients for one (basis, component) pair.
# Storage: RT1_REF_COEFFS[monomial_idx, component, basis]
# component: 1=x, 2=y, 3=z;  basis: 1..15;  monomial: 1..10.

const RT1_REF_COEFFS = let
    # phi_0 (face 0, DOF 0)
    c0x = @SVector [ 30.,  0.,  0.,  30.,  30.,  0., -24.,   0.,   0.,  0.]
    c0y = @SVector [  0., 30.,  0.,  30.,   0., 30.,   0., -24.,   0.,  0.]
    c0z = @SVector [  0.,  0., 30.,   0.,  30., 30., -24., -24., -48., 18.]
    # phi_1 (face 0, DOF 1)
    c1x = @SVector [-30.,  0.,  0.,   0.,   0.,  0.,  12.,   0.,   0.,  0.]
    c1y = @SVector [  0.,  0.,  0., -30.,   0.,  0.,   0.,   6.,   0.,  0.]
    c1z = @SVector [  0.,  0.,  0.,   0., -30.,  0.,  24.,   0.,   6., -6.]
    # phi_2 (face 0, DOF 2)
    c2x = @SVector [  0.,  0.,  0., -30.,   0.,  0.,   6.,   0.,   0.,  0.]
    c2y = @SVector [  0.,-30.,  0.,   0.,   0.,  0.,   0.,  12.,   0.,  0.]
    c2z = @SVector [  0.,  0.,  0.,   0.,   0.,-30.,   0.,  24.,   6., -6.]
    # phi_3 (face 1, DOF 0)
    c3x = @SVector [-30.,  0.,  0., -30., -30.,  0.,  24.,   0.,   0.,  0.]
    c3y = @SVector [  0.,-30.,  0., -30.,   0.,-30.,  24.,  48.,  24.,-18.]
    c3z = @SVector [  0.,  0.,-30.,   0., -30.,-30.,   0.,   0.,  24.,  0.]
    # phi_4 (face 1, DOF 1)
    c4x = @SVector [ 30.,  0.,  0.,   0.,   0.,  0., -12.,   0.,   0.,  0.]
    c4y = @SVector [  0.,  0.,  0.,  30.,   0.,  0., -24.,  -6.,   0.,  6.]
    c4z = @SVector [  0.,  0.,  0.,   0.,  30.,  0.,   0.,   0.,  -6.,  0.]
    # phi_5 (face 1, DOF 2)
    c5x = @SVector [  0.,  0.,  0.,   0.,  30.,  0.,  -6.,   0.,   0.,  0.]
    c5y = @SVector [  0.,  0.,  0.,   0.,   0., 30.,   0.,  -6., -24.,  6.]
    c5z = @SVector [  0.,  0., 30.,   0.,   0.,  0.,   0.,   0., -12.,  0.]
    # phi_6 (face 2, DOF 0)
    c6x = @SVector [ 30.,  0.,  0.,  30.,  30.,  0., -48., -24., -24., 18.]
    c6y = @SVector [  0., 30.,  0.,  30.,   0., 30.,   0., -24.,   0.,  0.]
    c6z = @SVector [  0.,  0., 30.,   0.,  30., 30.,   0.,   0., -24.,  0.]
    # phi_7 (face 2, DOF 1)
    c7x = @SVector [  0.,  0.,  0., -30.,   0.,  0.,   6.,  24.,   0., -6.]
    c7y = @SVector [  0.,-30.,  0.,   0.,   0.,  0.,   0.,  12.,   0.,  0.]
    c7z = @SVector [  0.,  0.,  0.,   0.,   0.,-30.,   0.,   0.,   6.,  0.]
    # phi_8 (face 2, DOF 2)
    c8x = @SVector [  0.,  0.,  0.,   0., -30.,  0.,   6.,   0.,  24., -6.]
    c8y = @SVector [  0.,  0.,  0.,   0.,   0.,-30.,   0.,   6.,   0.,  0.]
    c8z = @SVector [  0.,  0.,-30.,   0.,   0.,  0.,   0.,   0.,  12.,  0.]
    # phi_9 (face 3, DOF 0)
    c9x = @SVector [ 30.,  0.,  0.,   0.,   0.,  0., -12.,   0.,   0.,  0.]
    c9y = @SVector [  0.,  0.,  0.,  30.,   0.,  0.,   0.,  -6.,   0.,  0.]
    c9z = @SVector [  0.,  0.,  0.,   0.,  30.,  0.,   0.,   0.,  -6.,  0.]
    # phi_10 (face 3, DOF 1)
    cAx = @SVector [  0.,  0.,  0.,  30.,   0.,  0.,  -6.,   0.,   0.,  0.]
    cAy = @SVector [  0., 30.,  0.,   0.,   0.,  0.,   0., -12.,   0.,  0.]
    cAz = @SVector [  0.,  0.,  0.,   0.,   0., 30.,   0.,   0.,  -6.,  0.]
    # phi_11 (face 3, DOF 2)
    cBx = @SVector [  0.,  0.,  0.,   0.,  30.,  0.,  -6.,   0.,   0.,  0.]
    cBy = @SVector [  0.,  0.,  0.,   0.,   0., 30.,   0.,  -6.,   0.,  0.]
    cBz = @SVector [  0.,  0., 30.,   0.,   0.,  0.,   0.,   0., -12.,  0.]
    # phi_12 (interior, x-moment)
    cCx = @SVector [-60.,  0.,  0., -30., -30.,  0.,  60.,   0.,   0.,  0.]
    cCy = @SVector [  0.,-30.,  0., -60.,   0.,-30.,   0.,  30.,   0.,  0.]
    cCz = @SVector [  0.,  0.,-30.,   0., -60.,-30.,   0.,   0.,  30.,  0.]
    # phi_13 (interior, y-moment)
    cDx = @SVector [-30.,  0.,  0., -60., -30.,  0.,  30.,   0.,   0.,  0.]
    cDy = @SVector [  0.,-60.,  0., -30.,   0.,-30.,   0.,  60.,   0.,  0.]
    cDz = @SVector [  0.,  0.,-30.,   0., -30.,-60.,   0.,   0.,  30.,  0.]
    # phi_14 (interior, z-moment)
    cEx = @SVector [-30.,  0.,  0., -30., -60.,  0.,  30.,   0.,   0.,  0.]
    cEy = @SVector [  0.,-30.,  0., -30.,   0.,-60.,   0.,  30.,   0.,  0.]
    cEz = @SVector [  0.,  0.,-60.,   0., -30.,-30.,   0.,   0.,  60.,  0.]

    # Pack into a 3D array: [10 monomials × 3 components × 15 basis]
    # Using a tuple of tuples for type stability
    ntuple(15) do i
        cs = ((c0x,c0y,c0z),(c1x,c1y,c1z),(c2x,c2y,c2z),
              (c3x,c3y,c3z),(c4x,c4y,c4z),(c5x,c5y,c5z),
              (c6x,c6y,c6z),(c7x,c7y,c7z),(c8x,c8y,c8z),
              (c9x,c9y,c9z),(cAx,cAy,cAz),(cBx,cBy,cBz),
              (cCx,cCy,cCz),(cDx,cDy,cDz),(cEx,cEy,cEz))
        cs[i]
    end
end

# ---------------------------------------------------------------------------
# Divergence coefficients: 15 basis × 4 terms [coeff_x, coeff_y, coeff_z, const]
# div(phi_i) = d[1]*x + d[2]*y + d[3]*z + d[4]
# ---------------------------------------------------------------------------

const RT1_REF_DIV = let
    SVector{4,Float64}[
        SA[ 120.,  120.,  120., -96.],  # phi_0
        SA[-120.,    0.,    0.,  24.],  # phi_1
        SA[   0., -120.,    0.,  24.],  # phi_2
        SA[-120., -120., -120.,  96.],  # phi_3
        SA[ 120.,    0.,    0., -24.],  # phi_4
        SA[   0.,    0.,  120., -24.],  # phi_5
        SA[ 120.,  120.,  120., -96.],  # phi_6
        SA[   0., -120.,    0.,  24.],  # phi_7
        SA[   0.,    0., -120.,  24.],  # phi_8
        SA[ 120.,    0.,    0., -24.],  # phi_9
        SA[   0.,  120.,    0., -24.],  # phi_10
        SA[   0.,    0.,  120., -24.],  # phi_11
        SA[-240., -120., -120., 120.],  # phi_12
        SA[-120., -240., -120., 120.],  # phi_13
        SA[-120., -120., -240., 120.],  # phi_14
    ]
end

# ---------------------------------------------------------------------------
# Evaluation functions on the reference element
# ---------------------------------------------------------------------------

"""
    _eval_monomial_vec(x, y, z, cx, cy, cz) -> SVector{3,Float64}

Evaluate a quadratic vector polynomial at (x,y,z) from coefficient vectors.
"""
@inline function _eval_monomial_vec(x::Float64, y::Float64, z::Float64,
                                     cx::SVector{10,Float64},
                                     cy::SVector{10,Float64},
                                     cz::SVector{10,Float64})
    # Precompute monomials: [x², y², z², xy, xz, yz, x, y, z, 1]
    m = SVector{10,Float64}(x*x, y*y, z*z, x*y, x*z, y*z, x, y, z, 1.0)
    return SVector{3,Float64}(dot(cx, m), dot(cy, m), dot(cz, m))
end

"""
    rt1_ref_evaluate(i::Int, x::Float64, y::Float64, z::Float64) -> SVector{3,Float64}

Evaluate the `i`-th (1-based) RT1 basis function on the reference tetrahedron
at point `(x, y, z)`.
"""
@inline function rt1_ref_evaluate(i::Int, x::Float64, y::Float64, z::Float64)
    cx, cy, cz = RT1_REF_COEFFS[i]
    return _eval_monomial_vec(x, y, z, cx, cy, cz)
end

"""
    rt1_ref_divergence(i::Int, x::Float64, y::Float64, z::Float64) -> Float64

Evaluate the divergence of the `i`-th RT1 basis function on the reference
tetrahedron at point `(x, y, z)`.
"""
@inline function rt1_ref_divergence(i::Int, x::Float64, y::Float64, z::Float64)
    d = RT1_REF_DIV[i]
    return d[1] * x + d[2] * y + d[3] * z + d[4]
end
