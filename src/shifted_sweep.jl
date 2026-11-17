# Refractive-index sweep of an isotropic, homogeneous particle with one block
# Arnoldi process (shifted block GMRES).
#
# With kappa = (eps_p − eps_bg)/eps_p and tau = eps_bg/(eps_p − eps_bg), the
# impedance operator factors as
#
#     Z(eps_p) = (1/eps_p) M − (kappa/eps_bg) K0 = (kappa/eps_bg) (tau M − K0),
#
# where the mass matrix M (real, sparse, SPD) and the radiation operator K0
# do not depend on eps_p. Right preconditioning with M^{-1} gives
#
#     (tau I − K0 M^{-1}) Yhat = B,      X = (eps_bg/kappa) M^{-1} Yhat,
#
# which is a shift of the fixed operator C = K0 M^{-1}. Since
# B − Z X = B − (tau I − C) Yhat, the residual that GMRES minimises is the
# residual of the original system, `‖B − Z X‖_F / ‖B‖_F`.
#
# This file holds the operator pieces. The sweep driver that uses them is
# added separately.

using LinearAlgebra: cholesky, Symmetric
using SparseArrays: SparseMatrixCSC

# Sparse Cholesky factor of the mass matrix. `F` is the CHOLMOD factor type.
struct _MassCholesky{F}
    fac::F
end
_MassCholesky(M::SparseMatrixCSC{Float64,Int}) = _MassCholesky(cholesky(Symmetric(M)))

# M^{-1} X for a complex block X of shape (N, L). The factor is real, so the
# real and imaginary parts are solved separately.
function _mass_solve(Mi::_MassCholesky, X::AbstractMatrix{<:Complex})
    Xr = (Mi.fac \ Matrix{Float64}(real(X)))::Matrix{Float64}
    Xi = (Mi.fac \ Matrix{Float64}(imag(X)))::Matrix{Float64}
    return complex.(Xr, Xi)
end

# C = K0 M^{-1}, the operator handed to `block_gmres_shifted`.
struct _K0MinvOp{F}
    op::AIMOperator
    minv::_MassCholesky{F}
end
_K0MinvOp(op::AIMOperator) = _K0MinvOp(op, _MassCholesky(op.mass))

Base.:*(A::_K0MinvOp, X::AbstractMatrix) = aim_K0_mvp(A.op, _mass_solve(A.minv, X))

# Shift and contrast of an isotropic permittivity.
_shift_tau(eps_p::Number, eps_bg::Number) =
    ComplexF64(eps_bg) / (ComplexF64(eps_p) - ComplexF64(eps_bg))
_shift_kappa(eps_p::Number, eps_bg::Number) =
    (ComplexF64(eps_p) - ComplexF64(eps_bg)) / ComplexF64(eps_p)
