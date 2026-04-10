# Toeplitz Green's function kernel for the AIM grid and FFT-based
# convolution (Goodman et al. 1991).
#
# Because the weakened VIEM formulation (technical_note.md §4) uses the
# *scalar* Helmholtz Green's function G(R) multiplied by either
# (k0² f_m·f_n') or (∇·f_m)(∇'·f_n'), the grid-grid interaction is a
# single scalar Toeplitz convolution — no 3×3 dyadic tensor is needed.
#
# The Toeplitz kernel is embedded into a circulant matrix of twice the
# grid size in each dimension, then convolution is carried out via
# FFT → pointwise multiply → IFFT.

using FFTW

"""
    build_green_toeplitz(grid::AIMGrid, k0::Number) -> Array{ComplexF64,3}

Construct the circulant-embedded scalar Helmholtz Green's function on a
doubled grid of size `(2Nx, 2Ny, 2Nz)`.

For each index `(i, j, k)` the displacement is mapped to the "folded"
distance `min(i, 2Nx-i)` etc., which gives the symmetric (Toeplitz)
structure. The self-displacement `(0,0,0)` is set to zero because the
self-interaction is handled by the precorrection matrix (direct Z minus
AIM on near pairs).

Returns the **spatial-domain** kernel; call [`precompute_green_fft`](@ref)
to obtain its FFT.
"""
function build_green_toeplitz(grid::AIMGrid, k0::Number)
    Nx, Ny, Nz = grid.dims
    h = grid.pitch
    k0c = ComplexF64(k0)
    G = Array{ComplexF64}(undef, 2Nx, 2Ny, 2Nz)
    @inbounds for ki in 0:2Nz-1, ji in 0:2Ny-1, ii in 0:2Nx-1
        di = min(ii, 2Nx - ii)
        dj = min(ji, 2Ny - ji)
        dk = min(ki, 2Nz - ki)
        R = h * sqrt(Float64(di^2 + dj^2 + dk^2))
        if R > 0
            G[ii + 1, ji + 1, ki + 1] = helmholtz_green(R, k0c)
        else
            G[ii + 1, ji + 1, ki + 1] = zero(ComplexF64)
        end
    end
    return G
end

"""
    precompute_green_fft(G_toeplitz::Array{ComplexF64,3}) -> Array{ComplexF64,3}

Return the 3-D FFT of the circulant-embedded Green kernel. This is
precomputed once and reused in every [`fft_convolve`](@ref) call.
"""
function precompute_green_fft(G_toeplitz::Array{ComplexF64,3})
    G_hat = copy(G_toeplitz)
    fft!(G_hat)
    return G_hat
end

"""
    fft_convolve(G_hat::Array{ComplexF64,3}, u::AbstractArray{<:Number,3})
        -> Array{ComplexF64,3}

Compute the Toeplitz convolution `(G ⊛ u)[i,j,k] = Σ_{i',j',k'} G(Δ) u(i',j',k')`
via the standard zero-pad → FFT → multiply → IFFT → extract recipe.

`G_hat` is the pre-FFT'd circulant kernel of size `(2Nx,2Ny,2Nz)`.
`u` is the grid source of size `(Nx,Ny,Nz)`.
Returns an `(Nx,Ny,Nz)` complex array.
"""
function fft_convolve(G_hat::Array{ComplexF64,3},
                      u::AbstractArray{<:Number,3})
    Nx, Ny, Nz = size(u)
    N2x, N2y, N2z = size(G_hat)
    @assert (N2x, N2y, N2z) == (2Nx, 2Ny, 2Nz) "G_hat size must be 2× the grid size"

    buf = zeros(ComplexF64, N2x, N2y, N2z)
    buf[1:Nx, 1:Ny, 1:Nz] .= u
    fft!(buf)
    buf .*= G_hat
    ifft!(buf)
    return buf[1:Nx, 1:Ny, 1:Nz]
end

"""
    fft_convolve!(result::AbstractArray{ComplexF64,3},
                  G_hat::Array{ComplexF64,3},
                  u::AbstractArray{<:Number,3},
                  buf::Array{ComplexF64,3})

In-place variant of [`fft_convolve`](@ref). `buf` is a pre-allocated
`(2Nx,2Ny,2Nz)` workspace; `result` is `(Nx,Ny,Nz)`.
"""
function fft_convolve!(result::AbstractArray{ComplexF64,3},
                       G_hat::Array{ComplexF64,3},
                       u::AbstractArray{<:Number,3},
                       buf::Array{ComplexF64,3})
    Nx, Ny, Nz = size(u)
    buf .= 0
    buf[1:Nx, 1:Ny, 1:Nz] .= u
    fft!(buf)
    buf .*= G_hat
    ifft!(buf)
    result .= @view buf[1:Nx, 1:Ny, 1:Nz]
    return result
end
