# Phase A.7: verify the direct-element K_AIM formula matches the FFT
# version exactly. If they differ for a given (m, n), the precorrection
# is inconsistent with the MVP and produces residual error.

using LinearAlgebra: norm
using Printf
using BlockVIEM
using BlockVIEM: Vec3, AIMProjection, aim_far_mvp
import Gmsh: gmsh

function sphere_mesh_a7(lc)
    path = joinpath(tempdir(), "sph_a7_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("a7_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

mesh = read_msh(sphere_mesh_a7(0.7))
basis = build_swg_basis(mesh)
N = n_basis(basis)

h = 0.8
pitch = 0.25 * h

k0 = 1e-4
grid = BlockVIEM.aim_grid(mesh; pitch=pitch, padding=5)
proj = BlockVIEM.build_aim_projection(basis, grid; poly_order=2, stencil=3)
G_toep = BlockVIEM.build_green_toeplitz(grid, ComplexF64(k0))
G_hat  = BlockVIEM.precompute_green_fft(G_toep)

# FFT K_AIM for col n (apply aim_far_mvp to unit vector)
e_n = zeros(ComplexF64, N)
n_test = 5
e_n[n_test] = 1.0
k_aim_col_fft = aim_far_mvp(proj, G_hat, k0, e_n)

# Direct K_AIM for col n
Nx, Ny, Nz = grid.dims
k0_c = ComplexF64(k0)
k0_sq = k0_c^2
Wx_T = copy(transpose(proj.Wx))
Wy = proj.Wy; Wz = proj.Wz; Wdiv = proj.Wdiv
ci_all = CartesianIndices((Nx, Ny, Nz))
N2x = 2Nx; N2y = 2Ny; N2z = 2Nz

function direct_k_aim_col(m, n)
    # Stencils
    col_ptr_x = Wx_T.colptr
    rv = Wx_T.rowval
    nv = Wx_T.nzval

    # Column n and m stencils
    function stencil(b)
        ks = Int[]
        vs_x = ComplexF64[]; vs_y = ComplexF64[]; vs_z = ComplexF64[]; vs_d = ComplexF64[]
        idxs = NTuple{3,Int}[]
        for p in col_ptr_x[b]:(col_ptr_x[b+1]-1)
            lin = rv[p]
            push!(ks, lin)
            ci = ci_all[lin]
            push!(idxs, (ci[1], ci[2], ci[3]))
            push!(vs_x, nv[p])
            push!(vs_y, proj.Wy[b, lin])
            push!(vs_z, proj.Wz[b, lin])
            push!(vs_d, proj.Wdiv[b, lin])
        end
        return idxs, vs_x, vs_y, vs_z, vs_d
    end

    idx_m, vxm, vym, vzm, vdm = stencil(m)
    idx_n, vxn, vyn, vzn, vdn = stencil(n)

    s = zero(ComplexF64)
    for a in eachindex(idx_m)
        im_, jm_, km_ = idx_m[a]
        for b in eachindex(idx_n)
            in_, jn_, kn_ = idx_n[b]
            di = im_ - in_
            dj = jm_ - jn_
            dk = km_ - kn_
            a_idx = di >= 0 ? di : di + N2x
            b_idx = dj >= 0 ? dj : dj + N2y
            c_idx = dk >= 0 ? dk : dk + N2z
            G = G_toep[a_idx + 1, b_idx + 1, c_idx + 1]
            dot_xyz = vxm[a]*vxn[b] + vym[a]*vyn[b] + vzm[a]*vzn[b]
            s += (k0_sq * dot_xyz - vdm[a] * vdn[b]) * G
        end
    end
    return s
end

# Compare for several m values
println("Comparing FFT vs direct K_AIM for col n=$n_test:")
@printf("%5s | %-30s %-30s %-12s\n", "m", "FFT", "direct", "rel diff")
println("-"^86)
max_diff = Ref(0.0)
for m in 1:min(N, 30)
    k_fft = k_aim_col_fft[m]
    k_dir = direct_k_aim_col(m, n_test)
    rel = abs(k_fft - k_dir) / max(abs(k_fft), 1e-30)
    max_diff[] = max(max_diff[], rel)
    @printf("%5d | %+.6e%+.6e  %+.6e%+.6e  %.2e\n",
            m, real(k_fft), imag(k_fft), real(k_dir), imag(k_dir), rel)
end
println("\nmax rel diff: $(max_diff[])")
