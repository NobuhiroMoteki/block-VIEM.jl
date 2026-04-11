# Incident plane-wave field projected onto the SWG basis.
#
# The RHS of the EFVIE-D system is
#
#   E_m = ⟨f_m, ε E^inc⟩ = ∫ f_m(r) · (ε(r) E^inc(r)) dV
#
# For a homogeneous particle with scalar ε_p this simplifies to
#
#   E_m = ε_p ∫ f_m(r) · E^inc(r) dV
#
# The plane-wave incident field is
#
#   E^inc(r) = E0 exp(-j k_bg k̂ · r)
#
# where E0 is the complex polarization vector (perpendicular to k̂),
# k_bg = k0 (the background medium wavenumber since k0 is already the
# background wavenumber in our formulation), and k̂ is the unit propagation
# direction.

"""
    project_plane_wave(basis::AbstractDivBasis;
                       k_hat::Vec3,
                       E0::SVector{3,ComplexF64},
                       k_bg::Number,
                       rule::TetQuadRule = TET_QUAD_5PT) -> Vector{ComplexF64}

Compute the RHS vector `b` whose `m`-th entry is the Galerkin testing of
the incident field against the `m`-th basis function:

```
b_m = ∫ f_m(r) · E^inc(r) dV
```

where `E^inc(r) = E0 exp(-j k_bg k̂·r)`. Works for any `AbstractDivBasis`.
"""
function project_plane_wave(basis::AbstractDivBasis;
                            k_hat::Vec3,
                            E0::SVector{3,ComplexF64},
                            k_bg::Number,
                            rule::TetQuadRule = TET_QUAD_5PT)
    N = n_basis(basis)
    b = Vector{ComplexF64}(undef, N)
    k_bg_c = ComplexF64(k_bg)
    @inbounds for n in 1:N
        s = zero(ComplexF64)
        for tet in support_tets(basis, n)
            tet == 0 && continue
            verts = _tet_vertices(basis.mesh, tet)
            V = tet_volume(verts...)
            for i in 1:rule.n
                r = bary_to_point(rule.bary[i], verts)
                fn = evaluate(basis, n, r, tet)
                E_inc = E0 * exp(-im * k_bg_c * dot(k_hat, r))
                s += rule.weights[i] * V * dot(fn, E_inc)
            end
        end
        b[n] = s
    end
    return b
end
