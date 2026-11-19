# CAS-v2 2-sphere doublet benchmark: block-VIEM.jl vs MSTMforCAS.jl

**Geometry**: two disjoint spheres of radius 0.03 μm with centre-to-centre
separation 0.063 μm (gap 0.003 μm along the doublet axis).
Vacuum wavelength 0.638 μm; background medium refractive index 1.0.

**Orientation angle** β is the angle between the incidence direction and the
doublet axis.  In block-VIEM the particle is fixed along lab-z and the incidence
is rotated via `cas_orientation(0, β, 0)`; in MSTMforCAS the doublet axis is
rotated to make angle β with the lab-z incidence direction — physically
equivalent configurations.

**Observable**: CAS-v2 forward amplitudes (block-DDA_Py / block-VIEM
convention, RCP incidence):
    S_fw_θ = S11 + i·S12,  S_fw_φ = S22 − i·S21,
    S_fw_mean = (S_fw_θ + S_fw_φ)/2,
where the MI02 scattering amplitudes follow from BH83 via
    S11 = S₂/(-ik),  S12 = S₃/(ik),  S21 = S₄/(ik),  S22 = S₁/(-ik).
Combined:
    S_fw_θ = (S₂ − i·S₃)/(-ik),  S_fw_φ = (S₁ + i·S₄)/(-ik).



## Material: PS  (m_p = 1.6 + 0.01im)

**Summary** (complex S_fw_mean magnitude + phase):

| β [rad] | \|S_fw_mean\| VIEM | \|S_fw_mean\| MSTM | rel \|·\| err | complex rel err | phase err [rad] |
|---------|----------------------|----------------------|-----------------|-----------------|-----------------|

| 0.0000 | 1.7930e-03 | 1.7939e-03 | 4.85e-04 | 4.85e-04 | 3.77e-06 |
| 0.7854 | 1.8473e-03 | 1.8489e-03 | 9.06e-04 | 9.06e-04 | 3.46e-06 |
| 1.5708 | 1.9056e-03 | 1.9081e-03 | 1.32e-03 | 1.32e-03 | 2.33e-05 |

**S_fw_θ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +1.7925e-03 | +1.7934e-03 | 5.03e-04 | +4.1903e-05 | +4.1991e-05 | 2.10e-03 |
| 0.7854 | +1.9041e-03 | +1.9066e-03 | 1.30e-03 | +4.8930e-05 | +4.9082e-05 | 3.10e-03 |
| 1.5708 | +2.0239e-03 | +2.0279e-03 | 1.96e-03 | +5.6501e-05 | +5.6771e-05 | 4.75e-03 |

**S_fw_φ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +1.7926e-03 | +1.7934e-03 | 4.67e-04 | +4.2025e-05 | +4.1991e-05 | 8.07e-04 |
| 0.7854 | +1.7893e-03 | +1.7902e-03 | 4.90e-04 | +4.2780e-05 | +4.2724e-05 | 1.31e-03 |
| 1.5708 | +1.7860e-03 | +1.7870e-03 | 5.85e-04 | +4.3512e-05 | +4.3463e-05 | 1.13e-03 |

## Material: Au  (m_p = 0.17525 + 3.483im)

**Summary** (complex S_fw_mean magnitude + phase):

| β [rad] | \|S_fw_mean\| VIEM | \|S_fw_mean\| MSTM | rel \|·\| err | complex rel err | phase err [rad] |
|---------|----------------------|----------------------|-----------------|-----------------|-----------------|

| 0.0000 | 6.5042e-03 | 6.5039e-03 | 4.07e-05 | 5.31e-05 | 3.41e-05 |
| 0.7854 | 8.2293e-03 | 8.2865e-03 | 6.90e-03 | 6.96e-03 | 8.93e-04 |
| 1.5708 | 1.0085e-02 | 1.0197e-02 | 1.09e-02 | 1.10e-02 | 1.14e-03 |

**S_fw_θ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +6.4886e-03 | +6.4886e-03 | 1.32e-05 | +4.4603e-04 | +4.4705e-04 | 2.28e-03 |
| 0.7854 | +9.9280e-03 | +1.0037e-02 | 1.08e-02 | +1.3105e-03 | +1.3381e-03 | 2.07e-02 |
| 1.5708 | +1.3603e-02 | +1.3818e-02 | 1.55e-02 | +2.2322e-03 | +2.2857e-03 | 2.34e-02 |

**S_fw_φ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +6.4890e-03 | +6.4886e-03 | 7.30e-05 | +4.4766e-04 | +4.4705e-04 | 1.37e-03 |
| 0.7854 | +6.4356e-03 | +6.4392e-03 | 5.63e-04 | +4.5567e-04 | +4.5502e-04 | 1.43e-03 |
| 1.5708 | +6.3862e-03 | +6.3893e-03 | 4.84e-04 | +4.6397e-04 | +4.6332e-04 | 1.40e-03 |
