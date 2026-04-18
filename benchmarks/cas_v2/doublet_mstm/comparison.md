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

| 0.0000 | 1.7687e-03 | 1.7942e-03 | 1.42e-02 | 1.42e-02 | 1.31e-04 |
| 0.7854 | 1.8215e-03 | 1.8491e-03 | 1.49e-02 | 1.49e-02 | 1.55e-04 |
| 1.5708 | 1.8782e-03 | 1.9083e-03 | 1.57e-02 | 1.57e-02 | 2.00e-04 |

**S_fw_θ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +1.7682e-03 | +1.7937e-03 | 1.42e-02 | +4.1063e-05 | +4.2002e-05 | 2.24e-02 |
| 0.7854 | +1.8768e-03 | +1.9068e-03 | 1.57e-02 | +4.7872e-05 | +4.9085e-05 | 2.47e-02 |
| 1.5708 | +1.9933e-03 | +2.0282e-03 | 1.72e-02 | +5.5225e-05 | +5.6770e-05 | 2.72e-02 |

**S_fw_φ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +1.7683e-03 | +1.7937e-03 | 1.42e-02 | +4.1285e-05 | +4.2002e-05 | 1.71e-02 |
| 0.7854 | +1.7651e-03 | +1.7902e-03 | 1.40e-02 | +4.2005e-05 | +4.2723e-05 | 1.68e-02 |
| 1.5708 | +1.7619e-03 | +1.7870e-03 | 1.41e-02 | +4.2680e-05 | +4.3462e-05 | 1.80e-02 |

## Material: Au  (m_p = 0.17525 + 3.483im)

**Summary** (complex S_fw_mean magnitude + phase):

| β [rad] | \|S_fw_mean\| VIEM | \|S_fw_mean\| MSTM | rel \|·\| err | complex rel err | phase err [rad] |
|---------|----------------------|----------------------|-----------------|-----------------|-----------------|

| 0.0000 | 6.4197e-03 | 6.5146e-03 | 1.46e-02 | 1.46e-02 | 5.32e-04 |
| 0.7854 | 8.0428e-03 | 8.2443e-03 | 2.44e-02 | 2.46e-02 | 3.15e-03 |
| 1.5708 | 9.7853e-03 | 1.0106e-02 | 3.18e-02 | 3.21e-02 | 4.39e-03 |

**S_fw_θ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +6.4046e-03 | +6.4992e-03 | 1.46e-02 | +4.3635e-04 | +4.4794e-04 | 2.59e-02 |
| 0.7854 | +9.6439e-03 | +9.9561e-03 | 3.14e-02 | +1.2147e-03 | +1.3008e-03 | 6.62e-02 |
| 1.5708 | +1.3102e-02 | +1.3646e-02 | 3.99e-02 | +2.0467e-03 | +2.2088e-03 | 7.34e-02 |

**S_fw_φ** — real and imaginary parts:

| β [rad] | Re VIEM | Re MSTM | rel err Re | Im VIEM | Im MSTM | rel err Im |
|---------|---------|---------|------------|---------|---------|------------|

| 0.0000 | +6.4049e-03 | +6.4992e-03 | 1.45e-02 | +4.3967e-04 | +4.4794e-04 | 1.85e-02 |
| 0.7854 | +6.3556e-03 | +6.4387e-03 | 1.29e-02 | +4.4777e-04 | +4.5496e-04 | 1.58e-02 |
| 1.5708 | +6.3084e-03 | +6.3892e-03 | 1.26e-02 | +4.5536e-04 | +4.6332e-04 | 1.72e-02 |
