from __future__ import annotations

import numpy as np

GAMMA_MS_MTM = 267.52218744  # ms^-1 mT^-1


def M_nogse_free(TE_ms, G_mTm, N, x_ms, M0, D0_m2_per_ms):
    """
    Modelo "free" usado en tu código:
    M = M0 * exp(-(1/12)*gamma^2*G^2*D0 * ((N-1)*x^3 + y^3))
    con y = TE - (N-1)*x

    Unidades:
      TE_ms, x_ms en ms
      G_mTm en mT/m
      D0 en m^2/ms
    """
    g = GAMMA_MS_MTM

    TE = np.asarray(TE_ms, dtype=float)
    x = np.asarray(x_ms, dtype=float)
    N = np.asarray(N, dtype=float)
    G = np.asarray(G_mTm, dtype=float)

    y = TE - (N - 1.0) * x
    return M0 * np.exp(-(1.0 / 12.0) * (g**2) * (G**2) * D0_m2_per_ms * (((N - 1.0) * (x**3)) + (y**3)))


def OGSE_contrast_vs_g_free(TE_ms, G1_mTm, G2_mTm, N1, N2, M0, D0_m2_per_ms):
    """
    Contraste OGSE entre 2 adquisiciones (N1 vs N2), usando el modelo free.
    """
    return (
        M_nogse_free(TE_ms, G1_mTm, N1, TE_ms / N1, M0, D0_m2_per_ms)
        - M_nogse_free(TE_ms, G2_mTm, N2, TE_ms / N2, M0, D0_m2_per_ms)
    )


def OGSE_contrast_vs_g_tort(TE_ms, G1_mTm, G2_mTm, N1, N2, alpha, M0, D0_m2_per_ms):
    """
    Variante 'tort': D_eff = alpha * D0.
    """
    return OGSE_contrast_vs_g_free(TE_ms, G1_mTm, G2_mTm, N1, N2, M0, alpha * D0_m2_per_ms)
