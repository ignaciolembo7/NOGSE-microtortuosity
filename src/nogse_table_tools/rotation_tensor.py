from __future__ import annotations

from nogse_table_tools.dirs import load_dirs_csv, load_default_dirs
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

def design_matrix(n_dirs: np.ndarray) -> np.ndarray:
    """A @ d = y, con d = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]."""
    nx, ny, nz = n_dirs[:, 0], n_dirs[:, 1], n_dirs[:, 2]
    A = np.stack([nx*nx, ny*ny, nz*nz, 2*nx*ny, 2*nx*nz, 2*ny*nz], axis=1)
    return A

def vec6_to_tensor(d: np.ndarray) -> np.ndarray:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = d
    return np.array([
        [Dxx, Dxy, Dxz],
        [Dxy, Dyy, Dyz],
        [Dxz, Dyz, Dzz],
    ], dtype=float)

def fit_tensor_from_signals(
    b: float,
    s_norm: np.ndarray,
    n_dirs: np.ndarray,
    *,
    solver: str = "lstsq",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Ajusta tensor D desde señales normalizadas (S/S0) por dirección:
      -log(S/S0)/b = n^T D n
    """
    s_norm = np.asarray(s_norm, dtype=float)
    if np.any(~np.isfinite(s_norm)):
        raise ValueError("s_norm contiene NaN/inf")

    s_clip = np.clip(s_norm, eps, None)
    y = -np.log(s_clip) / float(b)

    A = design_matrix(n_dirs)

    if solver == "solve" and A.shape[0] == A.shape[1]:
        d = np.linalg.solve(A, y)
    else:
        d, *_ = np.linalg.lstsq(A, y, rcond=None)

    return vec6_to_tensor(d)

def D_proj(D: np.ndarray, n: np.ndarray) -> float:
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-15)
    return float(n.T @ D @ n)

@dataclass(frozen=True)
class RotResult:
    rotated_signal_long: pd.DataFrame
    dproj_long: pd.DataFrame

def rotate_signals_tensor(
    df_long: pd.DataFrame,
    *,
    stat_avg: str = "avg",
    s0_mode: str = "dir1",
    solver: str = "lstsq",
    b_col: str = "bvalue",
    dirs_csv: str | Path | None = None,   # <-- nuevo
) -> RotResult:

    """
    Toma el long DF de 1 archivo (1 N) y produce:
      1) df_rot: señales rotadas en formato long (axis = x/y/z/longitudinal/transversal_1/transversal_2)
      2) df_dproj: D_proj por axis (útil para QC)
    """
    Nval = None
    if "param_N" in df_long.columns:
        u = pd.Series(df_long["param_N"]).dropna().unique()
        if len(u) == 1:
            Nval = int(u[0])
    req = {"stat", "direction", "b_step", "roi", "value", b_col}
    missing = req - set(df_long.columns)
    if missing:
        raise ValueError(f"Faltan columnas en df_long: {missing}")

    # Filtramos a avg (como notebook: lee sheet 'avg')
    dfa = df_long[df_long["stat"] == stat_avg].copy()
    if dfa.empty:
        raise ValueError(f"No hay filas con stat='{stat_avg}'.")

    ndirs = int(pd.Series(dfa["direction"]).dropna().nunique())

    if dirs_csv is not None:
        n_dirs = load_dirs_csv(dirs_csv)
        if n_dirs.shape[0] != ndirs:
            raise ValueError(f"dirs_csv tiene {n_dirs.shape[0]} filas, pero el dataset tiene ndirs={ndirs}.")
    else:
        n_dirs = load_default_dirs(ndirs)

    # Ejes destino
    axes = {
        "x": np.array([1, 0, 0], dtype=float),
        "y": np.array([0, 1, 0], dtype=float),
        "z": np.array([0, 0, 1], dtype=float),
    }

    # Asegurar numéricos
    dfa["value"] = pd.to_numeric(dfa["value"], errors="coerce")
    dfa[b_col] = pd.to_numeric(dfa[b_col], errors="coerce")

    out_rows = []
    dproj_rows = []

    # Procesar por ROI
    for roi, d_roi in dfa.groupby("roi", sort=False):
        # S0
        d_b0 = d_roi[d_roi["b_step"] == 0].copy()
        if d_b0.empty:
            raise ValueError(f"ROI={roi}: no encontré b_step==0 (S0).")

        if s0_mode == "dir1":
            s0_row = d_b0[d_b0["direction"] == 1]
            if s0_row.empty:
                raise ValueError(f"ROI={roi}: no encontré direction==1 en b0 para s0_mode='dir1'.")
            S0 = float(s0_row["value"].iloc[0])
        elif s0_mode == "mean":
            S0 = float(d_b0["value"].mean())
        else:
            raise ValueError("s0_mode debe ser 'dir1' o 'mean'.")

        # Para cada b_step > 0, fit tensor con señales por dirección
        for b_step, d_bs in d_roi[d_roi["b_step"] > 0].groupby("b_step", sort=False):
            # bvalue (asumimos igual para todas las dirs dentro del b_step)
            b_vals = d_bs[b_col].dropna().unique()
            if len(b_vals) != 1:
                raise ValueError(f"ROI={roi}, b_step={b_step}: b_col='{b_col}' no es único (encontré {b_vals}).")
            b = float(b_vals[0])

            # señales por dirección, ordenadas 1..ndirs
            d_bs = d_bs.sort_values("direction", kind="stable")
            if len(d_bs) != ndirs:
                raise ValueError(f"ROI={roi}, b_step={b_step}: esperaba {ndirs} dirs, tengo {len(d_bs)}.")

            s = d_bs["value"].to_numpy(dtype=float)
            s_norm = s / S0

            D = fit_tensor_from_signals(b=b, s_norm=s_norm, n_dirs=n_dirs, solver=solver)

            # autovects para long/trans
            e_vals, e_vecs = np.linalg.eigh(D)
            idx = np.argsort(e_vals)[::-1]
            v_long = e_vecs[:, idx[0]]
            v_t1   = e_vecs[:, idx[1]]
            v_t2   = e_vecs[:, idx[2]]

            axes_full = dict(axes)
            axes_full["longitudinal"]  = v_long
            axes_full["transversal_1"] = v_t1
            axes_full["transversal_2"] = v_t2

            # guardar D_proj y señal reconstruida en cada axis
            for axis_name, axis_vec in axes_full.items():
                Dp = D_proj(D, axis_vec)
                S  = S0 * np.exp(-b * Dp)

                out_rows.append({
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b,
                    "signal": S,
                    "signal_norm": S / S0,
                    "S0": S0,
                    "N": Nval,
                })
                dproj_rows.append({
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b,
                    "D_proj": Dp,
                })

        # también incluimos el punto b0 en la salida rotada (útil para plots)
        out_rows.append({
            "roi": roi,
            "axis": "b0",
            "b_step": 0,
            "bvalue": float(d_b0[b_col].dropna().unique()[0]) if d_b0[b_col].notna().any() else 0.0,
            "signal": S0,
            "signal_norm": 1.0,
            "S0": S0,
        })

    df_rot = pd.DataFrame(out_rows).sort_values(["roi", "axis", "b_step"], kind="stable")
    df_dproj = pd.DataFrame(dproj_rows).sort_values(["roi", "axis", "b_step"], kind="stable")
    return RotResult(rotated_signal_long=df_rot, dproj_long=df_dproj)