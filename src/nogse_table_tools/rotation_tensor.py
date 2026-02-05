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

def b_from_g(g: np.ndarray, *, N: float, gamma: float, delta_ms: float, delta_app_ms: float, g_type: str) -> np.ndarray:
    """
    Replica tu notebook:
      if g_type == 'gthorsten': g = sqrt(2)*g
      b = N * gamma^2 * delta^2 * delta_app * g^2 / 1e9
    Asume: delta y delta_app en ms, g en mT/m (como en tus tablas).
    """
    g = np.asarray(g, dtype=float)
    if g_type == "gthorsten":
        g = np.sqrt(2.0) * g
    return N * (gamma**2) * (delta_ms**2) * (delta_app_ms) * (g**2) / 1e9

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
    gamma: float = 267.5221900, # 1/ms.mT 
    g_type: str = "g_lin_max",   # "g", "g_lin_max", "gthorsten"
    dirs_csv: str | Path | None = None, 
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

    delta_ms = None
    delta_app_ms = None

    if "param_delta_ms" in df_long.columns:
        u = pd.Series(df_long["param_delta_ms"]).dropna().unique()
        if len(u) == 1:
            delta_ms = float(u[0])

    if "param_delta_app_ms" in df_long.columns:
        u = pd.Series(df_long["param_delta_app_ms"]).dropna().unique()
        if len(u) == 1:
            delta_app_ms = float(u[0])

    if Nval is None or delta_ms is None or delta_app_ms is None:
        raise ValueError("Faltan parámetros para b_from_g: necesito param_N, param_delta_ms y param_delta_app_ms.")

    req = {"stat", "direction", "b_step", "roi", "value", b_col}
    missing = req - set(df_long.columns)
    if missing:
        raise ValueError(f"Faltan columnas en df_long: {missing}")

    # Filtramos a avg (como notebook: lee sheet 'avg')
    dfa = df_long[df_long["stat"] == stat_avg].copy()
    if dfa.empty:
        raise ValueError(f"No hay filas con stat='{stat_avg}'.")
    param_cols = [c for c in dfa.columns if c.startswith("param_")]

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
    # Convertimos columnas de b si existen
    if "bvalue" in dfa.columns:
        dfa["bvalue"] = pd.to_numeric(dfa["bvalue"], errors="coerce")
    if "bvalue_thorsten" in dfa.columns:
        dfa["bvalue_thorsten"] = pd.to_numeric(dfa["bvalue_thorsten"], errors="coerce")

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
            
            # señales por dirección, ordenadas 1..ndirs
            d_bs = d_bs.sort_values("direction", kind="stable")
            if len(d_bs) != ndirs:
                raise ValueError(f"ROI={roi}, b_step={b_step}: esperaba {ndirs} dirs, tengo {len(d_bs)}.")

            # --- elegimos la columna de gradiente según g_type
            g_col = None
            if g_type == "g" and "g" in d_bs.columns:
                g_col = "g"
            elif g_type == "g_max" and "g_max" in d_bs.columns:
                g_col = "g_max"
            elif g_type == "gthorsten":
                if "gthorsten" in d_bs.columns:
                    g_col = "gthorsten"
                elif "gthorsten_mTm" in d_bs.columns:
                    g_col = "gthorsten_mTm"
            elif g_type == "g_lin_max" and "g_lin_max" in d_bs.columns:
                g_col = "g_lin_max"

            # b original (para guardar, comparar)
            b_orig_series = pd.to_numeric(d_bs.loc[d_bs["direction"] == 1, b_col], errors="coerce")
            b_report_orig = float(b_orig_series.iloc[0]) if (not b_orig_series.empty and pd.notna(b_orig_series.iloc[0])) else None

            # b para el fit: calculado como en el notebook desde g_type
            if g_col is None:
                raise ValueError(f"No encontré columna de gradiente para g_type='{g_type}'. Busqué: g, g_max, gthorsten, gthorsten_mTm.")

            g_dir1 = pd.to_numeric(d_bs.loc[d_bs["direction"] == 1, g_col], errors="coerce").iloc[0]
            if not np.isfinite(g_dir1):
                raise ValueError(f"ROI={roi}, b_step={b_step}: g_dir1 es NaN/inf en col {g_col}.")

            b = float(b_from_g(np.array([g_dir1]), N=float(Nval), gamma=float(gamma), delta_ms=float(delta_ms), delta_app_ms=float(delta_app_ms), g_type=g_type)[0])
            b_report = b  # este es el bvalue 'thorsten' (o el que elijas con g_type)

            s = d_bs["value"].to_numpy(dtype=float)
            s_norm = s / S0

            D = fit_tensor_from_signals(b=b, s_norm=s_norm, n_dirs=n_dirs, solver=solver)

            # eigen-decomp
            e_vals, e_vecs = np.linalg.eigh(D)
            idx = np.argsort(e_vals)[::-1] 
            v1 = e_vecs[:, idx[0]]
            v2 = e_vecs[:, idx[1]]
            v3 = e_vecs[:, idx[2]]

            # ejes canónicos
            axes_full = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
                # eigen-directions (autovalores)
                "eig1": v1,
                "eig2": v2,
                "eig3": v3,
                # definiciones tuyas
                "long": np.array([1.0, 0.0, 0.0]),
            }

            # --- generar filas por eje
            S_y = None
            S_z = None

            # --- extras por b_step (se copian a todas las filas de salida)
            extras = {}

            # 1) arrastrar todos los param_*
            for c in param_cols:
                vals = d_bs[c].dropna()
                if vals.empty:
                    continue
                uniq = vals.astype(str).unique()
                if len(uniq) == 1:
                    extras[c] = vals.iloc[0]
                else:
                    nums = pd.to_numeric(vals, errors="coerce")
                    extras[c] = float(nums.median()) if nums.notna().any() else vals.iloc[0]

            # 2) arrastrar g / g_max / gthorsten (si existen)
            for c in ["g", "g_lin_max", "gthorsten", "gthorsten_mTm"]:
                if c not in d_bs.columns:
                    continue
                vals = pd.to_numeric(d_bs[c], errors="coerce")
                if vals.notna().any():
                    key = "gthorsten" if c == "gthorsten_mTm" else c
                    extras[key] = float(vals.median())


            for axis_name, axis_vec in axes_full.items():
                Dp = D_proj(D, axis_vec)
                S  = S0 * np.exp(-b * Dp)

                if axis_name == "y":
                    S_y = S
                elif axis_name == "z":
                    S_z = S

                row_dict = {
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "signal": S,
                    "signal_norm": S / S0,
                    "S0": S0,
                }
                row_dict.update(extras)
                if b_report_orig is not None:
                    row_dict["bvalue_orig"] = b_report_orig
                out_rows.append(row_dict)

                dproj_dict = {
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "D_proj": Dp,
                }
                dproj_dict.update(extras)
                if b_report_orig is not None:
                    dproj_dict["bvalue_orig"] = b_report_orig
                
                if axis_name in {"x", "y", "z", "eig1", "eig2", "eig3"}:
                    dproj_rows.append(dproj_dict)

            # --- Dproj también en las direcciones medidas (dir1..dirN)
            for k in range(ndirs):
                Dp_dir = D_proj(D, n_dirs[k])
                dproj_dict = {
                    "roi": roi,
                    "axis": f"dir{k+1}",
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "D_proj": Dp_dir,
                }
                dproj_dict.update(extras)
                if b_report_orig is not None:
                    dproj_dict["bvalue_orig"] = b_report_orig
                dproj_rows.append(dproj_dict)

            # --- tra = promedio de señales y y z (NO vector, NO D_proj)
            if (S_y is not None) and (S_z is not None):
                S_tra = 0.5 * (S_y + S_z)
                row_dict = {
                    "roi": roi,
                    "axis": "tra",
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "signal": S_tra,
                    "signal_norm": S_tra / S0,
                    "S0": S0,
                }
                row_dict.update(extras)
                if b_report_orig is not None:
                    row_dict["bvalue_orig"] = b_report_orig
                out_rows.append(row_dict)

    df_rot = pd.DataFrame(out_rows).sort_values(["roi", "axis", "b_step"], kind="stable")
    df_dproj = pd.DataFrame(dproj_rows).sort_values(["roi", "axis", "b_step"], kind="stable")
    return RotResult(rotated_signal_long=df_rot, dproj_long=df_dproj)