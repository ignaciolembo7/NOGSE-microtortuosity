from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nogse_table_tools.models import OGSE_contrast_vs_g_free, OGSE_contrast_vs_g_tort



def _sort_by_x(G1: np.ndarray, G2: np.ndarray, y: np.ndarray, *, xplot: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = G1 if str(xplot) == "1" else G2
    order = np.argsort(x)
    return G1[order], G2[order], y[order]

def _pick_axis_col(df: pd.DataFrame) -> str:
    if "axis" in df.columns:
        return "axis"
    if "direction" in df.columns:
        return "direction"
    raise KeyError("No encuentro columna 'axis' ni 'direction' en el contraste long.")


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _scalar_from_group(g: pd.DataFrame, col: str) -> float | None:
    if col not in g.columns:
        return None
    u = pd.Series(g[col]).dropna().unique()
    if len(u) == 0:
        return None
    return float(u[0])


def _gcols(base: str) -> tuple[str, str]:
    """
    base='g_lin_max' -> ('g_lin_max_1','g_lin_max_2')
    base='g_lin_max_1' -> idem (normaliza)
    """
    b = base
    if b.endswith("_1"):
        b = b[:-2]
    if b.endswith("_2"):
        b = b[:-2]
    return f"{b}_1", f"{b}_2"


def _maybe_scale_gthorsten(base: str, arr: np.ndarray) -> np.ndarray:

     b = base
     if b.endswith("_1") or b.endswith("_2"):
         b = b[:-2]
     if b == "gthorsten":
         return np.sqrt(2.0) * np.abs(arr)
     return arr


@dataclass(frozen=True)
class FitRow:
    roi: str
    axis: str
    model: str
    ycol: str
    gbase: str
    xplot: str
    n_points: int
    n_fit: int
    f_corr: float    
    TE_ms: float
    N1: int
    N2: int

    # fitted params
    M0: float | None = None
    M0_err: float | None = None
    D0: float | None = None
    D0_err: float | None = None
    alpha: float | None = None
    alpha_err: float | None = None

    # fit metrics
    rmse: float | None = None
    chi2: float | None = None
    method: str | None = None
    ok: bool = True
    msg: str = ""
def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _chi2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


def _fit_free_numpy_bruteforce(
    TE: float,
    G1: np.ndarray,
    G2: np.ndarray,
    N1: int,
    N2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
) -> tuple[float, float, float, float, str, float | None, float | None]:
    """Fallback sin SciPy (grilla sobre D0 y/o M0).
    Devuelve: M0, D0, rmse, chi2, method, M0_err, D0_err
    """
    if not M0_vary and not D0_vary:
        yhat = OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, M0_value, D0_value)
        return float(M0_value), float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "fixed", None, None

    D_grid = np.logspace(np.log10(D0_value / 100.0), np.log10(D0_value * 10.0), 240) if D0_vary else np.array([D0_value], float)
    M_grid = np.linspace(0.0, 2.0, 240) if M0_vary else np.array([M0_value], float)

    best = (np.inf, None, None)  # mse, M0, D0
    for D0 in D_grid:
        v = OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, 1.0, float(D0))
        if M0_vary:
            denom = float(np.dot(v, v))
            if denom <= 0:
                continue
            M0_ls = float(np.dot(v, y) / denom)
            M0_ls = float(np.clip(M0_ls, 0.0, 2.0))
            yhat = M0_ls * v
            mse = float(np.mean((y - yhat) ** 2))
            if mse < best[0]:
                best = (mse, M0_ls, float(D0))
        else:
            yhat = float(M0_value) * v
            mse = float(np.mean((y - yhat) ** 2))
            if mse < best[0]:
                best = (mse, float(M0_value), float(D0))

    if best[1] is None:
        raise RuntimeError("No pude encontrar mínimo en la grilla (free).")

    yhat = OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, best[1], best[2])
    return best[1], best[2], _rmse(y, yhat), _chi2(y, yhat), "numpy_bruteforce", None, None
 
def _fit_free(
    TE: float,
    G1: np.ndarray,
    G2: np.ndarray,
    N1: int,
    N2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
) -> tuple[float, float, float, float, str, float | None, float | None]:
    """Ajuste 'free' estilo notebook:
    - bounds M0: [0,2]
    - bounds D0: [D0/100, 10*D0]
    - permite fijar M0 o D0 (vary flags)
    Devuelve: M0, D0, rmse, chi2, method, M0_err, D0_err
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception:
        return _fit_free_numpy_bruteforce(
            TE, G1, G2, N1, N2, y,
            M0_vary=M0_vary, D0_vary=D0_vary, M0_value=M0_value, D0_value=D0_value,
        )

    if not M0_vary and not D0_vary:
        yhat = OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, M0_value, D0_value)
        return float(M0_value), float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "fixed", None, None

    D_lo, D_hi = float(D0_value / 100.0), float(D0_value * 10.0)

    if M0_vary and D0_vary:
        def f(_dummy, M0, D0):
            return OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, M0, D0)
        p0 = [float(M0_value), float(D0_value)]
        bounds = ([0.0, D_lo], [2.0, D_hi])
        popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
        yhat = f(None, *popt)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan])
        return float(popt[0]), float(popt[1]), _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", float(perr[0]), float(perr[1])

    if (not M0_vary) and D0_vary:
        def f(_dummy, D0):
            return OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, float(M0_value), D0)
        p0 = [float(D0_value)]
        bounds = ([D_lo], [D_hi])
        popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
        yhat = f(None, *popt)
        D0 = float(popt[0])
        D0_err = float(np.sqrt(pcov[0, 0])) if pcov is not None and np.isfinite(pcov[0, 0]) else None
        return float(M0_value), D0, _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", None, D0_err

    def f(_dummy, M0):
        return OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, M0, float(D0_value))
    p0 = [float(M0_value)]
    bounds = ([0.0], [2.0])
    popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
    yhat = f(None, *popt)
    M0 = float(popt[0])
    M0_err = float(np.sqrt(pcov[0, 0])) if pcov is not None and np.isfinite(pcov[0, 0]) else None
    return M0, float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", M0_err, None


def _fit_tort(TE, G1, G2, N1, N2, y):
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception as e:
        raise RuntimeError("Para model='tort' necesito SciPy (pip install scipy).") from e

    def f(_dummy, alpha, M0, D0):
        return OGSE_contrast_vs_g_tort(TE, G1, G2, N1, N2, alpha, M0, D0)

    p0 = [0.7, 1.0, 1e-12]
    bounds = ([0.0, 0.0, 1e-14], [2.0, 5.0, 1e-9])

    popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=600000)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan, np.nan])
    yhat = f(None, *popt)
    rmse = _rmse(y, yhat)
    chi2 = _chi2(y, yhat)
    alpha, M0, D0 = map(float, popt)
    return alpha, M0, D0, rmse, chi2, "scipy_curve_fit", float(perr[0]), float(perr[1]), float(perr[2])

def fit_ogse_contrast_long(
    df: pd.DataFrame,
    *,
    model: str = "free",
    gbase: str = "g_lin_max",
    ycol: str = "contrast_norm",
    axes: list[str] | None = None,
    rois: list[str] | None = None,
    xplot: str = "1",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_axis: dict[str, float] | None = None,
    TE_override_ms: float | None = None,
    M0_vary: bool = True,
    D0_vary: bool = True,
    M0_value: float = 1.0,
    D0_value: float = 1e-12,
) -> pd.DataFrame:
    axis_col = _pick_axis_col(df)

    if axes is not None:
        df = df[df[axis_col].isin(axes)].copy()

    if rois is not None and not (len(rois) == 1 and rois[0].upper() == "ALL"):
        df = df[df["roi"].isin(rois)].copy()

    g1c, g2c = _gcols(gbase)
    if g1c not in df.columns or g2c not in df.columns:
        raise KeyError(f"Faltan columnas '{g1c}' y/o '{g2c}' en el contraste long.")

    # params que esperamos del merge del contraste:
    TEc = _pick_col(df, ["param_TE_ms_1", "param_TE_ms", "param_delta_app_ms_1", "param_delta_app_ms"])
    N1c = _pick_col(df, ["param_N_1", "param_N1", "param_N_ref", "param_N"])
    N2c = _pick_col(df, ["param_N_2", "param_N2", "param_N_cmp"])

    if TEc is None or N1c is None or N2c is None:
        raise KeyError(f"No encuentro TE/N1/N2 en columnas. TEc={TEc}, N1c={N1c}, N2c={N2c}")

    group_cols = ["roi", axis_col]
    rows: list[FitRow] = []

    for (roi, ax), g in df.groupby(group_cols, sort=False):
        gg = g.copy()

        # arrays
        y = pd.to_numeric(gg[ycol], errors="coerce").to_numpy(dtype=float)
        G1 = pd.to_numeric(gg[g1c], errors="coerce").to_numpy(dtype=float)
        G2 = pd.to_numeric(gg[g2c], errors="coerce").to_numpy(dtype=float)

        # scale gthorsten si corresponde
        G1 = _maybe_scale_gthorsten(gbase, G1)
        G2 = _maybe_scale_gthorsten(gbase, G2)

        m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
        y, G1, G2 = y[m], G1[m], G2[m]



        TE = _scalar_from_group(gg, TEc)
        N1 = _scalar_from_group(gg, N1c)
        N2 = _scalar_from_group(gg, N2c)

        if TE is None or N1 is None or N2 is None:
            rows.append(FitRow(roi=str(roi), axis=str(ax), model=model, ycol=ycol, gbase=gbase,
                               n_points=int(len(y)), TE_ms=np.nan, N1=-1, N2=-1,
                               ok=False, msg="No pude extraer TE/N1/N2 únicos del grupo."))
            continue

        TE = float(TE)
        N1 = int(round(N1))
        N2 = int(round(N2))

        try:
            if model == "free":
                M0, D0, rmse, method = _fit_free(TE, G1, G2, N1, N2, y)
                rows.append(FitRow(roi=str(roi), axis=str(ax), model=model, ycol=ycol, gbase=gbase,
                                   n_points=int(len(y)), TE_ms=TE, N1=N1, N2=N2,
                                   M0=M0, D0=D0, rmse=rmse, method=method))
            elif model == "tort":
                alpha, M0, D0, rmse, method = _fit_tort(TE, G1, G2, N1, N2, y)
                rows.append(FitRow(roi=str(roi), axis=str(ax), model=model, ycol=ycol, gbase=gbase,
                                   n_points=int(len(y)), TE_ms=TE, N1=N1, N2=N2,
                                   alpha=alpha, M0=M0, D0=D0, rmse=rmse, method=method))
            else:
                rows.append(FitRow(roi=str(roi), axis=str(ax), model=model, ycol=ycol, gbase=gbase,
                                   n_points=int(len(y)), TE_ms=TE, N1=N1, N2=N2,
                                   ok=False, msg=f"Modelo '{model}' no implementado todavía."))
        except Exception as e:
            rows.append(FitRow(roi=str(roi), axis=str(ax), model=model, ycol=ycol, gbase=gbase,
                               n_points=int(len(y)), TE_ms=TE, N1=N1, N2=N2,
                               ok=False, msg=str(e)))

    return pd.DataFrame([r.__dict__ for r in rows])


def plot_fit_one_group(
    df_group: pd.DataFrame,
    fit_row: dict,
    *,
    out_png: Path,
    gbase: str,
    ycol: str,
):
    import matplotlib.pyplot as plt

    axis_col = "axis" if "axis" in df_group.columns else "direction"
    g1c, g2c = _gcols(gbase)

    y = pd.to_numeric(df_group[ycol], errors="coerce").to_numpy(dtype=float)
    G1 = pd.to_numeric(df_group[g1c], errors="coerce").to_numpy(dtype=float)
    G2 = pd.to_numeric(df_group[g2c], errors="coerce").to_numpy(dtype=float)

    G1 = _maybe_scale_gthorsten(gbase, G1)
    G2 = _maybe_scale_gthorsten(gbase, G2)

    m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
    y, G1, G2 = y[m], G1[m], G2[m]

    TE = float(fit_row["TE_ms"])
    N1 = int(fit_row["N1"])
    N2 = int(fit_row["N2"])
    model = fit_row["model"]

    # curva suave: escalamos por fracción (funciona bien para g_lin_max / gthorsten)
    G1max = float(np.nanmax(G1))
    G2max = float(np.nanmax(G2))
    f = np.linspace(0, 1, 250)
    G1s = f * G1max
    G2s = f * G2max

    if model == "free" and fit_row.get("ok", True):
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0"])
        ys = OGSE_contrast_vs_g_free(TE, G1s, G2s, N1, N2, M0, D0)
        label = f"free: M0={M0:.3g}, D0={D0:.3g}, rmse={fit_row.get('rmse', np.nan):.3g}"
    elif model == "tort" and fit_row.get("ok", True):
        alpha = float(fit_row["alpha"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0"])
        ys = OGSE_contrast_vs_g_tort(TE, G1s, G2s, N1, N2, alpha, M0, D0)
        label = f"tort: a={alpha:.3g}, M0={M0:.3g}, D0={D0:.3g}, rmse={fit_row.get('rmse', np.nan):.3g}"
    else:
        ys = None
        label = f"{model} (no fit)"

    plt.figure()
    plt.plot(G1, y, "o", label="data")
    if ys is not None:
        plt.plot(G1s, ys, "-", label=label)

    roi = fit_row.get("roi", "roi")
    axn = fit_row.get("axis", fit_row.get(axis_col, "axis"))
    plt.title(f"OGSE contrast fit | ROI={roi} | axis={axn}")
    plt.xlabel(f"{gbase}_1 (mT/m)")
    plt.ylabel(ycol)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
