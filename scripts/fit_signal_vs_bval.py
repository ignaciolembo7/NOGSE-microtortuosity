from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nogse_table_tools.bvalues import b_from_g


def exp_model(b, M0, D0):
    return M0 * np.exp(-b * D0)


def exp_model_M0fixed(b, D0, *, M0):
    return M0 * np.exp(-b * D0)


def infer_exp_id(p: Path) -> str:
    name = p.name
    suf = ".rot_tensor.long.parquet"
    if name.endswith(suf):
        return name[: -len(suf)]
    return p.stem


def get_unique_float(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")
    u = pd.Series(df[col]).dropna().unique()
    if len(u) != 1:
        raise ValueError(f"Esperaba 1 valor único en {col}, encontré {u}")
    return float(u[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rot_parquet", help="Archivo .rot_tensor.long.parquet")
    ap.add_argument("--fit_points", type=int, default=6, help="Cantidad de puntos iniciales a fitear (incluye b_step=0 si existe).")
    ap.add_argument("--axes", nargs="+", default=["long", "tra"], help="Ejes a fitear (ej: long tra, o x y z, etc.)")
    ap.add_argument("--roi", default="ALL", help="ROI a fitear, o ALL")
    ap.add_argument("--ycol", default="signal_norm", help="Columna de señal: signal_norm o signal")
    ap.add_argument("--g_type", default="gthorsten", choices=["bvalue", "g", "g_max", "g_lin_max", "gthorsten"],
                    help="Cómo construir bvalue para el fit (igual que notebook).")
    ap.add_argument("--gamma", type=float, default=267.5221900, help="gamma en 1/(ms*mT), igual notebook.")
    ap.add_argument("--D0_init", type=float, default=0.0023, help="Inicial D0 en mm^2/s (para b en s/mm^2).")
    ap.add_argument("--fix_M0", type=float, default=1.0, help="Si querés fijar M0, dejalo en 1.0 (default). Para liberar M0: usa --free_M0.")
    ap.add_argument("--free_M0", action="store_true", help="Si se activa, ajusta M0 y D0.")
    ap.add_argument("--out_root", default="plots", help="Raíz de salida (default plots/)")
    args = ap.parse_args()

    p = Path(args.rot_parquet)
    df = pd.read_parquet(p)

    # filtros básicos
    if "axis" not in df.columns or "roi" not in df.columns or "b_step" not in df.columns:
        raise ValueError("El parquet no parece ser rot_tensor.long.parquet (faltan axis/roi/b_step).")

    df = df[df["axis"].isin(args.axes)].copy()
    if args.roi != "ALL":
        df = df[df["roi"] == args.roi].copy()

    if df.empty:
        raise ValueError("No quedaron filas luego del filtro axis/roi.")

    # params (para convertir g -> b)
    Nval = get_unique_float(df, "param_N")
    delta_ms = get_unique_float(df, "param_delta_ms")
    delta_app_ms = get_unique_float(df, "param_delta_app_ms")

    exp_id = infer_exp_id(p)
    outdir = Path(args.out_root) / exp_id / "fit_signal"
    outdir.mkdir(parents=True, exist_ok=True)

    # qué usar como x (bvalue)
    if args.g_type == "bvalue":
        x_mode = "bvalue"
    else:
        x_mode = args.g_type
        if x_mode not in df.columns:
            raise ValueError(f"No encuentro columna '{x_mode}' en el parquet. Columnas: {list(df.columns)}")

    # rois / axes a iterar
    rois = sorted(df["roi"].dropna().unique())
    axes = args.axes

    results = []
    fit_rows = []

    for axis in axes:
        for roi in rois:
            d = df[(df["axis"] == axis) & (df["roi"] == roi)].sort_values("b_step", kind="stable").copy()
            if d.empty:
                continue

            y = pd.to_numeric(d[args.ycol], errors="coerce").to_numpy(dtype=float)

            if x_mode == "bvalue":
                b = pd.to_numeric(d["bvalue"], errors="coerce").to_numpy(dtype=float)
            else:
                g = pd.to_numeric(d[x_mode], errors="coerce").to_numpy(dtype=float)
                b = b_from_g(g, N=Nval, gamma=args.gamma, delta_ms=delta_ms, delta_app_ms=delta_app_ms, g_type=args.g_type)

            # seleccionar primeros fit_points
            k = min(args.fit_points, len(b))
            b_fit = b[:k].copy()
            y_fit = y[:k].copy()

            # filtrar puntos inválidos (y<=0 rompe log y la exp)
            m = np.isfinite(b_fit) & np.isfinite(y_fit) & (y_fit > 0)
            b_fit = b_fit[m]
            y_fit = y_fit[m]

            if len(b_fit) < 3:
                print(f"[skip] axis={axis} roi={roi}: no hay suficientes puntos válidos para fit.")
                continue

            # FIT
            M0_fixed = args.fix_M0
            D0_init = args.D0_init

            if args.free_M0:
                p0 = [1.0, D0_init]
                bounds = ([0.0, D0_init / 10], [2.0, 2 * D0_init])
                popt, pcov = curve_fit(exp_model, b_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                M0_hat, D0_hat = float(popt[0]), float(popt[1])
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
                M0_err, D0_err = float(perr[0]), float(perr[1])
            else:
                # M0 fijo
                p0 = [D0_init]
                bounds = ([D0_init / 10], [2 * D0_init])
                f = lambda bb, D0: exp_model_M0fixed(bb, D0, M0=M0_fixed)
                popt, pcov = curve_fit(f, b_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                M0_hat, D0_hat = float(M0_fixed), float(popt[0])
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]
                M0_err, D0_err = np.nan, float(perr[0])

            # curva para plot
            b_dense = np.linspace(0, np.nanmax(b), 1000)
            y_dense = exp_model(b_dense, M0_hat, D0_hat)

            # guardar resumen
            results.append({
                "exp": exp_id,
                "axis": axis,
                "roi": roi,
                "g_type": args.g_type,
                "fit_points": args.fit_points,
                "M0": M0_hat,
                "M0_err": M0_err,
                "D0_mm2_s": D0_hat,
                "D0_err_mm2_s": D0_err,
                "param_N": Nval,
                "param_delta_ms": delta_ms,
                "param_delta_app_ms": delta_app_ms,
            })

            # tabla “long” para comparar/depurar
            for bi, yi, bs in zip(b, y, d["b_step"].to_numpy()):
                fit_rows.append({
                    "exp": exp_id,
                    "axis": axis,
                    "roi": roi,
                    "b_step": int(bs),
                    "bvalue_fit": float(bi),
                    "y": float(yi) if np.isfinite(yi) else np.nan,
                    "used_for_fit": bool(int(bs) < args.fit_points),  # aproximación: los primeros steps
                    "M0": M0_hat,
                    "D0_mm2_s": D0_hat,
                })

            # PLOT (log y)
            plt.figure(figsize=(8, 6))
            plt.plot(b, y, "o", markersize=6, label="data")

            # resaltar puntos usados
            plt.plot(b[:k], y[:k], "o", markersize=8, label=f"fit points (first {args.fit_points})")

            plt.plot(b_dense, y_dense, "-", linewidth=2, label=f"fit: D0={D0_hat:.3e} mm²/s")

            plt.yscale("log")
            plt.xlabel("bvalue [s/mm²]")
            plt.ylabel(args.ycol)
            plt.title(f"{exp_id} | axis={axis} | roi={roi} | g_type={args.g_type}")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(fontsize=9)
            plt.tight_layout()

            out_png = outdir / f"{exp_id}.fit_exp.axis-{axis}.ROI-{roi}.g-{args.g_type}.k-{args.fit_points}.png"
            plt.savefig(out_png, dpi=300)
            plt.close()
            print("Saved:", out_png)

    # Guardar outputs tabulares
    df_res = pd.DataFrame(results).sort_values(["axis", "roi"])
    df_fit = pd.DataFrame(fit_rows)

    out_csv = outdir / f"{exp_id}.fit_params.csv"
    df_res.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    out_xlsx = outdir / f"{exp_id}.fit_params.xlsx"
    with pd.ExcelWriter(out_xlsx) as w:
        df_res.to_excel(w, sheet_name="fit_params", index=False)
        df_fit.to_excel(w, sheet_name="fit_table", index=False)
    print("Saved:", out_xlsx)


if __name__ == "__main__":
    main()