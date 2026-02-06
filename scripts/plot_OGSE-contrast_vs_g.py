from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = [
    '#377eb8',  # azul
    '#984ea3',  # púrpura
    '#e41a1c',  # rojo
    '#ff7f00',  # naranja
    '#a65628',  # marrón
    '#999999',  # gris
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet")
    ap.add_argument("--xcol", default="g_lin_max_1", help="ej: g_lin_max_1 | g_max_1 | gthorsten_1")
    ap.add_argument("--y", default="contrast_norm", choices=["contrast", "contrast_norm"])
    ap.add_argument("--out_root", default="plots")
    ap.add_argument("--exp", default=None, help="nombre carpeta experimento (opcional)")
    ap.add_argument("--axes", nargs="+", default=["long", "tra"])
    args = ap.parse_args()

    p = Path(args.contrast_parquet)
    df = pd.read_parquet(p)

    exp_id = args.exp or p.stem.split(".contrast_")[0]
    outdir = Path(args.out_root) /  "contrast" / exp_id
    outdir.mkdir(parents=True, exist_ok=True)

    # ROIs y Ns disponibles (N se toma del ref: param_N_1)
    rois = sorted(df["roi"].dropna().unique())
    if "param_N_1" in df.columns:
        N1 = int(pd.Series(df["param_N_1"]).dropna().unique()[0])
    else:
        N1 = None
    if "param_N_2" in df.columns:
        N2 = int(pd.Series(df["param_N_2"]).dropna().unique()[0])
    else:
        N2 = None

    # =========================
    # FIG A: por axis -> todas las ROIs
    # =========================
    for ax_name in args.axes:
        d = df[df["axis"] == ax_name].copy()
        if d.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.title(f"axis={ax_name} | d  =  |  N{N1}-N{N2}", fontsize=14)
        plt.xlabel(f'Gradient strength ({args.xcol}) [mT/m]')
        plt.ylabel(f'OGSE contrast $\Delta M_{{N{N1}-N{N2}}}$')
        plt.grid(True, linestyle="--", alpha=0.3)

        for i, roi in enumerate(rois):
            dr = d[d["roi"] == roi].sort_values(args.xcol)
            if dr.empty:
                continue
            if args.xcol == "gthorsten_1":
                dr[args.xcol] = np.sqrt(2*dr[args.xcol]**2)
            
            plt.plot(dr[args.xcol], dr[args.y], marker="o", label=roi, color=PALETTE[i % len(PALETTE)])

        plt.legend(fontsize=9, title="ROI")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
        plt.tight_layout()
        outpath = outdir / f"{exp_id}.{args.y}_vs_{args.xcol}.axis-{ax_name}.allROIs.png"
        plt.savefig(outpath, dpi=300)
        plt.close()
        print("Saved:", outpath)

    # =========================
    # FIG B: por ROI -> todas las axes (long/tra)
    # =========================
    for roi in rois:
        d = df[(df["roi"] == roi) & (df["axis"].isin(args.axes))].copy()
        if d.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.title(f"ROI={roi} | N{N1}-N{N2}", fontsize=14)
        plt.xlabel(f'Gradient strength ({args.xcol}) [mT/m]')
        plt.ylabel(f'OGSE contrast $\Delta M_{{N{N1}-N{N2}}}$')
        plt.grid(True, linestyle="--", alpha=0.3)

        for ax_i, ax_name in enumerate(args.axes):
            da = d[d["axis"] == ax_name].sort_values(args.xcol)
            if args.xcol == "gthorsten_1":
                da[args.xcol] = np.sqrt(2*da[args.xcol]**2)
            plt.plot(da[args.xcol], da[args.y], marker="o", label=ax_name)

        plt.legend(fontsize=10, title="axis")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
        plt.tight_layout()
        outpath = outdir / f"{exp_id}.{args.y}_vs_{args.xcol}.ROI-{roi}.axes.png"
        plt.savefig(outpath, dpi=300)
        plt.close()
        print("Saved:", outpath)


if __name__ == "__main__":
    main()