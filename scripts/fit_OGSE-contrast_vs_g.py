from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nogse_table_tools.fit_ogse_contrast import (
    fit_ogse_contrast_long,
    plot_fit_one_group,
)


def _analysis_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet", type=Path, help="Parquet long de contraste (salida de make_contrast.py)")
    ap.add_argument("--model", choices=["free", "tort"], default="free")
    ap.add_argument("--gbase", default="g_lin_max", help="base de g: g | g_lin_max | gthorsten (usa _1/_2 internamente)")
    ap.add_argument("--ycol", default="contrast_norm")
    ap.add_argument("--axes", nargs="*", default=None, help="filtra ejes (ej: long tra eig1 eig2 ...). Si no, usa todos.")
    ap.add_argument("--rois", nargs="*", default=None, help="filtra rois. Si no, usa todas. Usa 'ALL' para todas.")
    ap.add_argument("--out_root", default="OGSE_signal/fits")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    p = args.contrast_parquet
    df = pd.read_parquet(p)

    analysis_id = _analysis_id_from_path(p)
    outdir = Path(args.out_root) / analysis_id
    outdir.mkdir(parents=True, exist_ok=True)

    fit_df = fit_ogse_contrast_long(
        df,
        model=args.model,
        gbase=args.gbase,
        ycol=args.ycol,
        axes=args.axes,
        rois=args.rois,
    )

    out_parquet = outdir / f"{analysis_id}.fit_{args.model}.{args.gbase}.{args.ycol}.parquet"
    out_xlsx = out_parquet.with_suffix(".xlsx")
    fit_df.to_parquet(out_parquet, index=False)
    fit_df.to_excel(out_xlsx, index=False)
    print("Saved fit table:", out_parquet)

    if args.no_plots:
        return

    axis_col = "axis" if "axis" in df.columns else "direction"

    # plots por fila (ROI+axis)
    plots_dir = outdir / "plots"
    for _, r in fit_df.iterrows():
        roi = r["roi"]
        ax = r["axis"]

        g = df[(df["roi"] == roi) & (df[axis_col] == ax)].copy()
        if g.empty:
            continue

        out_png = plots_dir / f"ROI_{roi}" / f"{analysis_id}.{args.model}.{args.gbase}.{args.ycol}.axis_{ax}.png"
        plot_fit_one_group(g, r.to_dict(), out_png=out_png, gbase=args.gbase, ycol=args.ycol)

    print("Saved plots in:", plots_dir)


if __name__ == "__main__":
    main()