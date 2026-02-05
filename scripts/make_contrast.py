from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from nogse_table_tools.contrast import make_contrast


def infer_exp_id(p: Path) -> str:
    name = p.name
    for suf in [".rot_tensor.long.parquet", ".rot_prom.long.parquet"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_parquet", help="rotated signal parquet (ref)")
    ap.add_argument("cmp_parquet", help="rotated signal parquet (comp)")
    ap.add_argument("--axes", nargs="+", default=["long", "tra"])
    ap.add_argument("--out_root", default="OGSE_signal/contrast")
    ap.add_argument("--exp", default=None, help="nombre carpeta experimento (opcional)")
    args = ap.parse_args()

    ref_p = Path(args.ref_parquet)
    cmp_p = Path(args.cmp_parquet)

    df_ref = pd.read_parquet(ref_p)
    df_cmp = pd.read_parquet(cmp_p)

    # N para naming (si existe)
    N1 = int(pd.Series(df_ref.get("param_N", [])).dropna().unique()[0]) if "param_N" in df_ref.columns else -1
    N2 = int(pd.Series(df_cmp.get("param_N", [])).dropna().unique()[0]) if "param_N" in df_cmp.columns else -1

    exp_id = args.exp or infer_exp_id(ref_p)

    res = make_contrast(df_ref, df_cmp, axes=tuple(args.axes))

    outdir = Path(args.out_root) / exp_id / f"contrast_N{N1}-N{N2}"
    outdir.mkdir(parents=True, exist_ok=True)

    out_parquet = outdir / f"{exp_id}.contrast_N{N1}-N{N2}.long.parquet"
    res.df.to_parquet(out_parquet, index=False)
    print("Saved:", out_parquet)


if __name__ == "__main__":
    main()
