from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from nogse_table_tools.contrast import make_contrast
import re
import hashlib


def infer_exp_id(p: Path) -> str:
    name = p.name
    for suf in [".rot_tensor.long.parquet", ".rot_prom.long.parquet"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return p.stem

def _one(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        return default
    u = pd.Series(df[col]).dropna().unique()
    return u[0] if len(u) else default

def _fmt_num(x) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    # entero si es casi entero
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    # redondeo corto + reemplazo '.' -> 'p'
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def build_analysis_id(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    ref_name: str,
    cmp_name: str,
    axes: list[str],
    N1: int,
    N2: int,
) -> tuple[str, str]:
    # experimento corto
    sheet = str(_one(df_ref, "param_sheet", _one(df_cmp, "param_sheet", "EXP")))

    # parámetros cortos que suelen distinguir bien
    s1 = _fmt_num(_one(df_ref, "param_seq"))
    s2 = _fmt_num(_one(df_cmp, "param_seq"))
    hz1 = _fmt_num(_one(df_ref, "param_Hz"))
    hz2 = _fmt_num(_one(df_cmp, "param_Hz"))
    dms = _fmt_num(_one(df_ref, "param_d_ms", _one(df_ref, "param_delta_ms")))
    b1 = _fmt_num(_one(df_ref, "param_bmax"))
    b2 = _fmt_num(_one(df_cmp, "param_bmax"))

    axes_tag = "-".join(axes)

    # hash corto para unicidad (aunque todo lo demás coincida)
    # h = hashlib.blake2s(f"{ref_name}|{cmp_name}|{axes_tag}".encode("utf-8"), digest_size=4).hexdigest()

    analysis = f"{sheet}_N{N1}-N{N2}_d{dms}" #_s{s1}-{s2}_Hz{hz1}-{hz2}_b{b1}-{b2}_{axes_tag}_{h}"
    analysis = re.sub(r"[^A-Za-z0-9._-]+", "_", analysis)
    analysis = analysis[:120]  # recorte por seguridad Windows
    return sheet, analysis


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

    res = make_contrast(
        df_ref, df_cmp,
        axes=tuple(args.axes),
        y_col="value",
        key_cols=("stat", "roi", "direction", "b_step"),
    )
    
    sheet, analysis_id = build_analysis_id(
        df_ref, df_cmp,
        ref_p.name, cmp_p.name,
        args.axes, N1, N2
    )

    # permite override manual si querés
    if args.exp is not None:
        sheet = args.exp

    outdir = Path(args.out_root) / analysis_id
    outdir.mkdir(parents=True, exist_ok=True)

    out_parquet = outdir / f"{analysis_id}.long.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    res.df.to_parquet(out_parquet, index=False)
    df=pd.read_parquet(out_parquet)
    df.to_excel(out_parquet.with_suffix(".xlsx"), index=False)

    print("Saved:", out_parquet)

if __name__ == "__main__":
    main()
