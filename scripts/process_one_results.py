from __future__ import annotations
from pathlib import Path
import argparse

from nogse_table_tools.io import read_result_xls
from nogse_table_tools.reshape import to_long
from nogse_table_tools.params import read_sequence_params_xlsx
from nogse_table_tools.match_params import parse_results_filename, select_params_row
from nogse_table_tools.features import add_ogse_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_xls", type=Path)
    ap.add_argument("params_xlsx", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("OGSE_signal/data"))
    ap.add_argument("--gamma", type=float, default=267.522)  # 1/(ms*mT)
    args = ap.parse_args()

    meta = parse_results_filename(args.results_xls)

    if meta.ndirs is None or meta.nbvals is None:
        raise SystemExit(f"No pude inferir ndirs/nbvals del filename: {args.results_xls.name}")

    stats = read_result_xls(args.results_xls)
    df_long = to_long(
        stats,
        ndirs=meta.ndirs,
        nbvals=meta.nbvals,
        source_file=args.results_xls.name,
    )

    params = read_sequence_params_xlsx(args.params_xlsx)
    row = select_params_row(params, meta)

    # Attach metadata + params a todas las filas (útil para tracking)
    for k, v in row.to_dict().items():
        df_long[f"param_{k}"] = v
    df_long["meta_sheet"] = meta.sheet
    df_long["meta_seq"] = meta.seq
    df_long["meta_Hz"] = meta.Hz
    df_long["meta_bmax"] = meta.bmax
    df_long["meta_d_ms"] = meta.d_ms
    df_long["meta_encoding"] = meta.encoding

    # Compute features (igual lógica que notebook)
    df_long = add_ogse_features(
        df_long,
        gamma=args.gamma,
        N=int(row["N"]),
        delta_ms=float(row["delta_ms"]),
        delta_app_ms=float(row["delta_app_ms"]),
        gthorsten_val=float(row["gthorsten_mTm"]) if "gthorsten_mTm" in row and row["gthorsten_mTm"] == row["gthorsten_mTm"] else None,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (args.results_xls.stem + ".long.parquet")
    df_long.to_parquet(out_path, index=False)

    print("Selected params row:")
    print(row[["sheet","seq","Hz","bmax","d_ms","delta_ms","delta_app_ms","N","gthorsten_mTm"]].to_string())
    print("Saved:", out_path)

if __name__ == "__main__":
    main()