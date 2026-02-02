from pathlib import Path
from nogse_table_tools.io import infer_layout_from_filename, read_result_xls
from nogse_table_tools.reshape import to_long

if __name__ == "__main__":
    xls_path = Path("PATH/A/TU/ARCHIVO.xls")  # <-- cambiá esto
    layout = infer_layout_from_filename(xls_path)
    if layout.nbvals is None or layout.ndirs is None:
        raise SystemExit("No pude inferir nbvals/ndirs del nombre. Setealos a mano en el script.")

    stats = read_result_xls(xls_path)
    df_long = to_long(
        stats,
        ndirs=layout.ndirs,
        nbvals=layout.nbvals,
        source_file=xls_path.name,
    )

    out_path = xls_path.with_suffix(".long.parquet")
    df_long.to_parquet(out_path, index=False)
    print("Guardado:", out_path)
    print(df_long.head())
