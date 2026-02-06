import pandas as pd

def sort_curves(df: pd.DataFrame, *, curve_cols: tuple[str, ...], step_col: str) -> pd.DataFrame:
    cols = [c for c in curve_cols if c in df.columns]
    if step_col in df.columns:
        cols.append(step_col)
    if not cols:
        return df
    return df.sort_values(cols, kind="stable").reset_index(drop=True)