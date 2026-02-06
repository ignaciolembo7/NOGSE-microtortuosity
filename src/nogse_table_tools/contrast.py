from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from nogse_table_tools.order import sort_curves


@dataclass
class ContrastResult:
    df: pd.DataFrame


def make_contrast(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    *,
    axes: tuple[str, ...] = ("long", "tra"),
    y_col: str = "signal",
    y_norm_col: str = "signal_norm",
    key_cols: tuple[str, ...] = ("axis", "roi", "b_step"),
) -> ContrastResult:
    """
    Replica el notebook:
      contrast      = S_ref - S_cmp
      contrast_norm = Sref/Sref0 - Scmp/Scmp0

    En nuestro pipeline, si existe signal_norm, entonces:
      contrast_norm = signal_norm_ref - signal_norm_cmp
    """
    a = df_ref.copy()
    b = df_cmp.copy()

    # filtrar ejes pedidos
    a = a[a["axis"].isin(axes)]
    b = b[b["axis"].isin(axes)]

    # columnas que quiero arrastrar con sufijo _1 / _2
    keep_cols = set(key_cols)

    # gradientes típicos + params (lo mismo que venís arrastrando)
    for c in ["bvalue", "bvalue_orig", "bvalue_raw", "g", "g_max", "g_lin_max", "gthorsten", "g_type"]:
        if c in a.columns:
            keep_cols.add(c)
        if c in b.columns:
            keep_cols.add(c)

    for c in a.columns:
        if c.startswith("param_"):
            keep_cols.add(c)
    for c in b.columns:
        if c.startswith("param_"):
            keep_cols.add(c)

    # señales
    keep_cols.update([y_col])
    if y_norm_col in a.columns and y_norm_col in b.columns:
        keep_cols.add(y_norm_col)
    if "S0" in a.columns and "S0" in b.columns:
        keep_cols.add("S0")

    a = a[[c for c in a.columns if c in keep_cols]]
    b = b[[c for c in b.columns if c in keep_cols]]

    m = a.merge(b, on=list(key_cols), suffixes=("_1", "_2"), how="inner")

    # contrastes
    m["contrast"] = m[f"{y_col}_1"] - m[f"{y_col}_2"]

    if f"{y_norm_col}_1" in m.columns and f"{y_norm_col}_2" in m.columns:
        m["contrast_norm"] = m[f"{y_norm_col}_1"] - m[f"{y_norm_col}_2"]
    else:
        # fallback: normalizar con S0
        m["contrast_norm"] = (m[f"{y_col}_1"] / m["S0_1"]) - (m[f"{y_col}_2"] / m["S0_2"])

    # Orden: cada curva (axis, roi) con b_step creciendo (b_step=0 primero)
    sort_cols = [c for c in ["axis", "roi", "b_step"] if c in m.columns]
    if sort_cols:
        m = m.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    m = sort_curves(m, curve_cols=("axis", "roi"), step_col="b_step")

    return ContrastResult(df=m)
