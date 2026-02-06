from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class ContrastResult:
    df: pd.DataFrame


def make_contrast(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    *,
    axes: tuple[str, ...] = ("long", "tra"),
    y_col: str = "value",
    y_norm_col: str = "signal_norm",
    key_cols: tuple[str, ...] = ("stat", "roi", "direction", "b_step"),
) -> ContrastResult:
    """
    contrast      = S_ref - S_cmp
    contrast_norm = Sref/Sref0 - Scmp/Scmp0

    Si existe signal_norm en ambas tablas:
      contrast_norm = signal_norm_ref - signal_norm_cmp
    Si no, fallback:
      contrast_norm = (S_ref/S0_ref) - (S_cmp/S0_cmp)
    """
    a = df_ref.copy()
    b = df_cmp.copy()

    # --- Normalizar nombre de dirección entre tablas (preferimos "direction")
    if "direction" in a.columns or "direction" in b.columns:
        if "axis" in a.columns and "direction" not in a.columns:
            a = a.rename(columns={"axis": "direction"})
        if "axis" in b.columns and "direction" not in b.columns:
            b = b.rename(columns={"axis": "direction"})
        dir_col = "direction"
    elif "axis" in a.columns and "axis" in b.columns:
        dir_col = "axis"
    else:
        raise KeyError(
            "No encuentro columna de dirección compatible (axis/direction). "
            f"df_ref cols={sorted(a.columns)}; df_cmp cols={sorted(b.columns)}"
        )

    # --- Unificar nombre de gthorsten
    for df in (a, b):
        if "gthorsten" not in df.columns and "gthorsten_mTm" in df.columns:
            df["gthorsten"] = df["gthorsten_mTm"]

    # --- Fallback de nombre de señal
    if y_col not in a.columns and "signal" in a.columns:
        y_col = "signal"
    if y_col not in b.columns and "signal" in b.columns:
        y_col = "signal"

    # --- Key cols efectivos (si alguien pasa "axis" en key_cols pero usamos direction, o viceversa)
    key_cols_eff = list(key_cols)
    if dir_col == "direction":
        key_cols_eff = ["direction" if c == "axis" else c for c in key_cols_eff]
    else:
        key_cols_eff = ["axis" if c == "direction" else c for c in key_cols_eff]

    # --- Filtrar direcciones requeridas
    a = a[a[dir_col].isin(axes)]
    b = b[b[dir_col].isin(axes)]

    # --- Asegurar que existan columnas de gradiente en ambos (para sufijos _1/_2)
    grad_cols = ["g", "g_lin_max", "gthorsten"]
    for c in grad_cols:
        if c in a.columns and c not in b.columns:
            b[c] = pd.NA
        if c in b.columns and c not in a.columns:
            a[c] = pd.NA

    # --- Columnas a arrastrar con sufijos _1/_2 (NO bvalues)
    keep_cols = set(key_cols_eff)
    for c in grad_cols:
        if c in a.columns:
            keep_cols.add(c)
        if c in b.columns:
            keep_cols.add(c)

    keep_cols.add(y_col)
    if y_norm_col in a.columns and y_norm_col in b.columns:
        keep_cols.add(y_norm_col)
    if "S0" in a.columns and "S0" in b.columns:
        keep_cols.add("S0")

    for c in a.columns:
        if c.startswith("param_"):
            keep_cols.add(c)
    for c in b.columns:
        if c.startswith("param_"):
            keep_cols.add(c)

    a = a[[c for c in keep_cols if c in a.columns]].copy()
    b = b[[c for c in keep_cols if c in b.columns]].copy()

    # --- Merge
    m = a.merge(b, on=key_cols_eff, suffixes=("_1", "_2"), how="inner")

    # --- Contrastes
    m["contrast"] = m[f"{y_col}_1"] - m[f"{y_col}_2"]

    if f"{y_norm_col}_1" in m.columns and f"{y_norm_col}_2" in m.columns:
        m["contrast_norm"] = m[f"{y_norm_col}_1"] - m[f"{y_norm_col}_2"]
    else:
        if "S0_1" not in m.columns or "S0_2" not in m.columns:
            raise KeyError("Faltan signal_norm y también faltan S0_1/S0_2 para el fallback de contrast_norm.")
        m["contrast_norm"] = (m[f"{y_col}_1"] / m["S0_1"]) - (m[f"{y_col}_2"] / m["S0_2"])

    # --- Orden final
    sort_cols = [c for c in ["stat", "roi", dir_col, "b_step"] if c in m.columns]
    dir_order = ["eig1", "eig2", "eig3", "x", "y", "z", "tra", "long"]
    if dir_col in m.columns:
        u = set(pd.Series(m[dir_col]).dropna().unique())
        if u and u.issubset(set(dir_order)):
            m[dir_col] = pd.Categorical(m[dir_col], categories=dir_order, ordered=True)

    if sort_cols:
        m = m.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    if dir_col in m.columns and hasattr(m[dir_col].dtype, "categories"):
        m[dir_col] = m[dir_col].astype(str)

    return ContrastResult(df=m)
