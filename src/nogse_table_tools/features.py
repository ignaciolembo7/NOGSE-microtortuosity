from __future__ import annotations
import numpy as np
import pandas as pd

def add_ogse_features(
    df: pd.DataFrame,
    *,
    gamma: float,          # 1/(ms*mT)
    N: int,
    delta_ms: float,
    delta_app_ms: float,
    gthorsten_val: float | None = None,
    norm_stat: str = "avg",
) -> pd.DataFrame:
    out = df.copy()

    # g como en tu notebook: sqrt((b*1e9) / (N*gamma^2*delta^2*delta_app))
    # bvalue en s/mm^2
    b = out["bvalue"].to_numpy(dtype=float)
    g = np.sqrt((b * 1e9) / (N * (gamma**2) * (delta_ms**2) * delta_app_ms))
    g[b == 0] = 0.0
    out["g"] = g

    # g_max por b_step: máximo g entre direcciones en ese b_step
    gmax_by_step = out.loc[out["b_step"] > 0].groupby("b_step")["g"].max()
    out["g_max"] = out["b_step"].map(gmax_by_step).fillna(0.0)

    # columna equiespaciada 0..g_max con nbvals+1 puntos (como linspace)
    # b_step va 0..nbvals
    out["g_lin_max"] = out["g_max"] * (out["b_step"] / out["b_step"].max())

    if gthorsten_val is not None:
        out["gthorsten"] = gthorsten_val * (out["b_step"] / out["b_step"].max())

    # Normalización SOLO para stat==avg (igual a tu notebook)
    out["value_norm"] = np.nan
    mask_avg = out["stat"] == norm_stat
    # b0 por (direction, roi)
    b0 = out.loc[mask_avg & (out["b_step"] == 0), ["direction", "roi", "value"]]
    b0 = b0.rename(columns={"value": "b0_value"})
    out = out.merge(b0, on=["direction", "roi"], how="left")
    out.loc[mask_avg, "value_norm"] = out.loc[mask_avg, "value"] / out.loc[mask_avg, "b0_value"]
    out = out.drop(columns=["b0_value"])

    return out
