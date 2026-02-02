from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

@dataclass(frozen=True)
class ResultMeta:
    sheet: str | None
    seq: int | None
    Hz: int | None
    bmax: int | None
    d_ms: int | None
    ndirs: int | None
    nbvals: int | None
    encoding: str | None

def parse_results_filename(path: str | Path) -> ResultMeta:
    p = Path(path)
    name = p.name

    # sheet/experimento: todo antes de "_ep2d" (si existe)
    sheet = None
    if "_ep2d" in name:
        sheet = name.split("_ep2d")[0]
    else:
        # fallback: antes del primer "_"
        sheet = p.stem.split("_")[0]

    def _int(rx: str) -> int | None:
        m = re.search(rx, name, re.IGNORECASE)
        return int(m.group(1)) if m else None

    nbvals = _int(r"(\d+)bval")
    ndirs  = _int(r"(\d+)dir")
    d_ms   = _int(r"_d(\d+)")
    Hz     = _int(r"Hz(\d+)")
    bmax   = _int(r"_b(\d+)")
    seq    = _int(r"_(\d+)_results")  # sirve para _10_resultsa / _5_results

    encoding = "OGSE" if re.search(r"OGSE", name, re.IGNORECASE) else ("PGSE" if re.search(r"PGSE", name, re.IGNORECASE) else None)

    return ResultMeta(sheet=sheet, seq=seq, Hz=Hz, bmax=bmax, d_ms=d_ms, ndirs=ndirs, nbvals=nbvals, encoding=encoding)

def select_params_row(params: pd.DataFrame, meta: ResultMeta) -> pd.Series:
    """
    Devuelve UNA fila de params que matchee el archivo de resultados.
    Estrategia:
      1) filtra por sheet si existe
      2) si hay seq, matchea por seq
      3) si no, matchea por (Hz, bmax, d_ms) cuando estén
    """
    df = params.copy()

    if meta.sheet and "sheet" in df.columns:
        df = df[df["sheet"].astype(str) == str(meta.sheet)]

    if meta.seq is not None and "seq" in df.columns:
        cand = df[df["seq"].fillna(-1).astype(int) == int(meta.seq)]
        if len(cand) == 1:
            return cand.iloc[0]
        if len(cand) > 1:
            # desambiguar con Hz/bmax/d si están
            df = cand
        # si 0 filas, seguimos a heurística

    # Para PGSE: vos decidiste convención Hz=0
    Hz = 0 if (meta.Hz is None and meta.encoding == "PGSE") else meta.Hz

    if Hz is not None and "Hz" in df.columns:
        df = df[df["Hz"].fillna(-999).astype(float) == float(Hz)]
    if meta.bmax is not None and "bmax" in df.columns:
        df = df[df["bmax"].fillna(-999).astype(float) == float(meta.bmax)]
    if meta.d_ms is not None and "d_ms" in df.columns:
        df = df[df["d_ms"].fillna(-999).astype(float) == float(meta.d_ms)]

    if len(df) != 1:
        raise ValueError(
            f"No pude seleccionar 1 fila única de parámetros. Quedaron {len(df)} candidatos.\n"
            f"Meta: {meta}\n"
            f"Tip: revisá sheet/seq/Hz/bmax/d_ms o si hay duplicados en la tabla."
        )

    return df.iloc[0]
