from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

DEFAULT_STAT_SHEETS = {0: "avg", 1: "std", 2: "med", 3: "mad", 4: "mode"}

@dataclass(frozen=True)
class Layout:
    nbvals: int | None
    ndirs: int | None

def infer_layout_from_filename(path: str | Path) -> Layout:
    p = Path(path)
    m_bval = re.search(r"(\d+)bval", p.name)
    m_dir  = re.search(r"(\d+)dir", p.name)
    nbvals = int(m_bval.group(1)) if m_bval else None
    ndirs  = int(m_dir.group(1)) if m_dir else None
    return Layout(nbvals=nbvals, ndirs=ndirs)

def read_result_xls(path: str | Path, stat_sheets=DEFAULT_STAT_SHEETS) -> dict[str, pd.DataFrame]:
    """
    Lee un .xls con 5 sheets tipo:
      Sheet1..Sheet5  -> avg/std/med/mad/mode por orden.
    Requiere: pandas + xlrd (para .xls).
    """
    xls = pd.ExcelFile(path)
    out: dict[str, pd.DataFrame] = {}
    for sheet_idx, stat_name in stat_sheets.items():
        out[stat_name] = pd.read_excel(xls, sheet_name=sheet_idx)
    return out
