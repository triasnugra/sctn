"""Indikator dasar & primitif pattern (pivot, ATR, OBV, volume).
Detektor spesifik (deviation, fib, dll) ada di src/analysis/."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    unit, num = tf[-1], int(tf[:-1])
    if unit not in _TF_SECONDS:
        raise ValueError(f"Timeframe tidak didukung: {tf}")
    return pd.Timedelta(seconds=num * _TF_SECONDS[unit])


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def last_atr(df: pd.DataFrame, period: int = 14) -> float:
    v = atr(df, period).iloc[-1]
    if v is None or np.isnan(v):
        v = float((df["high"] - df["low"]).tail(period).mean())
    return float(v)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, recent: int = 3, base: int = 50) -> float:
    v = df["volume"]
    if len(v) < recent + 10:
        return float("nan")
    base = min(base, len(v) - recent)
    b = v.iloc[-(recent + base):-recent].mean()
    if b <= 0 or np.isnan(b):
        return float("nan")
    return float(v.iloc[-recent:].mean() / b)


def vol_sma_prior(v: np.ndarray, n: int = 20) -> np.ndarray:
    """SMA volume dari n candle SEBELUM tiap indeks (eksklusif)."""
    return pd.Series(v).rolling(n, min_periods=5).mean().shift(1).to_numpy()


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # 'H' | 'L'


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Pivot]:
    """Swing high/low terkonfirmasi (butuh `right` candle setelahnya).
    High: > semua `left` candle sebelumnya dan >= semua `right` candle sesudahnya."""
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(df)
    out: List[Pivot] = []
    for i in range(left, n - right):
        if h[i] > h[i - left:i].max() and h[i] >= h[i + 1:i + right + 1].max():
            out.append(Pivot(i, float(h[i]), "H"))
        if l[i] < l[i - left:i].min() and l[i] <= l[i + 1:i + right + 1].min():
            out.append(Pivot(i, float(l[i]), "L"))
    out.sort(key=lambda p: p.idx)
    return out


def merge_alternating(pivots: List[Pivot]) -> List[Pivot]:
    """Pastikan pivot selang-seling H/L; jika dua sejenis berurutan ambil yang lebih ekstrem."""
    out: List[Pivot] = []
    for p in sorted(pivots, key=lambda x: x.idx):
        if out and out[-1].kind == p.kind:
            if (p.kind == "H" and p.price > out[-1].price) or (p.kind == "L" and p.price < out[-1].price):
                out[-1] = p
        else:
            out.append(p)
    return out
