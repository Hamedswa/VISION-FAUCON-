# core/indicators.py
# ─────────────────────────────────────────────────────────────────
#  Calculs mathématiques des indicateurs techniques
#  100% formules pures — aucune dépendance IA
#  RSI(14), MACD(12,26,9), EMA(20,50), ATR(14)
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class IndicatorResult:
    # Valeurs brutes
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema20: float
    ema50: float
    atr: float

    # Dérivés — interprétation directe
    rsi_zone: str        # "oversold" | "neutral" | "overbought"
    macd_bullish: bool   # True si MACD > Signal
    trend_ema: str       # "bullish" | "bearish" | "neutral"
    ema20_slope: float   # % variation sur 3 bougies (positif = hausse)
    ema50_slope: float


class TechnicalIndicators:
    """
    Calcule tous les indicateurs techniques sur un DataFrame OHLCV.
    Le DataFrame doit avoir les colonnes : open, high, low, close
    L'index doit être un DatetimeIndex.
    """

    # ── RSI ───────────────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI de Wilder — méthode EWM (identique à TradingView).
        Formule : RSI = 100 - (100 / (1 + RS))
                  RS  = Moyenne des gains / Moyenne des pertes
        """
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        # EWM avec alpha = 1/period (méthode Wilder)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    # ── EMA ───────────────────────────────────────────────────────

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        """
        EMA standard.
        alpha = 2 / (period + 1)
        """
        return close.ewm(span=period, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD classique.
        Retourne : (macd_line, signal_line, histogram)
        """
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── ATR ───────────────────────────────────────────────────────

    @staticmethod
    def atr(
        high:  pd.Series,
        low:   pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        ATR de Wilder.
        TR = max(H-L, |H-Cprev|, |L-Cprev|)
        ATR = EWM(TR, alpha=1/period)
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # ── ANALYSE COMPLÈTE ──────────────────────────────────────────

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> IndicatorResult:
        """
        Calcule tous les indicateurs et retourne les valeurs
        de la DERNIÈRE bougie du DataFrame.

        Args:
            df: DataFrame OHLCV avec colonnes open, high, low, close

        Returns:
            IndicatorResult avec toutes les valeurs calculées
        """
        if len(df) < 60:
            raise ValueError(
                f"DataFrame trop court ({len(df)} bougies). "
                "Minimum 60 requis pour des calculs fiables."
            )

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── Calculs bruts ─────────────────────────────────────────
        rsi_series               = cls.rsi(close)
        ema20_series             = cls.ema(close, 20)
        ema50_series             = cls.ema(close, 50)
        macd_line, signal_line, histogram = cls.macd(close)
        atr_series               = cls.atr(high, low, close)

        # ── Valeurs finales (dernière bougie) ─────────────────────
        rsi_val     = float(rsi_series.iloc[-1])
        ema20_val   = float(ema20_series.iloc[-1])
        ema50_val   = float(ema50_series.iloc[-1])
        macd_val    = float(macd_line.iloc[-1])
        signal_val  = float(signal_line.iloc[-1])
        hist_val    = float(histogram.iloc[-1])
        atr_val     = float(atr_series.iloc[-1])

        # ── Pente EMA sur 4 bougies (% variation) ─────────────────
        ema20_slope = (
            (ema20_series.iloc[-1] - ema20_series.iloc[-4])
            / ema20_series.iloc[-4] * 100
        ) if len(ema20_series) >= 4 else 0.0

        ema50_slope = (
            (ema50_series.iloc[-1] - ema50_series.iloc[-4])
            / ema50_series.iloc[-4] * 100
        ) if len(ema50_series) >= 4 else 0.0

        # ── Interprétation ────────────────────────────────────────
        rsi_zone = (
            "oversold"   if rsi_val < 35 else
            "overbought" if rsi_val > 65 else
            "neutral"
        )

        macd_bullish = macd_val > signal_val

        # Tendance EMA : ema20 > ema50 + pente positive = haussier
        if ema20_val > ema50_val and ema20_slope > 0:
            trend_ema = "bullish"
        elif ema20_val < ema50_val and ema20_slope < 0:
            trend_ema = "bearish"
        else:
            trend_ema = "neutral"

        return IndicatorResult(
            rsi          = rsi_val,
            macd         = macd_val,
            macd_signal  = signal_val,
            macd_hist    = hist_val,
            ema20        = ema20_val,
            ema50        = ema50_val,
            atr          = atr_val,
            rsi_zone     = rsi_zone,
            macd_bullish = macd_bullish,
            trend_ema    = trend_ema,
            ema20_slope  = float(ema20_slope),
            ema50_slope  = float(ema50_slope),
        )
