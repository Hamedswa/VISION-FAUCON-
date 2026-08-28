# core/smc_detector.py
# ─────────────────────────────────────────────────────────────────
#  Détecteur SMC/ICT — 100% mathématique, zéro IA
#
#  Concepts implémentés :
#   • Swing Highs / Swing Lows (structure de marché)
#   • BOS  — Break of Structure (continuation)
#   • CHoCH — Change of Character (retournement)
#   • OB   — Order Block (zone d'offre / demande)
#   • FVG  — Fair Value Gap / Imbalance
#   • BSL/SSL — Liquidités (Buyside / Sellside)
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── DATA CLASSES (structures de résultats) ────────────────────────

@dataclass
class SwingPoint:
    index:     int
    timestamp: pd.Timestamp
    price:     float
    kind:      str   # "high" | "low"


@dataclass
class BOSResult:
    confirmed:       bool
    direction:       str    # "bullish" | "bearish"
    broken_level:    float  # Niveau de structure cassé
    break_idx:       int    # Index de la bougie qui a cassé
    strength:        float  # Distance au-delà du niveau (en prix)


@dataclass
class CHoCHResult:
    confirmed:    bool
    direction:    str    # "bullish" (retournement haussier) | "bearish"
    broken_level: float
    break_idx:    int


@dataclass
class OrderBlock:
    valid:      bool
    direction:  str    # "bullish" | "bearish"
    zone_high:  float
    zone_low:   float
    zone_mid:   float
    candle_idx: int
    mitigated:  bool   # True = prix déjà repassé par le 50% du corps


@dataclass
class FVGResult:
    valid:          bool
    direction:      str    # "bullish" | "bearish"
    gap_high:       float
    gap_low:        float
    gap_mid:        float
    gap_size:       float  # Taille du gap en points de prix
    candle_idx:     int    # Index de la bougie d'impulsion centrale
    in_current_zone: bool  # Prix actuel dans le FVG ?


@dataclass
class LiquidityResult:
    bsl_levels:        list[float]  # Niveaux BSL (au-dessus des swing highs)
    ssl_levels:        list[float]  # Niveaux SSL (sous les swing lows)
    recent_bsl_swept:  bool         # Sweep BSL récent (signal short)
    recent_ssl_swept:  bool         # Sweep SSL récent (signal long)
    sweep_level:       float | None # Niveau qui vient d'être sweepé


@dataclass
class SMCResult:
    bos:           BOSResult | None      = None
    choch:         CHoCHResult | None    = None
    order_block:   OrderBlock | None     = None
    fvg:           FVGResult | None      = None
    liquidity:     LiquidityResult | None = None
    current_trend: str                   = "neutral"
    swing_highs:   list[SwingPoint]      = field(default_factory=list)
    swing_lows:    list[SwingPoint]      = field(default_factory=list)


# ── DÉTECTEUR PRINCIPAL ───────────────────────────────────────────

class SMCDetector:
    """
    Analyse SMC/ICT sur un DataFrame OHLCV pandas.
    Toutes les détections sont des calculs déterministes purs.

    Args:
        swing_lookback: Nombre de bougies de chaque côté pour valider un swing.
                       5 pour H4/H1, 3 pour M15/M5.
    """

    def __init__(self, swing_lookback: int = 5):
        self.lookback = swing_lookback

    # ── SWING POINTS ──────────────────────────────────────────────

    def detect_swing_highs(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Swing High = bougie dont le HIGH est ≥ aux N bougies
        de gauche ET N bougies de droite.
        """
        n = self.lookback
        points = []
        for i in range(n, len(df) - n):
            h = df["high"].iloc[i]
            left  = df["high"].iloc[i - n : i]
            right = df["high"].iloc[i + 1 : i + n + 1]
            if h >= left.max() and h >= right.max():
                points.append(SwingPoint(
                    index     = i,
                    timestamp = df.index[i],
                    price     = float(h),
                    kind      = "high",
                ))
        return points

    def detect_swing_lows(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Swing Low = bougie dont le LOW est ≤ aux N bougies
        de gauche ET N bougies de droite.
        """
        n = self.lookback
        points = []
        for i in range(n, len(df) - n):
            l = df["low"].iloc[i]
            left  = df["low"].iloc[i - n : i]
            right = df["low"].iloc[i + 1 : i + n + 1]
            if l <= left.min() and l <= right.min():
                points.append(SwingPoint(
                    index     = i,
                    timestamp = df.index[i],
                    price     = float(l),
                    kind      = "low",
                ))
        return points

    # ── STRUCTURE DE MARCHÉ ───────────────────────────────────────

    def identify_trend(
        self,
        swing_highs: list[SwingPoint],
        swing_lows:  list[SwingPoint],
    ) -> str:
        """
        Identifie la tendance via l'analyse des swings.

        Haussier : HH + HL (Higher High + Higher Low)
        Baissier  : LH + LL (Lower High + Lower Low)
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"

        # 3 derniers de chaque type, triés par index
        highs = sorted(swing_highs, key=lambda x: x.index)[-3:]
        lows  = sorted(swing_lows,  key=lambda x: x.index)[-3:]

        hh = highs[-1].price > highs[-2].price  # Higher High
        hl = lows[-1].price  > lows[-2].price   # Higher Low
        lh = highs[-1].price < highs[-2].price  # Lower High
        ll = lows[-1].price  < lows[-2].price   # Lower Low

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "neutral"

    # ── BOS — BREAK OF STRUCTURE ──────────────────────────────────

    def detect_bos(
        self,
        df:          pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows:  list[SwingPoint],
        direction:   str,
    ) -> BOSResult:
        """
        BOS Haussier : clôture au-dessus du dernier swing high.
        BOS Baissier : clôture en-dessous du dernier swing low.
        On exclut les 2 dernières bougies (non confirmées).
        """
        current_close = float(df["close"].iloc[-1])
        confirmed_limit = len(df) - 2

        empty = BOSResult(
            confirmed=False, direction=direction,
            broken_level=0.0, break_idx=-1, strength=0.0
        )

        if direction == "bullish":
            valid = [sh for sh in swing_highs if sh.index < confirmed_limit]
            if not valid:
                return empty
            last_sh = max(valid, key=lambda x: x.index)
            if current_close > last_sh.price:
                return BOSResult(
                    confirmed=True, direction="bullish",
                    broken_level=last_sh.price,
                    break_idx=len(df) - 1,
                    strength=current_close - last_sh.price,
                )

        elif direction == "bearish":
            valid = [sl for sl in swing_lows if sl.index < confirmed_limit]
            if not valid:
                return empty
            last_sl = max(valid, key=lambda x: x.index)
            if current_close < last_sl.price:
                return BOSResult(
                    confirmed=True, direction="bearish",
                    broken_level=last_sl.price,
                    break_idx=len(df) - 1,
                    strength=last_sl.price - current_close,
                )

        return empty

    # ── CHoCH — CHANGE OF CHARACTER ───────────────────────────────

    def detect_choch(
        self,
        df:            pd.DataFrame,
        swing_highs:   list[SwingPoint],
        swing_lows:    list[SwingPoint],
        current_trend: str,
    ) -> CHoCHResult | None:
        """
        CHoCH = retournement de structure.

        En tendance haussière : clôture sous le dernier HL → CHoCH baissier.
        En tendance baissière : clôture au-dessus du dernier LH → CHoCH haussier.
        """
        current_close   = float(df["close"].iloc[-1])
        confirmed_limit = len(df) - 2

        if current_trend == "bullish":
            # Le CHoCH baissier casse le dernier swing low (Higher Low)
            valid = [sl for sl in swing_lows if sl.index < confirmed_limit]
            if not valid:
                return None
            last_sl = max(valid, key=lambda x: x.index)
            if current_close < last_sl.price:
                return CHoCHResult(
                    confirmed=True, direction="bearish",
                    broken_level=last_sl.price,
                    break_idx=len(df) - 1,
                )

        elif current_trend == "bearish":
            # Le CHoCH haussier casse le dernier swing high (Lower High)
            valid = [sh for sh in swing_highs if sh.index < confirmed_limit]
            if not valid:
                return None
            last_sh = max(valid, key=lambda x: x.index)
            if current_close > last_sh.price:
                return CHoCHResult(
                    confirmed=True, direction="bullish",
                    broken_level=last_sh.price,
                    break_idx=len(df) - 1,
                )

        return None

    # ── ORDER BLOCK ───────────────────────────────────────────────

    def detect_order_block(
        self,
        df:        pd.DataFrame,
        direction: str,
        lookback:  int = 60,
    ) -> OrderBlock | None:
        """
        OB Haussier : dernière bougie baissière avant une impulsion
                      haussière forte (≥ 1.5× la taille moyenne).
        OB Baissier : dernière bougie haussière avant une impulsion
                      baissière forte.

        L'OB est invalide s'il est déjà mitigé (prix repassé
        au-delà de 50% du corps depuis la formation).
        """
        current_price = float(df["close"].iloc[-1])
        search = df.iloc[-lookback:].copy()
        avg_body = (search["high"] - search["low"]).mean()

        ob_idx   = None
        ob_candle = None

        if direction == "bullish":
            # Cherche : bougie baissière suivie d'une impulsion haussière forte
            for i in range(len(search) - 3, 1, -1):
                c   = search.iloc[i]
                nxt = search.iloc[i + 1]
                if (c["close"] < c["open"] and          # Bougie baissière
                    nxt["close"] > nxt["open"] and      # Suite haussière
                    (nxt["close"] - nxt["open"]) > avg_body * 1.5):  # Fort
                    ob_idx    = i
                    ob_candle = c
                    break

            if ob_candle is None:
                return None

            zone_high = float(max(ob_candle["open"], ob_candle["close"]))
            zone_low  = float(min(ob_candle["open"], ob_candle["close"]))
            zone_mid  = (zone_high + zone_low) / 2.0

            # Mitigé si une bougie APRÈS l'OB a son low ≤ zone_mid
            post_ob   = search.iloc[ob_idx + 1:]
            mitigated = bool((post_ob["low"] <= zone_mid).any())

            # Pertinent si le prix actuel est proche (≤ 1.5 × ATR)
            atr_approx = avg_body
            near       = abs(current_price - zone_mid) <= atr_approx * 1.5

            return OrderBlock(
                valid      = (not mitigated) and near,
                direction  = "bullish",
                zone_high  = zone_high,
                zone_low   = zone_low,
                zone_mid   = zone_mid,
                candle_idx = ob_idx,
                mitigated  = mitigated,
            )

        elif direction == "bearish":
            for i in range(len(search) - 3, 1, -1):
                c   = search.iloc[i]
                nxt = search.iloc[i + 1]
                if (c["close"] > c["open"] and
                    nxt["close"] < nxt["open"] and
                    (nxt["open"] - nxt["close"]) > avg_body * 1.5):
                    ob_idx    = i
                    ob_candle = c
                    break

            if ob_candle is None:
                return None

            zone_high = float(max(ob_candle["open"], ob_candle["close"]))
            zone_low  = float(min(ob_candle["open"], ob_candle["close"]))
            zone_mid  = (zone_high + zone_low) / 2.0

            post_ob   = search.iloc[ob_idx + 1:]
            mitigated = bool((post_ob["high"] >= zone_mid).any())

            atr_approx = avg_body
            near       = abs(current_price - zone_mid) <= atr_approx * 1.5

            return OrderBlock(
                valid      = (not mitigated) and near,
                direction  = "bearish",
                zone_high  = zone_high,
                zone_low   = zone_low,
                zone_mid   = zone_mid,
                candle_idx = ob_idx,
                mitigated  = mitigated,
            )

        return None

    # ── FVG — FAIR VALUE GAP ──────────────────────────────────────

    def detect_fvg(
        self,
        df:        pd.DataFrame,
        direction: str,
        lookback:  int = 40,
    ) -> FVGResult | None:
        """
        FVG Haussier : candle[i-1].high < candle[i+1].low
                       (gap non comblé entre 3 bougies consécutives)
        FVG Baissier : candle[i-1].low > candle[i+1].high

        Retourne le FVG le plus récent ET le plus proche du prix actuel.
        """
        current_price = float(df["close"].iloc[-1])
        search        = df.iloc[-lookback:]

        if len(search) < 3:
            return None

        best_fvg     = None
        best_distance = float("inf")

        for i in range(1, len(search) - 1):
            prev = search.iloc[i - 1]
            nxt  = search.iloc[i + 1]

            if direction == "bullish":
                # Gap haussier : high de la précédente < low de la suivante
                if prev["high"] < nxt["low"]:
                    gap_low  = float(prev["high"])
                    gap_high = float(nxt["low"])
                    gap_mid  = (gap_low + gap_high) / 2.0
                    gap_size = gap_high - gap_low

                    in_zone  = gap_low <= current_price <= gap_high
                    distance = abs(current_price - gap_mid)
                    near     = distance <= gap_size * 3

                    if near and distance < best_distance:
                        best_distance = distance
                        best_fvg = FVGResult(
                            valid           = True,
                            direction       = "bullish",
                            gap_high        = gap_high,
                            gap_low         = gap_low,
                            gap_mid         = gap_mid,
                            gap_size        = gap_size,
                            candle_idx      = i,
                            in_current_zone = in_zone,
                        )

            elif direction == "bearish":
                # Gap baissier : low de la précédente > high de la suivante
                if prev["low"] > nxt["high"]:
                    gap_low  = float(nxt["high"])
                    gap_high = float(prev["low"])
                    gap_mid  = (gap_low + gap_high) / 2.0
                    gap_size = gap_high - gap_low

                    in_zone  = gap_low <= current_price <= gap_high
                    distance = abs(current_price - gap_mid)
                    near     = distance <= gap_size * 3

                    if near and distance < best_distance:
                        best_distance = distance
                        best_fvg = FVGResult(
                            valid           = True,
                            direction       = "bearish",
                            gap_high        = gap_high,
                            gap_low         = gap_low,
                            gap_mid         = gap_mid,
                            gap_size        = gap_size,
                            candle_idx      = i,
                            in_current_zone = in_zone,
                        )

        return best_fvg

    # ── LIQUIDITÉS BSL / SSL ──────────────────────────────────────

    def detect_liquidity(
        self,
        df:          pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows:  list[SwingPoint],
        lookback:    int = 30,
    ) -> LiquidityResult:
        """
        BSL = Buyside Liquidity  → au-dessus des swing highs (stops acheteurs)
        SSL = Sellside Liquidity → sous les swing lows (stops vendeurs)

        Sweep Haussier (SSL) :
            low[i] < ssl_level ET close[i] > ssl_level
            → mèche basse, clôture au-dessus → signal long

        Sweep Baissier (BSL) :
            high[i] > bsl_level ET close[i] < bsl_level
            → mèche haute, clôture en-dessous → signal short
        """
        recent_limit  = len(df) - lookback
        recent_highs  = [sh for sh in swing_highs if sh.index >= recent_limit]
        recent_lows   = [sl for sl in swing_lows  if sl.index >= recent_limit]

        bsl_levels = [sh.price for sh in recent_highs]
        ssl_levels = [sl.price for sl in recent_lows]

        # Vérifier les 5 dernières bougies pour un sweep récent
        last_candles = df.iloc[-5:]
        recent_bsl_swept = False
        recent_ssl_swept = False
        sweep_level: float | None = None

        for _, candle in last_candles.iterrows():
            # SSL Sweep (signal LONG)
            for ssl in ssl_levels:
                if candle["low"] < ssl and candle["close"] > ssl:
                    recent_ssl_swept = True
                    sweep_level = ssl
                    break

            # BSL Sweep (signal SHORT)
            for bsl in bsl_levels:
                if candle["high"] > bsl and candle["close"] < bsl:
                    recent_bsl_swept = True
                    sweep_level = bsl
                    break

        return LiquidityResult(
            bsl_levels       = bsl_levels,
            ssl_levels       = ssl_levels,
            recent_bsl_swept = recent_bsl_swept,
            recent_ssl_swept = recent_ssl_swept,
            sweep_level      = sweep_level,
        )

    # ── ANALYSE COMPLÈTE ──────────────────────────────────────────

    def analyze(
        self,
        df:        pd.DataFrame,
        direction: str = "auto",
    ) -> SMCResult:
        """
        Lance l'analyse SMC complète sur le DataFrame.

        Args:
            df:        DataFrame OHLCV (colonnes: open, high, low, close)
            direction: "bullish" | "bearish" | "auto" (détection automatique)

        Returns:
            SMCResult avec tous les éléments détectés.
        """
        if len(df) < 50:
            return SMCResult(current_trend="neutral")

        # ── Swings ────────────────────────────────────────────────
        swing_highs = self.detect_swing_highs(df)
        swing_lows  = self.detect_swing_lows(df)

        # ── Tendance ──────────────────────────────────────────────
        trend = self.identify_trend(swing_highs, swing_lows)

        if direction == "auto":
            direction = trend

        if direction == "neutral":
            return SMCResult(
                current_trend = "neutral",
                swing_highs   = swing_highs,
                swing_lows    = swing_lows,
            )

        # ── Détections ────────────────────────────────────────────
        bos       = self.detect_bos(df, swing_highs, swing_lows, direction)
        choch     = self.detect_choch(df, swing_highs, swing_lows, trend)
        ob        = self.detect_order_block(df, direction)
        fvg       = self.detect_fvg(df, direction)
        liquidity = self.detect_liquidity(df, swing_highs, swing_lows)

        return SMCResult(
            bos           = bos,
            choch         = choch,
            order_block   = ob,
            fvg           = fvg,
            liquidity     = liquidity,
            current_trend = trend,
            swing_highs   = swing_highs,
            swing_lows    = swing_lows,
        )
