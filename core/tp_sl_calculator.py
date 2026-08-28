# core/tp_sl_calculator.py
# ─────────────────────────────────────────────────────────────────
#  Calcul mathématique des niveaux TP et SL
#
#  Formules :
#   SL   = entry ± (ATR × multiplier)  [ajusté par OB si dispo]
#   TP1  = entry ± risk × 1.0          [1:1 — sécurité]
#   TP2  = entry ± risk × 1.618        [Golden Ratio Fibonacci]
#   TP3  = prochain OB/swing OU entry ± risk × 2.5
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from .smc_detector import OrderBlock, SwingPoint


@dataclass
class TPSLResult:
    entry:       float
    direction:   str

    sl:          float
    tp1:         float
    tp2:         float
    tp3:         float

    risk:        float   # Distance entry → SL en points de prix
    rr_tp1:      float   # R:R vers TP1
    rr_tp2:      float   # R:R vers TP2
    rr_tp3:      float   # R:R vers TP3


class TPSLCalculator:
    """
    Calcule les niveaux TP/SL pour un setup donné.
    Combine ATR, Fibonacci et niveaux SMC pour des niveaux optimaux.
    """

    def calculate(
        self,
        entry:           float,
        direction:       str,
        atr:             float,
        order_block:     OrderBlock | None = None,
        swing_highs:     list[SwingPoint]  = None,
        swing_lows:      list[SwingPoint]  = None,
        atr_multiplier:  float = 1.5,
        tp1_ratio:       float = 1.0,
        tp2_ratio:       float = 1.618,
        tp3_ratio:       float = 2.5,
    ) -> TPSLResult:
        """
        Args:
            entry:          Prix d'entrée
            direction:      "bullish" | "bearish"
            atr:            ATR(14) sur le timeframe d'entrée
            order_block:    OB détecté (pour affiner le SL)
            swing_highs:    Swing highs (pour TP3 sur résistance)
            swing_lows:     Swing lows (pour TP3 sur support)
            atr_multiplier: Multiplicateur ATR pour le SL (défaut 1.5)
            tp1_ratio:      Ratio TP1 (défaut 1.0 = 1:1)
            tp2_ratio:      Ratio TP2 (défaut 1.618 = Fibonacci)
            tp3_ratio:      Ratio TP3 fallback (défaut 2.5)
        """
        sl_distance = atr * atr_multiplier
        buffer      = atr * 0.08   # 8% ATR de marge de sécurité

        if direction == "bullish":
            # ── SL ─────────────────────────────────────────────
            if order_block and order_block.valid:
                # SL sous le bas de l'OB avec buffer
                sl = order_block.zone_low - buffer
            else:
                sl = entry - sl_distance

            risk = entry - sl
            if risk <= 0:
                risk = sl_distance   # Sécurité

            # ── TP ─────────────────────────────────────────────
            tp1 = entry + risk * tp1_ratio
            tp2 = entry + risk * tp2_ratio

            # TP3 : prochain swing high au-dessus de TP2 (résistance)
            tp3 = entry + risk * tp3_ratio  # Fallback
            if swing_highs:
                above_tp2 = [
                    sh.price for sh in swing_highs
                    if sh.price > tp2
                ]
                if above_tp2:
                    tp3 = min(above_tp2)  # Résistance la plus proche

        else:  # bearish
            # ── SL ─────────────────────────────────────────────
            if order_block and order_block.valid:
                sl = order_block.zone_high + buffer
            else:
                sl = entry + sl_distance

            risk = sl - entry
            if risk <= 0:
                risk = sl_distance

            # ── TP ─────────────────────────────────────────────
            tp1 = entry - risk * tp1_ratio
            tp2 = entry - risk * tp2_ratio

            # TP3 : prochain swing low en-dessous de TP2 (support)
            tp3 = entry - risk * tp3_ratio  # Fallback
            if swing_lows:
                below_tp2 = [
                    sl.price for sl in swing_lows
                    if sl.price < tp2
                ]
                if below_tp2:
                    tp3 = max(below_tp2)  # Support le plus proche

        # ── R:R ────────────────────────────────────────────────
        rr_tp1 = abs(tp1 - entry) / risk if risk > 0 else 0.0
        rr_tp2 = abs(tp2 - entry) / risk if risk > 0 else 0.0
        rr_tp3 = abs(tp3 - entry) / risk if risk > 0 else 0.0

        return TPSLResult(
            entry     = round(entry, 5),
            direction = direction,
            sl        = round(sl,    5),
            tp1       = round(tp1,   5),
            tp2       = round(tp2,   5),
            tp3       = round(tp3,   5),
            risk      = round(risk,  5),
            rr_tp1    = round(rr_tp1, 2),
            rr_tp2    = round(rr_tp2, 2),
            rr_tp3    = round(rr_tp3, 2),
        )
