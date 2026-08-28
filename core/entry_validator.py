# core/entry_validator.py
# ─────────────────────────────────────────────────────────────────
#  Validation finale de l'entrée
#
#  Vérifie 4 conditions OBLIGATOIRES :
#   1. R:R minimum respecté (≥ 1.5 vers TP2)
#   2. Prix dans ou proche de l'OB (si disponible)
#   3. FVG confirme la zone (si disponible)
#   4. Bougie de confirmation sur M5 (engulf ou pin bar)
# ─────────────────────────────────────────────────────────────────

import pandas as pd
from dataclasses import dataclass
from .smc_detector import OrderBlock, FVGResult


@dataclass
class ValidationResult:
    is_valid:         bool
    rejection_reason: str | None    # None si valide
    rr_ratio:         float         # R:R calculé (vers TP2)
    quality:          str           # "excellent" | "good" | "acceptable" | "rejected"
    checks:           dict          # Détail de chaque vérification


class EntryValidator:
    """
    Valide qu'un setup remplit toutes les conditions d'entrée.
    Chaque vérification est un calcul mathématique pur.
    """

    def validate(
        self,
        direction:   str,
        entry:       float,
        sl_price:    float,
        tp2:         float,
        df_m5:       pd.DataFrame | None,
        order_block: OrderBlock | None = None,
        fvg:         FVGResult | None  = None,
        min_rr:      float = 1.5,
    ) -> ValidationResult:
        """
        Args:
            direction:   "bullish" | "bearish"
            entry:       Prix d'entrée prévu
            sl_price:    Stop loss calculé
            tp2:         TP2 calculé (R:R cible)
            df_m5:       DataFrame M5 pour la confirmation de bougie
            order_block: OB détecté (optionnel)
            fvg:         FVG détecté (optionnel)
            min_rr:      R:R minimum (défaut 1.5)
        """
        checks = {
            "rr_ok":      False,
            "in_ob":      None,    # None = non vérifié (pas d'OB)
            "fvg_ok":     None,    # None = non vérifié
            "m5_confirm": False,
        }

        risk = abs(entry - sl_price)
        if risk == 0:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Risk nul — SL égal à l'entrée",
                rr_ratio=0.0,
                quality="rejected",
                checks=checks,
            )

        rr = abs(tp2 - entry) / risk

        # ── Vérification 1 : R:R minimum ──────────────────────────
        checks["rr_ok"] = rr >= min_rr
        if not checks["rr_ok"]:
            return ValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"R:R insuffisant : {rr:.2f} < {min_rr} minimum"
                ),
                rr_ratio=round(rr, 2),
                quality="rejected",
                checks=checks,
            )

        # ── Vérification 2 : Prix dans l'OB ──────────────────────
        if order_block and order_block.valid:
            if direction == "bullish":
                # Entrée doit être dans la zone OB ou juste au-dessus
                tolerance = (order_block.zone_high - order_block.zone_low) * 0.3
                checks["in_ob"] = (
                    order_block.zone_low - tolerance <= entry
                    <= order_block.zone_high + tolerance
                )
            else:
                tolerance = (order_block.zone_high - order_block.zone_low) * 0.3
                checks["in_ob"] = (
                    order_block.zone_low - tolerance <= entry
                    <= order_block.zone_high + tolerance
                )

        # ── Vérification 3 : FVG confirme la zone ─────────────────
        if fvg and fvg.valid:
            checks["fvg_ok"] = fvg.in_current_zone

        # ── Vérification 4 : Confirmation M5 ─────────────────────
        checks["m5_confirm"] = self._check_m5_confirmation(df_m5, direction)

        if not checks["m5_confirm"]:
            return ValidationResult(
                is_valid=False,
                rejection_reason=(
                    "Pas de bougie de confirmation M5 "
                    "(engulfing ou pin bar requis)"
                ),
                rr_ratio=round(rr, 2),
                quality="rejected",
                checks=checks,
            )

        # ── Qualité du setup ──────────────────────────────────────
        quality = self._assess_quality(rr, checks)

        return ValidationResult(
            is_valid=True,
            rejection_reason=None,
            rr_ratio=round(rr, 2),
            quality=quality,
            checks=checks,
        )

    # ── Confirmation M5 ───────────────────────────────────────────

    @staticmethod
    def _check_m5_confirmation(
        df_m5: pd.DataFrame | None,
        direction: str,
    ) -> bool:
        """
        Vérifie la présence d'une bougie de confirmation sur M5.

        Patterns reconnus :
          • Engulfing : la bougie englobe entièrement la précédente
          • Pin Bar   : la mèche = 2× le corps (rejet de zone)
          • Marubozu  : bougie directionnelle solide (corps > 70% range)
        """
        if df_m5 is None or len(df_m5) < 2:
            return True   # Pas de données M5 → on passe (non bloquant)

        curr = df_m5.iloc[-1]
        prev = df_m5.iloc[-2]

        c_open  = float(curr["open"])
        c_close = float(curr["close"])
        c_high  = float(curr["high"])
        c_low   = float(curr["low"])

        p_open  = float(prev["open"])
        p_close = float(prev["close"])

        body       = abs(c_close - c_open)
        full_range = c_high - c_low
        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)

        if direction == "bullish":
            # Engulfing haussier
            engulf = (
                c_close > c_open and                    # Bougie haussière
                c_close > max(p_open, p_close) and      # Close au-dessus du max précédent
                c_open  <= min(p_open, p_close)         # Open en-dessous du min précédent
            )
            # Pin bar haussier (longue mèche basse)
            pin_bar = (
                c_close > c_open and
                lower_wick >= body * 2 and
                full_range > 0
            )
            # Marubozu haussier
            marubozu = (
                c_close > c_open and
                body >= full_range * 0.7
            )
            return engulf or pin_bar or marubozu

        else:  # bearish
            # Engulfing baissier
            engulf = (
                c_close < c_open and
                c_close < min(p_open, p_close) and
                c_open  >= max(p_open, p_close)
            )
            # Pin bar baissier (longue mèche haute)
            pin_bar = (
                c_close < c_open and
                upper_wick >= body * 2 and
                full_range > 0
            )
            # Marubozu baissier
            marubozu = (
                c_close < c_open and
                body >= full_range * 0.7
            )
            return engulf or pin_bar or marubozu

    # ── Qualité ───────────────────────────────────────────────────

    @staticmethod
    def _assess_quality(rr: float, checks: dict) -> str:
        """Évalue la qualité du setup en fonction du R:R et des checks."""
        positive_checks = sum(
            1 for v in checks.values()
            if v is True
        )

        if rr >= 3.0 and positive_checks >= 3:
            return "excellent"
        if rr >= 2.0 and positive_checks >= 2:
            return "good"
        return "acceptable"
