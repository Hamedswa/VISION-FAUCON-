# core/scoring.py
# ─────────────────────────────────────────────────────────────────
#  Système de scoring de confluence — /100 points
#
#  Barème :
#   BOS confirmé              → 20 pts
#   CHoCH présent             → 15 pts
#   OB valide (non mitigé)    → 20 pts
#   FVG dans la zone          → 10 pts
#   Liquidité sweepée         → 10 pts
#   Alignement Multi-TF       → 15 pts  (3TF=15, 2TF=10, 1TF=5)
#   RSI confirme direction    →  5 pts
#   MACD confirme direction   →  5 pts
#   ─────────────────────────────────────────────────────────────
#   TOTAL MAX                 → 100 pts
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from .smc_detector import SMCResult
from .indicators import IndicatorResult


@dataclass
class ScoreResult:
    total:     int    # Score sur 100
    direction: str    # "bullish" | "bearish"
    is_valid:  bool   # True si >= seuil minimum

    # Détail par critère
    score_bos:        int
    score_choch:      int
    score_ob:         int
    score_fvg:        int
    score_liquidity:  int
    score_mtf:        int
    score_rsi:        int
    score_macd:       int

    mtf_aligned: int  # Nombre de TF alignés (0-3)
    summary:     str  # Résumé lisible par humain


class ScoringEngine:
    """
    Calcule le score de confluence d'un setup.
    Prend les résultats SMC de 3 timeframes (H4, H1, M15)
    et les indicateurs techniques pour produire un score /100.
    """

    def calculate(
        self,
        smc_h4:    SMCResult,
        smc_h1:    SMCResult,
        smc_m15:   SMCResult,
        indicators: IndicatorResult,
        direction:  str,
        min_score:  int = 75,
    ) -> ScoreResult:
        """
        Args:
            smc_h4:     Analyse SMC sur H4 (direction principale)
            smc_h1:     Analyse SMC sur H1 (confirmation)
            smc_m15:    Analyse SMC sur M15 (setup d'entrée)
            indicators: Indicateurs techniques (calculés sur H1 ou M15)
            direction:  "bullish" | "bearish"
            min_score:  Seuil de validation (défaut: 75)

        Returns:
            ScoreResult avec le total et le détail.
        """
        # ── 1. BOS (20 pts) ───────────────────────────────────────
        # Le BOS doit être confirmé sur M15 dans la bonne direction
        score_bos = 0
        if (smc_m15.bos
                and smc_m15.bos.confirmed
                and smc_m15.bos.direction == direction):
            score_bos = 20

        # ── 2. CHoCH (15 pts) ─────────────────────────────────────
        # CHoCH sur H1 dans la direction attendue (signal de retournement)
        score_choch = 0
        if (smc_h1.choch
                and smc_h1.choch.confirmed
                and smc_h1.choch.direction == direction):
            score_choch = 15

        # ── 3. Order Block (20 pts) ───────────────────────────────
        # OB valide et non mitigé sur M15
        score_ob = 0
        if (smc_m15.order_block
                and smc_m15.order_block.valid
                and not smc_m15.order_block.mitigated
                and smc_m15.order_block.direction == direction):
            score_ob = 20

        # ── 4. FVG (10 pts) ───────────────────────────────────────
        # FVG dans la zone actuelle sur M15
        score_fvg = 0
        if (smc_m15.fvg
                and smc_m15.fvg.valid
                and smc_m15.fvg.direction == direction):
            if smc_m15.fvg.in_current_zone:
                score_fvg = 10
            else:
                score_fvg = 5   # FVG proche mais pas encore atteint

        # ── 5. Liquidité (10 pts) ─────────────────────────────────
        # Sweep de liquidité dans le sens du trade sur H1
        score_liq = 0
        if smc_h1.liquidity:
            if direction == "bullish" and smc_h1.liquidity.recent_ssl_swept:
                score_liq = 10   # SSL sweepé = signal long fort
            elif direction == "bearish" and smc_h1.liquidity.recent_bsl_swept:
                score_liq = 10   # BSL sweepé = signal short fort

        # ── 6. Alignement Multi-TF (15 pts) ──────────────────────
        # H4 = direction principale (poids le plus fort)
        # H1 + M15 = confirmation
        aligned = 0
        if smc_h4.current_trend  == direction: aligned += 1
        if smc_h1.current_trend  == direction: aligned += 1
        if smc_m15.current_trend == direction: aligned += 1

        score_mtf = {3: 15, 2: 10, 1: 5, 0: 0}[aligned]

        # ── 7. RSI (5 pts) ────────────────────────────────────────
        score_rsi = 0
        if direction == "bullish":
            if indicators.rsi_zone == "oversold":
                score_rsi = 5   # Survendu → rebond attendu
            elif indicators.rsi < 50 and indicators.rsi > 30:
                score_rsi = 3   # Zone favorable sans être extrême
        elif direction == "bearish":
            if indicators.rsi_zone == "overbought":
                score_rsi = 5   # Suracheté → retournement attendu
            elif indicators.rsi > 50 and indicators.rsi < 70:
                score_rsi = 3

        # ── 8. MACD (5 pts) ───────────────────────────────────────
        score_macd = 0
        if direction == "bullish" and indicators.macd_bullish:
            score_macd = 5
        elif direction == "bearish" and not indicators.macd_bullish:
            score_macd = 5

        # ── Total ─────────────────────────────────────────────────
        total = min(
            score_bos + score_choch + score_ob + score_fvg
            + score_liq + score_mtf + score_rsi + score_macd,
            100,
        )

        # ── Résumé lisible ────────────────────────────────────────
        checks = []
        if score_bos:    checks.append(f"BOS ✅(+{score_bos})")
        if score_choch:  checks.append(f"CHoCH ✅(+{score_choch})")
        if score_ob:     checks.append(f"OB ✅(+{score_ob})")
        if score_fvg:    checks.append(f"FVG ✅(+{score_fvg})")
        if score_liq:    checks.append(f"Liquidité ✅(+{score_liq})")
        checks.append(f"MTF({aligned}/3)(+{score_mtf})")
        if score_rsi:    checks.append(f"RSI ✅(+{score_rsi})")
        if score_macd:   checks.append(f"MACD ✅(+{score_macd})")

        missing = []
        if not score_bos:   missing.append("BOS ❌")
        if not score_choch: missing.append("CHoCH ❌")
        if not score_ob:    missing.append("OB ❌")
        if not score_fvg:   missing.append("FVG ❌")
        if not score_liq:   missing.append("Liquidité ❌")

        summary = (
            f"Score {total}/100 | {direction.upper()} | "
            f"{'VALIDE ✅' if total >= min_score else 'REJETÉ ❌'}\n"
            f"Présents : {', '.join(checks)}\n"
            f"Absents  : {', '.join(missing) if missing else 'Aucun'}"
        )

        return ScoreResult(
            total            = total,
            direction        = direction,
            is_valid         = total >= min_score,
            score_bos        = score_bos,
            score_choch      = score_choch,
            score_ob         = score_ob,
            score_fvg        = score_fvg,
            score_liquidity  = score_liq,
            score_mtf        = score_mtf,
            score_rsi        = score_rsi,
            score_macd       = score_macd,
            mtf_aligned      = aligned,
            summary          = summary,
        )
