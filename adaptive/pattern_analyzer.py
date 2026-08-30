# adaptive/pattern_analyzer.py
# ─────────────────────────────────────────────────────────────────
#  Analyseur de patterns — Intelligence adaptative
#
#  Identifie quels patterns SMC performent le mieux
#  pour ajuster les poids du scoring en conséquence.
#
#  Questions auxquelles il répond :
#   • Quel score minimum donne le meilleur WR ?
#   • Quel instrument performe le mieux ?
#   • Quel élément SMC (BOS, OB, FVG...) corrèle le plus avec les wins ?
#   • Quelle session donne le meilleur WR ?
#   • Le R:R minimum optimal ?
# ─────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from database.db_manager import DatabaseManager
from database.models import SignalStatus


@dataclass
class PatternInsight:
    """Résultat d'une analyse de pattern."""
    category:    str       # "session" | "score" | "instrument" | "element"
    name:        str       # Ex: "london", "85-90", "XAUUSD", "ob"
    trades:      int
    win_rate:    float
    profit_factor: float
    avg_pnl_usd: float
    recommendation: str    # Conseil actionnable


@dataclass
class PatternAnalysisResult:
    best_session:       PatternInsight | None
    best_instrument:    PatternInsight | None
    optimal_min_score:  int              # Score minimum recommandé
    optimal_min_rr:     float            # R:R minimum recommandé
    top_smc_elements:   list[str]        # Éléments SMC les plus corrélés
    insights:           list[PatternInsight] = field(default_factory=list)
    generated_at:       datetime = field(default_factory=datetime.utcnow)
    summary:            str = ""


class PatternAnalyzer:
    """
    Analyse les patterns de performance pour guider l'optimisation.
    Fonctionne sur les données historiques en base.
    """

    WIN_STATUSES = {
        SignalStatus.TP1_HIT,
        SignalStatus.TP2_HIT,
        SignalStatus.TP3_HIT,
    }

    def __init__(self, db: DatabaseManager):
        self._db = db

    # ── ANALYSE PRINCIPALE ────────────────────────────────────────

    async def analyze(
        self,
        months: int = 3,
    ) -> PatternAnalysisResult:
        """
        Analyse complète des patterns sur X mois.
        Minimum 20 trades requis pour des insights fiables.
        """
        signals = await self._db.get_signals_for_backtest(months=months)

        finished = [
            s for s in signals
            if s.status in (
                *self.WIN_STATUSES,
                SignalStatus.SL_HIT,
            )
        ]

        if len(finished) < 20:
            logger.warning(
                f"PatternAnalyzer — seulement {len(finished)} trades, "
                "minimum 20 requis pour des insights fiables."
            )
            return PatternAnalysisResult(
                best_session=None,
                best_instrument=None,
                optimal_min_score=75,
                optimal_min_rr=1.5,
                top_smc_elements=[],
                summary="Données insuffisantes — minimum 20 trades requis.",
            )

        insights = []

        # ── 1. Par session ────────────────────────────────────────
        session_insights = self._analyze_by_session(finished)
        insights.extend(session_insights)
        best_session = max(
            session_insights, key=lambda x: x.win_rate, default=None
        )

        # ── 2. Par instrument ─────────────────────────────────────
        instr_insights = self._analyze_by_instrument(finished)
        insights.extend(instr_insights)
        best_instrument = max(
            instr_insights, key=lambda x: x.win_rate, default=None
        )

        # ── 3. Score optimal ──────────────────────────────────────
        optimal_score = self._find_optimal_min_score(finished)

        # ── 4. R:R optimal ────────────────────────────────────────
        optimal_rr = self._find_optimal_rr(finished)

        # ── 5. Éléments SMC corrélés ──────────────────────────────
        top_elements = self._analyze_smc_elements(finished)

        # ── Résumé ────────────────────────────────────────────────
        summary = self._build_summary(
            finished, best_session, best_instrument,
            optimal_score, optimal_rr, top_elements,
        )

        logger.info(
            f"🧩 PatternAnalyzer — {len(finished)} trades analysés | "
            f"Score optimal: {optimal_score} | "
            f"R:R optimal: {optimal_rr} | "
            f"Meilleure session: {best_session.name if best_session else 'N/A'}"
        )

        return PatternAnalysisResult(
            best_session      = best_session,
            best_instrument   = best_instrument,
            optimal_min_score = optimal_score,
            optimal_min_rr    = optimal_rr,
            top_smc_elements  = top_elements,
            insights          = insights,
            summary           = summary,
        )

    # ── PAR SESSION ───────────────────────────────────────────────

    def _analyze_by_session(
        self, signals: list
    ) -> list[PatternInsight]:
        """Calcule le WR et PF par session de trading."""
        groups: dict[str, list] = {}

        for sig in signals:
            sess = sig.session or "unknown"
            groups.setdefault(sess, []).append(sig)

        insights = []
        for sess, sigs in groups.items():
            if len(sigs) < 3:
                continue

            wins   = [s for s in sigs if s.status in self.WIN_STATUSES]
            losses = [s for s in sigs if s.status == SignalStatus.SL_HIT]
            wr     = len(wins) / len(sigs)

            gross_profit = sum(s.pnl_usd or 0 for s in wins if (s.pnl_usd or 0) > 0)
            gross_loss   = abs(sum(s.pnl_usd or 0 for s in losses if (s.pnl_usd or 0) < 0))
            pf           = gross_profit / gross_loss if gross_loss > 0 else 999.0

            avg_pnl = sum(s.pnl_usd or 0 for s in sigs) / len(sigs)

            rec = (
                f"Session '{sess}' : WR {wr:.1%} sur {len(sigs)} trades — "
                + ("✅ Prioriser" if wr >= 0.60 else "⚠️ Prudence recommandée")
            )

            insights.append(PatternInsight(
                category       = "session",
                name           = sess,
                trades         = len(sigs),
                win_rate       = round(wr, 4),
                profit_factor  = round(pf, 2),
                avg_pnl_usd    = round(avg_pnl, 2),
                recommendation = rec,
            ))

        return sorted(insights, key=lambda x: x.win_rate, reverse=True)

    # ── PAR INSTRUMENT ────────────────────────────────────────────

    def _analyze_by_instrument(
        self, signals: list
    ) -> list[PatternInsight]:
        """Calcule le WR et PF par instrument."""
        groups: dict[str, list] = {}
        for sig in signals:
            groups.setdefault(sig.instrument, []).append(sig)

        insights = []
        for instr, sigs in groups.items():
            if len(sigs) < 3:
                continue

            wins   = [s for s in sigs if s.status in self.WIN_STATUSES]
            losses = [s for s in sigs if s.status == SignalStatus.SL_HIT]
            wr     = len(wins) / len(sigs)

            gross_profit = sum(s.pnl_usd or 0 for s in wins if (s.pnl_usd or 0) > 0)
            gross_loss   = abs(sum(s.pnl_usd or 0 for s in losses if (s.pnl_usd or 0) < 0))
            pf           = gross_profit / gross_loss if gross_loss > 0 else 999.0
            avg_pnl      = sum(s.pnl_usd or 0 for s in sigs) / len(sigs)

            rec = (
                f"{instr} : WR {wr:.1%} — "
                + ("✅ Excellent — à maintenir" if wr >= 0.65 else
                   "✅ Bon — continuer" if wr >= 0.60 else
                   "⚠️ Sous-performant — revoir paramètres")
            )

            insights.append(PatternInsight(
                category       = "instrument",
                name           = instr,
                trades         = len(sigs),
                win_rate       = round(wr, 4),
                profit_factor  = round(pf, 2),
                avg_pnl_usd    = round(avg_pnl, 2),
                recommendation = rec,
            ))

        return sorted(insights, key=lambda x: x.win_rate, reverse=True)

    # ── SCORE OPTIMAL ─────────────────────────────────────────────

    def _find_optimal_min_score(self, signals: list) -> int:
        """
        Teste différents seuils de score minimum (75 à 95)
        et retourne celui qui maximise le WR tout en
        maintenant un minimum de 10 trades.
        """
        best_score = 75
        best_wr    = 0.0

        for threshold in range(75, 96, 5):
            filtered = [
                s for s in signals
                if (s.confluence_score or 0) >= threshold
            ]

            if len(filtered) < 10:
                break   # Trop peu de trades pour être fiable

            wins = [s for s in filtered if s.status in self.WIN_STATUSES]
            wr   = len(wins) / len(filtered)

            # On cherche le meilleur WR avec suffisamment de trades
            if wr > best_wr:
                best_wr    = wr
                best_score = threshold

        logger.debug(
            f"Score optimal trouvé: {best_score} "
            f"(WR: {best_wr:.1%})"
        )
        return best_score

    # ── R:R OPTIMAL ───────────────────────────────────────────────

    def _find_optimal_rr(self, signals: list) -> float:
        """
        Teste différents R:R minimums (1.0 à 3.0)
        et retourne celui qui maximise le profit factor.
        """
        best_rr = 1.5
        best_pf = 0.0

        for rr_min in [1.0, 1.5, 2.0, 2.5, 3.0]:
            filtered = [
                s for s in signals
                if (s.rr_ratio or 0) >= rr_min
            ]

            if len(filtered) < 8:
                continue

            wins   = [s for s in filtered if s.status in self.WIN_STATUSES]
            losses = [s for s in filtered if s.status == SignalStatus.SL_HIT]

            gross_profit = sum(s.pnl_usd or 0 for s in wins if (s.pnl_usd or 0) > 0)
            gross_loss   = abs(sum(s.pnl_usd or 0 for s in losses if (s.pnl_usd or 0) < 0))
            pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0

            if pf > best_pf:
                best_pf = pf
                best_rr = rr_min

        return best_rr

    # ── ÉLÉMENTS SMC CORRÉLÉS ─────────────────────────────────────

    def _analyze_smc_elements(self, signals: list) -> list[str]:
        """
        Identifie quels éléments SMC sont les plus corrélés
        avec les trades gagnants.
        Utilise les scores individuels stockés en DB.
        """
        elements = {
            "bos":       {"wins": 0, "total": 0},
            "choch":     {"wins": 0, "total": 0},
            "ob":        {"wins": 0, "total": 0},
            "fvg":       {"wins": 0, "total": 0},
            "liquidity": {"wins": 0, "total": 0},
        }

        score_fields = {
            "bos":       "score_bos",
            "choch":     "score_choch",
            "ob":        "score_ob",
            "fvg":       "score_fvg",
            "liquidity": "score_liquidity",
        }

        for sig in signals:
            is_win = sig.status in self.WIN_STATUSES
            for elem, field_name in score_fields.items():
                score = getattr(sig, field_name, 0) or 0
                if score > 0:   # L'élément était présent
                    elements[elem]["total"] += 1
                    if is_win:
                        elements[elem]["wins"] += 1

        # Calcul du WR par élément
        elem_wr = {}
        for elem, data in elements.items():
            if data["total"] >= 5:
                elem_wr[elem] = data["wins"] / data["total"]

        # Tri par WR décroissant
        top = sorted(elem_wr.items(), key=lambda x: x[1], reverse=True)
        return [elem for elem, _ in top[:3]]

    # ── RÉSUMÉ ────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        signals: list,
        best_session: PatternInsight | None,
        best_instrument: PatternInsight | None,
        optimal_score: int,
        optimal_rr: float,
        top_elements: list[str],
    ) -> str:
        wins  = sum(1 for s in signals if s.status in PatternAnalyzer.WIN_STATUSES)
        total = len(signals)
        wr    = wins / total if total > 0 else 0.0

        lines = [
            f"📊 Analyse {total} trades — WR global: {wr:.1%}",
            f"🎯 Score min optimal   : {optimal_score}/100",
            f"📐 R:R min optimal     : {optimal_rr:.1f}",
            f"🔑 Éléments SMC clés   : {', '.join(top_elements) or 'N/A'}",
            f"📍 Meilleure session   : {best_session.name if best_session else 'N/A'}"
            + (f" (WR {best_session.win_rate:.1%})" if best_session else ""),
            f"🏆 Meilleur instrument : {best_instrument.name if best_instrument else 'N/A'}"
            + (f" (WR {best_instrument.win_rate:.1%})" if best_instrument else ""),
        ]
        return "\n".join(lines)
