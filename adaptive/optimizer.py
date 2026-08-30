# adaptive/optimizer.py
# ─────────────────────────────────────────────────────────────────
#  Optimiseur adaptatif — Cerveau évolutif du bot
#
#  Utilise Optuna (optimisation bayésienne) pour trouver
#  les meilleurs paramètres à partir des données historiques.
#
#  Paramètres optimisés :
#   • min_confluence_score  (seuil de validation du signal)
#   • atr_sl_multiplier     (taille du Stop Loss)
#   • tp1_ratio, tp2_ratio  (niveaux TP)
#   • min_rr_ratio          (R:R minimum)
#   • swing_lookback        (sensibilité détection SMC)
#
#  Stratégie :
#   • S'exécute toutes les X jours (défaut: 7)
#   • Backteste chaque combinaison sur les 3 derniers mois
#   • Applique les meilleurs paramètres si WR amélioré
#   • Sauvegarde chaque run en DB pour traçabilité complète
# ─────────────────────────────────────────────────────────────────

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import settings
from database.db_manager import DatabaseManager
from database.models import SignalStatus
from .performance_tracker import PerformanceTracker
from .pattern_analyzer import PatternAnalyzer


@dataclass
class OptimizationResult:
    success:       bool
    best_params:   dict
    best_wr:       float
    best_pf:       float
    trials_done:   int
    improved:      bool       # True si meilleur que les params actuels
    improvement:   float      # % d'amélioration du WR
    summary:       str
    run_id:        int | None = None


class AdaptiveOptimizer:
    """
    Optimise les paramètres du bot via Optuna.
    Tourne en tâche de fond, n'interfère pas avec le trading live.
    """

    # Paramètres par défaut (fallback si pas encore de run)
    DEFAULT_PARAMS = {
        "min_confluence_score": 75,
        "atr_sl_multiplier":    1.5,
        "tp1_ratio":            1.0,
        "tp2_ratio":            1.618,
        "tp3_ratio":            2.5,
        "min_rr_ratio":         1.5,
        "swing_lookback":       5,
    }

    # Espaces de recherche pour chaque paramètre
    PARAM_SPACE = {
        "min_confluence_score": {"type": "int",   "low": 70,  "high": 92,  "step": 1},
        "atr_sl_multiplier":    {"type": "float", "low": 1.0, "high": 2.5, "step": 0.1},
        "tp1_ratio":            {"type": "float", "low": 0.8, "high": 1.5, "step": 0.1},
        "tp2_ratio":            {"type": "float", "low": 1.3, "high": 2.5, "step": 0.05},
        "tp3_ratio":            {"type": "float", "low": 2.0, "high": 4.0, "step": 0.25},
        "min_rr_ratio":         {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
        "swing_lookback":       {"type": "int",   "low": 3,   "high": 8,   "step": 1},
    }

    def __init__(
        self,
        db:      DatabaseManager,
        tracker: PerformanceTracker,
        analyzer: PatternAnalyzer,
    ):
        self._db       = db
        self._tracker  = tracker
        self._analyzer = analyzer

        # Paramètres actuellement actifs
        self._current_params = self.DEFAULT_PARAMS.copy()

    # ── DÉMARRAGE ────────────────────────────────────────────────

    async def start_scheduler(self):
        """
        Lance la boucle d'optimisation périodique.
        S'exécute en arrière-plan toutes les X jours.
        """
        if not settings.OPTIMIZER_ENABLED:
            logger.info("🔒 Optimiseur désactivé (OPTIMIZER_ENABLED=false)")
            return

        logger.info(
            f"🧠 Optimiseur démarré — "
            f"Re-optimisation tous les {settings.OPTIMIZER_INTERVAL_DAYS} jours"
        )

        while True:
            try:
                # Attend X jours avant la première optimisation
                await asyncio.sleep(
                    settings.OPTIMIZER_INTERVAL_DAYS * 86400
                )
                await self.run_optimization()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimizer scheduler error: {e}")
                await asyncio.sleep(3600)   # Retry dans 1h

    # ── OPTIMISATION PRINCIPALE ───────────────────────────────────

    async def run_optimization(
        self,
        instrument: str | None = None,
        n_trials:   int | None = None,
    ) -> OptimizationResult:
        """
        Lance une session d'optimisation Optuna complète.

        Args:
            instrument: Optimiser pour un instrument spécifique
                        (None = paramètres globaux)
            n_trials:   Nombre d'essais Optuna
        """
        n_trials = n_trials or settings.OPTIMIZER_TRIALS

        logger.info(
            f"🧠 Lancement optimisation — "
            f"{n_trials} trials | "
            f"Instrument: {instrument or 'GLOBAL'}"
        )

        # ── Récupère les données historiques ──────────────────────
        signals = await self._db.get_signals_for_backtest(
            instrument = instrument,
            months     = settings.BACKTEST_MONTHS,
        )

        finished = [
            s for s in signals
            if s.status in (
                SignalStatus.TP1_HIT, SignalStatus.TP2_HIT,
                SignalStatus.TP3_HIT, SignalStatus.SL_HIT,
            )
        ]

        if len(finished) < 30:
            logger.warning(
                f"Optimisation annulée — seulement {len(finished)} trades "
                "(minimum 30 requis)"
            )
            return OptimizationResult(
                success=False,
                best_params=self._current_params.copy(),
                best_wr=0.0, best_pf=0.0, trials_done=0,
                improved=False, improvement=0.0,
                summary=f"Données insuffisantes: {len(finished)} trades (min 30)",
            )

        # ── Performance actuelle (baseline) ───────────────────────
        baseline_wr = await self._tracker.get_win_rate(
            days=settings.BACKTEST_MONTHS * 30,
            instrument=instrument,
        )

        logger.info(f"📊 Baseline WR actuel: {baseline_wr:.1%}")

        # ── Analyse des patterns pour guider la recherche ─────────
        pattern_result = await self._analyzer.analyze(
            months=settings.BACKTEST_MONTHS
        )

        # Réduit l'espace de recherche grâce aux insights
        search_space = self._build_search_space(pattern_result)

        # ── Création de l'étude Optuna ────────────────────────────
        study = optuna.create_study(
            direction           = "maximize",    # Maximise le score objectif
            sampler             = optuna.samplers.TPESampler(seed=42),
            pruner              = optuna.pruners.MedianPruner(
                n_startup_trials = 10,
                n_warmup_steps   = 5,
            ),
        )

        # Objectif à maximiser
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, search_space)
            return self._evaluate_params(params, finished)

        # ── Exécution dans un thread pool ────────────────────────
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: study.optimize(
                objective,
                n_trials    = n_trials,
                show_progress_bar = False,
                n_jobs      = 1,    # Single-threaded pour stabilité
            )
        )

        # ── Meilleurs paramètres ──────────────────────────────────
        best_trial  = study.best_trial
        best_params = best_trial.params
        best_score  = best_trial.value   # Score objectif

        # Recalcule WR et PF sur les meilleurs params
        best_wr, best_pf = self._calc_metrics(best_params, finished)

        improved   = best_wr > baseline_wr
        improvement = (best_wr - baseline_wr) / baseline_wr * 100 if baseline_wr > 0 else 0.0

        logger.info(
            f"✅ Optimisation terminée — "
            f"{n_trials} trials | "
            f"Baseline WR: {baseline_wr:.1%} → "
            f"Optimal WR: {best_wr:.1%} "
            f"({'↑' if improved else '↓'}{abs(improvement):.1f}%)"
        )

        # ── Sauvegarde en base de données ─────────────────────────
        db_run = await self._db.save_optimizer_run({
            "instrument":   instrument,
            "trials_count": n_trials,
            "best_score":   best_score,
            "best_params":  best_params,
            "win_rate":     best_wr,
            "profit_factor": best_pf,
            "sharpe_ratio": None,
            "total_trades": len(finished),
            "period_months": settings.BACKTEST_MONTHS,
            "notes": (
                f"Baseline WR: {baseline_wr:.1%} | "
                f"Amélioration: {improvement:+.1f}%"
            ),
        })

        # ── Application si amélioré ───────────────────────────────
        if improved and best_wr >= 0.55:   # Minimum 55% WR
            self._apply_params(best_params)
            logger.success(
                f"🚀 Nouveaux paramètres appliqués — WR: {best_wr:.1%}"
            )

        summary = (
            f"Optimisation {instrument or 'GLOBAL'} — "
            f"{n_trials} trials\n"
            f"WR: {baseline_wr:.1%} → {best_wr:.1%} "
            f"({'↑' if improved else '↓'}{abs(improvement):.1f}%)\n"
            f"PF: {best_pf:.2f}\n"
            f"Params optimaux: {best_params}"
        )

        return OptimizationResult(
            success     = True,
            best_params = best_params,
            best_wr     = best_wr,
            best_pf     = best_pf,
            trials_done = n_trials,
            improved    = improved,
            improvement = round(improvement, 2),
            summary     = summary,
            run_id      = db_run.id,
        )

    # ── ÉVALUATION D'UN JEU DE PARAMÈTRES ────────────────────────

    def _evaluate_params(
        self,
        params:  dict,
        signals: list,
    ) -> float:
        """
        Score objectif pour Optuna.
        Simule l'application des paramètres sur l'historique.

        Score = WR × 0.5 + PF_norm × 0.3 + Sharpe_norm × 0.2
        Pénalité si < 15 trades (trop restrictif).
        """
        min_score = params.get("min_confluence_score", 75)
        min_rr    = params.get("min_rr_ratio", 1.5)

        # Filtre les signaux qui passeraient avec ces paramètres
        filtered = [
            s for s in signals
            if (s.confluence_score or 0) >= min_score
            and (s.rr_ratio or 0) >= min_rr
        ]

        if len(filtered) < 15:
            return 0.0   # Pénalité sévère si trop peu de trades

        win_statuses = {
            SignalStatus.TP1_HIT,
            SignalStatus.TP2_HIT,
            SignalStatus.TP3_HIT,
        }

        wins   = [s for s in filtered if s.status in win_statuses]
        losses = [s for s in filtered if s.status == SignalStatus.SL_HIT]

        wr = len(wins) / len(filtered)

        # Profit Factor
        gross_profit = sum(s.pnl_usd or 0 for s in wins if (s.pnl_usd or 0) > 0)
        gross_loss   = abs(sum(s.pnl_usd or 0 for s in losses if (s.pnl_usd or 0) < 0))
        pf           = min(gross_profit / gross_loss, 5.0) if gross_loss > 0 else 2.0

        # Normalisation PF (0 à 1, où 3.0 = parfait)
        pf_norm = min(pf / 3.0, 1.0)

        # Bonus pour suffisamment de trades
        trade_bonus = min(len(filtered) / 50.0, 0.1)

        # Score composite
        score = wr * 0.5 + pf_norm * 0.3 + trade_bonus

        return round(score, 4)

    def _calc_metrics(
        self,
        params:  dict,
        signals: list,
    ) -> tuple[float, float]:
        """Calcule WR et PF pour un jeu de paramètres."""
        min_score = params.get("min_confluence_score", 75)
        min_rr    = params.get("min_rr_ratio", 1.5)

        filtered = [
            s for s in signals
            if (s.confluence_score or 0) >= min_score
            and (s.rr_ratio or 0) >= min_rr
        ]

        if not filtered:
            return 0.0, 0.0

        win_statuses = {
            SignalStatus.TP1_HIT,
            SignalStatus.TP2_HIT,
            SignalStatus.TP3_HIT,
        }
        wins   = [s for s in filtered if s.status in win_statuses]
        losses = [s for s in filtered if s.status == SignalStatus.SL_HIT]

        wr           = len(wins) / len(filtered)
        gross_profit = sum(s.pnl_usd or 0 for s in wins if (s.pnl_usd or 0) > 0)
        gross_loss   = abs(sum(s.pnl_usd or 0 for s in losses if (s.pnl_usd or 0) < 0))
        pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return round(wr, 4), round(pf, 2)

    # ── GESTION DES PARAMÈTRES ────────────────────────────────────

    def _suggest_params(
        self,
        trial: optuna.Trial,
        search_space: dict,
    ) -> dict:
        """Propose un jeu de paramètres pour un trial Optuna."""
        params = {}
        for name, space in search_space.items():
            if space["type"] == "int":
                params[name] = trial.suggest_int(
                    name, space["low"], space["high"], step=space.get("step", 1)
                )
            else:
                params[name] = trial.suggest_float(
                    name, space["low"], space["high"], step=space.get("step", 0.1)
                )
        return params

    def _apply_params(self, params: dict):
        """
        Applique les paramètres optimisés.
        Met à jour les settings en mémoire.
        """
        self._current_params.update(params)

        # Mise à jour des settings runtime
        if "min_confluence_score" in params:
            settings.MIN_CONFLUENCE_SCORE = params["min_confluence_score"]
        if "atr_sl_multiplier" in params:
            settings.ATR_SL_MULTIPLIER = params["atr_sl_multiplier"]
        if "tp1_ratio" in params:
            settings.TP1_RATIO = params["tp1_ratio"]
        if "tp2_ratio" in params:
            settings.TP2_RATIO = params["tp2_ratio"]
        if "tp3_ratio" in params:
            settings.TP3_RATIO = params["tp3_ratio"]
        if "min_rr_ratio" in params:
            settings.MIN_RR_RATIO = params["min_rr_ratio"]

        logger.info(f"⚙️ Paramètres mis à jour: {params}")

    def get_current_params(self) -> dict:
        """Retourne les paramètres actuellement actifs."""
        return self._current_params.copy()

    # ── ESPACE DE RECHERCHE ADAPTATIF ─────────────────────────────

    def _build_search_space(self, pattern_result) -> dict:
        """
        Réduit l'espace de recherche en utilisant
        les insights du PatternAnalyzer.

        Si on sait que score >= 82 performe mieux,
        on ne cherche pas en dessous de 80.
        """
        space = self.PARAM_SPACE.copy()

        # Ajuste le score minimum selon les insights
        optimal_score = pattern_result.optimal_min_score
        if optimal_score > 75:
            space["min_confluence_score"] = {
                "type": "int",
                "low":  max(70, optimal_score - 5),
                "high": min(95, optimal_score + 5),
                "step": 1,
            }

        # Ajuste le R:R minimum
        optimal_rr = pattern_result.optimal_min_rr
        space["min_rr_ratio"] = {
            "type": "float",
            "low":  max(1.0, optimal_rr - 0.5),
            "high": min(3.0, optimal_rr + 1.0),
            "step": 0.25,
        }

        return space

    # ── CHARGEMENT DES MEILLEURS PARAMS DEPUIS LA DB ──────────────

    async def load_best_params(self) -> dict:
        """
        Charge les meilleurs paramètres du dernier run Optuna
        depuis la base de données.
        Utile au redémarrage du bot.
        """
        db_params = await self._db.get_latest_optimizer_params()

        if db_params:
            self._apply_params(db_params)
            logger.info(
                f"✅ Paramètres chargés depuis DB: {db_params}"
            )
            return db_params

        logger.info("📋 Paramètres par défaut utilisés (aucun run précédent)")
        return self.DEFAULT_PARAMS.copy()
