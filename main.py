# main.py
# ─────────────────────────────────────────────────────────────────
#  Point d'entrée principal — Boucle 24/7 du bot
#
#  Séquence de démarrage :
#   1. Initialisation DB + connexions API
#   2. Chargement des meilleurs paramètres (Optimizer)
#   3. Démarrage OrderManager (monitoring trades)
#   4. Démarrage TelegramBot
#   5. Démarrage Dashboard FastAPI
#   6. Boucle principale d'analyse (toutes les X minutes)
#   7. Scheduler : rapport journalier + snapshot
#   8. Scheduler : optimisation adaptative (tous les 7 jours)
#
#  Arrêt propre sur SIGINT / SIGTERM (Ctrl+C ou Railway stop)
# ─────────────────────────────────────────────────────────────────

import asyncio
import signal
import sys
from datetime import datetime

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import settings
from config.instruments import get_active_instruments
from database.db_manager import DatabaseManager
from data.data_manager import DataManager
from execution.risk_manager import RiskManager
from execution.order_manager import OrderManager
from signals.signal_generator import SignalGenerator
from adaptive.performance_tracker import PerformanceTracker
from adaptive.pattern_analyzer import PatternAnalyzer
from adaptive.optimizer import AdaptiveOptimizer
from notifications.telegram_bot import TelegramBot
from backtesting.backtester import Backtester, BacktestConfig
from dashboard.app import create_app


# ── Logging ───────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> — {message}"
    ),
    level=settings.LOG_LEVEL,
    colorize=True,
)
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    level="DEBUG",
    encoding="utf-8",
)

console = Console()


# ─────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE DU BOT
# ─────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Orchestrateur central du bot de trading.
    Gère le cycle de vie complet : démarrage → analyse → arrêt.
    """

    # Intervalle entre chaque cycle d'analyse (en secondes)
    # 5 min = 300s — synchronisé avec M5
    ANALYSIS_INTERVAL = 300

    def __init__(self):
        # ── Composants principaux ─────────────────────────────────
        self.db        = DatabaseManager()
        self.data      = DataManager()
        self.risk      = RiskManager(self.db)
        self.orders    = OrderManager(self.db, self.data, self.risk)
        self.generator = SignalGenerator(self.data, self.db)
        self.tracker   = PerformanceTracker(self.db)
        self.analyzer  = PatternAnalyzer(self.db)
        self.optimizer = AdaptiveOptimizer(self.db, self.tracker, self.analyzer)
        self.telegram  = TelegramBot()
        self.backtester= Backtester(self.data, self.db)

        # ── Scheduler APScheduler ─────────────────────────────────
        self.scheduler = AsyncIOScheduler(timezone="UTC")

        # ── État ──────────────────────────────────────────────────
        self._running       = False
        self._paused        = False
        self._analysis_task : asyncio.Task | None = None
        self._dashboard_task: asyncio.Task | None = None
        self._start_time    = datetime.utcnow()
        self._cycle_count   = 0
        self._signals_today = 0

    # ── DÉMARRAGE ────────────────────────────────────────────────

    async def start(self):
        """
        Démarre tous les composants dans l'ordre correct.
        En cas d'erreur critique, lève une exception.
        """
        self._print_banner()

        try:
            # ── 1. Base de données ────────────────────────────────
            logger.info("📦 Initialisation de la base de données...")
            await self.db.init()
            logger.info("✅ Base de données prête")

            # ── 2. Validation des connexions API ──────────────────
            logger.info("🔌 Vérification des connexions API...")
            connections = await self.data.validate_connection()

            if not connections["twelve_data"]:
                raise RuntimeError(
                    "❌ Twelve Data API inaccessible — "
                    "Vérifie TWELVE_DATA_KEY dans .env"
                )
            if not connections["oanda"]:
                logger.warning(
                    "⚠️ OANDA indisponible — "
                    "Mode signaux uniquement (pas d'exécution auto)"
                )

            # ── 3. Chargement paramètres optimisés ────────────────
            logger.info("🧠 Chargement des paramètres optimisés...")
            await self.optimizer.load_best_params()

            # ── 4. Balance de référence journalière ───────────────
            try:
                balance = await self.data.get_balance("oanda")
                self.risk.update_daily_balance(balance)
                logger.info(f"💰 Balance OANDA: {balance:.2f} USD")
            except Exception as e:
                logger.warning(f"Balance OANDA non disponible: {e}")

            # ── 5. OrderManager (monitoring trades ouverts) ───────
            logger.info("🔄 Démarrage du monitoring des trades...")
            await self.orders.start()

            # ── 6. Telegram Bot ───────────────────────────────────
            logger.info("📱 Démarrage du bot Telegram...")
            await self.telegram.start(
                on_pause  = self._on_pause,
                on_resume = self._on_resume,
                on_report = self._on_report,
            )

            # ── 7. Scheduler ──────────────────────────────────────
            self._setup_scheduler()
            self.scheduler.start()
            logger.info("⏰ Scheduler démarré")

            # ── 8. Dashboard FastAPI ───────────────────────────────
            app = create_app(self.db, self.data, self.tracker)
            config = uvicorn.Config(
                app     = app,
                host    = settings.DASHBOARD_HOST,
                port    = settings.DASHBOARD_PORT,
                log_level = "warning",
            )
            server = uvicorn.Server(config)
            self._dashboard_task = asyncio.create_task(server.serve())
            logger.info(
                f"🌐 Dashboard disponible sur "
                f"http://{settings.DASHBOARD_HOST}:{settings.DASHBOARD_PORT}"
            )

            # ── 9. Backtest de validation au démarrage ────────────
            if settings.BOT_ENV == "production":
                logger.info("🔬 Backtest de validation au démarrage...")
                asyncio.create_task(self._startup_backtest())

            # ── 10. Boucle principale ─────────────────────────────
            self._running = True
            self._analysis_task = asyncio.create_task(self._analysis_loop())

            logger.success(
                f"🚀 Bot démarré en mode {settings.BOT_ENV.upper()} — "
                f"Analyse toutes les {self.ANALYSIS_INTERVAL // 60} min"
            )

            # Attend la fin (SIGINT ou SIGTERM)
            await asyncio.gather(
                self._analysis_task,
                self._dashboard_task,
                return_exceptions=True,
            )

        except Exception as e:
            logger.critical(f"❌ Erreur critique au démarrage: {e}")
            await self.stop()
            raise

    # ── ARRÊT ─────────────────────────────────────────────────────

    async def stop(self):
        """Arrête tous les composants proprement."""
        logger.info("⏹️ Arrêt du bot en cours...")
        self._running = False

        # Annule les tâches principales
        for task in [self._analysis_task, self._dashboard_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Arrêt des composants
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass

        try:
            await self.orders.stop()
        except Exception:
            pass

        try:
            await self.telegram.stop()
        except Exception:
            pass

        try:
            await self.data.close()
        except Exception:
            pass

        try:
            await self.db.close()
        except Exception:
            pass

        uptime = datetime.utcnow() - self._start_time
        hours  = int(uptime.total_seconds() // 3600)
        mins   = int((uptime.total_seconds() % 3600) // 60)

        logger.info(
            f"✅ Bot arrêté proprement — "
            f"Uptime: {hours}h {mins}min | "
            f"Cycles: {self._cycle_count}"
        )

    # ── BOUCLE D'ANALYSE PRINCIPALE ───────────────────────────────

    async def _analysis_loop(self):
        """
        Boucle principale d'analyse — tourne en continu.

        Chaque cycle :
          1. Vérifie si le bot est en pause
          2. Met à jour la balance journalière
          3. Analyse tous les instruments actifs
          4. Pour chaque signal valide : exécute le trade
          5. Envoie les signaux sur Telegram
          6. Attend le prochain cycle
        """
        logger.info(
            f"🔁 Boucle d'analyse démarrée — "
            f"Intervalle: {self.ANALYSIS_INTERVAL}s"
        )

        # Attente initiale courte (10s) pour laisser tout démarrer
        await asyncio.sleep(10)

        while self._running:
            cycle_start = datetime.utcnow()
            self._cycle_count += 1

            try:
                await self._run_analysis_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur cycle #{self._cycle_count}: {e}")
                await self.telegram.send_alert(
                    f"Erreur cycle analyse #{self._cycle_count}:\n{e}",
                    critical=False,
                )

            # Calcule le temps d'attente jusqu'au prochain cycle
            elapsed = (datetime.utcnow() - cycle_start).total_seconds()
            wait    = max(0, self.ANALYSIS_INTERVAL - elapsed)

            logger.debug(
                f"Cycle #{self._cycle_count} terminé en {elapsed:.1f}s — "
                f"Prochain dans {wait:.0f}s"
            )

            await asyncio.sleep(wait)

    async def _run_analysis_cycle(self):
        """
        Exécute un cycle d'analyse complet sur tous les instruments.
        """
        if self._paused:
            logger.debug("⏸️ Bot en pause — cycle ignoré")
            return

        logger.info(
            f"🔍 Cycle #{self._cycle_count} — "
            f"{datetime.utcnow().strftime('%H:%M UTC')}"
        )

        # ── Mise à jour balance journalière ───────────────────────
        try:
            balance = await self.data.get_balance("oanda")
            self.risk.update_daily_balance(balance)
        except Exception as e:
            logger.warning(f"Balance update error: {e}")

        # ── Analyse multi-instruments ─────────────────────────────
        instruments = [
            instr.symbol
            for instr in get_active_instruments()
        ]

        signals = await self.generator.analyze_all(instruments)

        if not signals:
            logger.info("📊 Aucun signal ce cycle")
            return

        # ── Traitement de chaque signal ───────────────────────────
        for signal in signals:
            try:
                await self._process_signal(signal)
            except Exception as e:
                logger.error(
                    f"Erreur traitement signal "
                    f"{signal.instrument}: {e}"
                )

    async def _process_signal(self, signal):
        """
        Traite un signal généré :
          1. Envoie le signal sur Telegram
          2. Exécute le trade (si mode auto activé)
          3. Enregistre le message Telegram ID
        """
        logger.info(
            f"📡 Signal traité — {signal.instrument} "
            f"{signal.direction.upper()} | "
            f"Score: {signal.score}/100 | "
            f"R:R: {signal.rr_ratio:.2f}"
        )

        # ── Envoi Telegram ────────────────────────────────────────
        msg_id = await self.telegram.send_signal(
            message    = signal.telegram_message,
            instrument = signal.instrument,
            direction  = signal.direction,
            entry      = signal.entry,
            sl         = signal.sl,
            tp1        = signal.tp1,
            tp2        = signal.tp2,
            tp3        = signal.tp3,
            score      = signal.score,
            quality    = signal.quality,
            session    = signal.session,
            signal_id  = signal.signal_id,
        )

        # Sauvegarde l'ID du message Telegram dans la DB
        if msg_id and signal.signal_id:
            await self.db.update_signal_status(
                signal.signal_id,
                (await self.db.get_signal(signal.signal_id)).status,
                extra={"telegram_msg_id": msg_id},
            )

        # ── Exécution automatique ─────────────────────────────────
        if settings.BOT_ENV == "production":
            order_result = await self.orders.execute_signal(
                signal_id  = signal.signal_id,
                instrument = signal.instrument,
                direction  = signal.direction,
                entry      = signal.entry,
                sl_price   = signal.sl,
                tp1        = signal.tp1,
                tp2        = signal.tp2,
                tp3        = signal.tp3,
            )

            if order_result.success:
                logger.success(
                    f"✅ Ordre exécuté — {signal.instrument} "
                    f"@ {order_result.entry_price}"
                )
            else:
                logger.warning(
                    f"⚠️ Ordre non exécuté — {signal.instrument}: "
                    f"{order_result.error}"
                )
        else:
            logger.info(
                f"🧪 Mode development — signal envoyé "
                f"sans exécution réelle"
            )

        self._signals_today += 1

    # ── SCHEDULER — TÂCHES PLANIFIÉES ────────────────────────────

    def _setup_scheduler(self):
        """Configure les tâches planifiées."""

        # ── Rapport journalier à minuit UTC ───────────────────────
        self.scheduler.add_job(
            self._daily_report,
            "cron",
            hour=0, minute=5,
            id="daily_report",
            name="Rapport journalier",
        )

        # ── Snapshot de performance à 23h55 UTC ───────────────────
        self.scheduler.add_job(
            self._daily_snapshot,
            "cron",
            hour=23, minute=55,
            id="daily_snapshot",
            name="Snapshot journalier",
        )

        # ── Optimisation adaptative (tous les X jours) ────────────
        self.scheduler.add_job(
            self._run_optimization,
            "interval",
            days=settings.OPTIMIZER_INTERVAL_DAYS,
            id="optimizer",
            name="Optimisation adaptative",
        )

        # ── Reset du compteur journalier à minuit UTC ─────────────
        self.scheduler.add_job(
            self._daily_reset,
            "cron",
            hour=0, minute=0,
            id="daily_reset",
            name="Reset journalier",
        )

        # ── Health check toutes les heures ────────────────────────
        self.scheduler.add_job(
            self._health_check,
            "interval",
            hours=1,
            id="health_check",
            name="Health check",
        )

        logger.info(
            f"⏰ Scheduler configuré — "
            f"5 tâches planifiées"
        )

    # ── TÂCHES PLANIFIÉES ────────────────────────────────────────

    async def _daily_report(self):
        """Génère et envoie le rapport de performance journalier."""
        logger.info("📊 Génération du rapport journalier...")
        try:
            report = await self.tracker.generate_report(days=1)
            await self.telegram.send_daily_report(report)
            logger.info("✅ Rapport journalier envoyé")
        except Exception as e:
            logger.error(f"Daily report error: {e}")

    async def _daily_snapshot(self):
        """Sauvegarde un snapshot de performance en base."""
        try:
            await self.tracker.save_daily_snapshot()
        except Exception as e:
            logger.error(f"Daily snapshot error: {e}")

    async def _run_optimization(self):
        """Lance l'optimisation adaptative en arrière-plan."""
        if not settings.OPTIMIZER_ENABLED:
            return
        logger.info("🧠 Lancement de l'optimisation adaptative...")
        try:
            result = await self.optimizer.run_optimization()
            if result.success and result.improved:
                await self.telegram.send_alert(
                    f"🧠 Optimisation terminée\n"
                    f"WR amélioré: {result.improvement:+.1f}%\n"
                    f"Nouveau WR: {result.best_wr:.1%}\n"
                    f"PF: {result.best_pf:.2f}",
                    critical=False,
                )
        except Exception as e:
            logger.error(f"Optimization error: {e}")

    async def _daily_reset(self):
        """Reset des compteurs journaliers."""
        self._signals_today = 0
        try:
            balance = await self.data.get_balance("oanda")
            self.risk.update_daily_balance(balance)
        except Exception:
            pass
        logger.info("🔄 Compteurs journaliers réinitialisés")

    async def _health_check(self):
        """
        Vérifie que tout fonctionne correctement.
        Envoie une alerte si quelque chose est cassé.
        """
        try:
            connections = await self.data.validate_connection()
            issues      = [k for k, v in connections.items() if not v]

            if issues:
                await self.telegram.send_alert(
                    f"⚠️ Connexions perdues: {', '.join(issues)}\n"
                    f"Le bot continue en mode dégradé.",
                    critical=False,
                )
        except Exception as e:
            logger.error(f"Health check error: {e}")

    # ── BACKTEST DE VALIDATION ────────────────────────────────────

    async def _startup_backtest(self):
        """
        Lance un backtest rapide au démarrage pour valider
        que les paramètres actuels performent bien.
        """
        await asyncio.sleep(30)   # Laisse le bot démarrer complètement

        logger.info("🔬 Backtest de validation démarré...")
        try:
            config = BacktestConfig(
                instrument        = "XAUUSD",
                months            = 1,
                initial_balance   = 10000.0,
                risk_pct          = settings.RISK_PER_TRADE,
                min_score         = settings.MIN_CONFLUENCE_SCORE,
                atr_sl_multiplier = settings.ATR_SL_MULTIPLIER,
                tp1_ratio         = settings.TP1_RATIO,
                tp2_ratio         = settings.TP2_RATIO,
                tp3_ratio         = settings.TP3_RATIO,
                min_rr            = settings.MIN_RR_RATIO,
            )

            result = await self.backtester.run(config)
            m      = result.metrics

            status = "✅ Params validés" if result.meets_target else "⚠️ Params sous-optimaux"

            await self.telegram.send_alert(
                f"🔬 Backtest validation (1 mois)\n"
                f"{status}\n"
                f"WR: {m.win_rate:.1%} | PF: {m.profit_factor:.2f}\n"
                f"Grade: {m.grade} | Trades: {m.total_trades}",
                critical=not result.meets_target,
            )

        except Exception as e:
            logger.error(f"Startup backtest error: {e}")

    # ── CALLBACKS ADMIN ───────────────────────────────────────────

    async def _on_pause(self):
        """Callback appelé par /pause sur Telegram."""
        self._paused = True
        logger.warning("⏸️ Bot mis en pause par l'admin")

    async def _on_resume(self):
        """Callback appelé par /resume sur Telegram."""
        self._paused = False
        logger.info("▶️ Bot relancé par l'admin")

    async def _on_report(self):
        """Callback appelé par /report sur Telegram."""
        await self._daily_report()

    # ── BANNER ────────────────────────────────────────────────────

    def _print_banner(self):
        """Affiche le banner de démarrage dans le terminal."""
        text = Text()
        text.append("TRADING BOT SMC/ICT\n", style="bold gold1")
        text.append(f"  Env     : {settings.BOT_ENV}\n", style="cyan")
        text.append(f"  Score   : ≥ {settings.MIN_CONFLUENCE_SCORE}/100\n", style="cyan")
        text.append(f"  Risque  : {settings.RISK_PER_TRADE}% / trade\n", style="cyan")
        text.append(f"  R:R min : {settings.MIN_RR_RATIO}\n", style="cyan")
        text.append(f"  Objectif: ≥ 60% Win Rate\n", style="green")
        text.append(f"  Build   : v1.0.0\n", style="dim")

        console.print(Panel(
            text,
            border_style = "gold1",
            padding      = (1, 4),
        ))


# ─────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────

async def main():
    """Fonction principale — gère le bot et les signaux système."""
    bot  = TradingBot()
    loop = asyncio.get_running_loop()

    # ── Gestion des signaux système (arrêt propre) ────────────────
    def _signal_handler():
        logger.info("🛑 Signal d'arrêt reçu")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows ne supporte pas add_signal_handler
            pass

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt reçu")
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        sys.exit(1)
    finally:
        await bot.stop()


if __name__ == "__main__":
    # Crée le dossier logs si inexistant
    import os
    os.makedirs("logs", exist_ok=True)

    asyncio.run(main())
