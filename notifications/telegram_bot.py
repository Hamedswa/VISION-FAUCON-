# notifications/telegram_bot.py
# ─────────────────────────────────────────────────────────────────
#  Bot Telegram — Diffusion des signaux et alertes
#
#  Types de messages :
#   • SIGNAL      — Nouveau signal de trading (avec niveaux)
#   • TP_HIT      — Take Profit atteint (TP1, TP2 ou TP3)
#   • SL_HIT      — Stop Loss touché
#   • BREAKEVEN   — SL déplacé au breakeven
#   • DAILY_REPORT— Rapport de performance journalier
#   • ALERT       — Alerte critique (admin uniquement)
#   • STATUS      — Statut du bot (démarrage, erreur...)
#
#  Fonctionnalités :
#   • File d'attente async (pas de blocage de la boucle principale)
#   • Rate limiting (max 30 msg/s Telegram API)
#   • Retry automatique (3 tentatives)
#   • Mise à jour des messages existants (édition)
#   • Commandes admin (/status, /report, /pause, /resume)
# ─────────────────────────────────────────────────────────────────

import asyncio
from datetime import datetime
from enum import Enum
from typing import Callable
from loguru import logger
from telegram import (
    Bot,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

from config import settings
from adaptive.performance_tracker import PerformanceReport


class MessageType(Enum):
    SIGNAL       = "signal"
    TP_HIT       = "tp_hit"
    SL_HIT       = "sl_hit"
    BREAKEVEN    = "breakeven"
    DAILY_REPORT = "daily_report"
    ALERT        = "alert"
    STATUS       = "status"


class TelegramBot:
    """
    Gestionnaire complet du bot Telegram.
    Gère l'envoi de signaux, les mises à jour de trades
    et les commandes admin.
    """

    MAX_RETRIES    = 3
    RETRY_DELAY    = 2.0     # secondes entre retries
    RATE_LIMIT     = 0.05    # 50ms entre messages (20/s max)

    def __init__(self):
        self._bot         = Bot(token=settings.TELEGRAM_TOKEN)
        self._channel_id  = settings.TELEGRAM_CHANNEL_ID
        self._admin_id    = settings.TELEGRAM_ADMIN_ID
        self._app         = None

        # File d'attente async pour les messages
        self._queue:      asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

        # Callbacks pour les commandes admin
        self._on_pause:   Callable | None = None
        self._on_resume:  Callable | None = None
        self._on_report:  Callable | None = None

        # État du bot
        self._is_paused = False
        self._start_time = datetime.utcnow()

    # ── DÉMARRAGE / ARRÊT ─────────────────────────────────────────

    async def start(
        self,
        on_pause:  Callable | None = None,
        on_resume: Callable | None = None,
        on_report: Callable | None = None,
    ):
        """
        Démarre le bot Telegram et le worker de la file d'attente.

        Args:
            on_pause:  Callback quand admin tape /pause
            on_resume: Callback quand admin tape /resume
            on_report: Callback quand admin tape /report
        """
        self._on_pause  = on_pause
        self._on_resume = on_resume
        self._on_report = on_report

        # Lance le worker de messages
        self._worker_task = asyncio.create_task(self._message_worker())

        # Démarre le listener de commandes admin
        asyncio.create_task(self._start_command_listener())

        # Message de démarrage
        await self.send_status(
            f"🟢 Bot démarré\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"📊 Instruments: {', '.join(['XAUUSD', 'BTCUSD', 'EURUSD'])}\n"
            f"🎯 Score min: {settings.MIN_CONFLUENCE_SCORE}/100\n"
            f"⚙️ Env: {settings.BOT_ENV}"
        )

        logger.info("✅ TelegramBot démarré")

    async def stop(self):
        """Arrête le bot proprement."""
        if self._worker_task:
            self._worker_task.cancel()

        await self.send_status("🔴 Bot arrêté")
        logger.info("⏹️ TelegramBot arrêté")

    # ── ENVOI DE SIGNAUX ──────────────────────────────────────────

    async def send_signal(
        self,
        message:       str,
        instrument:    str,
        direction:     str,
        entry:         float,
        sl:            float,
        tp1:           float,
        tp2:           float,
        tp3:           float,
        score:         int,
        quality:       str,
        session:       str | None = None,
        signal_id:     int | None = None,
    ) -> int | None:
        """
        Envoie un signal de trading sur le canal.
        Retourne l'ID du message Telegram envoyé.

        Si le message Groq est fourni (via `message`), il est utilisé.
        Sinon, un message structuré est généré automatiquement.
        """
        if self._is_paused:
            logger.warning(f"Bot en pause — signal {instrument} non envoyé")
            return None

        # Clavier inline avec les niveaux clés
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Dashboard", url="http://localhost:8000"),
                InlineKeyboardButton("✅ Pris", callback_data=f"taken_{signal_id}"),
            ],
            [
                InlineKeyboardButton("❌ Ignoré", callback_data=f"ignored_{signal_id}"),
            ],
        ])

        # Utilise le message Groq s'il est riche, sinon génère un fallback
        if len(message) > 100:
            final_message = message
        else:
            final_message = self._build_signal_message(
                instrument = instrument,
                direction  = direction,
                entry      = entry,
                sl         = sl,
                tp1        = tp1,
                tp2        = tp2,
                tp3        = tp3,
                score      = score,
                quality    = quality,
                session    = session,
            )

        msg_id = await self._enqueue_and_wait(
            chat_id      = self._channel_id,
            text         = final_message,
            parse_mode   = ParseMode.MARKDOWN,
            reply_markup = keyboard,
            msg_type     = MessageType.SIGNAL,
        )

        logger.info(
            f"📤 Signal Telegram envoyé — {instrument} | "
            f"Msg ID: {msg_id}"
        )
        return msg_id

    # ── MISES À JOUR DE TRADES ────────────────────────────────────

    async def send_tp_hit(
        self,
        instrument:   str,
        direction:    str,
        tp_level:     int,
        entry:        float,
        exit_price:   float,
        pnl_usd:      float,
        pnl_pips:     float,
        rr_achieved:  float,
        original_msg_id: int | None = None,
    ):
        """
        Notifie qu'un Take Profit a été atteint.
        Essaie d'éditer le message original, sinon envoie un nouveau.
        """
        tp_emojis = {1: "🥉", 2: "🥈", 3: "🏆"}
        emoji = tp_emojis.get(tp_level, "🎯")

        text = (
            f"{emoji} *TP{tp_level} ATTEINT — {instrument}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"📍 Entrée   : `{entry}`\n"
            f"✅ Sortie   : `{exit_price}`\n"
            f"━━━━━━━━━━━━━━\n"
            f"💰 P&L      : *+{pnl_usd:.2f} USD* (+{pnl_pips:.1f} pips)\n"
            f"📐 R:R réel : {rr_achieved:.2f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"{'🔒 SL déplacé au BE — reste 50% position' if tp_level == 1 else '✅ Position fermée'}\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M UTC')}"
        )

        # Tente d'éditer le message original
        if original_msg_id:
            edited = await self._edit_message(
                chat_id    = self._channel_id,
                message_id = original_msg_id,
                text       = text,
            )
            if edited:
                return

        # Fallback : nouveau message
        await self._enqueue_and_wait(
            chat_id    = self._channel_id,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.TP_HIT,
        )

    async def send_sl_hit(
        self,
        instrument:      str,
        direction:       str,
        entry:           float,
        sl_price:        float,
        pnl_usd:         float,
        pnl_pips:        float,
        was_breakeven:   bool = False,
        original_msg_id: int | None = None,
    ):
        """Notifie qu'un Stop Loss a été touché."""
        if was_breakeven:
            title = f"🔒 *BE CLÔTURÉ — {instrument}*"
            pnl_text = f"±0.00 USD (breakeven)"
        else:
            title = f"🔴 *SL TOUCHÉ — {instrument}*"
            pnl_text = f"*{pnl_usd:.2f} USD* ({pnl_pips:.1f} pips)"

        text = (
            f"{title}\n"
            f"━━━━━━━━━━━━━━\n"
            f"📍 Entrée : `{entry}`\n"
            f"🛑 SL     : `{sl_price}`\n"
            f"━━━━━━━━━━━━━━\n"
            f"💸 P&L    : {pnl_text}\n"
            f"━━━━━━━━━━━━━━\n"
            f"📊 _Le bot analyse le prochain setup_\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M UTC')}"
        )

        if original_msg_id:
            edited = await self._edit_message(
                chat_id    = self._channel_id,
                message_id = original_msg_id,
                text       = text,
            )
            if edited:
                return

        await self._enqueue_and_wait(
            chat_id    = self._channel_id,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.SL_HIT,
        )

    async def send_breakeven(
        self,
        instrument:      str,
        entry:           float,
        original_msg_id: int | None = None,
    ):
        """Notifie le déplacement du SL au breakeven."""
        text = (
            f"🔒 *BREAKEVEN ACTIVÉ — {instrument}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"✅ TP1 atteint — SL déplacé à `{entry}`\n"
            f"📊 50% position fermée (profit sécurisé)\n"
            f"🎯 Reste 50% vers TP2 et TP3\n"
            f"🕐 {datetime.utcnow().strftime('%H:%M UTC')}"
        )

        if original_msg_id:
            await self._edit_message(
                chat_id    = self._channel_id,
                message_id = original_msg_id,
                text       = f"{text}\n\n_(Message original mis à jour)_",
            )
            return

        await self._enqueue_and_wait(
            chat_id    = self._channel_id,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.BREAKEVEN,
        )

    # ── RAPPORT DE PERFORMANCE ────────────────────────────────────

    async def send_daily_report(self, report: PerformanceReport):
        """
        Envoie le rapport de performance journalier.
        Format adapté pour lecture rapide sur mobile.
        """
        wr_emoji = (
            "🏆" if report.win_rate >= 0.70 else
            "✅" if report.win_rate >= 0.60 else
            "⚠️" if report.win_rate >= 0.50 else
            "🔴"
        )

        # Barre de progression visuelle pour le WR
        wr_bar  = self._progress_bar(report.win_rate, 1.0, length=10)
        pnl_sign = "+" if report.total_pnl_usd >= 0 else ""

        # Performance par session
        session_text = ""
        for sess, data in report.by_session.items():
            sess_wr = data.get("win_rate", 0)
            session_text += (
                f"  • {sess.upper():<8} : "
                f"{sess_wr:.0%} ({data.get('trades', 0)} trades)\n"
            )

        # Score ranges
        score_text = ""
        for rng, data in report.by_score_range.items():
            if data.get("trades", 0) > 0:
                score_text += (
                    f"  • Score {rng} : "
                    f"{data.get('win_rate', 0):.0%} "
                    f"({data.get('trades', 0)} trades)\n"
                )

        text = (
            f"📊 *RAPPORT JOURNALIER — {datetime.utcnow().strftime('%d/%m/%Y')}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"*Performance globale*\n"
            f"{wr_emoji} Win Rate    : *{report.win_rate:.1%}* {wr_bar}\n"
            f"📐 Profit Factor: *{report.profit_factor:.2f}*\n"
            f"📈 Sharpe Ratio : *{report.sharpe_ratio:.2f}*\n"
            f"\n"
            f"*Trades*\n"
            f"📌 Total        : {report.total_trades}\n"
            f"✅ Gagnants     : {report.winning_trades}\n"
            f"❌ Perdants     : {report.losing_trades}\n"
            f"  ├ TP1        : {report.tp1_hit_count}\n"
            f"  ├ TP2        : {report.tp2_hit_count}\n"
            f"  └ TP3        : {report.tp3_hit_count}\n"
            f"\n"
            f"*P&L*\n"
            f"💰 Total        : *{pnl_sign}{report.total_pnl_usd:.2f} USD*\n"
            f"📊 Moy. gain    : +{report.avg_win_usd:.2f} USD\n"
            f"📉 Moy. perte   : {report.avg_loss_usd:.2f} USD\n"
            f"🎯 Espérance    : {report.expectancy_usd:.2f} USD/trade\n"
            f"\n"
            f"*Drawdown*\n"
            f"📉 Max DD       : {report.max_drawdown_pct:.1f}%\n"
            f"📊 Actuel       : {report.current_drawdown:.1f}%\n"
            f"\n"
            f"*Par session*\n"
            f"{session_text or '  Aucune donnée'}\n"
            f"*Par score*\n"
            f"{score_text or '  Aucune donnée'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅ Objectif 60% WR ATTEINT' if report.target_wr_reached else '⚠️ Objectif 60% WR non atteint'}\n"
            f"🕐 Généré : {datetime.utcnow().strftime('%H:%M UTC')}"
        )

        await self._enqueue_and_wait(
            chat_id    = self._channel_id,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.DAILY_REPORT,
        )

        logger.info("📤 Rapport journalier envoyé sur Telegram")

    # ── ALERTES ADMIN ─────────────────────────────────────────────

    async def send_alert(self, message: str, critical: bool = False):
        """
        Envoie une alerte critique à l'admin uniquement.
        Ne passe pas par le canal public.
        """
        if not self._admin_id:
            logger.warning("TELEGRAM_ADMIN_ID non configuré")
            return

        prefix = "🚨 *ALERTE CRITIQUE*" if critical else "⚠️ *ALERTE*"
        text = (
            f"{prefix}\n"
            f"━━━━━━━━━━━━━━\n"
            f"{message}\n"
            f"━━━━━━━━━━━━━━\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        await self._enqueue_and_wait(
            chat_id    = self._admin_id,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.ALERT,
        )

    async def send_status(self, message: str):
        """Envoie un message de statut (admin uniquement si disponible)."""
        target = self._admin_id or self._channel_id
        text = (
            f"ℹ️ *STATUS BOT*\n"
            f"━━━━━━━━━━━━━━\n"
            f"{message}"
        )

        await self._enqueue_and_wait(
            chat_id    = target,
            text       = text,
            parse_mode = ParseMode.MARKDOWN,
            msg_type   = MessageType.STATUS,
        )

    # ── COMMANDES ADMIN ───────────────────────────────────────────

    async def _start_command_listener(self):
        """Lance le listener de commandes Telegram."""
        try:
            self._app = (
                Application.builder()
                .token(settings.TELEGRAM_TOKEN)
                .build()
            )

            # Enregistrement des commandes
            self._app.add_handler(
                CommandHandler("start",  self._cmd_start)
            )
            self._app.add_handler(
                CommandHandler("status", self._cmd_status)
            )
            self._app.add_handler(
                CommandHandler("report", self._cmd_report)
            )
            self._app.add_handler(
                CommandHandler("pause",  self._cmd_pause)
            )
            self._app.add_handler(
                CommandHandler("resume", self._cmd_resume)
            )
            self._app.add_handler(
                CommandHandler("params", self._cmd_params)
            )

            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(
                drop_pending_updates = True,
            )

            logger.info("✅ Listener commandes Telegram actif")

        except Exception as e:
            logger.error(f"Command listener error: {e}")

    async def _cmd_start(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return
        await update.message.reply_text(
            "🤖 *Bot Trading SMC/ICT*\n\n"
            "Commandes disponibles :\n"
            "/status — État du bot\n"
            "/report — Rapport de performance\n"
            "/pause  — Mettre en pause\n"
            "/resume — Reprendre\n"
            "/params — Paramètres actuels",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_status(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return

        uptime = datetime.utcnow() - self._start_time
        hours  = int(uptime.total_seconds() // 3600)
        mins   = int((uptime.total_seconds() % 3600) // 60)

        status_text = (
            f"🤖 *Statut du bot*\n"
            f"━━━━━━━━━━━━━━\n"
            f"{'⏸️ EN PAUSE' if self._is_paused else '▶️ ACTIF'}\n"
            f"⏱️ Uptime    : {hours}h {mins}min\n"
            f"🎯 Min score : {settings.MIN_CONFLUENCE_SCORE}/100\n"
            f"📐 Min R:R   : {settings.MIN_RR_RATIO}\n"
            f"💰 Risque    : {settings.RISK_PER_TRADE}%\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        await update.message.reply_text(
            status_text, parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_report(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return
        if self._on_report:
            await self._on_report()
        else:
            await update.message.reply_text(
                "📊 Rapport en cours de génération..."
            )

    async def _cmd_pause(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return
        self._is_paused = True
        if self._on_pause:
            await self._on_pause()
        await update.message.reply_text(
            "⏸️ Bot mis en pause — aucun nouveau signal ne sera émis."
        )
        logger.warning("⏸️ Bot mis en pause par l'admin Telegram")

    async def _cmd_resume(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return
        self._is_paused = False
        if self._on_resume:
            await self._on_resume()
        await update.message.reply_text(
            "▶️ Bot relancé — analyse reprend normalement."
        )
        logger.info("▶️ Bot relancé par l'admin Telegram")

    async def _cmd_params(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ):
        if not self._is_admin(update):
            return
        params_text = (
            f"⚙️ *Paramètres actifs*\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 Score min    : {settings.MIN_CONFLUENCE_SCORE}\n"
            f"📐 R:R min      : {settings.MIN_RR_RATIO}\n"
            f"📊 ATR × SL     : {settings.ATR_SL_MULTIPLIER}\n"
            f"🎯 TP1 ratio    : {settings.TP1_RATIO}\n"
            f"🎯 TP2 ratio    : {settings.TP2_RATIO}\n"
            f"🎯 TP3 ratio    : {settings.TP3_RATIO}\n"
            f"💰 Risque/trade : {settings.RISK_PER_TRADE}%\n"
            f"📉 Max DD/jour  : {settings.MAX_DAILY_LOSS}%\n"
            f"🔁 Max trades   : {settings.MAX_OPEN_TRADES}\n"
        )
        await update.message.reply_text(
            params_text, parse_mode=ParseMode.MARKDOWN
        )

    def _is_admin(self, update: Update) -> bool:
        """Vérifie si l'utilisateur est l'admin configuré."""
        if not self._admin_id:
            return True   # Pas d'admin configuré = tous autorisés
        return str(update.effective_user.id) == str(self._admin_id)

    # ── FILE D'ATTENTE ET WORKER ──────────────────────────────────

    async def _enqueue_and_wait(
        self,
        chat_id:     str,
        text:        str,
        parse_mode:  str = ParseMode.MARKDOWN,
        reply_markup = None,
        msg_type:    MessageType = MessageType.STATUS,
    ) -> int | None:
        """
        Ajoute un message à la file d'attente et attend le résultat.
        Retourne l'ID du message envoyé.
        """
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        await self._queue.put({
            "chat_id":      chat_id,
            "text":         text,
            "parse_mode":   parse_mode,
            "reply_markup": reply_markup,
            "msg_type":     msg_type,
            "future":       future,
        })

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Telegram message timeout ({msg_type.value})")
            return None

    async def _message_worker(self):
        """
        Worker qui consomme la file d'attente et envoie les messages.
        Gère le rate limiting et les retries automatiquement.
        """
        logger.info("📬 Telegram message worker démarré")

        while True:
            try:
                item = await self._queue.get()
                future    = item.pop("future")
                msg_type  = item.pop("msg_type")

                msg_id = await self._send_with_retry(**item)

                if not future.done():
                    future.set_result(msg_id)

                # Rate limiting
                await asyncio.sleep(self.RATE_LIMIT)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message worker error: {e}")

    async def _send_with_retry(
        self,
        chat_id:      str,
        text:         str,
        parse_mode:   str = ParseMode.MARKDOWN,
        reply_markup  = None,
        **kwargs,
    ) -> int | None:
        """
        Envoie un message avec retry automatique.
        Gère les erreurs RetryAfter (flood control Telegram).
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                msg = await self._bot.send_message(
                    chat_id      = chat_id,
                    text         = text,
                    parse_mode   = parse_mode,
                    reply_markup = reply_markup,
                    disable_web_page_preview = True,
                )
                return msg.message_id

            except RetryAfter as e:
                wait = e.retry_after + 1
                logger.warning(
                    f"Telegram flood control — attente {wait}s "
                    f"(tentative {attempt}/{self.MAX_RETRIES})"
                )
                await asyncio.sleep(wait)

            except TelegramError as e:
                if attempt == self.MAX_RETRIES:
                    logger.error(
                        f"Telegram send_message FAILED après "
                        f"{self.MAX_RETRIES} tentatives: {e}"
                    )
                    return None
                logger.warning(
                    f"Telegram error (tentative {attempt}): {e} — "
                    f"retry dans {self.RETRY_DELAY}s"
                )
                await asyncio.sleep(self.RETRY_DELAY)

        return None

    async def _edit_message(
        self,
        chat_id:    str,
        message_id: int,
        text:       str,
    ) -> bool:
        """
        Édite un message existant.
        Retourne True si réussi, False sinon.
        """
        try:
            await self._bot.edit_message_text(
                chat_id    = chat_id,
                message_id = message_id,
                text       = text,
                parse_mode = ParseMode.MARKDOWN,
                disable_web_page_preview = True,
            )
            return True
        except TelegramError as e:
            logger.warning(f"Edit message failed: {e}")
            return False

    # ── CONSTRUCTEUR DE MESSAGES ──────────────────────────────────

    @staticmethod
    def _build_signal_message(
        instrument: str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp1:        float,
        tp2:        float,
        tp3:        float,
        score:      int,
        quality:    str,
        session:    str | None,
    ) -> str:
        """Message de signal structuré (fallback si Groq indisponible)."""
        dir_emoji  = "🟢📈" if direction == "bullish" else "🔴📉"
        dir_txt    = "LONG" if direction == "bullish" else "SHORT"
        qual_emoji = {"excellent": "💎", "good": "⭐", "acceptable": "✅"}.get(quality, "✅")
        sess_txt   = (session or "MARKET").upper()

        risk_pips  = abs(entry - sl)
        rr1        = abs(tp1 - entry) / risk_pips if risk_pips > 0 else 0
        rr2        = abs(tp2 - entry) / risk_pips if risk_pips > 0 else 0
        rr3        = abs(tp3 - entry) / risk_pips if risk_pips > 0 else 0

        return (
            f"{dir_emoji} *{instrument} — {dir_txt}* {qual_emoji}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"📍 Entrée  : `{entry}`\n"
            f"🛡️ SL      : `{sl}`\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🎯 TP1     : `{tp1}` *(R:R {rr1:.1f})*\n"
            f"🎯 TP2     : `{tp2}` *(R:R {rr2:.1f})*\n"
            f"🎯 TP3     : `{tp3}` *(R:R {rr3:.1f})*\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"📊 Score   : *{score}/100*\n"
            f"🕐 Session : {sess_txt}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"💡 _Clôture 50% à TP1 — SL → BE automatique_\n"
            f"⚠️ _Trading à risque — Respectez votre money management_"
        )

    # ── UTILITAIRES ───────────────────────────────────────────────

    @staticmethod
    def _progress_bar(
        value: float,
        max_val: float,
        length: int = 10,
    ) -> str:
        """Génère une barre de progression textuelle."""
        filled  = int(round(value / max_val * length))
        filled  = max(0, min(length, filled))
        empty   = length - filled
        return f"[{'█' * filled}{'░' * empty}]"
