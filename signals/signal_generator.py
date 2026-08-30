# signal_generator.py
# ─────────────────────────────────────────────────────────────────
#  Générateur de signaux — Chef d'orchestre du système
#
#  Pipeline d'analyse complet pour un instrument :
#
#   1. Filtres pré-analyse   (session, news, anti-spam)
#   2. Fetch multi-TF        (H4, H1, M15, M5)
#   3. Indicateurs           (RSI, MACD, EMA, ATR sur H1)
#   4. SMC Détection         (BOS, CHoCH, OB, FVG, Liquidité)
#      → sur H4, H1, M15 séparément
#   5. Direction             (auto depuis H4)
#   6. Scoring confluence    (/100 → rejeté si < MIN_SCORE)
#   7. TP/SL Calcul          (ATR + Fibonacci + SMC)
#   8. Validation entrée     (R:R, OB, FVG, confirmation M5)
#   9. Sauvegarde DB
#  10. Formatage message     (Groq optionnel)
# ─────────────────────────────────────────────────────────────────

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from groq import Groq
from loguru import logger

from config import settings
from config.instruments import get_instrument, InstrumentConfig
from core import (
    TechnicalIndicators,
    SMCDetector,
    ScoringEngine,
    TPSLCalculator,
    EntryValidator,
)
from core.smc_detector import SMCResult
from data.data_manager import DataManager
from database.db_manager import DatabaseManager
from database.models import SignalDirection
from .filters import SignalFilters, FilterResult


@dataclass
class GeneratedSignal:
    # ── Identification ────────────────────────────────────────────
    instrument:     str
    direction:      str           # "bullish" | "bearish"
    session:        str | None

    # ── Prix ──────────────────────────────────────────────────────
    entry:          float
    sl:             float
    tp1:            float
    tp2:            float
    tp3:            float

    # ── Risque ────────────────────────────────────────────────────
    rr_ratio:       float
    risk_pips:      float

    # ── Score ─────────────────────────────────────────────────────
    score:          int
    score_detail:   dict
    quality:        str           # "excellent" | "good" | "acceptable"

    # ── Contexte SMC ──────────────────────────────────────────────
    smc_details:    dict = field(default_factory=dict)

    # ── Indicateurs ───────────────────────────────────────────────
    rsi:            float = 0.0
    atr:            float = 0.0

    # ── Message formaté pour Telegram ─────────────────────────────
    telegram_message: str = ""

    # ── DB ────────────────────────────────────────────────────────
    signal_id:      int | None = None

    # ── Timestamp ─────────────────────────────────────────────────
    generated_at:   datetime = field(default_factory=datetime.utcnow)


class SignalGenerator:
    """
    Orchestre l'analyse complète et génère des signaux de trading.
    Un signal n'est généré que si TOUS les critères sont remplis.
    """

    def __init__(
        self,
        data: DataManager,
        db:   DatabaseManager,
    ):
        self._data     = data
        self._db       = db
        self._filters  = SignalFilters()
        self._smc      = SMCDetector(swing_lookback=5)
        self._scoring  = ScoringEngine()
        self._tpsl     = TPSLCalculator()
        self._validator = EntryValidator()
        self._groq     = Groq(api_key=settings.GROQ_API_KEY)

    # ── ANALYSE D'UN INSTRUMENT ───────────────────────────────────

    async def analyze(self, symbol: str) -> GeneratedSignal | None:
        """
        Pipeline complet pour un instrument.
        Retourne un GeneratedSignal ou None si aucun signal.

        Args:
            symbol: Ex "XAUUSD", "BTCUSD", "EURUSD"
        """
        instrument_cfg = get_instrument(symbol)
        start_time     = datetime.utcnow()

        logger.info(f"🔍 Analyse en cours — {symbol}")

        # ── Étape 1 : Filtres pré-analyse ─────────────────────────
        signals_today = await self._db.count_signals_today(symbol)
        filter_result = await self._filters.check(
            instrument     = symbol,
            instrument_cfg = instrument_cfg,
            signals_today  = signals_today,
        )

        if not filter_result.passed:
            logger.debug(
                f"⛔ {symbol} bloqué — {filter_result.blocked_by}"
            )
            return None

        # ── Étape 2 : Fetch multi-TF ──────────────────────────────
        try:
            tf_data = await self._data.get_multi_tf(
                symbol      = symbol,
                output_size = 200,
            )
        except Exception as e:
            logger.error(f"Fetch data error {symbol}: {e}")
            return None

        df_h4   = tf_data.get("4h")
        df_h1   = tf_data.get("1h")
        df_m15  = tf_data.get("15min")
        df_m5   = tf_data.get("5min")

        # Vérification données suffisantes
        for tf, df in [("4h", df_h4), ("1h", df_h1), ("15min", df_m15)]:
            if df is None or len(df) < 60:
                logger.warning(f"Données insuffisantes {symbol} {tf}")
                return None

        # ── Étape 3 : Indicateurs techniques (sur H1) ─────────────
        try:
            indicators = TechnicalIndicators.analyze(df_h1)
        except Exception as e:
            logger.error(f"Indicators error {symbol}: {e}")
            return None

        # ── Étape 4 : Détection SMC sur chaque timeframe ──────────
        try:
            # H4 — Direction principale
            smc_h4 = self._smc.analyze(df_h4, direction="auto")

            # Direction déterminée par H4
            direction = smc_h4.current_trend
            if direction == "neutral":
                logger.debug(f"📊 {symbol} — H4 neutre, pas de signal")
                return None

            # H1 — Confirmation
            smc_h1 = self._smc.analyze(df_h1, direction=direction)

            # M15 — Setup d'entrée
            smc_m15 = self._smc.analyze(df_m15, direction=direction)

        except Exception as e:
            logger.error(f"SMC detection error {symbol}: {e}")
            return None

        # ── Étape 5 : Scoring de confluence ───────────────────────
        score_result = self._scoring.calculate(
            smc_h4     = smc_h4,
            smc_h1     = smc_h1,
            smc_m15    = smc_m15,
            indicators = indicators,
            direction  = direction,
            min_score  = settings.MIN_CONFLUENCE_SCORE,
        )

        logger.info(
            f"📊 {symbol} {direction.upper()} — "
            f"Score: {score_result.total}/100 "
            f"({'✅ VALIDE' if score_result.is_valid else '❌ REJETÉ'})"
        )

        if not score_result.is_valid:
            return None

        # ── Étape 6 : Prix d'entrée actuel ────────────────────────
        try:
            current_price = await self._data.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Price fetch error {symbol}: {e}")
            return None

        # ── Étape 7 : Calcul TP/SL ────────────────────────────────
        tpsl = self._tpsl.calculate(
            entry          = current_price,
            direction      = direction,
            atr            = indicators.atr,
            order_block    = smc_m15.order_block,
            swing_highs    = smc_h1.swing_highs,
            swing_lows     = smc_h1.swing_lows,
            atr_multiplier = settings.ATR_SL_MULTIPLIER,
            tp1_ratio      = settings.TP1_RATIO,
            tp2_ratio      = settings.TP2_RATIO,
            tp3_ratio      = settings.TP3_RATIO,
        )

        # ── Étape 8 : Validation de l'entrée ──────────────────────
        validation = self._validator.validate(
            direction   = direction,
            entry       = current_price,
            sl_price    = tpsl.sl,
            tp2         = tpsl.tp2,
            df_m5       = df_m5,
            order_block = smc_m15.order_block,
            fvg         = smc_m15.fvg,
            min_rr      = settings.MIN_RR_RATIO,
        )

        if not validation.is_valid:
            logger.debug(
                f"❌ {symbol} — Validation échouée: "
                f"{validation.rejection_reason}"
            )
            return None

        # ── Étape 9 : Détails SMC pour la DB ──────────────────────
        smc_details = self._build_smc_details(smc_h4, smc_h1, smc_m15)

        # ── Étape 10 : Sauvegarde en base de données ───────────────
        db_direction = (
            SignalDirection.LONG
            if direction == "bullish"
            else SignalDirection.SHORT
        )

        try:
            signal_db = await self._db.save_signal({
                "instrument":          symbol,
                "direction":           db_direction,
                "timeframe":           "15min",
                "entry_price":         tpsl.entry,
                "sl_price":            tpsl.sl,
                "tp1_price":           tpsl.tp1,
                "tp2_price":           tpsl.tp2,
                "tp3_price":           tpsl.tp3,
                "rr_ratio":            tpsl.rr_tp2,
                "risk_pips":           abs(current_price - tpsl.sl) / instrument_cfg.pip_value,
                "reward_pips_tp1":     abs(tpsl.tp1 - tpsl.entry) / instrument_cfg.pip_value,
                "reward_pips_tp2":     abs(tpsl.tp2 - tpsl.entry) / instrument_cfg.pip_value,
                "reward_pips_tp3":     abs(tpsl.tp3 - tpsl.entry) / instrument_cfg.pip_value,
                "confluence_score":    score_result.total,
                "score_bos":           score_result.score_bos,
                "score_choch":         score_result.score_choch,
                "score_ob":            score_result.score_ob,
                "score_fvg":           score_result.score_fvg,
                "score_liquidity":     score_result.score_liquidity,
                "score_mtf_alignment": score_result.score_mtf,
                "score_rsi":           score_result.score_rsi,
                "score_macd":          score_result.score_macd,
                "smc_details":         smc_details,
                "rsi_value":           indicators.rsi,
                "macd_value":          indicators.macd,
                "macd_signal":         indicators.macd_signal,
                "atr_value":           indicators.atr,
                "ema20":               indicators.ema20,
                "ema50":               indicators.ema50,
                "session":             filter_result.session,
                "news_nearby":         filter_result.news_nearby,
            })
        except Exception as e:
            logger.error(f"DB save_signal error {symbol}: {e}")
            return None

        # ── Étape 11 : Formatage message Telegram (Groq) ──────────
        telegram_msg = await self._format_telegram_message(
            symbol       = symbol,
            direction    = direction,
            tpsl         = tpsl,
            score        = score_result.total,
            quality      = validation.quality,
            session      = filter_result.session,
            indicators   = indicators,
            smc_details  = smc_details,
        )

        # ── Résumé dans les logs ───────────────────────────────────
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.success(
            f"🚀 SIGNAL GÉNÉRÉ — {symbol} {direction.upper()} "
            f"@ {current_price} | Score: {score_result.total}/100 "
            f"| R:R {tpsl.rr_tp2:.2f} | Qualité: {validation.quality} "
            f"| {elapsed:.1f}s"
        )

        return GeneratedSignal(
            instrument      = symbol,
            direction       = direction,
            session         = filter_result.session,
            entry           = tpsl.entry,
            sl              = tpsl.sl,
            tp1             = tpsl.tp1,
            tp2             = tpsl.tp2,
            tp3             = tpsl.tp3,
            rr_ratio        = tpsl.rr_tp2,
            risk_pips       = abs(current_price - tpsl.sl) / instrument_cfg.pip_value,
            score           = score_result.total,
            score_detail    = {
                "bos":        score_result.score_bos,
                "choch":      score_result.score_choch,
                "ob":         score_result.score_ob,
                "fvg":        score_result.score_fvg,
                "liquidity":  score_result.score_liquidity,
                "mtf":        score_result.score_mtf,
                "rsi":        score_result.score_rsi,
                "macd":       score_result.score_macd,
            },
            quality          = validation.quality,
            smc_details      = smc_details,
            rsi              = indicators.rsi,
            atr              = indicators.atr,
            telegram_message = telegram_msg,
            signal_id        = signal_db.id,
        )

    # ── ANALYSE MULTI-INSTRUMENTS ─────────────────────────────────

    async def analyze_all(
        self,
        symbols: list[str] | None = None,
    ) -> list[GeneratedSignal]:
        """
        Analyse tous les instruments en parallèle.
        Retourne la liste des signaux générés (peut être vide).
        """
        from config.instruments import get_active_instruments
        if symbols is None:
            symbols = [i.symbol for i in get_active_instruments()]

        tasks = [
            asyncio.create_task(self.analyze(s))
            for s in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"analyze_all error {symbol}: {result}")
            elif result is not None:
                signals.append(result)

        if signals:
            logger.info(
                f"✅ Analyse complète — "
                f"{len(signals)}/{len(symbols)} signaux générés"
            )
        else:
            logger.info(
                f"📊 Analyse complète — "
                f"Aucun signal sur {len(symbols)} instruments"
            )

        return signals

    # ── FORMAT TELEGRAM (GROQ) ────────────────────────────────────

    async def _format_telegram_message(
        self,
        symbol:      str,
        direction:   str,
        tpsl:        object,
        score:       int,
        quality:     str,
        session:     str | None,
        indicators:  object,
        smc_details: dict,
    ) -> str:
        """
        Utilise Groq (Mixtral) pour générer un message Telegram
        professionnel et structuré.

        Groq est UNIQUEMENT utilisé pour la présentation du message.
        La décision de trading a déjà été prise par le moteur math.
        """
        emoji_dir   = "🟢📈 LONG" if direction == "bullish" else "🔴📉 SHORT"
        emoji_qual  = {"excellent": "💎", "good": "⭐", "acceptable": "✅"}.get(quality, "✅")
        session_txt = session.upper() if session else "MARKET"

        # Contexte structuré pour Groq
        prompt = f"""
Tu es un expert en trading SMC/ICT. Génère un message Telegram professionnel
pour ce signal. Utilise des emojis. Maximum 400 caractères hors tableau.

Signal:
- Instrument: {symbol}
- Direction: {direction.upper()} ({emoji_dir})
- Score confluence: {score}/100
- Qualité: {quality} {emoji_qual}
- Session: {session_txt}

Niveaux (déjà calculés, ne pas modifier):
- Entrée: {tpsl.entry}
- SL: {tpsl.sl}
- TP1: {tpsl.tp1} (R:R {tpsl.rr_tp1:.1f})
- TP2: {tpsl.tp2} (R:R {tpsl.rr_tp2:.1f})
- TP3: {tpsl.tp3} (R:R {tpsl.rr_tp3:.1f})

Contexte SMC: {', '.join(f'{k}={v}' for k, v in smc_details.items() if v)}
RSI: {indicators.rsi:.1f} | ATR: {indicators.atr:.4f}

Format souhaité:
1. Titre accrocheur (instrument + direction + emoji)
2. Tableau des niveaux (Entrée / SL / TP1 / TP2 / TP3)
3. Score et éléments SMC validés
4. Mention de la session
5. Disclaimer court (trading à risque)

Réponds UNIQUEMENT avec le message Telegram, rien d'autre.
"""

        try:
            loop     = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._groq.chat.completions.create(
                    model    = settings.GROQ_MODEL,
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens  = 500,
                    temperature = 0.7,
                ),
            )
            msg = response.choices[0].message.content.strip()
            logger.debug(f"✅ Message Groq généré ({len(msg)} chars)")
            return msg

        except Exception as e:
            logger.warning(f"Groq error (fallback manuel): {e}")
            # Fallback — message manuel si Groq échoue
            return self._fallback_message(
                symbol, direction, tpsl, score, quality, session
            )

    @staticmethod
    def _fallback_message(
        symbol: str, direction: str, tpsl, score: int,
        quality: str, session: str | None,
    ) -> str:
        """Message de fallback si Groq est indisponible."""
        dir_emoji = "🟢📈" if direction == "bullish" else "🔴📉"
        dir_txt   = "LONG" if direction == "bullish" else "SHORT"
        sess      = (session or "MARKET").upper()

        return (
            f"{dir_emoji} *{symbol} — {dir_txt}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"📍 Entrée : `{tpsl.entry}`\n"
            f"🛡️ SL     : `{tpsl.sl}`\n"
            f"🎯 TP1    : `{tpsl.tp1}` (R:R {tpsl.rr_tp1:.1f})\n"
            f"🎯 TP2    : `{tpsl.tp2}` (R:R {tpsl.rr_tp2:.1f})\n"
            f"🎯 TP3    : `{tpsl.tp3}` (R:R {tpsl.rr_tp3:.1f})\n"
            f"━━━━━━━━━━━━━━\n"
            f"📊 Score  : {score}/100 | Qualité: {quality}\n"
            f"🕐 Session: {sess}\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚠️ _Trading à risque — Gérez votre capital._"
        )

    # ── UTILITAIRES ───────────────────────────────────────────────

    @staticmethod
    def _build_smc_details(
        smc_h4: SMCResult,
        smc_h1: SMCResult,
        smc_m15: SMCResult,
    ) -> dict:
        """Construit un dict JSON-serialisable des détails SMC."""
        details: dict = {
            "trend_h4":  smc_h4.current_trend,
            "trend_h1":  smc_h1.current_trend,
            "trend_m15": smc_m15.current_trend,
        }

        if smc_m15.bos and smc_m15.bos.confirmed:
            details["bos_level"]     = smc_m15.bos.broken_level
            details["bos_direction"] = smc_m15.bos.direction

        if smc_h1.choch and smc_h1.choch.confirmed:
            details["choch_level"]     = smc_h1.choch.broken_level
            details["choch_direction"] = smc_h1.choch.direction

        if smc_m15.order_block and smc_m15.order_block.valid:
            details["ob_high"] = smc_m15.order_block.zone_high
            details["ob_low"]  = smc_m15.order_block.zone_low
            details["ob_mid"]  = smc_m15.order_block.zone_mid

        if smc_m15.fvg and smc_m15.fvg.valid:
            details["fvg_high"] = smc_m15.fvg.gap_high
            details["fvg_low"]  = smc_m15.fvg.gap_low
            details["fvg_in_zone"] = smc_m15.fvg.in_current_zone

        if smc_h1.liquidity:
            details["ssl_swept"] = smc_h1.liquidity.recent_ssl_swept
            details["bsl_swept"] = smc_h1.liquidity.recent_bsl_swept
            if smc_h1.liquidity.sweep_level:
                details["sweep_level"] = smc_h1.liquidity.sweep_level

        return details
