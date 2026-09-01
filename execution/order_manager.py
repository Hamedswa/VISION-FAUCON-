# execution/order_manager.py
# ─────────────────────────────────────────────────────────────────
#  Gestionnaire des ordres — Version MT5 (Exness) + Binance
#
#  Responsabilités :
#   • Placement des ordres (route vers MT5 ou CCXT)
#   • Clôture partielle à TP1 (50% de la position)
#   • Déplacement SL au breakeven après TP1
#   • Trailing stop actif
#   • Monitoring des trades ouverts (loop 30s)
#   • Mise à jour base de données après chaque événement
# ─────────────────────────────────────────────────────────────────

import asyncio
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from config import settings
from config.instruments import get_instrument
from database.db_manager import DatabaseManager
from database.models import SignalStatus, TradeResult
from data.data_manager import DataManager
from .risk_manager import RiskManager, PositionSize


@dataclass
class OrderResult:
    success:        bool
    signal_id:      int
    trade_id:       int | None
    broker_id:      str | None
    entry_price:    float | None
    error:          str | None


class OrderManager:
    """
    Exécute les ordres et surveille les positions ouvertes.
    Travaille en coordination avec RiskManager et DataManager.
    """

    def __init__(
        self,
        db:   DatabaseManager,
        data: DataManager,
        risk: RiskManager,
    ):
        self._db   = db
        self._data = data
        self._risk = risk
        self._monitoring_task: asyncio.Task | None = None

    # ── DÉMARRAGE / ARRÊT ─────────────────────────────────────────

    async def start(self):
        """Lance la boucle de monitoring des trades ouverts."""
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop()
        )
        logger.info("🔄 OrderManager démarré — monitoring actif")

    async def stop(self):
        """Arrête le monitoring proprement."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ OrderManager arrêté")

    # ── PLACEMENT D'UN ORDRE ──────────────────────────────────────

    async def execute_signal(
        self,
        signal_id:  int,
        instrument: str,
        direction:  str,        # "bullish" | "bearish"
        entry:      float,
        sl_price:   float,
        tp1:        float,
        tp2:        float,
        tp3:        float,
    ) -> OrderResult:
        """
        Exécute un signal validé :
          1. Vérifie le risque
          2. Calcule le sizing
          3. Place l'ordre sur MT5 ou Binance
          4. Enregistre le trade en base
        """
        instrument_cfg = get_instrument(instrument)
        broker         = instrument_cfg.broker

        # ── Balance actuelle ──────────────────────────────────────
        try:
            balance = await self._data.get_balance(broker)
        except Exception as e:
            return OrderResult(
                success=False, signal_id=signal_id,
                trade_id=None, broker_id=None,
                entry_price=None,
                error=f"Impossible de récupérer le solde: {e}"
            )

        # ── Vérification du risque ────────────────────────────────
        risk_check = await self._risk.check(instrument, direction, balance)
        if not risk_check.approved:
            await self._db.update_signal_status(
                signal_id, SignalStatus.CANCELLED,
                extra={"pnl_usd": 0}
            )
            return OrderResult(
                success=False, signal_id=signal_id,
                trade_id=None, broker_id=None,
                entry_price=None,
                error=f"Risk check refusé: {risk_check.rejection_reason}"
            )

        # ── Sizing ────────────────────────────────────────────────
        sizing = self._risk.calculate_position_size(
            balance        = balance,
            entry          = entry,
            sl_price       = sl_price,
            instrument_cfg = instrument_cfg,
        )

        if not sizing.valid:
            return OrderResult(
                success=False, signal_id=signal_id,
                trade_id=None, broker_id=None,
                entry_price=None,
                error=f"Sizing invalide: {sizing.reason}"
            )

        # ── Direction broker ──────────────────────────────────────
        broker_direction = "buy" if direction == "bullish" else "sell"

        # ── Placement selon le broker ─────────────────────────────
        try:
            if broker == "mt5":
                # ── Exness via MT5 ────────────────────────────────
                order = await self._data.get_mt5().place_market_order(
                    mt5_symbol = instrument_cfg.mt5_symbol,
                    direction  = broker_direction,
                    volume     = sizing.lot_size,
                    sl_price   = sl_price,
                    tp_price   = tp1,   # TP1 pour clôture auto à 50%
                    comment    = f"Bot#{signal_id}",
                )
                broker_order_id = str(order.ticket)
                broker_trade_id = str(order.ticket)
                actual_entry    = order.entry

            elif broker == "ccxt":
                # ── Binance via CCXT ──────────────────────────────
                order = await self._data.get_ccxt().place_market_order(
                    symbol    = instrument_cfg.ccxt_symbol,
                    direction = broker_direction,
                    amount    = sizing.lot_size,
                    sl_price  = sl_price,
                    tp_price  = tp1,
                )
                broker_order_id = order.order_id
                broker_trade_id = order.order_id
                actual_entry    = order.price or entry

            else:
                raise ValueError(f"Broker inconnu: {broker}")

        except Exception as e:
            logger.error(f"Erreur placement ordre {instrument}: {e}")
            await self._db.update_signal_status(
                signal_id, SignalStatus.CANCELLED
            )
            return OrderResult(
                success=False, signal_id=signal_id,
                trade_id=None, broker_id=None,
                entry_price=None,
                error=str(e)
            )

        # ── Enregistrement Trade en DB ────────────────────────────
        trade = await self._db.save_trade({
            "signal_id":       signal_id,
            "broker":          broker,
            "broker_order_id": broker_order_id,
            "broker_trade_id": broker_trade_id,
            "lot_size":        sizing.lot_size,
            "open_price":      actual_entry,
            "open_at":         datetime.utcnow(),
            "balance_before":  balance,
        })

        # ── Mise à jour signal → ACTIVE ───────────────────────────
        await self._db.update_signal_status(
            signal_id,
            SignalStatus.ACTIVE,
            extra={"actual_entry": actual_entry},
        )

        logger.info(
            f"🚀 Trade exécuté — {instrument} {direction} "
            f"@ {actual_entry} | SL: {sl_price} | TP1: {tp1} "
            f"| TP2: {tp2} | TP3: {tp3} "
            f"| Lots: {sizing.lot_size} "
            f"| Risque: {sizing.risk_usd:.2f} USD"
        )

        return OrderResult(
            success     = True,
            signal_id   = signal_id,
            trade_id    = trade.id,
            broker_id   = broker_trade_id,
            entry_price = actual_entry,
            error       = None,
        )

    # ── MONITORING DES TRADES OUVERTS ─────────────────────────────

    async def _monitoring_loop(self):
        """
        Boucle de surveillance des trades ouverts.
        S'exécute toutes les 30 secondes.
        """
        logger.info("👁️ Monitoring loop démarrée")

        while True:
            try:
                await self._check_all_active_trades()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            await asyncio.sleep(30)

    async def _check_all_active_trades(self):
        """Vérifie tous les trades actifs en DB."""
        active_signals = await self._db.get_active_signals()

        if not active_signals:
            return

        instruments = list({s.instrument for s in active_signals})
        prices      = await self._data.get_current_prices(instruments)

        for signal in active_signals:
            current_price = prices.get(signal.instrument, 0)
            if current_price == 0:
                continue
            try:
                await self._check_single_trade(signal, current_price)
            except Exception as e:
                logger.error(
                    f"Erreur check trade signal#{signal.id}: {e}"
                )

    async def _check_single_trade(self, signal, current_price: float):
        """
        Vérifie un trade individuel.
        Gère TP1, TP2, TP3, SL, breakeven et trailing.
        """
        direction = signal.direction.value.lower()
        is_long   = direction == "long"

        # ── SL ────────────────────────────────────────────────────
        if is_long and current_price <= signal.sl_price:
            await self._close_trade_on_sl(signal, current_price)
            return
        if not is_long and current_price >= signal.sl_price:
            await self._close_trade_on_sl(signal, current_price)
            return

        # ── TP3 ───────────────────────────────────────────────────
        if is_long and current_price >= signal.tp3_price:
            await self._close_trade_on_tp(signal, current_price, tp_level=3)
            return
        if not is_long and current_price <= signal.tp3_price:
            await self._close_trade_on_tp(signal, current_price, tp_level=3)
            return

        # ── TP2 ───────────────────────────────────────────────────
        if is_long and current_price >= signal.tp2_price:
            await self._close_trade_on_tp(signal, current_price, tp_level=2)
            return
        if not is_long and current_price <= signal.tp2_price:
            await self._close_trade_on_tp(signal, current_price, tp_level=2)
            return

        # ── TP1 → Breakeven + Clôture 50% ────────────────────────
        tp1_hit = (
            (is_long     and current_price >= signal.tp1_price) or
            (not is_long and current_price <= signal.tp1_price)
        )

        if tp1_hit and signal.status.value == "ACTIVE":
            await self._handle_tp1(signal, current_price)
            return

        # ── Breakeven actif ───────────────────────────────────────
        if signal.status.value == "TP1_HIT":
            be_price = self._risk.should_move_to_breakeven(
                direction     = "bullish" if is_long else "bearish",
                entry         = signal.actual_entry or signal.entry_price,
                current_price = current_price,
                tp1           = signal.tp1_price,
                current_sl    = signal.sl_price,
            )
            if be_price:
                await self._move_sl(signal, be_price, "breakeven")

    # ── HANDLERS TP / SL ──────────────────────────────────────────

    async def _handle_tp1(self, signal, current_price: float):
        """
        TP1 atteint :
          • Clôture 50% de la position
          • SL → Breakeven
          • Statut → TP1_HIT
        """
        logger.info(
            f"🎯 TP1 atteint — Signal#{signal.id} "
            f"{signal.instrument} @ {current_price}"
        )

        await self._partial_close_broker(signal, fraction=0.5)
        entry = signal.actual_entry or signal.entry_price
        await self._move_sl(signal, entry, "breakeven après TP1")

        await self._db.update_signal_status(
            signal.id, SignalStatus.TP1_HIT
        )

    async def _close_trade_on_tp(
        self, signal, current_price: float, tp_level: int
    ):
        """Clôture complète sur TP2 ou TP3."""
        pnl_pips = self._calc_pnl_pips(signal, current_price)
        pnl_usd  = self._calc_pnl_usd(signal, pnl_pips)

        status_map = {
            2: SignalStatus.TP2_HIT,
            3: SignalStatus.TP3_HIT,
        }

        logger.info(
            f"🏆 TP{tp_level} atteint — Signal#{signal.id} "
            f"{signal.instrument} @ {current_price} | "
            f"P&L: +{pnl_usd:.2f} USD"
        )

        await self._close_broker_position(signal)

        balance_after = await self._data.get_balance(
            get_instrument(signal.instrument).broker
        )

        await self._db.update_signal_status(
            signal.id,
            status_map[tp_level],
            extra={
                "exit_price":       current_price,
                "exit_at":          datetime.utcnow(),
                "pnl_pips":         pnl_pips,
                "pnl_usd":          pnl_usd,
                "duration_minutes": self._calc_duration(signal),
            }
        )

        if signal.trade:
            await self._db.close_trade(
                trade_id      = signal.trade.id,
                close_price   = current_price,
                result        = TradeResult.WIN,
                pnl_usd       = pnl_usd,
                pnl_pips      = pnl_pips,
                tp_level_hit  = tp_level,
                balance_after = balance_after,
            )

    async def _close_trade_on_sl(self, signal, current_price: float):
        """Clôture sur SL."""
        pnl_pips = self._calc_pnl_pips(signal, current_price)
        pnl_usd  = self._calc_pnl_usd(signal, pnl_pips)

        logger.warning(
            f"🔴 SL touché — Signal#{signal.id} "
            f"{signal.instrument} @ {current_price} | "
            f"P&L: {pnl_usd:.2f} USD"
        )

        balance_after = await self._data.get_balance(
            get_instrument(signal.instrument).broker
        )

        await self._close_broker_position(signal)

        result = (
            TradeResult.PARTIAL
            if signal.status.value == "TP1_HIT"
            else TradeResult.LOSS
        )

        await self._db.update_signal_status(
            signal.id,
            SignalStatus.SL_HIT,
            extra={
                "exit_price":       current_price,
                "exit_at":          datetime.utcnow(),
                "pnl_pips":         pnl_pips,
                "pnl_usd":          pnl_usd,
                "duration_minutes": self._calc_duration(signal),
            }
        )

        if signal.trade:
            await self._db.close_trade(
                trade_id      = signal.trade.id,
                close_price   = current_price,
                result        = result,
                pnl_usd       = pnl_usd,
                pnl_pips      = pnl_pips,
                balance_after = balance_after,
            )

    # ── ACTIONS BROKER ────────────────────────────────────────────

    async def _partial_close_broker(self, signal, fraction: float = 0.5):
        """Clôture partielle d'une position (50% à TP1)."""
        instrument_cfg = get_instrument(signal.instrument)

        try:
            if instrument_cfg.broker == "mt5":
                if signal.trade and signal.trade.broker_trade_id:
                    ticket   = int(signal.trade.broker_trade_id)
                    position = await self._data.get_mt5().get_position(ticket)

                    if position:
                        close_vol = round(position["volume"] * fraction, 2)
                        close_vol = max(instrument_cfg.min_lot, close_vol)

                        await self._data.get_mt5().close_partial(
                            ticket    = ticket,
                            symbol    = instrument_cfg.mt5_symbol,
                            direction = position["direction"],
                            volume    = close_vol,
                        )
                        logger.info(
                            f"✂️ Clôture partielle {fraction*100:.0f}% "
                            f"— {close_vol} lots | Ticket {ticket}"
                        )

            elif instrument_cfg.broker == "ccxt":
                if signal.trade and signal.trade.broker_trade_id:
                    pos = await self._data.get_ccxt().get_position(
                        instrument_cfg.ccxt_symbol
                    )
                    if pos:
                        amount = abs(float(pos.get("contracts", 0))) * fraction
                        direction = signal.direction.value.lower()
                        await self._data.get_ccxt().close_position(
                            symbol    = instrument_cfg.ccxt_symbol,
                            direction = "buy" if direction == "long" else "sell",
                            amount    = amount,
                        )

        except Exception as e:
            logger.error(f"Partial close error: {e}")

    async def _close_broker_position(self, signal):
        """Clôture complète d'une position."""
        instrument_cfg = get_instrument(signal.instrument)

        try:
            if instrument_cfg.broker == "mt5":
                if signal.trade and signal.trade.broker_trade_id:
                    ticket   = int(signal.trade.broker_trade_id)
                    position = await self._data.get_mt5().get_position(ticket)

                    if position:
                        await self._data.get_mt5().close_position(
                            ticket    = ticket,
                            symbol    = instrument_cfg.mt5_symbol,
                            direction = position["direction"],
                            volume    = position["volume"],
                        )

            elif instrument_cfg.broker == "ccxt":
                if signal.trade and signal.trade.broker_trade_id:
                    pos = await self._data.get_ccxt().get_position(
                        instrument_cfg.ccxt_symbol
                    )
                    if pos:
                        direction = signal.direction.value.lower()
                        await self._data.get_ccxt().close_position(
                            symbol    = instrument_cfg.ccxt_symbol,
                            direction = "buy" if direction == "long" else "sell",
                            amount    = abs(float(pos.get("contracts", 0))),
                        )

        except Exception as e:
            logger.error(f"Close position error: {e}")

    async def _move_sl(self, signal, new_sl: float, reason: str):
        """Déplace le SL sur le broker."""
        instrument_cfg = get_instrument(signal.instrument)

        try:
            if instrument_cfg.broker == "mt5":
                if signal.trade and signal.trade.broker_trade_id:
                    ticket = int(signal.trade.broker_trade_id)
                    await self._data.get_mt5().modify_sl_tp(
                        ticket  = ticket,
                        symbol  = instrument_cfg.mt5_symbol,
                        new_sl  = new_sl,
                    )

            logger.info(
                f"📍 SL déplacé ({reason}) — "
                f"Signal#{signal.id}: "
                f"{signal.sl_price} → {new_sl}"
            )

        except Exception as e:
            logger.error(f"Move SL error: {e}")

    # ── CALCULS P&L ───────────────────────────────────────────────

    @staticmethod
    def _calc_pnl_pips(signal, current_price: float) -> float:
        """Calcule le P&L en pips."""
        direction      = signal.direction.value.lower()
        entry          = signal.actual_entry or signal.entry_price
        instrument_cfg = get_instrument(signal.instrument)

        diff = (
            current_price - entry if direction == "long"
            else entry - current_price
        )
        pips = diff / instrument_cfg.pip_value
        return round(pips, 1)

    @staticmethod
    def _calc_pnl_usd(signal, pnl_pips: float) -> float:
        """Estime le P&L en USD."""
        instrument_cfg = get_instrument(signal.instrument)
        lot_size = signal.trade.lot_size if signal.trade else 0.01
        return round(pnl_pips * instrument_cfg.pip_value * lot_size, 2)

    @staticmethod
    def _calc_duration(signal) -> int:
        """Calcule la durée du trade en minutes."""
        if signal.trade and signal.trade.open_at:
            delta = datetime.utcnow() - signal.trade.open_at
            return int(delta.total_seconds() / 60)
        return 0
