# data/mt5_client.py
# ─────────────────────────────────────────────────────────────────
#  Client MetaTrader 5 — Exness
#  Remplace oanda_client.py
#
#  ⚠️  Fonctionne uniquement sur Windows avec MT5 installé
#
#  Responsabilités :
#   • Connexion à MT5 via login Exness
#   • Prix en temps réel
#   • Placement / modification / clôture d'ordres
#   • Informations du compte (solde, équité, marge)
# ─────────────────────────────────────────────────────────────────

import asyncio
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

import MetaTrader5 as mt5

from config import settings


@dataclass
class AccountInfo:
    balance:     float
    equity:      float
    margin_used: float
    margin_free: float
    unrealized:  float
    currency:    str


@dataclass
class MT5Order:
    ticket:     int
    symbol:     str
    direction:  str      # "buy" | "sell"
    volume:     float
    entry:      float
    sl:         float
    tp:         float
    status:     str      # "filled" | "pending" | "cancelled"


class MT5Client:
    """
    Client MetaTrader 5 pour Exness.
    Toutes les opérations bloquantes sont exécutées
    dans un thread pool pour ne pas bloquer asyncio.
    """

    def __init__(self):
        self._login    = settings.MT5_LOGIN
        self._password = settings.MT5_PASSWORD
        self._server   = settings.MT5_SERVER
        self._connected = False

    # ── CONNEXION ─────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Initialise et connecte MT5."""
        loop = asyncio.get_event_loop()

        def _connect():
            if not mt5.initialize():
                logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
                return False

            if not mt5.login(
                login    = self._login,
                password = self._password,
                server   = self._server,
            ):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

            return True

        self._connected = await loop.run_in_executor(None, _connect)

        if self._connected:
            info = mt5.account_info()
            logger.info(
                f"✅ MT5 connecté — Exness | "
                f"Login: {self._login} | "
                f"Balance: {info.balance:.2f} {info.currency}"
            )
        return self._connected

    async def disconnect(self):
        """Déconnecte MT5."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, mt5.shutdown)
        self._connected = False
        logger.info("🔌 MT5 déconnecté")

    async def _run(self, func, *args, **kwargs):
        """Exécute une fonction MT5 dans un thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    # ── COMPTE ────────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        """Retourne les infos du compte Exness."""
        info = await self._run(mt5.account_info)
        if info is None:
            raise RuntimeError(f"MT5 account_info failed: {mt5.last_error()}")

        return AccountInfo(
            balance     = float(info.balance),
            equity      = float(info.equity),
            margin_used = float(info.margin),
            margin_free = float(info.margin_free),
            unrealized  = float(info.profit),
            currency    = info.currency,
        )

    async def get_balance(self) -> float:
        acc = await self.get_account()
        return acc.balance

    # ── PRIX EN TEMPS RÉEL ────────────────────────────────────────

    async def get_price(self, mt5_symbol: str) -> dict:
        """
        Retourne le bid/ask actuel d'un symbole MT5.
        mt5_symbol: ex "XAUUSD", "EURUSDm"
        """
        tick = await self._run(mt5.symbol_info_tick, mt5_symbol)
        if tick is None:
            raise RuntimeError(
                f"MT5 symbol_info_tick({mt5_symbol}) failed: "
                f"{mt5.last_error()}"
            )

        return {
            "symbol": mt5_symbol,
            "bid":    float(tick.bid),
            "ask":    float(tick.ask),
            "mid":    (float(tick.bid) + float(tick.ask)) / 2,
            "time":   datetime.fromtimestamp(tick.time).isoformat(),
        }

    # ── PLACEMENT D'ORDRES ────────────────────────────────────────

    async def place_market_order(
        self,
        mt5_symbol: str,
        direction:  str,      # "buy" | "sell"
        volume:     float,
        sl_price:   float,
        tp_price:   float | None = None,
        comment:    str = "TradingBot",
    ) -> MT5Order:
        """
        Place un ordre Market sur MT5.

        direction: "buy" | "sell"
        volume:    Taille en lots (ex: 0.01)
        """
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL

        # Prix d'exécution
        tick = await self._run(mt5.symbol_info_tick, mt5_symbol)
        price = tick.ask if direction == "buy" else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_symbol,
            "volume":       round(volume, 2),
            "type":         order_type,
            "price":        price,
            "sl":           round(sl_price, 5),
            "deviation":    20,        # Slippage max en points
            "magic":        20240101,  # Identifiant du bot
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if tp_price:
            request["tp"] = round(tp_price, 5)

        result = await self._run(mt5.order_send, request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            retcode = result.retcode if result else "None"
            raise RuntimeError(
                f"MT5 order_send failed — retcode: {retcode} | "
                f"error: {error}"
            )

        logger.info(
            f"✅ MT5 ordre placé — {mt5_symbol} {direction} "
            f"vol={volume} @ {price} | Ticket: {result.order}"
        )

        return MT5Order(
            ticket    = result.order,
            symbol    = mt5_symbol,
            direction = direction,
            volume    = volume,
            entry     = float(price),
            sl        = sl_price,
            tp        = tp_price or 0.0,
            status    = "filled",
        )

    # ── MODIFICATION SL/TP ────────────────────────────────────────

    async def modify_sl_tp(
        self,
        ticket:   int,
        symbol:   str,
        new_sl:   float,
        new_tp:   float | None = None,
    ) -> bool:
        """Modifie le SL (et TP optionnel) d'un ordre ouvert."""

        # Récupère le TP actuel si non fourni
        if new_tp is None:
            position = await self._run(
                mt5.positions_get, ticket=ticket
            )
            if position:
                new_tp = position[0].tp

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "ticket":   ticket,
            "symbol":   symbol,
            "sl":       round(new_sl, 5),
            "tp":       round(new_tp or 0.0, 5),
        }

        result = await self._run(mt5.order_send, request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ MT5 SL modifié — Ticket {ticket} → SL: {new_sl}")
            return True

        logger.error(
            f"MT5 modify_sl_tp failed — "
            f"Ticket {ticket} | {mt5.last_error()}"
        )
        return False

    # ── CLÔTURE ───────────────────────────────────────────────────

    async def close_position(
        self,
        ticket:    int,
        symbol:    str,
        direction: str,    # direction du trade OUVERT
        volume:    float,
    ) -> dict:
        """Clôture une position ouverte (totalement)."""
        close_type = (
            mt5.ORDER_TYPE_SELL if direction == "buy"
            else mt5.ORDER_TYPE_BUY
        )

        tick = await self._run(mt5.symbol_info_tick, symbol)
        price = tick.bid if direction == "buy" else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       round(volume, 2),
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        20240101,
            "comment":      "TradingBot Close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = await self._run(mt5.order_send, request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"✅ MT5 position clôturée — "
                f"Ticket {ticket} @ {price}"
            )
            return {"ticket": ticket, "price": price, "success": True}

        logger.error(
            f"MT5 close_position failed — "
            f"Ticket {ticket} | {mt5.last_error()}"
        )
        return {"ticket": ticket, "price": price, "success": False}

    async def close_partial(
        self,
        ticket:    int,
        symbol:    str,
        direction: str,
        volume:    float,     # Volume partiel à fermer
    ) -> dict:
        """Clôture partielle d'une position (ex: 50% à TP1)."""
        return await self.close_position(ticket, symbol, direction, volume)

    # ── POSITIONS OUVERTES ────────────────────────────────────────

    async def get_open_positions(self) -> list[dict]:
        """Retourne toutes les positions ouvertes."""
        positions = await self._run(mt5.positions_get)
        if positions is None:
            return []

        result = []
        for p in positions:
            result.append({
                "ticket":    p.ticket,
                "symbol":    p.symbol,
                "direction": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                "volume":    p.volume,
                "open_price":p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "profit":    p.profit,
                "swap":      p.swap,
                "comment":   p.comment,
            })
        return result

    async def get_position(self, ticket: int) -> dict | None:
        """Retourne une position par son ticket."""
        positions = await self._run(mt5.positions_get, ticket=ticket)
        if not positions:
            return None
        p = positions[0]
        return {
            "ticket":    p.ticket,
            "symbol":    p.symbol,
            "direction": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume":    p.volume,
            "open_price":p.price_open,
            "current":   p.price_current,
            "sl":        p.sl,
            "tp":        p.tp,
            "profit":    p.profit,
        }

    async def count_open_positions(self) -> int:
        positions = await self._run(mt5.positions_get)
        return len(positions) if positions else 0

    # ── SIZING ────────────────────────────────────────────────────

    async def calculate_volume(
        self,
        symbol:    str,
        risk_usd:  float,
        sl_pips:   float,
        pip_value: float,
    ) -> float:
        """
        Calcule le volume en lots pour un risque donné.

        volume = risk_usd / (sl_pips × pip_value × 100000)
        """
        if sl_pips <= 0 or pip_value <= 0:
            return 0.01

        # Récupère les infos du symbole pour les limites
        info = await self._run(mt5.symbol_info, symbol)
        if info is None:
            return 0.01

        volume_usd = risk_usd / (sl_pips * pip_value)
        volume     = volume_usd / 100000.0   # Conversion en lots

        # Arrondi au volume_step
        step   = info.volume_step or 0.01
        volume = round(round(volume / step) * step, 2)

        # Limites min/max
        volume = max(info.volume_min or 0.01, min(info.volume_max or 100.0, volume))

        return volume

    # ── VALIDATION CONNEXION ──────────────────────────────────────

    async def is_connected(self) -> bool:
        """Vérifie que la connexion MT5 est active."""
        try:
            info = await self._run(mt5.account_info)
            return info is not None
        except Exception:
            return False
