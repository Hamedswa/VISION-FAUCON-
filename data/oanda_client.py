# data/oanda_client.py
# ─────────────────────────────────────────────────────────────────
#  Client OANDA — Exécution ordres Forex + XAU/USD
#  Utilise oandapyV20 (API REST officielle OANDA)
#
#  Responsabilités :
#   • Gestion du compte (solde, équité, marge)
#   • Prix en temps réel
#   • Placement / modification / annulation d'ordres
#   • Gestion des positions ouvertes
# ─────────────────────────────────────────────────────────────────

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import oandapyV20
import oandapyV20.endpoints.accounts    as accounts_ep
import oandapyV20.endpoints.instruments as instruments_ep
import oandapyV20.endpoints.orders      as orders_ep
import oandapyV20.endpoints.trades      as trades_ep
import oandapyV20.endpoints.positions   as positions_ep
import oandapyV20.endpoints.pricing     as pricing_ep
from oandapyV20.exceptions import V20Error
from loguru import logger

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
class OandaOrder:
    order_id:  str
    trade_id:  str | None
    instrument: str
    direction:  str          # "buy" | "sell"
    units:      float
    entry:      float | None
    sl:         float | None
    tp:         float | None
    status:     str          # "PENDING" | "FILLED" | "CANCELLED"


class OandaClient:
    """
    Wrapper async autour de oandapyV20.
    Toutes les opérations bloquantes sont exécutées dans un thread pool
    pour ne pas bloquer la boucle asyncio principale.
    """

    def __init__(self):
        self._api = oandapyV20.API(
            access_token = settings.OANDA_API_KEY,
            environment  = settings.OANDA_ENVIRONMENT,
        )
        self._account_id = settings.OANDA_ACCOUNT_ID

    # ── Exécution thread-safe ──────────────────────────────────────

    async def _run(self, endpoint):
        """
        Exécute un endpoint oandapyV20 dans un thread pool
        pour rester non-bloquant.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._api.request(endpoint)
        )

    # ── COMPTE ────────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        """Retourne les infos du compte (solde, équité, marge)."""
        try:
            ep   = accounts_ep.AccountDetails(self._account_id)
            data = await self._run(ep)
            acc  = data["account"]

            return AccountInfo(
                balance     = float(acc["balance"]),
                equity      = float(acc["NAV"]),
                margin_used = float(acc["marginUsed"]),
                margin_free = float(acc["marginAvailable"]),
                unrealized  = float(acc["unrealizedPL"]),
                currency    = acc["currency"],
            )
        except V20Error as e:
            logger.error(f"OANDA get_account error: {e}")
            raise

    async def get_balance(self) -> float:
        """Retourne uniquement le solde du compte."""
        acc = await self.get_account()
        return acc.balance

    # ── PRIX EN TEMPS RÉEL ────────────────────────────────────────

    async def get_price(self, instrument: str) -> dict:
        """
        Retourne le bid/ask actuel d'un instrument OANDA.
        instrument: ex "XAU_USD", "EUR_USD"
        """
        try:
            ep   = pricing_ep.PricingInfo(
                accountID  = self._account_id,
                params     = {"instruments": instrument},
            )
            data = await self._run(ep)
            p    = data["prices"][0]
            return {
                "instrument": instrument,
                "bid":        float(p["bids"][0]["price"]),
                "ask":        float(p["asks"][0]["price"]),
                "mid":        (float(p["bids"][0]["price"]) + float(p["asks"][0]["price"])) / 2,
                "tradeable":  p["tradeable"],
                "timestamp":  p["time"],
            }
        except V20Error as e:
            logger.error(f"OANDA get_price({instrument}) error: {e}")
            raise

    # ── POSITIONS OUVERTES ────────────────────────────────────────

    async def get_open_trades(self) -> list[dict]:
        """Retourne tous les trades ouverts sur le compte."""
        try:
            ep   = trades_ep.OpenTrades(self._account_id)
            data = await self._run(ep)
            return data.get("trades", [])
        except V20Error as e:
            logger.error(f"OANDA get_open_trades error: {e}")
            raise

    async def get_open_trade(self, trade_id: str) -> dict | None:
        """Retourne un trade ouvert par son ID."""
        try:
            ep   = trades_ep.TradeDetails(self._account_id, trade_id)
            data = await self._run(ep)
            return data.get("trade")
        except V20Error:
            return None

    async def count_open_trades(self) -> int:
        """Nombre de trades actuellement ouverts."""
        trades = await self.get_open_trades()
        return len(trades)

    # ── PLACEMENT D'ORDRES ────────────────────────────────────────

    async def place_market_order(
        self,
        instrument:   str,
        direction:    str,         # "buy" | "sell"
        units:        float,
        sl_price:     float,
        tp_price:     float | None = None,
    ) -> OandaOrder:
        """
        Place un ordre Market avec SL/TP.

        units: positif pour buy, négatif pour sell (géré automatiquement).
        """
        signed_units = units if direction == "buy" else -units

        order_body = {
            "order": {
                "type":       "MARKET",
                "instrument": instrument,
                "units":      str(round(signed_units, 2)),
                "stopLossOnFill": {
                    "price":       str(round(sl_price, 5)),
                    "timeInForce": "GTC",
                },
            }
        }

        if tp_price:
            order_body["order"]["takeProfitOnFill"] = {
                "price":       str(round(tp_price, 5)),
                "timeInForce": "GTC",
            }

        try:
            ep   = orders_ep.OrderCreate(self._account_id, data=order_body)
            data = await self._run(ep)

            fill = data.get("orderFillTransaction", {})
            created = data.get("orderCreateTransaction", {})

            order_id = created.get("id", "")
            trade_id = fill.get("tradeOpened", {}).get("tradeID")
            fill_price = float(fill.get("price", 0)) if fill.get("price") else None

            logger.info(
                f"✅ OANDA ordre placé — {instrument} {direction} "
                f"units={units} | Trade ID: {trade_id}"
            )

            return OandaOrder(
                order_id   = order_id,
                trade_id   = trade_id,
                instrument = instrument,
                direction  = direction,
                units      = units,
                entry      = fill_price,
                sl         = sl_price,
                tp         = tp_price,
                status     = "FILLED" if trade_id else "PENDING",
            )

        except V20Error as e:
            logger.error(f"OANDA place_market_order error: {e}")
            raise

    async def place_limit_order(
        self,
        instrument: str,
        direction:  str,
        units:      float,
        price:      float,
        sl_price:   float,
        tp_price:   float | None = None,
        expiry_utc: datetime | None = None,
    ) -> OandaOrder:
        """Place un ordre limite avec expiration optionnelle."""
        signed_units = units if direction == "buy" else -units

        order_body: dict = {
            "order": {
                "type":        "LIMIT",
                "instrument":  instrument,
                "units":       str(round(signed_units, 2)),
                "price":       str(round(price, 5)),
                "timeInForce": "GTC" if not expiry_utc else "GTD",
                "stopLossOnFill": {
                    "price":       str(round(sl_price, 5)),
                    "timeInForce": "GTC",
                },
            }
        }

        if tp_price:
            order_body["order"]["takeProfitOnFill"] = {
                "price":       str(round(tp_price, 5)),
                "timeInForce": "GTC",
            }

        if expiry_utc:
            order_body["order"]["gtdTime"] = expiry_utc.strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )

        try:
            ep   = orders_ep.OrderCreate(self._account_id, data=order_body)
            data = await self._run(ep)
            created = data.get("orderCreateTransaction", {})
            order_id = created.get("id", "")

            logger.info(
                f"✅ OANDA limite placé — {instrument} {direction} "
                f"@ {price} | Order ID: {order_id}"
            )

            return OandaOrder(
                order_id   = order_id,
                trade_id   = None,
                instrument = instrument,
                direction  = direction,
                units      = units,
                entry      = price,
                sl         = sl_price,
                tp         = tp_price,
                status     = "PENDING",
            )

        except V20Error as e:
            logger.error(f"OANDA place_limit_order error: {e}")
            raise

    # ── MODIFICATION ET CLÔTURE ───────────────────────────────────

    async def modify_sl(self, trade_id: str, new_sl: float) -> bool:
        """Modifie le Stop Loss d'un trade ouvert."""
        try:
            body = {"stopLoss": {"price": str(round(new_sl, 5)), "timeInForce": "GTC"}}
            ep   = trades_ep.TradeCRCDO(self._account_id, trade_id, data=body)
            await self._run(ep)
            logger.info(f"✅ SL modifié — Trade {trade_id} → {new_sl}")
            return True
        except V20Error as e:
            logger.error(f"OANDA modify_sl error: {e}")
            return False

    async def close_trade(self, trade_id: str, partial_units: float | None = None) -> dict:
        """
        Clôture un trade (totalement ou partiellement).
        partial_units: si fourni, clôture seulement ce nombre d'unités.
        """
        try:
            if partial_units:
                body = {"units": str(round(partial_units, 2))}
            else:
                body = {"units": "ALL"}

            ep   = trades_ep.TradeClose(self._account_id, trade_id, data=body)
            data = await self._run(ep)
            fill = data.get("orderFillTransaction", {})

            logger.info(
                f"✅ Trade {trade_id} clôturé — "
                f"P&L: {fill.get('pl', 'N/A')}"
            )
            return fill

        except V20Error as e:
            logger.error(f"OANDA close_trade({trade_id}) error: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre en attente."""
        try:
            ep = orders_ep.OrderCancel(self._account_id, order_id)
            await self._run(ep)
            logger.info(f"✅ Ordre {order_id} annulé")
            return True
        except V20Error as e:
            logger.error(f"OANDA cancel_order error: {e}")
            return False

    # ── SIZING ────────────────────────────────────────────────────

    async def calculate_units(
        self,
        instrument:    str,
        risk_usd:      float,
        sl_pips:       float,
        pip_value_usd: float = 10.0,
    ) -> float:
        """
        Calcule le nombre d'unités pour un risque donné.

        units = risk_usd / (sl_pips × pip_value_per_unit)
        """
        if sl_pips <= 0:
            return 0.0

        units = risk_usd / (sl_pips * pip_value_usd / 10000)
        return round(units, 0)
