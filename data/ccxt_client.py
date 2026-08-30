# data/ccxt_client.py
# ─────────────────────────────────────────────────────────────────
#  Client CCXT — Exécution ordres BTC/USD (crypto)
#  Supporte Binance, Bybit, Kraken...
#
#  Responsabilités :
#   • Prix crypto en temps réel
#   • Placement / annulation d'ordres crypto
#   • Gestion des positions futures (si applicable)
#   • Solde du compte exchange
# ─────────────────────────────────────────────────────────────────

import asyncio
import ccxt.async_support as ccxt
from dataclasses import dataclass
from loguru import logger

from config import settings


@dataclass
class CCXTOrder:
    order_id:   str
    symbol:     str
    direction:  str       # "buy" | "sell"
    amount:     float
    price:      float | None
    status:     str       # "open" | "closed" | "canceled"
    filled:     float
    cost:       float


class CCXTClient:
    """
    Client async pour l'exécution d'ordres crypto via CCXT.
    Utilise l'API async native de CCXT.
    """

    def __init__(self):
        exchange_class = getattr(ccxt, settings.CCXT_EXCHANGE, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{settings.CCXT_EXCHANGE}' non supporté par CCXT."
            )

        self._exchange: ccxt.Exchange = exchange_class({
            "apiKey":    settings.CCXT_API_KEY,
            "secret":    settings.CCXT_API_SECRET,
            "sandbox":   settings.CCXT_SANDBOX,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",   # Utilise futures (margin)
            },
        })

        if settings.CCXT_SANDBOX:
            self._exchange.set_sandbox_mode(True)

    # ── LIFECYCLE ─────────────────────────────────────────────────

    async def close(self):
        """Ferme la connexion CCXT."""
        await self._exchange.close()

    # ── COMPTE ────────────────────────────────────────────────────

    async def get_balance(self, currency: str = "USDT") -> float:
        """Retourne le solde disponible."""
        try:
            balance = await self._exchange.fetch_balance()
            return float(balance.get(currency, {}).get("free", 0.0))
        except Exception as e:
            logger.error(f"CCXT get_balance error: {e}")
            raise

    async def get_full_balance(self) -> dict:
        """Retourne le solde complet (toutes les devises)."""
        try:
            return await self._exchange.fetch_balance()
        except Exception as e:
            logger.error(f"CCXT get_full_balance error: {e}")
            raise

    # ── PRIX EN TEMPS RÉEL ────────────────────────────────────────

    async def get_price(self, symbol: str) -> dict:
        """
        Retourne le ticker complet d'un symbole.
        symbol: ex "BTC/USDT"
        """
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "bid":    float(ticker["bid"] or 0),
                "ask":    float(ticker["ask"] or 0),
                "mid":    float(ticker["last"] or 0),
                "volume": float(ticker["baseVolume"] or 0),
                "change": float(ticker["percentage"] or 0),
            }
        except Exception as e:
            logger.error(f"CCXT get_price({symbol}) error: {e}")
            raise

    # ── PLACEMENT D'ORDRES ────────────────────────────────────────

    async def place_market_order(
        self,
        symbol:    str,
        direction: str,
        amount:    float,
        sl_price:  float | None = None,
        tp_price:  float | None = None,
    ) -> CCXTOrder:
        """
        Place un ordre Market.
        symbol:    "BTC/USDT"
        direction: "buy" | "sell"
        amount:    Quantité en unités de base (ex: 0.01 BTC)
        """
        try:
            side   = direction.lower()
            params = {}

            # Certains exchanges supportent SL/TP dans les params
            if sl_price:
                params["stopLoss"]   = {"triggerPrice": sl_price}
            if tp_price:
                params["takeProfit"] = {"triggerPrice": tp_price}

            order = await self._exchange.create_order(
                symbol = symbol,
                type   = "market",
                side   = side,
                amount = amount,
                params = params,
            )

            logger.info(
                f"✅ CCXT ordre placé — {symbol} {direction} "
                f"qty={amount} | ID: {order['id']}"
            )

            return CCXTOrder(
                order_id  = str(order["id"]),
                symbol    = symbol,
                direction = direction,
                amount    = amount,
                price     = float(order.get("average") or order.get("price") or 0),
                status    = order["status"],
                filled    = float(order.get("filled", 0)),
                cost      = float(order.get("cost", 0)),
            )

        except Exception as e:
            logger.error(f"CCXT place_market_order error: {e}")
            raise

    async def place_limit_order(
        self,
        symbol:    str,
        direction: str,
        amount:    float,
        price:     float,
        sl_price:  float | None = None,
        tp_price:  float | None = None,
    ) -> CCXTOrder:
        """Place un ordre limite."""
        try:
            side   = direction.lower()
            params = {}

            if sl_price:
                params["stopLoss"]   = {"triggerPrice": sl_price}
            if tp_price:
                params["takeProfit"] = {"triggerPrice": tp_price}

            order = await self._exchange.create_order(
                symbol = symbol,
                type   = "limit",
                side   = side,
                amount = amount,
                price  = price,
                params = params,
            )

            logger.info(
                f"✅ CCXT limite placé — {symbol} {direction} "
                f"@ {price} qty={amount} | ID: {order['id']}"
            )

            return CCXTOrder(
                order_id  = str(order["id"]),
                symbol    = symbol,
                direction = direction,
                amount    = amount,
                price     = price,
                status    = order["status"],
                filled    = float(order.get("filled", 0)),
                cost      = float(order.get("cost", 0)),
            )

        except Exception as e:
            logger.error(f"CCXT place_limit_order error: {e}")
            raise

    async def set_stop_loss(
        self,
        symbol:    str,
        direction: str,
        amount:    float,
        sl_price:  float,
    ) -> CCXTOrder | None:
        """
        Place un ordre Stop Loss séparé (pour exchanges
        qui ne supportent pas SL dans l'ordre initial).
        """
        try:
            side = "sell" if direction == "buy" else "buy"
            order = await self._exchange.create_order(
                symbol = symbol,
                type   = "stop",
                side   = side,
                amount = amount,
                price  = sl_price,
                params = {"stopPrice": sl_price},
            )
            return CCXTOrder(
                order_id  = str(order["id"]),
                symbol    = symbol,
                direction = side,
                amount    = amount,
                price     = sl_price,
                status    = order["status"],
                filled    = 0.0,
                cost      = 0.0,
            )
        except Exception as e:
            logger.warning(f"CCXT set_stop_loss error (non bloquant): {e}")
            return None

    # ── ANNULATION ET CLÔTURE ─────────────────────────────────────

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Annule un ordre en attente."""
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ CCXT ordre {order_id} annulé")
            return True
        except Exception as e:
            logger.error(f"CCXT cancel_order error: {e}")
            return False

    async def close_position(
        self,
        symbol:    str,
        direction: str,    # direction du trade OUVERT (on fait l'inverse)
        amount:    float,
    ) -> CCXTOrder:
        """Clôture une position au marché."""
        close_side = "sell" if direction == "buy" else "buy"
        return await self.place_market_order(
            symbol    = symbol,
            direction = close_side,
            amount    = amount,
        )

    # ── POSITIONS ─────────────────────────────────────────────────

    async def get_open_positions(self) -> list[dict]:
        """Retourne toutes les positions ouvertes."""
        try:
            positions = await self._exchange.fetch_positions()
            return [p for p in positions if float(p.get("contracts", 0)) != 0]
        except Exception as e:
            logger.error(f"CCXT get_open_positions error: {e}")
            return []

    async def get_position(self, symbol: str) -> dict | None:
        """Retourne la position ouverte sur un symbole."""
        try:
            positions = await self._exchange.fetch_positions([symbol])
            for p in positions:
                if float(p.get("contracts", 0)) != 0:
                    return p
            return None
        except Exception as e:
            logger.error(f"CCXT get_position({symbol}) error: {e}")
            return None

    # ── SIZING ────────────────────────────────────────────────────

    async def calculate_amount(
        self,
        symbol:    str,
        risk_usd:  float,
        sl_pips:   float,
    ) -> float:
        """
        Calcule la taille de position pour un risque en USD.
        Retourne le nombre d'unités (BTC, etc.) arrondi au step.
        """
        try:
            markets = await self._exchange.load_markets()
            market  = markets.get(symbol, {})

            price   = (await self.get_price(symbol))["mid"]
            if price == 0 or sl_pips == 0:
                return 0.0

            # risk_usd / (sl_distance_en_USD par unité)
            amount = risk_usd / (sl_pips * price / 100)

            # Arrondi au step minimum de l'exchange
            step = market.get("precision", {}).get("amount", 0.001)
            amount = round(round(amount / step) * step, 8)

            return max(amount, market.get("limits", {}).get("amount", {}).get("min", 0.001))

        except Exception as e:
            logger.error(f"CCXT calculate_amount error: {e}")
            return 0.001
