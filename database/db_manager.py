# database/db_manager.py
# ─────────────────────────────────────────────────────────────────
#  Gestionnaire de base de données
#  Toutes les opérations CRUD passent par cette classe
#  Utilise SQLAlchemy async pour ne pas bloquer la boucle principale
# ─────────────────────────────────────────────────────────────────

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, func, and_
from loguru import logger

from config import settings
from .models import (
    Base, Signal, Trade, Candle,
    OptimizerRun, PerformanceSnapshot,
    SignalStatus, TradeResult
)


class DatabaseManager:

    def __init__(self):
        # Convertit postgresql:// en postgresql+asyncpg://
        db_url = settings.DATABASE_URL.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        self._engine = create_async_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=(settings.LOG_LEVEL == "DEBUG"),
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    # ── INITIALISATION ────────────────────────────────────────────

    async def init(self):
        """Crée toutes les tables si elles n'existent pas."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Base de données initialisée")

    async def close(self):
        await self._engine.dispose()
        logger.info("🔌 Connexion base de données fermée")

    def session(self) -> AsyncSession:
        """Retourne une nouvelle session async."""
        return self._session_factory()

    # ── SIGNALS : ÉCRITURE ────────────────────────────────────────

    async def save_signal(self, signal_data: dict) -> Signal:
        """Enregistre un nouveau signal en base."""
        async with self.session() as sess:
            signal = Signal(**signal_data)
            sess.add(signal)
            await sess.commit()
            await sess.refresh(signal)
            logger.info(
                f"💾 Signal #{signal.id} enregistré — "
                f"{signal.instrument} {signal.direction} "
                f"score={signal.confluence_score}"
            )
            return signal

    async def update_signal_status(
        self,
        signal_id: int,
        status: SignalStatus,
        extra: dict | None = None,
    ) -> None:
        """Met à jour le statut d'un signal (et champs optionnels)."""
        values = {"status": status, "updated_at": datetime.utcnow()}
        if extra:
            values.update(extra)
        async with self.session() as sess:
            await sess.execute(
                update(Signal).where(Signal.id == signal_id).values(**values)
            )
            await sess.commit()

    # ── SIGNALS : LECTURE ─────────────────────────────────────────

    async def get_signal(self, signal_id: int) -> Optional[Signal]:
        async with self.session() as sess:
            result = await sess.execute(
                select(Signal).where(Signal.id == signal_id)
            )
            return result.scalar_one_or_none()

    async def get_active_signals(self, instrument: str | None = None) -> list[Signal]:
        """Retourne tous les signaux actifs (trade ouvert)."""
        async with self.session() as sess:
            query = select(Signal).where(Signal.status == SignalStatus.ACTIVE)
            if instrument:
                query = query.where(Signal.instrument == instrument)
            result = await sess.execute(query)
            return list(result.scalars().all())

    async def count_signals_today(self, instrument: str) -> int:
        """Nombre de signaux émis aujourd'hui pour cet instrument (anti-spam)."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.session() as sess:
            result = await sess.execute(
                select(func.count(Signal.id)).where(
                    and_(
                        Signal.instrument == instrument,
                        Signal.created_at >= today_start,
                    )
                )
            )
            return result.scalar_one() or 0

    async def get_signals_for_backtest(
        self,
        instrument: str | None = None,
        months: int = 6,
    ) -> list[Signal]:
        """Retourne les signaux des X derniers mois pour backtesting."""
        since = datetime.utcnow() - timedelta(days=months * 30)
        async with self.session() as sess:
            query = select(Signal).where(Signal.created_at >= since)
            if instrument:
                query = query.where(Signal.instrument == instrument)
            query = query.order_by(Signal.created_at.asc())
            result = await sess.execute(query)
            return list(result.scalars().all())

    # ── TRADES : ÉCRITURE ─────────────────────────────────────────

    async def save_trade(self, trade_data: dict) -> Trade:
        """Enregistre un trade exécuté."""
        async with self.session() as sess:
            trade = Trade(**trade_data)
            sess.add(trade)
            await sess.commit()
            await sess.refresh(trade)
            logger.info(
                f"💾 Trade #{trade.id} enregistré — "
                f"Signal #{trade.signal_id} | Broker: {trade.broker}"
            )
            return trade

    async def close_trade(
        self,
        trade_id: int,
        close_price: float,
        result: TradeResult,
        pnl_usd: float,
        pnl_pips: float,
        tp_level_hit: int | None = None,
        commission: float = 0.0,
        swap: float = 0.0,
        balance_after: float | None = None,
    ) -> None:
        """Clôture un trade avec son résultat final."""
        net_pnl = pnl_usd - commission - abs(swap)
        values = {
            "close_price":   close_price,
            "close_at":      datetime.utcnow(),
            "result":        result,
            "pnl_usd":       pnl_usd,
            "pnl_pips":      pnl_pips,
            "net_pnl_usd":   net_pnl,
            "tp_level_hit":  tp_level_hit,
            "commission":    commission,
            "swap":          swap,
            "balance_after": balance_after,
            "updated_at":    datetime.utcnow(),
        }
        async with self.session() as sess:
            await sess.execute(
                update(Trade).where(Trade.id == trade_id).values(**values)
            )
            await sess.commit()
        logger.info(
            f"✅ Trade #{trade_id} clôturé — "
            f"Résultat: {result} | Net P&L: {net_pnl:.2f} USD"
        )

    # ── CANDLES : CACHE ───────────────────────────────────────────

    async def save_candles(self, candles: list[dict]) -> int:
        """
        Insère des candles en base (ignore les doublons).
        Retourne le nombre de candles insérées.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        if not candles:
            return 0

        async with self.session() as sess:
            stmt = pg_insert(Candle).values(candles)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["instrument", "timeframe", "timestamp"]
            )
            result = await sess.execute(stmt)
            await sess.commit()
            return result.rowcount

    async def get_candles(
        self,
        instrument: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> list[Candle]:
        """Récupère les candles depuis le cache."""
        until = until or datetime.utcnow()
        async with self.session() as sess:
            result = await sess.execute(
                select(Candle).where(
                    and_(
                        Candle.instrument == instrument,
                        Candle.timeframe  == timeframe,
                        Candle.timestamp  >= since,
                        Candle.timestamp  <= until,
                    )
                ).order_by(Candle.timestamp.asc())
            )
            return list(result.scalars().all())

    # ── PERFORMANCE ───────────────────────────────────────────────

    async def save_performance_snapshot(self, data: dict) -> PerformanceSnapshot:
        async with self.session() as sess:
            snap = PerformanceSnapshot(**data)
            sess.add(snap)
            await sess.commit()
            await sess.refresh(snap)
            return snap

    async def get_win_rate(
        self,
        instrument: str | None = None,
        days: int = 30,
    ) -> float:
        """Calcule le win rate sur les X derniers jours."""
        since = datetime.utcnow() - timedelta(days=days)
        async with self.session() as sess:
            query = select(
                func.count(Signal.id).label("total"),
                func.sum(
                    func.cast(
                        Signal.status.in_([
                            SignalStatus.TP1_HIT,
                            SignalStatus.TP2_HIT,
                            SignalStatus.TP3_HIT,
                        ]),
                        Integer
                    )
                ).label("wins")
            ).where(
                and_(
                    Signal.created_at >= since,
                    Signal.status.in_([
                        SignalStatus.TP1_HIT,
                        SignalStatus.TP2_HIT,
                        SignalStatus.TP3_HIT,
                        SignalStatus.SL_HIT,
                    ])
                )
            )
            if instrument:
                query = query.where(Signal.instrument == instrument)
            result = await sess.execute(query)
            row = result.one()
            if not row.total or row.total == 0:
                return 0.0
            return (row.wins or 0) / row.total

    # ── OPTIMIZER ─────────────────────────────────────────────────

    async def save_optimizer_run(self, data: dict) -> OptimizerRun:
        async with self.session() as sess:
            run = OptimizerRun(**data)
            sess.add(run)
            await sess.commit()
            await sess.refresh(run)
            logger.info(
                f"🧠 OptimizerRun #{run.id} sauvegardé — "
                f"WR: {run.win_rate:.1%} | PF: {run.profit_factor:.2f}"
            )
            return run

    async def get_latest_optimizer_params(
        self,
        instrument: str | None = None,
    ) -> dict | None:
        """Retourne les meilleurs paramètres du dernier run Optuna."""
        async with self.session() as sess:
            query = select(OptimizerRun).order_by(
                OptimizerRun.run_at.desc()
            ).limit(1)
            if instrument:
                query = query.where(OptimizerRun.instrument == instrument)
            result = await sess.execute(query)
            run = result.scalar_one_or_none()
            return run.best_params if run else None
