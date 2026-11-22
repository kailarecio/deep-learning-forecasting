import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dukascopy_python.compat as duka
from dukascopy_python.compat import instruments

from src.config.logging_config import get_logger
from src.dto.config import VolConfig
from src.dto.enums import TZ_LONDON, TZ_NEW_YORK, Session

logger = get_logger(__name__)

SIDES: dict[str, str] = {
    'bid': duka.OFFER_SIDE_BID, 
    'ask': duka.OFFER_SIDE_ASK
    }

INSTRUMENTS: dict[str, str] = {
    'USDJPY': instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    'EURUSD': instruments.INSTRUMENT_FX_MAJORS_EUR_USD,
    'GBPUSD': instruments.INSTRUMENT_FX_MAJORS_GBP_USD,
    'AUDUSD': instruments.INSTRUMENT_FX_MAJORS_AUD_USD,
}


def fetch_bars(side: str, config: VolConfig) -> pd.DataFrame:
    """
    Dukascopy OHLCV bars of config.data.pair for one side, fetched a year at a time and cached as a pickle.

    Args:
        side: 'bid' or 'ask'
        config: VolConfig

    Returns:
        DataFrame indexed by UTC bar open time, columns open, high, low, close, volume.
    """
    # cache data load to reduce run time when iterating, read from cache when available
    data = config.data
    cache_dir = Path(data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{data.pair.lower()}_{data.base_minutes}m_{side}_{data.start:%Y%m%d}_{data.end:%Y%m%d}.pkl'
    if path.exists():
        return pd.read_pickle(path)
    if data.pair not in INSTRUMENTS:
        raise ValueError(f'Unknown pair {data.pair}; add its Dukascopy instrument to INSTRUMENTS')

    chunks, start, end = [], pd.Timestamp(data.start), pd.Timestamp(data.end)
    while start < end:
        stop = min(start + pd.DateOffset(years=1), end)
        t0 = time.perf_counter()
        logging.disable(logging.INFO)        # dukascopy_python logs every page it downloads to the root logger
        try:
            chunks.append(duka.fetch(INSTRUMENTS[data.pair], data.interval, SIDES[side],
                                     start.to_pydatetime(), stop.to_pydatetime()))
        finally:
            logging.disable(logging.NOTSET)
        logger.info(f'{data.pair} {side}: {start:%Y-%m-%d} -> {stop:%Y-%m-%d} ({len(chunks[-1]):,} bars, {time.perf_counter() - t0:.0f}s)')
        start = stop
    df = pd.concat(chunks).sort_index()
  
    df = df[~df.index.duplicated(keep='first')]
    df.to_pickle(path)
    return df


def get_session(ny_hour: int, london_hour: int) -> Session:
    """
    Map local New York / London hours to an FX trading session (DST handled by the caller's tz conversion).

    Args:
        ny_hour: hour of the bar in America/New_York
        london_hour: hour of the bar in Europe/London

    Returns:
        Session enum. Asia is the residual: every hour outside London / New York / rollover,
        i.e. 18:00 -> 02:00 New York (07:00/08:00 -> 15:00/16:00 Tokyo, depending on US DST).
    """
    match (ny_hour, london_hour):
        case (ny, _) if 16 <= ny < 18:          # around the 17:00 NY close / rollover
            return Session.ROLLOVER
        case (ny, _) if 8 <= ny < 12:           # London afternoon overlaps NY morning
            return Session.LDN_NY
        case (ny, _) if 12 <= ny < 16:
            return Session.NEW_YORK
        case (_, ldn) if 7 <= ldn < 13:
            return Session.LONDON
        case _:
            return Session.ASIA


def assign_sessions(index: pd.DatetimeIndex) -> np.ndarray:
    """Session name of every bar, DST-aware."""
    ny_hours = index.tz_convert(TZ_NEW_YORK).hour
    london_hours = index.tz_convert(TZ_LONDON).hour
    return np.array([get_session(ny, ldn).value for ny, ldn in zip(ny_hours, london_hours)], dtype=object)


def fx_trade_day(index: pd.DatetimeIndex) -> np.ndarray:
    """
    FX trade date's day of week (0 = Monday ... 4 = Friday).
    The FX day runs 17:00 Sunday -> 17:00 Friday New York, so the sunday session already belongs to Monday's trade date. 
    """
    return (index.tz_convert(TZ_NEW_YORK) + pd.Timedelta(hours=7)).day_of_week.to_numpy()


def five_minute_frame(df_bid: pd.DataFrame, df_ask: pd.DataFrame, config: VolConfig) -> pd.DataFrame:
    """
    5-minute mid prices, log returns and relative spread from the bid/ask close of each bar.

    Returns:
        DataFrame with bid, ask, mid, r (log mid return; NaN if it spans more than max_gap_minutes),
        r2 = r^2, rel_spread = (ask - bid) / mid.
    """
    five = pd.DataFrame({'bid': df_bid['close'], 
                         'ask': df_ask['close']}).dropna()

    # guard: fetched bar size must match the config
    step = five.index.to_series().diff().median() / pd.Timedelta(minutes=1)
    if step != config.data.base_minutes:
        raise ValueError(f'Data has {step:.0f}-min bars but config says {config.data.base_minutes}-min.')

    mid = (five['bid'] + five['ask']) / 2
    r5 = np.log(mid).diff()
    gap = five.index.to_series().diff() > pd.Timedelta(minutes=config.data.max_gap_minutes)
    r5 = r5.mask(gap.to_numpy())
    return five.assign(mid=mid, r=r5, r2=r5 ** 2, rel_spread=(five['ask'] - five['bid']) / mid)


def build_hourly_frame(df_bid: pd.DataFrame, df_ask: pd.DataFrame, config: VolConfig) -> pd.DataFrame:
    """
    Aggregate 5-minute bid/ask bars into hourly realized variance, returns, spread, features and the target.

    Args:
        df_bid: Dukascopy 5-min bid bars
        df_ask: Dukascopy 5-min ask bars
        config: VolConfig

    Returns:
        DataFrame indexed by the hour's start (UTC); target_rv / target_ret are NaN where the next row
        is not the next hour.
    """
    data_cfg, feat_cfg = config.data, config.features
    five = five_minute_frame(df_bid, df_ask, config)

  
    hourly = five.groupby(five.index.floor('h')).agg(
        rv=('r2', 'sum'), ret=('r', 'sum'), n_obs=('r', 'count'), mid=('mid', 'last'), rel_spread=('rel_spread', 'last'))
    hourly = hourly[hourly['n_obs'] >= data_cfg.min_obs_per_hour]
    hourly = hourly.assign(rv=(hourly['rv'] * data_cfg.bars_per_hour / hourly['n_obs']).clip(lower=feat_cfg.rv_floor))

    # volatility level, shock and trend (all known at the end of the hour)
    rv24 = hourly['rv'].rolling(feat_cfg.har_day).mean()
    hourly = hourly.assign(
        rv24=rv24,
        shock=np.log(hourly['rv']) - np.log(rv24),
        trend=hourly['ret'].rolling(feat_cfg.trend_window).sum().abs()
              / np.sqrt(hourly['rv'].rolling(feat_cfg.trend_window).sum()),
    )

    # calendar: session of this hour; NY hour of this and of the next hour (the forecast target's hour)
    idx = hourly.index
    ny_now = idx.tz_convert(TZ_NEW_YORK)
    ny_next = (idx + pd.Timedelta(hours=1)).tz_convert(TZ_NEW_YORK)
    hourly = hourly.assign(
        hour_ny=ny_now.hour, hour_ny_next=ny_next.hour,
        tod_sin=np.sin(2 * np.pi * ny_next.hour / 24), tod_cos=np.cos(2 * np.pi * ny_next.hour / 24),
        session=assign_sessions(idx),
        trade_dow=fx_trade_day(idx),
    )

    # target: next hour's realized variance and return, only if the next row really is the next hour
    contiguous = (idx.to_series().shift(-1) - idx.to_series()) == pd.Timedelta(hours=1)
    hourly = hourly.assign(target_rv=hourly['rv'].shift(-1).where(contiguous.to_numpy()),
                           target_ret=hourly['ret'].shift(-1).where(contiguous.to_numpy()))
    return hourly.dropna(subset=['rv24', 'trend'])
