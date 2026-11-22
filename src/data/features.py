import numpy as np
import pandas as pd

from src.dto.config import VolConfig

# the networks' inputs are set in config.features.nn_inputs (part of the cache hash)
HAR_FEATURES: list[str] = ['x', 'x_day', 'x_week']
EXTRA_INPUTS: list[str] = ['z_ret', 'log_spread', 'tod_sin', 'tod_cos']     
HARX_FEATURES: list[str] = HAR_FEATURES + EXTRA_INPUTS
HARX2_FEATURES: list[str] = HARX_FEATURES + ['abs_z', 'z_neg', 'z_day_neg', 'x_tod_sin', 'x_tod_cos', 'x_spread']


def seasonal_factors(market: pd.DataFrame, end_pos: int) -> np.ndarray:
    """Diurnal factor per New York hour (0-23): mean RV in that hour / overall mean RV, rows [0, end_pos]"""
    rv = market['rv'].iloc[:end_pos + 1]
    hours = market['hour_ny'].iloc[:end_pos + 1]
    s = (rv.groupby(hours).mean() / rv.mean()).reindex(range(24)).fillna(1.0)
    return s.to_numpy()


def vol_features(market: pd.DataFrame, s: np.ndarray, config: VolConfig) -> pd.DataFrame:
    """
    Deseasonalized volatility features (inputs to HAR / NN) and the log-scale target for one fold.

    Args:
        market: output of build_hourly_frame()
        s: diurnal factors from seasonal_factors()
        config: VolConfig

    Returns:
        DataFrame aligned with market: x, x_day, x_week, z_ret, log_spread, tod_sin, tod_cos (model inputs),
        the HAR-X2 terms (abs_z, z_neg, z_day_neg, x_tod_sin, x_tod_cos, x_spread),
        s_now, s_next and y = log(RV_{t+1} / s_{h(t+1)})
    """
    s_now = s[market['hour_ny'].to_numpy()]
    s_next = s[market['hour_ny_next'].to_numpy()]
    rvd = market['rv'] / s_now
    x = np.log(rvd)
    z_ret = (market['ret'] / np.sqrt(market['rv'])).clip(-5, 5) 
    d = config.features.har_day
    z_day = (market['ret'].rolling(d).sum() / np.sqrt(market['rv'].rolling(d).sum())).clip(-5, 5)
    log_spread = np.log(market['rel_spread'].clip(lower=1e-7))

    return pd.DataFrame({
        'x': x,
        'x_day': np.log(rvd.rolling(d).mean()),
        'x_week': np.log(rvd.rolling(config.features.har_week).mean()),
        'z_ret': z_ret,
        'log_spread': log_spread,
        'tod_sin': market['tod_sin'], 'tod_cos': market['tod_cos'],
        'abs_z': z_ret.abs(),
        'z_neg': np.minimum(z_ret, 0.0),
        'z_day_neg': np.minimum(z_day, 0.0),
        'x_tod_sin': x * market['tod_sin'], 'x_tod_cos': x * market['tod_cos'],
        'x_spread': x * log_spread,
        's_now': s_now, 
        's_next': s_next,
        'y': np.log(market['target_rv'] / s_next),
    }, index=market.index)


def to_ann_vol_pct(variance: np.ndarray | pd.Series, config: VolConfig) -> np.ndarray | pd.Series:
    """Hourly variance -> annualised volatility in %"""
    return np.sqrt(variance * config.hours_per_year) * 100
