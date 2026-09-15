"""
Phase 2A: 技術指標計算
嚴格遵守無未來函數 (No Look-ahead Bias) 原則：
  - RSI / MACD / 滾動報酬率 / 波動率 / 位階 全部使用 t 日之前資料
  - Target_Binary 使用 shift(-1) 僅作為訓練標籤，不作為特徵
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)


# ─── 單一指標函式 ────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    計算 RSI (相對強弱指標)。
    使用 Wilder EMA (com = period-1) — 與 TradingView 標準一致。
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename('RSI_14')


def compute_macd(series: pd.Series,
                 fast: int = MACD_FAST,
                 slow: int = MACD_SLOW,
                 signal: int = MACD_SIGNAL_P) -> tuple[pd.Series, pd.Series]:
    """
    計算 MACD 線與 Signal 線。
    回傳 (macd_line, signal_line)。
    """
    ema_fast   = series.ewm(span=fast,   min_periods=fast).mean()
    ema_slow   = series.ewm(span=slow,   min_periods=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line.rename('MACD'), signal_line.rename('MACD_Signal')


def compute_rolling_returns(series: pd.Series,
                             windows: list = ROLL_WINDOWS) -> pd.DataFrame:
    """
    計算多天期歷史報酬率: Ret_wD = (Close_t / Close_{t-w}) - 1。
    全為已知歷史資料，無未來函數。
    """
    result = {}
    for w in windows:
        result[f'Ret_{w}D'] = series.pct_change(periods=w)
    return pd.DataFrame(result, index=series.index)


def compute_volatility(series: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """
    滾動波動率 = 日報酬率標準差 × √252 (年化)。
    """
    daily_ret = series.pct_change()
    vol = daily_ret.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)
    return vol.rename('Vol_42D')


def compute_position(series: pd.Series, window: int = POSITION_WINDOW) -> pd.Series:
    """
    42 日相對位階 = (Close - 42D_Low) / (42D_High - 42D_Low)。
    回傳 0~1 之間的值 (0=42日最低, 1=42日最高)。
    """
    hi = series.rolling(window=window, min_periods=window // 2).max()
    lo = series.rolling(window=window, min_periods=window // 2).min()
    pos = (series - lo) / (hi - lo).replace(0, np.nan)
    return pos.clip(0, 1).rename('Position_42D')


# ─── 主函式 ──────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    對市場資料 DataFrame 加入所有技術指標。

    輸入: DataFrame (index=Date, 必含 Close / Volume)
    輸出: 原 DataFrame + 技術指標欄位 + Next_Day_Return + Target_Binary
    """
    result = df.copy()
    close  = result['Close']

    # RSI
    result['RSI_14'] = compute_rsi(close)

    # MACD / Signal
    result['MACD'], result['MACD_Signal'] = compute_macd(close)

    # 滾動報酬率
    ret_df = compute_rolling_returns(close)
    for col in ret_df.columns:
        result[col] = ret_df[col]

    # 波動率
    result['Vol_42D'] = compute_volatility(close)

    # 相對位階
    result['Position_42D'] = compute_position(close)

    # 預測標籤 (僅用於評估，訓練時嚴格切分避免洩漏)
    next_ret = close.pct_change().shift(-1)
    result['Next_Day_Return'] = next_ret
    result['Target_Binary']   = (next_ret > 0).astype(int)

    logger.info(
        f"技術指標計算完成: RSI, MACD, "
        f"Ret_5D~42D, Vol_42D, Position_42D "
        f"({len(result)} 筆)"
    )
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crawlers.market_data import collect_market_data
    mkt  = collect_market_data()
    tech = add_technical_indicators(mkt)
    print(tech.tail(5).to_string())
    print("\n缺失值:\n", tech.isnull().sum())
