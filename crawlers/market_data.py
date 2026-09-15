"""
Phase 1A: 市場資料爬蟲
來源: Yahoo Finance (via yfinance)
涵蓋: TSMC 2330.TW, VIX, SOX, TSM ADR, USD/TWD
時間: 2000-01-01 ~ 2026-04-24
"""
import os
import time
import logging
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)


def _download(symbol: str, name: str, retries: int = 3) -> pd.DataFrame:
    """帶重試機制的 yfinance 下載，統一回傳單層欄位 DataFrame。"""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  [{attempt}/{retries}] 下載 {name} ({symbol})...")
            df = yf.download(symbol, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False, timeout=30)
            # 壓平 MultiIndex (yfinance 新版會產生)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                logger.info(f"    ✓ {name}: {len(df)} 筆 "
                            f"({df.index[0].date()} ~ {df.index[-1].date()})")
                return df
            logger.warning(f"    ✗ {name}: 下載成功但無資料")
        except Exception as exc:
            logger.warning(f"    ✗ 嘗試 {attempt} 失敗: {exc}")
            time.sleep(3 * attempt)
    return pd.DataFrame()


def collect_market_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    下載並對齊所有市場資料至台灣交易日。

    欄位: Close, Volume, VIX_Close, SOX_Ret, ADR_Ret, FX_Ret
    """
    cache = os.path.join(RAW_DIR, 'market_data.csv')
    if not force_refresh and os.path.exists(cache):
        logger.info(f"載入市場資料快取: {cache}")
        df = pd.read_csv(cache, parse_dates=['Date'], index_col='Date')
        return df

    os.makedirs(RAW_DIR, exist_ok=True)
    logger.info("=== 下載市場資料 ===")

    # 1. TSMC 台股 (構成主要 index)
    tsmc = _download(TSMC_TW, 'TSMC 台股')
    if tsmc.empty:
        raise RuntimeError("TSMC 台股下載失敗，無法繼續")

    mkt = pd.DataFrame({'Close': tsmc['Close'], 'Volume': tsmc['Volume']})

    # 2. VIX 恐慌指數
    vix = _download(VIX_SYM, 'CBOE VIX')
    mkt['VIX_Close'] = (
        vix['Close'].reindex(mkt.index, method='ffill') if not vix.empty else np.nan
    )

    # 3. SOX 費半報酬率
    sox = _download(SOX_SYM, 'Philadelphia SOX')
    mkt['SOX_Ret'] = (
        sox['Close'].pct_change().reindex(mkt.index, method='ffill')
        if not sox.empty else np.nan
    )

    # 4. TSM ADR 報酬率
    tsm = _download(TSMC_US, 'TSM ADR')
    mkt['ADR_Ret'] = (
        tsm['Close'].pct_change().reindex(mkt.index, method='ffill')
        if not tsm.empty else np.nan
    )

    # 5. USD/TWD 匯率報酬率
    fx = _download(FX_SYM, 'USD/TWD FX')
    if fx.empty:                              # 嘗試備用代碼
        fx = _download('TWD=X', 'USD/TWD FX(alt)')
    mkt['FX_Ret'] = (
        fx['Close'].pct_change().reindex(mkt.index, method='ffill')
        if not fx.empty else np.nan
    )

    # 前向填補國際資料 (對齊台灣交易日)
    for col in ['VIX_Close', 'SOX_Ret', 'ADR_Ret', 'FX_Ret']:
        mkt[col] = mkt[col].ffill()

    mkt = mkt.dropna(subset=['Close'])
    mkt.index.name = 'Date'

    mkt.to_csv(cache)
    logger.info(f"市場資料已儲存: {cache} ({len(mkt)} 筆)")
    return mkt


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    df = collect_market_data()
    print(df.tail())
    print(df.isnull().sum())
