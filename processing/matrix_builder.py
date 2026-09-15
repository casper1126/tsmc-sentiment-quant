"""
Phase 3: 超級矩陣組裝
整合策略:
  1. 從 yfinance 取得 2000~2026 台股技術面 + 總經面 (Source 1 & 2)
  2. 從 sentiment_processor 取得 PTT + 新聞情緒日資料 (Source 3 & 4)
  3. 若已有現成 TSMC_Matrix_2000-2015.csv 直接沿用 (加速)
  4. 使用 pd.merge(on='Date', how='left') 嚴格日期對齊
  5. 無情緒資料的日期填補 PTT_Raw_Score/News_Raw_Score = 0.5 (中立)
  6. 輸出: output/TSMC_matrix_2000-2026.csv

最終欄位 (22 col):
  Date, Close, Volume, VIX_Close, ADR_Ret, SOX_Ret, FX_Ret,
  RSI_14, MACD, MACD_Signal,
  Ret_5D, Ret_10D, Ret_20D, Ret_42D, Vol_42D, Position_42D,
  PTT_Raw_Score, PTT_Volume, News_Raw_Score, News_Volume,
  Next_Day_Return, Target_Binary
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)

NEUTRAL       = 0.5
FINAL_COLUMNS = [
    'Date', 'Close', 'Volume',
    'VIX_Close', 'ADR_Ret', 'SOX_Ret', 'FX_Ret',
    'RSI_14', 'MACD', 'MACD_Signal',
    'Ret_5D', 'Ret_10D', 'Ret_20D', 'Ret_42D',
    'Vol_42D', 'Position_42D',
    'PTT_Raw_Score', 'PTT_Volume',
    'News_Raw_Score', 'News_Volume',
    'Next_Day_Return', 'Target_Binary',
]

# 現成歷史矩陣路徑 (若已有可直接沿用)
HIST_MATRIX_PATH = os.path.join(BASE_DIR, 'TSMC', 'TSMC_Matrix_2000-2015.csv')
E2E_MATRIX_PATH  = os.path.join(BASE_DIR, 'TSMC', 'TSMC_EndToEnd_Dynamic_4Sources.csv')


# ─── 輔助: 讀取/建立技術面基礎矩陣 ─────────────────────────────────────────

def _load_or_build_tech_matrix() -> pd.DataFrame:
    """
    優先讀取已有的歷史矩陣 + EndToEnd 矩陣合併，
    若都不存在則從 yfinance 重新計算。
    """
    frames = []

    # ── 2000-2015 ────────────────────────────────────────────────────────────
    _SENTIMENT_COLS = ('PTT_Raw_Score', 'PTT_Volume',
                       'PTT_Score',     'PTT_Sentiment',
                       'News_Raw_Score', 'News_Volume',
                       'News_Score',     'News_Sentiment')
    if os.path.exists(HIST_MATRIX_PATH):
        h = pd.read_csv(HIST_MATRIX_PATH, parse_dates=['Date'])
        # 統一欄位名稱
        h = h.rename(columns={'Target_Up': 'Target_Binary'})
        # 去除情緒欄位 (稍後由 ptt_df / news_df 重新填入，避免 merge 後 _x/_y 衝突)
        h = h.drop(columns=[c for c in _SENTIMENT_COLS if c in h.columns],
                   errors='ignore')
        frames.append(h)
        logger.info(f"讀取歷史矩陣: {len(h)} 筆 ({h.Date.min().date()}~{h.Date.max().date()})")

    # ── 2016-2026 (EndToEnd) ─────────────────────────────────────────────────
    if os.path.exists(E2E_MATRIX_PATH):
        e = pd.read_csv(E2E_MATRIX_PATH, parse_dates=['Date'])
        # 僅取技術/總經欄位，情緒欄位另外再 merge (重新計算 0-1 分數)
        tech_cols = [c for c in FINAL_COLUMNS
                     if c not in ('PTT_Raw_Score', 'PTT_Volume',
                                  'News_Raw_Score', 'News_Volume')]
        avail = [c for c in tech_cols if c in e.columns]
        frames.append(e[avail])
        logger.info(f"讀取 EndToEnd 矩陣: {len(e)} 筆 ({e.Date.min().date()}~{e.Date.max().date()})")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')
        return combined

    # ── 從頭建立 ─────────────────────────────────────────────────────────────
    logger.info("未找到現成矩陣，從 yfinance 重新建立...")
    from crawlers.market_data import collect_market_data
    from processing.technical_indicators import add_technical_indicators
    mkt  = collect_market_data()
    tech = add_technical_indicators(mkt)
    tech = tech.reset_index()
    tech = tech.rename(columns={'index': 'Date'})
    return tech


# ─── 主函式 ──────────────────────────────────────────────────────────────────

def build_super_matrix(ptt_df: pd.DataFrame,
                        news_df: pd.DataFrame,
                        force: bool = False) -> pd.DataFrame:
    """
    組裝最終超級矩陣。

    參數:
        ptt_df  : [Date, PTT_Raw_Score, PTT_Volume]
        news_df : [Date, News_Raw_Score, News_Volume]
        force   : 強制重新計算 (忽略快取)
    回傳:
        完整 DataFrame，欄位同 FINAL_COLUMNS
    """
    out_path = os.path.join(OUTPUT_DIR, 'TSMC_matrix_2000-2026.csv')
    if not force and os.path.exists(out_path):
        logger.info(f"載入矩陣快取: {out_path}")
        return pd.read_csv(out_path, parse_dates=['Date'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── Step 1: 技術面基礎矩陣 ─────────────────────────────────────────────
    logger.info("Step 1: 讀取/建立技術面基礎矩陣...")
    base = _load_or_build_tech_matrix()
    base['Date'] = pd.to_datetime(base['Date']).dt.normalize()

    # ─── Step 2: 合併 PTT 情緒 ─────────────────────────────────────────────
    logger.info("Step 2: 合併 PTT 情緒...")
    if not ptt_df.empty:
        ptt_clean = ptt_df.copy()
        ptt_clean['Date'] = pd.to_datetime(ptt_clean['Date']).dt.normalize()
        # 確保欄位名稱正確
        if 'PTT_Score' in ptt_clean.columns and 'PTT_Raw_Score' not in ptt_clean.columns:
            ptt_clean = ptt_clean.rename(columns={'PTT_Score': 'PTT_Raw_Score'})
        base = pd.merge(base,
                        ptt_clean[['Date', 'PTT_Raw_Score', 'PTT_Volume']],
                        on='Date', how='left')
    else:
        base['PTT_Raw_Score'] = np.nan
        base['PTT_Volume']    = 0

    # ─── Step 3: 合併新聞情緒 ─────────────────────────────────────────────
    logger.info("Step 3: 合併新聞情緒...")

    # 先嘗試讀取 Taiwan_News_Sentiment.csv + Yahoo_Sentiment.csv 作為補充來源
    # 這兩份已有 1,892 / 1,800 天的每日覆蓋 (2019~2026)，分數需標準化到 [0,1]
    _TSMC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TSMC')
    def _load_legacy_news_scores() -> pd.DataFrame:
        frames = []
        for fname, score_col in [('Taiwan_News_Sentiment.csv', 'News_Score'),
                                  ('Yahoo_Sentiment.csv',       'Yahoo_Score')]:
            p = os.path.join(_TSMC_DIR, fname)
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            vol_col = [c for c in df.columns if 'volume' in c.lower() or 'Volume' in c]
            df = df.rename(columns={score_col: '_raw_score'})
            if vol_col:
                df = df.rename(columns={vol_col[0]: 'News_Volume'})
            else:
                df['News_Volume'] = 1
            frames.append(df[['Date', '_raw_score', 'News_Volume']])
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames).groupby('Date').agg(
            _raw_score=('_raw_score', 'mean'),
            News_Volume=('News_Volume', 'sum'),
        ).reset_index()
        # Min-Max 標準化到 [0,1]，偏移至 0.5 中心 (正值=樂觀, 負值=悲觀)
        mn, mx = combined['_raw_score'].min(), combined['_raw_score'].max()
        combined['News_Raw_Score'] = ((combined['_raw_score'] - mn) / (mx - mn)).clip(0, 1)
        logger.info(f"  [補充新聞] 讀入 {len(combined)} 天舊版情緒分數 "
                    f"(MinMax→[{combined.News_Raw_Score.min():.3f}, "
                    f"{combined.News_Raw_Score.max():.3f}])")
        return combined[['Date', 'News_Raw_Score', 'News_Volume']]

    legacy_news = _load_legacy_news_scores()

    if not news_df.empty:
        news_clean = news_df.copy()
        news_clean['Date'] = pd.to_datetime(news_clean['Date']).dt.normalize()
        if 'News_Score' in news_clean.columns and 'News_Raw_Score' not in news_clean.columns:
            news_clean = news_clean.rename(columns={'News_Score': 'News_Raw_Score'})

        # 合併 Text2Vec 分數 (優先) + 舊版分數 (補充空白日)
        if not legacy_news.empty:
            # Text2Vec 覆蓋的日期用 Text2Vec，其餘用舊版
            all_news = pd.concat([news_clean[['Date', 'News_Raw_Score', 'News_Volume']],
                                   legacy_news], ignore_index=True)
            all_news = all_news.sort_values('Date')
            # 同一天以 Text2Vec (先出現) 為主，保留第一筆
            all_news = all_news.drop_duplicates(subset=['Date'], keep='first')
            logger.info(f"  新聞情緒: Text2Vec {len(news_clean)} 天 "
                        f"+ 舊版補充 → 合計 {len(all_news)} 天")
        else:
            all_news = news_clean[['Date', 'News_Raw_Score', 'News_Volume']]

        base = pd.merge(base, all_news, on='Date', how='left')
    elif not legacy_news.empty:
        # 完全沒有 Text2Vec 結果時，直接用舊版
        base = pd.merge(base, legacy_news, on='Date', how='left')
        logger.info(f"  新聞情緒: 僅使用舊版分數 {len(legacy_news)} 天")
    else:
        base['News_Raw_Score'] = np.nan
        base['News_Volume']    = 0

    # ─── Step 4: 嚴格填補缺失情緒值 ─────────────────────────────────────
    # 規則: 無資料 → 中立 0.5，不使用前後日插補
    logger.info("Step 4: 填補缺失情緒值 (中立=0.5)...")
    base['PTT_Raw_Score']  = base['PTT_Raw_Score'].fillna(NEUTRAL)
    base['News_Raw_Score'] = base['News_Raw_Score'].fillna(NEUTRAL)
    base['PTT_Volume']     = base['PTT_Volume'].fillna(0).astype(int)
    base['News_Volume']    = base['News_Volume'].fillna(0).astype(int)

    # ─── Step 5: 確保所有欄位存在並排序 ─────────────────────────────────
    for col in FINAL_COLUMNS:
        if col not in base.columns:
            logger.warning(f"  欄位 {col} 缺失，填補 NaN")
            base[col] = np.nan

    matrix = base[FINAL_COLUMNS].copy()

    # ─── Step 6: 清理 ────────────────────────────────────────────────────
    # 移除 Close 缺失列
    matrix = matrix.dropna(subset=['Close'])
    # 移除沒有技術指標的最初 ~60 列 (warming up 期間)
    matrix = matrix.dropna(subset=['RSI_14', 'MACD'])
    # 移除最後一列 (Target_Binary 為 NaN，因為沒有明日資料)
    matrix = matrix.dropna(subset=['Target_Binary'])
    matrix = matrix.reset_index(drop=True)

    # ─── Step 7: 儲存 ────────────────────────────────────────────────────
    matrix.to_csv(out_path, index=False, encoding='utf-8-sig')

    logger.info("=" * 60)
    logger.info(f"超級矩陣完成 → {out_path}")
    logger.info(f"筆數    : {len(matrix)}")
    logger.info(f"時間範圍: {matrix.Date.min()} ~ {matrix.Date.max()}")
    logger.info(f"欄位    : {list(matrix.columns)}")
    logger.info(f"\n情緒分數分布:")
    logger.info(f"  PTT_Raw_Score  — mean={matrix.PTT_Raw_Score.mean():.4f}, "
                f"std={matrix.PTT_Raw_Score.std():.4f}")
    logger.info(f"  News_Raw_Score — mean={matrix.News_Raw_Score.mean():.4f}, "
                f"std={matrix.News_Raw_Score.std():.4f}")
    logger.info(f"\nTarget_Binary 分布:\n"
                f"{matrix.Target_Binary.value_counts(normalize=True).round(3).to_string()}")
    logger.info("=" * 60)
    return matrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    from processing.sentiment_processor import compute_all_sentiments
    ptt, news = compute_all_sentiments()
    mat = build_super_matrix(ptt, news)
    print(mat.tail(5).to_string())
