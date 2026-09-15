"""
台積電跨時空量化預測系統 — 主執行程式
======================================
執行順序:
  Phase 1 : 爬蟲 (市場資料 + PTT + 財經新聞)
  Phase 2 : Text2Vec 情緒分析 (0~1 標準化)
  Phase 3 : 超級矩陣組裝 (2000-2026)
  Phase 4 : 消融實驗 (7 關鍵年份 × 5 模型) + 圖表

用法:
  python main.py                  # 執行全部 Phase
  python main.py --phase 4        # 只執行 Phase 4 (需已有矩陣)
  python main.py --phase 1 --force   # 強制重新爬蟲
  python main.py --skip-crawl     # 跳過爬蟲，直接從快取組裝
"""
import os
import sys
import logging
import argparse
import time

# 確保根目錄在 path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import *

# ─── 日誌設定 ────────────────────────────────────────────────────────────────
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
LOG_FILE   = os.path.join(BASE_DIR, 'tsmc_quant.log')

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ],
)
logger = logging.getLogger('main')


# ─── 目錄初始化 ──────────────────────────────────────────────────────────────

def _init_dirs() -> None:
    for d in [RAW_DIR,
              os.path.join(RAW_DIR, 'ptt'),
              os.path.join(RAW_DIR, 'news'),
              PROCESSED_DIR,
              OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── 現有快取自動轉換 (Bootstrap) ────────────────────────────────────────────

def bootstrap_caches() -> None:
    """
    將已有的 TSMC/ 目錄下舊版快取 CSV 自動轉換為
    Text2Vec 可讀的 JSONL 格式，並計算 0~1 情緒分數。

    處理邏輯:
      cache_ptt_2016_2026.csv   (Date, Title)
        → data/raw/ptt/ptt_titles.jsonl

      cache_news_2000_2015.csv + cache_news_2016_2026.csv
        (Date, News_Titles[], News_Volume)
        → data/raw/news/news_titles.jsonl   (合併去重，最大化覆蓋率)
    """
    import json
    import pandas as pd

    tsmc_dir    = os.path.join(BASE_DIR, 'TSMC')
    ptt_src     = os.path.join(tsmc_dir, 'cache_ptt_2016_2026.csv')
    news_src_a  = os.path.join(tsmc_dir, 'cache_news_2000_2015.csv')
    news_src_b  = os.path.join(tsmc_dir, 'cache_news_2016_2026.csv')
    ptt_jsonl   = os.path.join(RAW_DIR, 'ptt', 'ptt_titles.jsonl')
    news_jsonl  = os.path.join(RAW_DIR, 'news', 'news_titles.jsonl')

    # ── PTT 快取轉換 ─────────────────────────────────────────────────────────
    # 策略: 永遠將 cache_ptt_2016_2026.csv (2022-2025) 合併進現有 JSONL
    # 因為 live 爬蟲結果 (2008-2021) 與快取 (2022-2025) 互補，缺一不可
    if os.path.exists(ptt_src):
        logger.info("  [Bootstrap] 合併 PTT 快取 (cache_ptt_2016_2026.csv) → JSONL ...")

        # 讀取快取 (Date, Title)，用 PTT_KEYWORDS 過濾
        cache_df = pd.read_csv(ptt_src)
        cache_df.columns = [c.strip().lstrip('\ufeff') for c in cache_df.columns]
        cache_df['Date'] = pd.to_datetime(cache_df['Date']).dt.normalize()

        # 過濾關鍵字（快取可能已過濾，但以防萬一再過濾一次）
        mask = cache_df['Title'].apply(
            lambda t: any(kw in str(t) for kw in PTT_KEYWORDS))
        cache_df = cache_df[mask]

        # 按日期聚合成 {date: [titles]} dict
        cache_dict: dict = {}
        for _, row in cache_df.iterrows():
            d = str(row['Date'].date())
            cache_dict.setdefault(d, []).append(str(row['Title']))

        # 讀取現有 JSONL（live 爬蟲結果 2008-2021），合併進去
        existing: dict = {}
        if os.path.exists(ptt_jsonl):
            with open(ptt_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    existing[obj['date']] = obj.get('titles', [])

        # 合併：live 優先，快取補充空白日
        before = len(existing)
        for d, titles in cache_dict.items():
            if d not in existing:
                existing[d] = titles
            else:
                # 同日去重合併
                combined = list(dict.fromkeys(existing[d] + titles))
                existing[d] = combined

        os.makedirs(os.path.dirname(ptt_jsonl), exist_ok=True)
        with open(ptt_jsonl, 'w', encoding='utf-8') as f:
            for d in sorted(existing.keys()):
                titles = existing[d]
                json.dump({'date': d, 'titles': titles, 'volume': len(titles)},
                          f, ensure_ascii=False)
                f.write('\n')

        after = len(existing)
        new_days = after - before
        logger.info(f"  [Bootstrap] PTT JSONL 完成: 原 {before} 天 + 新增 {new_days} 天 = {after} 天")
    else:
        logger.info("  [Bootstrap] cache_ptt_2016_2026.csv 不存在，跳過合併")

    # ── News 快取轉換 (合併 2000-2015 + 2016-2026 兩份快取) ─────────────────
    def _parse_news_cache(path: str) -> dict:
        """讀取 cache_news_*.csv，回傳 {date_str: [titles]} dict (僅非空天)。"""
        result = {}
        if not os.path.exists(path):
            return result
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        for _, row in df.iterrows():
            try:
                titles = json.loads(row['News_Titles'])
                if isinstance(titles, str):
                    titles = [titles]
            except Exception:
                titles = [str(row['News_Titles'])]
            # 過濾掉空標題
            titles = [t for t in titles if str(t).strip() and str(t).strip() != '[]']
            if titles:
                d = str(row['Date'].date())
                result[d] = result.get(d, []) + titles
        return result

    if not os.path.exists(news_jsonl):
        logger.info("  [Bootstrap] 合併 cache_news_2000_2015 + cache_news_2016_2026 → JSONL ...")
        merged: dict = {}
        merged.update(_parse_news_cache(news_src_a))
        for d, titles in _parse_news_cache(news_src_b).items():
            merged[d] = list(dict.fromkeys(merged.get(d, []) + titles))  # 合併去重

        os.makedirs(os.path.dirname(news_jsonl), exist_ok=True)
        with open(news_jsonl, 'w', encoding='utf-8') as f:
            for d in sorted(merged.keys()):
                titles = merged[d]
                json.dump({'date': d, 'titles': titles, 'volume': len(titles)},
                          f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"  [Bootstrap] News JSONL 完成: {len(merged)} 天 (含標題的交易日)")
    else:
        logger.info("  [Bootstrap] News JSONL 已存在，跳過轉換")


# ─── Phase 1: 資料爬蟲 ───────────────────────────────────────────────────────

def phase1(force: bool = False, skip_crawl: bool = False) -> None:
    logger.info("\n" + "█" * 55)
    logger.info("  Phase 1 — 資料爬蟲")
    logger.info("█" * 55)

    if skip_crawl:
        logger.info("  --skip-crawl 旗標啟用，跳過爬蟲")
        return

    from crawlers.market_data import collect_market_data
    from crawlers.ptt_crawler import crawl_ptt
    from crawlers.news_crawler import crawl_news

    t0 = time.time()
    mkt  = collect_market_data(force_refresh=force)
    logger.info(f"  市場資料: {len(mkt)} 筆")

    ptt = crawl_ptt(force=force, max_pages=5000)
    logger.info(f"  PTT 資料: {len(ptt)} 天")

    news = crawl_news(force=force)
    logger.info(f"  新聞資料: {len(news)} 天")
    logger.info(f"  Phase 1 完成 ({time.time()-t0:.0f}s)")


# ─── Phase 2: 情緒分析 ───────────────────────────────────────────────────────

def phase2(force: bool = False) -> tuple:
    logger.info("\n" + "█" * 55)
    logger.info("  Phase 2 — Text2Vec 情緒分析")
    logger.info("█" * 55)

    # 自動把所有舊版快取 CSV 轉成 JSONL (PTT + News 一次處理)
    bootstrap_caches()

    from processing.sentiment_processor import compute_all_sentiments
    t0 = time.time()
    ptt_df, news_df = compute_all_sentiments(force=force)
    logger.info(f"  PTT 情緒 : {len(ptt_df)} 天")
    logger.info(f"  新聞情緒 : {len(news_df)} 天")
    logger.info(f"  Phase 2 完成 ({time.time()-t0:.0f}s)")
    return ptt_df, news_df


# ─── Phase 3: 矩陣組裝 ───────────────────────────────────────────────────────

def phase3(ptt_df, news_df, force: bool = False):
    logger.info("\n" + "█" * 55)
    logger.info("  Phase 3 — 超級矩陣組裝")
    logger.info("█" * 55)

    from processing.matrix_builder import build_super_matrix
    t0     = time.time()
    matrix = build_super_matrix(ptt_df, news_df, force=force)
    logger.info(f"  矩陣: {len(matrix)} 筆 "
                f"({matrix.Date.min().date()} ~ {matrix.Date.max().date()})")
    logger.info(f"  Phase 3 完成 ({time.time()-t0:.0f}s)")
    return matrix


# ─── Phase 4: 消融實驗 ───────────────────────────────────────────────────────

def phase4(matrix_path: str | None = None) -> None:
    logger.info("\n" + "█" * 55)
    logger.info("  Phase 4 — 消融實驗 & 圖表")
    logger.info("█" * 55)

    if matrix_path is None:
        matrix_path = os.path.join(OUTPUT_DIR, 'TSMC_matrix_2000-2026.csv')

    from models.prediction_model import run_prediction_study
    t0 = time.time()
    run_prediction_study(matrix_path)
    logger.info(f"  Phase 4 完成 ({time.time()-t0:.0f}s)")


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='台積電跨時空量化預測系統',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        '--phase', type=int, choices=[1, 2, 3, 4], default=None,
        help='只執行指定 Phase (省略 = 全部執行)',
    )
    p.add_argument(
        '--force', action='store_true',
        help='強制重新爬蟲 / 重新計算 (忽略快取)',
    )
    p.add_argument(
        '--skip-crawl', action='store_true',
        help='跳過 Phase 1 爬蟲 (直接使用現有快取)',
    )
    p.add_argument(
        '--matrix', type=str,
        default=os.path.join(OUTPUT_DIR, 'TSMC_matrix_2000-2026.csv'),
        help='矩陣 CSV 路徑 (僅 Phase 4 使用)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _init_dirs()

    total_start = time.time()
    run_all = args.phase is None

    logger.info("=" * 55)
    logger.info("  台積電跨時空量化預測系統  啟動")
    logger.info(f"  Phase = {args.phase or 'ALL'}"
                f"  force={args.force}  skip-crawl={args.skip_crawl}")
    logger.info("=" * 55)

    if run_all or args.phase == 1:
        phase1(force=args.force, skip_crawl=args.skip_crawl)

    ptt_df = news_df = None
    if run_all or args.phase == 2:
        ptt_df, news_df = phase2(force=args.force)

    if run_all or args.phase == 3:
        # 若直接執行 Phase 3，需先取得情緒資料
        if ptt_df is None or news_df is None:
            ptt_df, news_df = phase2(force=False)
        phase3(ptt_df, news_df, force=args.force)

    if run_all or args.phase == 4:
        phase4(matrix_path=args.matrix)

    elapsed = time.time() - total_start
    logger.info(f"\n{'='*55}")
    logger.info(f"  全部完成 — 總耗時 {elapsed:.0f}s")
    logger.info(f"  輸出目錄: {OUTPUT_DIR}")
    logger.info(f"{'='*55}")


if __name__ == '__main__':
    main()
