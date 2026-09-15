"""
Phase 1C: 財經新聞爬蟲
來源 1: 鉅亨網 (cnyes.com) — 台積電股票專區 API
  1a: 即時/近期新聞 (tw_stock_news, 無日期篩選, ~42 頁)
  1b: 歷史新聞   (tw_stock_news + startAt/endAt 時間戳, 2013-2025)
來源 2: Yahoo! Finance 台灣 — 台積電新聞

輸出:
  data/raw/news/news_articles_raw.csv        (Date, Title, Source)   — 逐筆，近期
  data/raw/news/news_articles_historical.csv (Date, Title, Source)   — 逐筆，歷史
  data/raw/news/news_daily_raw.csv           (Date, News_Volume)     — 日級
  data/raw/news/news_titles.jsonl            (date, titles[], volume) — 供 Text2Vec
"""
import os
import re
import sys
import json
import time
import logging
from datetime import datetime, date, timedelta
from calendar import monthrange

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)

NEWS_DIR        = os.path.join(RAW_DIR, 'news')
NEWS_CACHE      = os.path.join(NEWS_DIR, 'news_daily_raw.csv')
NEWS_JSONL      = os.path.join(NEWS_DIR, 'news_titles.jsonl')
NEWS_POSTS_CSV  = os.path.join(NEWS_DIR, 'news_articles_raw.csv')
NEWS_HIST_CSV   = os.path.join(NEWS_DIR, 'news_articles_historical.csv')  # 歷史爬蟲結果

_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'Chrome/122.0 Safari/537.36'),
    'Accept': 'application/json, text/html',
    'Referer': 'https://news.cnyes.com/',
}

_TSMC_CODES = {'2330', 'US-TSM', 'TSM'}
_TSMC_KW    = ['台積電', '2330', 'TSMC', 'TSM', 'CoWoS', 'SoIC', 'HBM', '台積']


def _is_tsmc(item: dict) -> bool:
    """判斷新聞是否與台積電相關 (stock code 或 title/keyword 關鍵字)。"""
    stocks = item.get('stock') or []
    if any(s in _TSMC_CODES for s in stocks):
        return True
    title   = item.get('title', '')
    keyword = str(item.get('keyword', ''))
    return any(kw in title or kw in keyword for kw in _TSMC_KW)


# ─── 鉅亨網 — 近期新聞 (無日期篩選) ────────────────────────────────────────────

_CNYES_URL = 'https://api.cnyes.com/media/api/v1/newslist/category/tw_stock_news'


def _fetch_cnyes_page(page: int,
                      start_ts: int | None = None,
                      end_ts: int | None = None,
                      limit: int = 30) -> tuple[list[dict], int]:
    """爬取鉅亨網台股新聞一頁，支援可選的 startAt/endAt 歷史篩選。"""
    params: dict = {'page': page, 'limit': limit}
    if start_ts is not None:
        params['startAt'] = start_ts
    if end_ts is not None:
        params['endAt'] = end_ts
    try:
        r = requests.get(_CNYES_URL, params=params,
                         headers=_HEADERS, timeout=20)
        if r.status_code == 200:
            obj = r.json().get('items', {})
            return obj.get('data', []), int(obj.get('last_page', 1))
    except Exception as exc:
        logger.warning(f"鉅亨網 API (page={page}): {exc}")
    return [], 1


def _parse_cnyes_item(item: dict, source: str = 'cnyes') -> dict | None:
    """解析單筆鉅亨網新聞，提取 date + title。"""
    try:
        title = item.get('title', '').strip()
        if not title:
            return None
        ts = item.get('publishAt', 0)
        pub_date = datetime.fromtimestamp(ts).date() if ts else None
        if pub_date is None:
            return None
        return {'date': pub_date, 'title': title, 'source': source}
    except Exception:
        return None


def crawl_cnyes(max_pages: int = 600, delay: float = 0.5) -> list[dict]:
    """
    爬取鉅亨網台股新聞 (近期，無日期篩選)，過濾出台積電相關文章。
    自動偵測總頁數，爬到 START_DATE 為止。
    """
    logger.info("  [鉅亨網-近期] 開始爬取...")
    records   = []
    start_dt  = datetime.strptime(START_DATE, '%Y-%m-%d').date()
    last_page = max_pages

    for pg in range(1, max_pages + 1):
        raw_items, lp = _fetch_cnyes_page(pg)

        if pg == 1:
            last_page = min(lp, max_pages)
            logger.info(f"  [鉅亨網-近期] 總頁數: {lp}，爬取上限: {last_page}")

        if not raw_items:
            logger.info(f"  [鉅亨網-近期] 第 {pg} 頁無資料，停止")
            break

        tsmc_items = [i for i in raw_items if _is_tsmc(i)]
        parsed = [p for p in (_parse_cnyes_item(i) for i in tsmc_items) if p]
        records.extend(parsed)

        if raw_items:
            oldest_ts   = min(i.get('publishAt', 9e9) for i in raw_items)
            oldest_date = datetime.fromtimestamp(oldest_ts).date()
            if oldest_date < start_dt:
                logger.info(f"  [鉅亨網-近期] 已達 {START_DATE}，停止")
                break

        if pg % 50 == 0:
            logger.info(f"  [鉅亨網-近期] {pg}/{last_page} 頁，累計 {len(records)} 筆")

        if pg >= last_page:
            break

        time.sleep(delay)

    logger.info(f"  [鉅亨網-近期] 完成: {len(records)} 筆台積電相關新聞")
    return records


# ─── 鉅亨網 — 歷史新聞 (startAt/endAt 月分批) ───────────────────────────────

def crawl_cnyes_historical(
        start_date: date | None = None,
        end_date:   date | None = None,
        delay: float = 0.25,
        force: bool = False,
) -> list[dict]:
    """
    使用 startAt/endAt 時間戳，逐月爬取鉅亨網歷史台積電新聞。

    API 特性:
      - 支援 2013-01-01 以後的歷史查詢
      - 每月 ~30-132 頁 (依新聞密度而定)
      - TSMC 相關約佔 5-10% (需逐頁過濾)

    策略: 逐月批次 → 每月分頁爬完 → 只保留 TSMC 相關文章。
    """
    if start_date is None:
        start_date = date(2013, 1, 1)   # cnyes 最早有效日期
    if end_date is None:
        end_date = date.today()

    # 若已有歷史 CSV 且不強制重新爬，直接讀取
    if not force and os.path.exists(NEWS_HIST_CSV):
        hist_df = pd.read_csv(NEWS_HIST_CSV, parse_dates=['Date'])
        hist_df.columns = [c.strip().lstrip('\ufeff') for c in hist_df.columns]
        existing_min = hist_df['Date'].min().date() if len(hist_df) else end_date
        existing_max = hist_df['Date'].max().date() if len(hist_df) else start_date

        logger.info(f"  [鉅亨網-歷史] 已有 CSV: {len(hist_df)} 筆 "
                    f"({existing_min} ~ {existing_max})")

        # 只補充尚未爬過的日期範圍
        need_earlier = start_date < existing_min
        need_later   = end_date   > existing_max + timedelta(days=7)  # 7天寬容

        if not need_earlier and not need_later:
            logger.info("  [鉅亨網-歷史] 已涵蓋所需範圍，跳過爬蟲")
            return hist_df.rename(columns={'Date': 'date', 'Title': 'title',
                                           'Source': 'source'}).to_dict('records')

        # 決定補充範圍
        fetch_start = start_date   if need_earlier else existing_max + timedelta(days=1)
        fetch_end   = end_date
        logger.info(f"  [鉅亨網-歷史] 補充範圍: {fetch_start} ~ {fetch_end}")
    else:
        existing_records: list[dict] = []
        fetch_start = start_date
        fetch_end   = end_date

    # ── 按月爬取 ──────────────────────────────────────────────────────────────
    new_records: list[dict] = []
    cur = date(fetch_start.year, fetch_start.month, 1)

    total_months = (fetch_end.year - cur.year) * 12 + (fetch_end.month - cur.month) + 1
    month_idx    = 0

    while cur <= fetch_end:
        month_idx += 1
        last_day = monthrange(cur.year, cur.month)[1]
        m_start  = cur
        m_end    = date(cur.year, cur.month, last_day)
        if m_end > fetch_end:
            m_end = fetch_end

        start_ts = int(datetime.combine(m_start, datetime.min.time()).timestamp())
        end_ts   = int(datetime.combine(m_end + timedelta(days=1),
                                        datetime.min.time()).timestamp())

        # 取得該月第 1 頁，確認總頁數
        raw_items, last_page = _fetch_cnyes_page(1, start_ts, end_ts, limit=30)
        if not raw_items:
            # 移至下個月
            cur = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
            continue

        month_records: list[dict] = []
        # 第 1 頁
        tsmc = [i for i in raw_items if _is_tsmc(i)]
        month_records.extend(filter(None, (_parse_cnyes_item(i, 'cnyes_hist') for i in tsmc)))

        # 後續頁
        for pg in range(2, last_page + 1):
            items, _ = _fetch_cnyes_page(pg, start_ts, end_ts, limit=30)
            if not items:
                break
            tsmc = [i for i in items if _is_tsmc(i)]
            month_records.extend(filter(None, (_parse_cnyes_item(i, 'cnyes_hist') for i in tsmc)))
            time.sleep(delay)

        new_records.extend(month_records)

        if month_idx % 6 == 0 or month_idx == total_months:
            logger.info(
                f"  [鉅亨網-歷史] {cur.year}/{cur.month:02d} "
                f"({month_idx}/{total_months}) "
                f"— 本月 {len(month_records)} 筆 TSMC，累計 {len(new_records)} 筆"
            )

        time.sleep(delay)
        # 移至下個月
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    logger.info(f"  [鉅亨網-歷史] 爬蟲完成: {len(new_records)} 筆新增 TSMC 新聞")

    # ── 合併既有 + 新增，儲存 ─────────────────────────────────────────────────
    all_records = new_records[:]
    if os.path.exists(NEWS_HIST_CSV):
        hist_df = pd.read_csv(NEWS_HIST_CSV, parse_dates=['Date'])
        hist_df.columns = [c.strip().lstrip('\ufeff') for c in hist_df.columns]
        existing_records = [
            {'date': r['Date'].date(), 'title': r['Title'], 'source': r['Source']}
            for _, r in hist_df.iterrows()
        ]
        all_records = existing_records + new_records

    if all_records:
        out_df = pd.DataFrame(all_records)
        out_df.columns = ['date', 'title', 'source'] if list(out_df.columns) == ['date', 'title', 'source'] else out_df.columns
        out_df = out_df.rename(columns={'date': 'Date', 'title': 'Title', 'source': 'Source'})
        out_df['Date'] = pd.to_datetime(out_df['Date'])
        out_df = out_df.drop_duplicates(subset=['Date', 'Title']).sort_values('Date')
        os.makedirs(NEWS_DIR, exist_ok=True)
        out_df.to_csv(NEWS_HIST_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"  [鉅亨網-歷史] 儲存 → {NEWS_HIST_CSV} ({len(out_df)} 筆)")

        return [
            {'date': r.Date.date(), 'title': r.Title, 'source': r.Source}
            for _, r in out_df.iterrows()
        ]

    return all_records


# ─── Yahoo Finance 台股新聞 ───────────────────────────────────────────────────

def _fetch_yahoo_news(delay: float = 1.5) -> list[dict]:
    """
    爬取 Yahoo Finance 台灣 2330 新聞 (近期 3~6 個月)。
    """
    logger.info("  [Yahoo] 開始爬取...")
    records  = []
    KEYWORDS = ['台積電', '2330', 'TSMC', 'TSM']

    # ── 方法 1: yfinance .news ─────────────────────────────────────────────────
    try:
        import yfinance as yf
        ticker = yf.Ticker('2330.TW')
        for item in ticker.news or []:
            title = item.get('title', '')
            if not any(k in title for k in KEYWORDS):
                continue
            ts = item.get('providerPublishTime', 0)
            if ts:
                records.append({'date': datetime.fromtimestamp(ts).date(),
                                 'title': title, 'source': 'yahoo_yf'})
        ticker_us = yf.Ticker('TSM')
        for item in ticker_us.news or []:
            title = item.get('title', '')
            ts    = item.get('providerPublishTime', 0)
            if ts and title:
                records.append({'date': datetime.fromtimestamp(ts).date(),
                                 'title': title, 'source': 'yahoo_yf_us'})
        logger.info(f"  [Yahoo-yfinance] {len(records)} 筆")
    except Exception as exc:
        logger.warning(f"  [Yahoo-yfinance] 失敗: {exc}")

    # ── 方法 2: Yahoo Finance RSS ─────────────────────────────────────────────
    rss_headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; QuantResearch/1.0)',
        'Accept': 'application/rss+xml, application/xml',
    }
    for rss_url in [YAHOO_RSS_TSM, YAHOO_RSS_2330]:
        try:
            r = requests.get(rss_url, headers=rss_headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'xml')
                for item in soup.find_all('item'):
                    title_tag = item.find('title')
                    pub_tag   = item.find('pubDate')
                    if not title_tag or not pub_tag:
                        continue
                    title = title_tag.get_text(strip=True)
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_date = parsedate_to_datetime(pub_tag.get_text()).date()
                    except Exception:
                        continue
                    records.append({'date': pub_date, 'title': title,
                                    'source': 'yahoo_rss'})
            time.sleep(delay)
        except Exception as exc:
            logger.warning(f"  [Yahoo-RSS] {rss_url}: {exc}")

    logger.info(f"  [Yahoo] 共取得 {len(records)} 筆 (含去重前)")
    return records


# ─── 整合並輸出 ───────────────────────────────────────────────────────────────

def _build_jsonl_from_all_sources(records: list[dict]) -> pd.DataFrame:
    """
    將新聞列表 (含歷史 + 近期 + Yahoo) 合併去重後，
    寫出 news_daily_raw.csv + news_titles.jsonl。

    與現有 JSONL 合併策略:
      - 讀取既有 JSONL (bootstrap 或上次爬蟲結果)
      - 新爬資料加入，同日去重
      - 輸出最終 JSONL
    """
    os.makedirs(NEWS_DIR, exist_ok=True)

    # ── 讀取現有 JSONL ─────────────────────────────────────────────────────────
    existing: dict[str, list[str]] = {}
    if os.path.exists(NEWS_JSONL):
        with open(NEWS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing[obj['date']] = obj.get('titles', [])
                except Exception:
                    pass

    # ── 合併新爬資料 ───────────────────────────────────────────────────────────
    new_df = pd.DataFrame(records)
    if new_df.empty:
        pass
    else:
        new_df['date'] = pd.to_datetime(new_df['date'])
        for dt, grp in new_df.groupby('date'):
            d = str(dt.date())
            new_titles = grp['title'].tolist()
            combined   = list(dict.fromkeys(existing.get(d, []) + new_titles))
            existing[d] = combined

    # ── 同時輸出逐筆 CSV (近期用) ─────────────────────────────────────────────
    if not new_df.empty:
        flat = new_df.rename(columns={'date': 'Date', 'title': 'Title',
                                       'source': 'Source'})
        flat = flat.sort_values('Date')
        flat[['Date', 'Title', 'Source']].to_csv(
            NEWS_POSTS_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"新聞逐筆資料: {NEWS_POSTS_CSV} ({len(flat)} 筆)")

    # ── 輸出 JSONL ─────────────────────────────────────────────────────────────
    daily_rows = []
    with open(NEWS_JSONL, 'w', encoding='utf-8') as f:
        for d in sorted(existing.keys()):
            titles = existing[d]
            if not titles:
                continue
            json.dump({'date': d, 'titles': titles, 'volume': len(titles)},
                      f, ensure_ascii=False)
            f.write('\n')
            daily_rows.append({'Date': pd.Timestamp(d), 'News_Volume': len(titles)})

    daily_df = pd.DataFrame(daily_rows).sort_values('Date')
    daily_df.to_csv(NEWS_CACHE, index=False, encoding='utf-8-sig')
    logger.info(f"新聞日級資料: {NEWS_CACHE} ({len(daily_df)} 天，JSONL 合計)")
    return daily_df[['Date', 'News_Volume']]


def crawl_news(force: bool = False) -> pd.DataFrame:
    """
    主函式: 爬取鉅亨網 (近期 + 歷史) + Yahoo 新聞，輸出日級資料。

    流程:
      1. crawl_cnyes()           → 近期全量 (~42 頁，最多 3-6 個月)
      2. crawl_cnyes_historical() → 歷史月分批 (2013~至今，有快取時跳過)
      3. _fetch_yahoo_news()     → Yahoo RSS / yfinance (最近 3 個月)
      4. _build_jsonl_from_all_sources() → 合併現有 JSONL + 新爬，輸出
    """
    if not force and os.path.exists(NEWS_CACHE) and os.path.exists(NEWS_JSONL):
        logger.info(f"載入新聞快取: {NEWS_CACHE}")
        return pd.read_csv(NEWS_CACHE, parse_dates=['Date'])

    logger.info("=== 財經新聞爬蟲啟動 ===")
    all_records: list[dict] = []

    # 1. 近期鉅亨網
    cnyes_recent = crawl_cnyes()
    all_records.extend(cnyes_recent)

    # 2. 歷史鉅亨網 (2013-01-01 ~ 今天，有快取則增量補充)
    hist_records = crawl_cnyes_historical(
        start_date=date(2013, 1, 1),
        end_date=date.today(),
        force=force,
    )
    all_records.extend(hist_records)

    # 3. Yahoo Finance
    yahoo_records = _fetch_yahoo_news()
    all_records.extend(yahoo_records)

    if not all_records:
        logger.warning("新聞爬蟲無資料，回傳空 DataFrame")
        return pd.DataFrame(columns=['Date', 'News_Volume'])

    # 4. 去重 + 合併現有 JSONL + 輸出
    df = pd.DataFrame(all_records).drop_duplicates(subset=['date', 'title'])
    return _build_jsonl_from_all_sources(df.to_dict('records'))


# ─── 快速補充: 僅跑歷史爬蟲 (不重爬近期) ──────────────────────────────────────

def refresh_historical_only(force: bool = False) -> pd.DataFrame:
    """
    只更新歷史爬蟲結果並重建 JSONL，不重新爬近期/Yahoo。
    適合在已有快取的情況下只補充歷史缺口。
    """
    logger.info("=== 歷史新聞補充爬蟲 ===")
    hist_records = crawl_cnyes_historical(
        start_date=date(2013, 1, 1),
        end_date=date.today(),
        force=force,
    )
    if not hist_records:
        logger.warning("歷史爬蟲無新資料")
        return pd.read_csv(NEWS_CACHE, parse_dates=['Date']) if os.path.exists(NEWS_CACHE) \
               else pd.DataFrame(columns=['Date', 'News_Volume'])

    df = pd.DataFrame(hist_records).drop_duplicates(subset=['date', 'title'])
    return _build_jsonl_from_all_sources(df.to_dict('records'))


# ─── 補充: 讀取已有快取標題 ──────────────────────────────────────────────────

def load_existing_cache(cache_csv: str) -> pd.DataFrame:
    if not os.path.exists(cache_csv):
        return pd.DataFrame()
    df = pd.read_csv(cache_csv, parse_dates=['Date'])
    if 'News_Titles' not in df.columns:
        return df
    os.makedirs(NEWS_DIR, exist_ok=True)
    records = []
    with open(NEWS_JSONL, 'a', encoding='utf-8') as f:
        for _, row in df.iterrows():
            try:
                titles = json.loads(row['News_Titles'])
                if isinstance(titles, str):
                    titles = [titles]
            except Exception:
                titles = [str(row['News_Titles'])]
            vol = len(titles)
            json.dump({'date': str(row['Date'].date()), 'titles': titles, 'volume': vol},
                      f, ensure_ascii=False)
            f.write('\n')
            records.append({'Date': row['Date'], 'News_Volume': vol})
    return pd.DataFrame(records)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # 直接執行時: 只跑歷史補充 (不重爬近期)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full',  action='store_true', help='完整爬蟲 (近期+歷史+Yahoo)')
    parser.add_argument('--force', action='store_true', help='強制重新爬蟲')
    args = parser.parse_args()

    if args.full:
        df = crawl_news(force=args.force)
    else:
        df = refresh_historical_only(force=args.force)

    print(f"\n完成! 日級資料 {len(df)} 天")
    print(df.groupby(df['Date'].dt.year)['News_Volume'].agg(['count','sum']).rename(
        columns={'count':'days','sum':'titles'}))
