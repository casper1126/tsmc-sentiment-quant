"""
Phase 1B: PTT 股版爬蟲
目標: 爬取 PTT Stock 板含「台積電/2330/TSMC」關鍵字的文章標題與留言數

策略:
  - 直接爬 https://www.ptt.cc/bbs/Stock (可回溯約 3000+ 頁, 覆蓋 2005~2026)
  - 利用文章 permalink 的 M 檔 timestamp 精確還原日期
  - 聚合為日級 (Date, PTT_Volume, titles_jsonl)
  - 2000-2005 無 PTT 資料 → 後續 Matrix Builder 填補中立值 0.5

注意: PTT 有 over18 cookie，本爬蟲自動附帶；請勿高速爬取 (delay ≥ 0.3s)
"""
import os
import re
import sys
import json
import time
import logging
from datetime import datetime, date

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)

PTT_DIR  = os.path.join(RAW_DIR, 'ptt')
PTT_CACHE = os.path.join(PTT_DIR, 'ptt_daily_raw.csv')
PTT_JSONL = os.path.join(PTT_DIR, 'ptt_titles.jsonl')

# ─── Session 設定 ────────────────────────────────────────────────────────────
_session = requests.Session()
_session.cookies.set('over18', '1', domain='www.ptt.cc')
_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/122.0 Safari/537.36'),
    'Referer': f'https://www.ptt.cc/bbs/{PTT_BOARD}/index.html',
}


# ─── 輔助函式 ────────────────────────────────────────────────────────────────

def _get_last_page() -> int:
    """取得 Stock 板最後頁碼。"""
    url = f'{PTT_BASE}/bbs/{PTT_BOARD}/index.html'
    try:
        r = _session.get(url, headers=_HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        for a in soup.select('.btn-group-paging a'):
            href = a.get('href', '')
            m = re.search(r'index(\d+)\.html', href)
            if m and '上頁' in a.text:
                return int(m.group(1)) + 1
    except Exception as e:
        logger.warning(f"無法取得最後頁碼: {e}")
    return 4000


def _parse_push_count(text: str) -> int:
    """將爆/X1/數字 → 整數推文數。"""
    text = text.strip()
    if text == '爆':
        return 100
    m = re.match(r'^X(\d+)$', text)
    if m:
        return -int(m.group(1)) * 10
    try:
        return int(text)
    except ValueError:
        return 0


def _parse_page(page_num: int) -> list[dict]:
    """解析一頁 Stock 板，回傳 TSMC 相關文章 list。"""
    url = f'{PTT_BASE}/bbs/{PTT_BOARD}/index{page_num}.html'
    posts = []
    try:
        r = _session.get(url, headers=_HEADERS, timeout=15)
        if r.status_code == 404:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')

        for ent in soup.find_all('div', class_='r-ent'):
            title_div = ent.find('div', class_='title')
            if not title_div or not title_div.a:
                continue
            title = title_div.a.get_text(strip=True)
            if not any(kw in title for kw in PTT_KEYWORDS):
                continue

            href = title_div.a.get('href', '')
            date_div = ent.find('div', class_='date')
            date_str = date_div.get_text(strip=True) if date_div else ''
            nrec_div = ent.find('div', class_='nrec')
            push = _parse_push_count(nrec_div.get_text() if nrec_div else '0')

            # 從文章連結提取日期 (格式: /bbs/Stock/M.XXXXXXXXXX.A.XXX.html)
            ts_match = re.search(r'M\.(\d+)\.', href)
            post_date = None
            if ts_match:
                try:
                    post_date = datetime.fromtimestamp(int(ts_match.group(1))).date()
                except (ValueError, OSError):
                    pass

            # 若無法從連結取得日期，用頁面日期字串估算
            if post_date is None:
                try:
                    m, d = map(int, date_str.split('/'))
                    # 估算年份: 以今天為基準，頁碼差估算年份
                    pages_from_now = _last_page_cache - page_num
                    years_ago = pages_from_now / 300  # ~300 頁/年
                    yr = datetime.now().year - int(years_ago)
                    yr = max(2005, min(datetime.now().year, yr))
                    post_date = date(yr, m, d)
                except Exception:
                    continue

            posts.append({
                'date': post_date,
                'title': title,
                'push_count': push,
            })
    except Exception as exc:
        logger.warning(f"頁面 {page_num} 解析失敗: {exc}")
    return posts


_last_page_cache = 4000  # 模組層快取，供估算年份使用


# ─── 主爬蟲 ─────────────────────────────────────────────────────────────────

PTT_POSTS_CSV = os.path.join(PTT_DIR, 'ptt_posts_raw.csv')   # ← 逐筆可讀


def crawl_ptt(max_pages: int = 5000, delay: float = 0.35,
              force: bool = False) -> pd.DataFrame:
    """
    爬取 PTT Stock 板，輸出日級 DataFrame。

    儲存檔案:
      ptt_posts_raw.csv   — 逐筆：Date, Title, push_count  ← 人工可讀檢查
      ptt_daily_raw.csv   — 日級：Date, PTT_Volume, avg_push
      ptt_titles.jsonl    — 日級 JSONL，供 Text2Vec 情緒計算使用
    """
    global _last_page_cache
    os.makedirs(PTT_DIR, exist_ok=True)

    if not force and os.path.exists(PTT_CACHE):
        logger.info(f"載入 PTT 快取: {PTT_CACHE}")
        return pd.read_csv(PTT_CACHE, parse_dates=['Date'])

    logger.info("=== PTT 股版爬蟲啟動 ===")
    last_page = _get_last_page()
    _last_page_cache = last_page
    logger.info(f"目前最後頁碼: {last_page}")

    all_posts: list[dict] = []
    stop_year = int(START_DATE[:4])

    for i, pg in enumerate(range(last_page, max(last_page - max_pages, 1), -1)):
        posts = _parse_page(pg)
        if posts:
            all_posts.extend(posts)
            oldest_year = min(p['date'].year for p in posts)
            if oldest_year < stop_year:
                logger.info(f"已達到目標起始年 {stop_year}，停止爬取")
                break

        if i % 200 == 0 and i > 0:
            logger.info(f"  已爬 {i} 頁，累計 {len(all_posts)} 篇 TSMC 文章")

        time.sleep(delay)

    if not all_posts:
        logger.warning("PTT 爬蟲無任何資料")
        return pd.DataFrame(columns=['Date', 'PTT_Volume', 'avg_push'])

    df = pd.DataFrame(all_posts)
    df['date'] = pd.to_datetime(df['date'])

    # ── 1. 逐筆可讀 CSV（人工檢查用）────────────────────────────────────────
    flat = df.rename(columns={'date': 'Date', 'title': 'Title',
                               'push_count': 'Push_Count'})
    flat = flat.sort_values(['Date', 'Push_Count'], ascending=[True, False])
    flat.to_csv(PTT_POSTS_CSV, index=False, encoding='utf-8-sig')
    logger.info(f"PTT 逐筆資料已儲存: {PTT_POSTS_CSV} ({len(flat)} 筆)")

    # ── 2. 日級聚合 ──────────────────────────────────────────────────────────
    daily = (
        df.groupby('date')
        .agg(PTT_Volume=('title', 'count'),
             avg_push=('push_count', 'mean'),
             titles=('title', list))
        .reset_index()
        .rename(columns={'date': 'Date'})
        .sort_values('Date')
    )

    # 儲存 JSONL (供 Text2Vec 使用)
    with open(PTT_JSONL, 'w', encoding='utf-8') as f:
        for _, row in daily.iterrows():
            json.dump({'date': str(row['Date'].date()),
                       'titles': row['titles'],
                       'volume': int(row['PTT_Volume'])},
                      f, ensure_ascii=False)
            f.write('\n')

    out = daily[['Date', 'PTT_Volume', 'avg_push']].copy()
    out.to_csv(PTT_CACHE, index=False, encoding='utf-8-sig')
    logger.info(f"PTT 資料已儲存: {PTT_CACHE} ({len(out)} 天)")
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    df = crawl_ptt(max_pages=500)
    print(df.tail(10))
