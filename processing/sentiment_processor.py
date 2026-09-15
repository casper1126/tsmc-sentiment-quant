"""
Phase 2B: Text2Vec 情緒分析處理器
模型 : shibing624/text2vec-base-chinese
方法 : 對每筆標題計算 embedding，分別與 Bullish / Bearish
       錨定向量 (多句平均) 計算 Cosine Similarity，
       再標準化為 0~1 情緒分數。
規則 :
  - 當日有資料 → score = sim_bull / (sim_bull + sim_bear)
  - 當日無資料 → 嚴格填補中立值 0.5 (不借用前後日，避免資料污染)
  - 若無法載入 Text2Vec → 自動退回關鍵字版本 (fallback)
"""
import os
import sys
import json
import logging
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)

NEUTRAL = 0.5          # 無資料填補值
_model  = None         # 懶加載快取
_bull_vec: np.ndarray | None = None
_bear_vec: np.ndarray | None = None

# ─── 關鍵字備援 (Fallback) ───────────────────────────────────────────────────
_BULL_KW = ['看漲', '買進', '做多', '大漲', '突破', '強勢', '加碼', '創高',
            '上漲', '漲停', '多頭', '樂觀', 'bullish', '上攻', '利多']
_BEAR_KW = ['看跌', '賣出', '做空', '暴跌', '跌破', '崩盤', '弱勢', '砍倉',
            '下跌', '跌停', '空頭', '恐慌', 'bearish', '利空', '重挫']


def _keyword_score(title: str) -> float:
    """簡易關鍵字情緒分數 (作為 Text2Vec 無法載入時的備援)。"""
    bull = sum(1 for kw in _BULL_KW if kw in title)
    bear = sum(1 for kw in _BEAR_KW if kw in title)
    if bull + bear == 0:
        return NEUTRAL
    return bull / (bull + bear)


# ─── Text2Vec 模型 ────────────────────────────────────────────────────────────

def _get_model():
    """懶加載 sentence-transformers 模型。"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"載入 Text2Vec 模型: {TEXT2VEC_MODEL}")
            _model = SentenceTransformer(TEXT2VEC_MODEL)
            logger.info("Text2Vec 模型載入完成")
        except Exception as exc:
            logger.warning(f"Text2Vec 載入失敗，改用關鍵字備援: {exc}")
            _model = 'FALLBACK'
    return _model


def _get_anchor_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """建構/快取 Bullish / Bearish 錨定向量。"""
    global _bull_vec, _bear_vec
    if _bull_vec is not None:
        return _bull_vec, _bear_vec

    mdl = _get_model()
    if mdl == 'FALLBACK':
        # 回傳假向量 (不會實際使用)
        _bull_vec = np.ones(128)
        _bear_vec = np.zeros(128)
        return _bull_vec, _bear_vec

    logger.info("計算 Bullish/Bearish 錨定向量...")
    bull_embs  = mdl.encode(BULLISH_SENTENCES, show_progress_bar=False,
                             normalize_embeddings=True)
    bear_embs  = mdl.encode(BEARISH_SENTENCES, show_progress_bar=False,
                             normalize_embeddings=True)
    _bull_vec  = np.mean(bull_embs, axis=0)
    _bear_vec  = np.mean(bear_embs, axis=0)
    # 正規化錨定向量
    _bull_vec /= np.linalg.norm(_bull_vec) + 1e-12
    _bear_vec /= np.linalg.norm(_bear_vec) + 1e-12
    logger.info("錨定向量計算完成")
    return _bull_vec, _bear_vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """計算兩向量的 Cosine Similarity。"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ─── 核心評分 ────────────────────────────────────────────────────────────────

def titles_to_score(titles: list[str]) -> float:
    """
    將標題列表轉換為 0~1 情緒分數。

    流程:
      1. 對每個標題取 embedding
      2. 分別與 bull_vec / bear_vec 計算 cosine similarity
      3. score = sim_bull / (sim_bull + sim_bear)  → [0, 1]
      4. 取所有標題分數的平均
    """
    if not titles:
        return NEUTRAL

    mdl = _get_model()
    if mdl == 'FALLBACK':
        return float(np.mean([_keyword_score(t) for t in titles]))

    bull_vec, bear_vec = _get_anchor_vectors()
    try:
        embs   = mdl.encode(titles, show_progress_bar=False,
                             normalize_embeddings=True)
        scores = []
        for emb in embs:
            sim_b  = _cosine(emb, bull_vec)
            sim_br = _cosine(emb, bear_vec)
            total  = sim_b + sim_br
            scores.append(sim_b / total if total > 1e-12 else NEUTRAL)
        return float(np.mean(scores))
    except Exception as exc:
        logger.warning(f"Text2Vec 評分失敗，改用關鍵字: {exc}")
        return float(np.mean([_keyword_score(t) for t in titles]))


# ─── JSONL 處理 ──────────────────────────────────────────────────────────────

def process_jsonl(jsonl_path: str,
                  score_col: str,
                  volume_col: str,
                  source: str) -> pd.DataFrame:
    """
    讀取 {date, titles[], volume} JSONL，逐日計算情緒分數。

    回傳 DataFrame [Date, score_col, volume_col]，
    缺失日期不在此補 (由 matrix_builder 統一補 0.5)。
    """
    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL 不存在: {jsonl_path}")
        return pd.DataFrame(columns=['Date', score_col, volume_col])

    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    logger.info(f"[{source}] 處理 {len(lines)} 天的標題...")
    _get_anchor_vectors()          # 預熱錨定向量

    for i, line in enumerate(lines):
        try:
            data   = json.loads(line)
            titles = data.get('titles', [])
            vol    = int(data.get('volume', 0))
            score  = titles_to_score(titles)
            records.append({
                'Date':     pd.to_datetime(data['date']),
                score_col:  round(score, 6),
                volume_col: vol,
            })
        except Exception as exc:
            logger.debug(f"[{source}] 第 {i} 行失敗: {exc}")

        if (i + 1) % 200 == 0:
            logger.info(f"  [{source}] {i+1}/{len(lines)} 完成")

    df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)

    # 儲存快取
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, f'{source.lower()}_sentiment.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    logger.info(f"[{source}] 情緒分數已儲存: {out}")
    return df


# ─── 主函式 ──────────────────────────────────────────────────────────────────

def compute_all_sentiments(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    計算並快取 PTT + 新聞 情緒分數。
    回傳 (ptt_df, news_df)，欄位分別為:
      ptt_df  → [Date, PTT_Raw_Score, PTT_Volume]
      news_df → [Date, News_Raw_Score, News_Volume]
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ptt_cache  = os.path.join(PROCESSED_DIR, 'ptt_sentiment.csv')
    news_cache = os.path.join(PROCESSED_DIR, 'news_sentiment.csv')

    # PTT
    if not force and os.path.exists(ptt_cache):
        logger.info(f"載入 PTT 情緒快取: {ptt_cache}")
        ptt_df = pd.read_csv(ptt_cache, parse_dates=['Date'])
    else:
        ptt_jsonl = os.path.join(RAW_DIR, 'ptt', 'ptt_titles.jsonl')
        ptt_df    = process_jsonl(ptt_jsonl, 'PTT_Raw_Score', 'PTT_Volume', 'PTT')

    # 新聞
    if not force and os.path.exists(news_cache):
        logger.info(f"載入新聞情緒快取: {news_cache}")
        news_df = pd.read_csv(news_cache, parse_dates=['Date'])
    else:
        news_jsonl = os.path.join(RAW_DIR, 'news', 'news_titles.jsonl')
        news_df    = process_jsonl(news_jsonl, 'News_Raw_Score', 'News_Volume', 'news')

    return ptt_df, news_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    ptt, news = compute_all_sentiments()
    print("PTT:\n",  ptt.describe())
    print("News:\n", news.describe())
