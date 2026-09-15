"""
全域設定檔 — 台積電跨時空量化預測系統
"""
import os

# ── 時間範圍 ─────────────────────────────────────────────────────────────────
START_DATE = '2000-01-01'
END_DATE   = '2026-04-24'

# ── 目錄路徑 ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, 'data')
RAW_DIR       = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'output')

# ── 股票代碼 ─────────────────────────────────────────────────────────────────
TSMC_TW = '2330.TW'   # 台灣交易所
TSMC_US = 'TSM'       # 美股 ADR
VIX_SYM = '^VIX'
SOX_SYM = '^SOX'
FX_SYM  = 'USDTWD=X'

# ── PTT 設定 ─────────────────────────────────────────────────────────────────
PTT_BASE     = 'https://www.ptt.cc'
PTT_BOARD    = 'Stock'
# 關鍵字擴充：涵蓋代號/英文縮寫/製程節點/封裝技術等近年常用詞
PTT_KEYWORDS = [
    '台積電', '2330', 'TSMC',          # 核心關鍵字
    'TSM',                              # 美股 ADR 代號
    'CoWoS', 'SoIC',                   # 先進封裝技術（2022+ 常見）
    'N3', 'N2', '2nm', '3nm',          # 製程節點討論
    'HBM',                             # 高頻寬記憶體（AI晶片需求）
    '台積',                             # 縮寫（需搭配過濾避免誤抓）
]

# ── 鉅亨網 API (2025版 正確 endpoint) ────────────────────────────────────────
# 注意: TWS:2330:STOCK 舊 endpoint 已廢棄 (HTTP 422)
# 正確: tw_stock_news 類別 + _is_tsmc() 過濾台積電相關新聞
CNYES_NEWS_URL   = 'https://api.cnyes.com/media/api/v1/newslist/category/tw_stock_news'

# ── Yahoo Finance 新聞 RSS ────────────────────────────────────────────────────
YAHOO_RSS_TSM   = 'https://finance.yahoo.com/rss/headline?s=TSM'
YAHOO_RSS_2330  = 'https://finance.yahoo.com/rss/headline?s=2330.TW'

# ── Text2Vec 模型 ─────────────────────────────────────────────────────────────
TEXT2VEC_MODEL = 'shibing624/text2vec-base-chinese'

# ── Bullish / Bearish 錨定句子 ────────────────────────────────────────────────
BULLISH_SENTENCES = [
    '台積電股價大漲，市場情緒極度樂觀，投資人大量買進，做多看漲',
    '台積電業績超預期，法人瘋狂加碼，股價創新高，多頭氣勢如虹',
    '半導體產業景氣大好，台積電訂單滿載，強勢上攻，全力買進',
    '台積電AI需求爆發，外資強力買超，突破壓力區，漲勢強勁',
    '台積電獲利創新高，股利大增，前景看漲，多方信心滿滿買進',
]

BEARISH_SENTENCES = [
    '台積電股價暴跌，市場恐慌性拋售，投資人爭相出逃，做空看跌',
    '台積電業績不如預期，外資大量賣超，股價重挫，空頭氣勢如虹',
    '半導體庫存過剩，台積電砍單，跌破支撐，全力賣出',
    '台積電面臨嚴重利空，崩盤風險極高，恐慌賣壓湧現，空方獲勝',
    '台積電獲利衰退，股利縮減，前景看空，空方信心滿滿放空',
]

# ── 技術指標參數 ──────────────────────────────────────────────────────────────
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL_P   = 9
ROLL_WINDOWS    = [5, 10, 20, 42]
VOL_WINDOW      = 42
POSITION_WINDOW = 42

# ── ML 模型參數 ───────────────────────────────────────────────────────────────
TRAIN_YEARS = 5
KEY_YEARS   = [2008, 2011, 2015, 2018, 2020, 2022, 2025]

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'min_samples_leaf': 20,
    'random_state': 42,
    'n_jobs': -1,
}

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}
