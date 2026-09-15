"""
台積電跨時空量化預測模型 — 消融實驗引擎
=========================================
演算法  : VotingClassifier (RandomForest + XGBoost, soft voting)
切分方式: 滾動時間窗格 (Train=過去5年, Test=當年)  ← 嚴格無 Look-ahead
消融組合: 5 種特徵子集
壓力測試: 7 大關鍵年份 (2008/2011/2015/2018/2020/2022/2025)
輸出    :
  1. Pandas DataFrame 對比表格 (console + CSV)
  2. 年度準確度折線圖 + Alpha 條形圖 (PNG)
"""
import os
import sys
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')              # 非互動後端，確保伺服器/遠端環境可用
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ── 設定中文字型 ──────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'Heiti TC', 'PingFang TC',
    'STHeiti', 'Microsoft JhengHei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

# ─── 消融實驗特徵定義 ─────────────────────────────────────────────────────────
TECH   = ['RSI_14', 'MACD', 'MACD_Signal',
          'Ret_5D', 'Ret_10D', 'Ret_20D', 'Ret_42D',
          'Vol_42D', 'Position_42D', 'Volume']
MACRO  = ['VIX_Close', 'SOX_Ret', 'ADR_Ret', 'FX_Ret']
NEWS   = ['News_Raw_Score', 'News_Volume']
PTT    = ['PTT_Raw_Score', 'PTT_Volume']

ABLATION_SETS = {
    '① 純NLP情緒版':        NEWS + PTT,
    '② 純技術大盤版':        TECH + MACRO,
    '③ 大盤+新聞版':         TECH + MACRO + NEWS,
    '④ 大盤+PTT版':          TECH + MACRO + PTT,
    '⑤ 終極完全體':          TECH + MACRO + NEWS + PTT,
}

# 年份情境標籤 (用於圖表)
YEAR_CONTEXT = {
    2008: '金融海嘯',
    2011: '歐債/庫存',
    2015: '半導體寒冬',
    2018: '中美貿易戰',
    2020: 'COVID-19',
    2022: '激進升息',
    2025: 'AI狂熱',
}

# 顏色 & 標記方案
STYLE = {
    'Baseline':          dict(color='#95a5a6', ls='--', marker='D', lw=1.8, ms=7),
    '① 純NLP情緒版':     dict(color='#e74c3c', ls='-',  marker='v', lw=2.2, ms=7),
    '② 純技術大盤版':    dict(color='#3498db', ls='-',  marker='s', lw=2.2, ms=7),
    '③ 大盤+新聞版':     dict(color='#2ecc71', ls='-',  marker='^', lw=2.2, ms=7),
    '④ 大盤+PTT版':      dict(color='#f39c12', ls='-',  marker='o', lw=2.2, ms=7),
    '⑤ 終極完全體':      dict(color='#9b59b6', ls='-',  marker='*', lw=2.8, ms=11),
}


# ─── 模型建構 ─────────────────────────────────────────────────────────────────

def _build_classifier() -> VotingClassifier:
    """建構 RF + XGBoost 軟投票分類器。"""
    rf = RandomForestClassifier(**RF_PARAMS)
    try:
        xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    except TypeError:
        # 舊版 XGBoost 不支援某些參數
        safe_params = {k: v for k, v in XGB_PARAMS.items()
                       if k not in ('use_label_encoder',)}
        xgb_clf = xgb.XGBClassifier(**safe_params)
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf)],
        voting='soft', weights=[1, 1],
    )


# ─── 時間序列切分 ─────────────────────────────────────────────────────────────

def _time_split(df: pd.DataFrame,
                test_year: int,
                train_years: int = TRAIN_YEARS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    滾動時間序列切分 (嚴格無 Look-ahead)。
      Train: [test_year - train_years, test_year - 1] 的完整曆年
      Test : test_year 全年
    """
    yr = df['Date'].dt.year
    train = df[(yr >= test_year - train_years) & (yr < test_year)].copy()
    test  = df[yr == test_year].copy()
    return train, test


# ─── 單次消融評估 ─────────────────────────────────────────────────────────────

def _run_one(train: pd.DataFrame,
             test: pd.DataFrame,
             features: list[str]) -> Tuple[float, float, int]:
    """
    訓練並評估一次模型。
    回傳 (model_accuracy%, baseline_accuracy%, n_test_days)
    """
    avail = [f for f in features
             if f in train.columns and f in test.columns]
    if not avail:
        return np.nan, np.nan, 0

    TARGET = 'Target_Binary'
    tr = train[avail + [TARGET]].dropna()
    te = test[avail + [TARGET]].dropna()

    if len(tr) < 60 or len(te) < 10:
        logger.debug(f"資料不足 (train={len(tr)}, test={len(te)})")
        return np.nan, np.nan, len(te)

    X_tr, y_tr = tr[avail].values, tr[TARGET].values
    X_te, y_te = te[avail].values, te[TARGET].values

    # StandardScaler 僅 fit 於訓練集
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = _build_classifier()
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    model_acc    = accuracy_score(y_te, y_pred) * 100
    # 「永遠猜漲」基準: 該年實際上漲天數比例
    # (與多數類別 baseline 不同，直接反映市場方向性)
    baseline_acc = y_te.mean() * 100
    return model_acc, baseline_acc, len(te)


# ─── 完整消融實驗 ─────────────────────────────────────────────────────────────

def run_ablation_study(matrix_path: str,
                        key_years: list[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    對 key_years 中每個年份執行 5 種消融模型訓練與測試。

    回傳:
        output_df   — 格式化對比表格 (供顯示)
        raw_df      — 原始數值 DataFrame (供繪圖)
    """
    if key_years is None:
        key_years = KEY_YEARS

    logger.info(f"載入矩陣: {matrix_path}")
    df = pd.read_csv(matrix_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    logger.info(f"矩陣: {len(df)} 筆 "
                f"({df.Date.min().date()} ~ {df.Date.max().date()})")

    rows = []
    for year in key_years:
        logger.info(f"\n{'='*55}")
        logger.info(f"壓力測試年份: {year}  [{YEAR_CONTEXT.get(year,'')}]")
        logger.info(f"{'='*55}")
        train, test = _time_split(df, year, TRAIN_YEARS)

        row = {'Year': year}
        baseline_pct = np.nan
        n_days       = 0

        for name, feats in ABLATION_SETS.items():
            acc, bl, nd = _run_one(train, test, feats)
            row[name] = acc
            if np.isnan(baseline_pct) and not np.isnan(bl):
                baseline_pct = bl
                n_days       = nd
            alpha = acc - baseline_pct if not np.isnan(acc) else np.nan
            logger.info(
                f"  {name:<18} acc={acc:5.1f}%  "
                f"baseline={baseline_pct:5.1f}%  alpha={alpha:+.1f}%"
            )

        row['Baseline']   = baseline_pct
        row['N_Test_Days'] = n_days
        row['Alpha_Complete'] = (row.get('⑤ 終極完全體', np.nan) - baseline_pct
                                  if not np.isnan(baseline_pct) else np.nan)
        rows.append(row)

    raw_df = pd.DataFrame(rows).set_index('Year')

    # ─── 格式化對比表 ─────────────────────────────────────────────────────
    out = pd.DataFrame(index=raw_df.index)
    out.index.name = 'Year'
    out['情境']          = [YEAR_CONTEXT.get(y, '') for y in raw_df.index]
    out['測試天數']       = raw_df['N_Test_Days'].astype(int)
    out['永遠猜漲(%)']   = raw_df['Baseline'].round(1)   # 該年實際上漲天數 %
    for name in ABLATION_SETS:
        out[f'{name}(%)'] = raw_df[name].round(1)
    out['⑤終極完全體_Alpha(%)'] = raw_df['Alpha_Complete'].round(1)

    # 儲存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_csv = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    out.to_csv(result_csv, encoding='utf-8-sig')
    logger.info(f"\n結果已儲存: {result_csv}")

    return out, raw_df


# ─── 視覺化 ───────────────────────────────────────────────────────────────────

def plot_ablation(output_df: pd.DataFrame,
                  raw_df: pd.DataFrame,
                  save_path: str | None = None) -> None:
    """
    繪製雙子圖:
      上圖: 年度準確度折線圖 (含 Baseline 灰帶)
      下圖: Alpha 消融階梯條形圖 (各模型 vs Baseline)
    """
    years = list(raw_df.index)
    x     = np.arange(len(years))
    x_labels = [f"{y}\n({YEAR_CONTEXT.get(y,'')})" for y in years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 13),
                                    gridspec_kw={'height_ratios': [1.2, 1]})
    fig.suptitle(
        '台積電 (2330.TW) 跨時空量化預測模型 — 7大關鍵年份消融實驗\n'
        '滾動訓練窗格: 5年 Train → 1年 Test  |  演算法: RF + XGBoost VotingClassifier',
        fontsize=13, fontweight='bold', y=0.99,
    )

    # ── 上圖: 準確度折線 ───────────────────────────────────────────────────
    bl_vals = raw_df['Baseline'].values
    ax1.fill_between(x, 40, bl_vals, alpha=0.12, color='#95a5a6')
    ax1.plot(x, bl_vals, label='永遠猜漲基準線 (該年實際漲天%)',
             **STYLE['Baseline'])
    ax1.axhline(50, color='black', lw=0.8, ls=':', alpha=0.5,
                label='50% 隨機猜測線')

    for name in ABLATION_SETS:
        vals = raw_df[name].values
        sty  = STYLE[name].copy()
        ax1.plot(x, vals, label=name, **sty)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax1.annotate(
                    f'{v:.0f}%',
                    xy=(xi, v),
                    xytext=(0, 9 if name == '⑤ Complete' else -13),
                    textcoords='offset points',
                    ha='center', fontsize=7.5,
                    color=sty['color'], fontweight='bold',
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_ylabel('預測準確度 (%)', fontsize=11)
    ax1.set_ylim(38, 88)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_facecolor('#fafafa')

    # ── 下圖: Alpha 條形 ───────────────────────────────────────────────────
    n_models  = len(ABLATION_SETS)
    bar_w     = 0.14
    names     = list(ABLATION_SETS.keys())

    for i, name in enumerate(names):
        offset = (i - n_models / 2 + 0.5) * bar_w
        alphas = (raw_df[name] - raw_df['Baseline']).values
        sty    = STYLE[name]
        bars   = ax2.bar(x + offset, alphas, bar_w,
                          color=sty['color'], alpha=0.82,
                          edgecolor='white', linewidth=0.5, label=name)
        for bar, av in zip(bars, alphas):
            if not np.isnan(av):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if av >= 0 else -1.0),
                    f'{av:+.1f}%',
                    ha='center', va='bottom', fontsize=6.5,
                    color=sty['color'], fontweight='bold',
                )

    ax2.axhline(0, color='black', lw=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_ylabel('Alpha (模型準確度 - Baseline %)', fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
    ax2.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_facecolor('#fafafa')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"圖表已儲存: {save_path}")
    plt.show()


# ─── 假說驗證報告 ─────────────────────────────────────────────────────────────

def print_hypothesis_report(raw_df: pd.DataFrame) -> None:
    """列印 3 大假說的驗證摘要。"""
    sep = '-' * 65
    print(f"\n{'='*65}")
    print("【研究假說驗證報告】")
    print(f"{'='*65}")

    groups = [
        ("假說 1【極端恐慌】 PTT 情緒捕捉非理性恐慌",
         [2008, 2020], '② 純技術大盤版', '④ 大盤+PTT版'),
        ("假說 2【總經巨變】 News 捕捉法人籌碼，大盤+新聞 更強",
         [2018, 2022], '② 純技術大盤版', '③ 大盤+新聞版'),
        ("假說 3【產業循環/散戶狂熱】 終極完全體應最強",
         [2011, 2015, 2025], '② 純技術大盤版', '⑤ 終極完全體'),
    ]

    for title, yrs, model_a, model_b in groups:
        print(f"\n{sep}")
        print(f" {title}")
        print(sep)
        print(f"  {'年份':<6} {'永遠猜漲':>9} {model_a:>15} {model_b:>15}  {'提升':>8}")
        for yr in yrs:
            if yr not in raw_df.index:
                continue
            bl = raw_df.loc[yr, 'Baseline']
            a  = raw_df.loc[yr, model_a]
            b  = raw_df.loc[yr, model_b]
            print(f"  {yr:<6} {bl:>8.1f}%  {a:>14.1f}%  {b:>14.1f}%  "
                  f"{b-a:>+7.1f}%")

    print(f"\n{'='*65}\n")


# ─── 主函式 ───────────────────────────────────────────────────────────────────

def print_data_coverage_report(matrix_path: str) -> None:
    """
    列印各年份情緒資料覆蓋率報告。
    供學術簡報使用，明確揭露 NLP 資料品質。
    """
    df = pd.read_csv(matrix_path, parse_dates=['Date'])
    NEUTRAL = 0.5
    df['year'] = df['Date'].dt.year

    sep = '─' * 65
    print(f"\n{'='*65}")
    print("【情緒資料覆蓋率報告 — 學術透明度揭露】")
    print(f"{'='*65}")
    print(f"  矩陣總筆數 : {len(df):,} 天  ({df.Date.min().date()} ~ {df.Date.max().date()})")
    ptt_total  = (df['PTT_Raw_Score']  != NEUTRAL).sum()
    news_total = (df['News_Raw_Score'] != NEUTRAL).sum()
    print(f"  PTT 情緒   : {ptt_total:4d} 天有真實資料 ({ptt_total/len(df)*100:.1f}%覆蓋)")
    print(f"  新聞情緒   : {news_total:4d} 天有真實資料 ({news_total/len(df)*100:.1f}%覆蓋)")
    print(f"\n  {'年份':^4}  {'交易日':^6}  {'PTT覆蓋':^9}  {'新聞覆蓋':^9}  {'NLP品質':^8}")
    print(f"  {sep}")

    for yr in sorted(df['year'].unique()):
        sub = df[df['year'] == yr]
        p = (sub['PTT_Raw_Score']  != NEUTRAL).mean() * 100
        n = (sub['News_Raw_Score'] != NEUTRAL).mean() * 100
        avg = (p + n) / 2
        quality = ('★★★ 高' if avg >= 50 else
                   '★★  中' if avg >= 15 else
                   '★   低' if avg >  0  else '✗ 無資料')
        star = '◀ 測試年' if yr in KEY_YEARS else ''
        print(f"  {yr}  {len(sub):6d}  {p:8.1f}%  {n:8.1f}%  {quality:<8} {star}")

    print(f"\n  ⚠ 注意: 無資料年份情緒欄位以中立值 0.5 填補，不影響技術指標計算。")
    print(f"  ✓ 結論: NLP 預測能力主要體現於 2013年後（新聞）及 2019年後（PTT）。")
    print(f"{'='*65}\n")


def run_prediction_study(matrix_path: str | None = None) -> None:
    """完整執行消融實驗 → 輸出表格 → 列印假說報告 → 繪圖。"""
    if matrix_path is None:
        matrix_path = os.path.join(OUTPUT_DIR, 'TSMC_matrix_2000-2026.csv')

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"矩陣檔不存在: {matrix_path}\n"
            "請先執行 main.py 完成 Phase 1~3。"
        )

    logger.info("\n" + "=" * 55)
    logger.info("台積電跨時空量化預測模型 — 消融實驗啟動")
    logger.info("=" * 55)

    # 先列印資料覆蓋率 (學術透明度)
    print_data_coverage_report(matrix_path)

    output_df, raw_df = run_ablation_study(matrix_path)

    # 輸出對比表格
    print("\n" + "=" * 80)
    print("【台積電量化預測模型 — 7大關鍵年份消融實驗結果】")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.1f}'.format)
    print(output_df.to_string())
    print("=" * 80)

    # 假說驗證
    print_hypothesis_report(raw_df)

    # 繪圖
    chart_path = os.path.join(OUTPUT_DIR, 'ablation_ladder_chart.png')
    plot_ablation(output_df, raw_df, save_path=chart_path)
    print(f"\n圖表已儲存 → {chart_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    run_prediction_study()
