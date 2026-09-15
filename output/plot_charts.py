"""
生成兩張精美學術圖表:
  1. ablation_bar_chart.png   — 消融實驗分組柱狀圖
  2. coverage_trend_chart.png — NLP資料覆蓋率時間趨勢線
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 字型 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'Heiti TC', 'PingFang TC',
    'STHeiti', 'Microsoft JhengHei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 圖一：消融實驗分組柱狀圖（六個近期滾動期間）
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_bar():
    df = pd.read_csv(os.path.join(OUT_DIR, 'best_model_2021_2026.csv'))

    labels = df['期間'].tolist()
    bl     = df['永遠猜漲(%)'].values

    models = [
        ('① 純NLP情緒版',  df['① 純NLP情緒版(%)'].values,  '#e74c3c'),
        ('② 純技術大盤版', df['② 純技術大盤版(%)'].values, '#2980b9'),
        ('③ 大盤+新聞版',  df['③ 大盤+新聞版(%)'].values,  '#27ae60'),
        ('④ 大盤+PTT版',   df['④ 大盤+PTT版(%)'].values,   '#e67e22'),
        ('⑤ 終極完全體',   df['⑤ 終極完全體(%)'].values,   '#8e44ad'),
    ]

    n      = len(labels)
    n_models = len(models)
    bar_w  = 0.14
    x      = np.arange(n)

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # 永遠猜漲虛線（每組一條）
    for xi, bv in zip(x, bl):
        ax.plot([xi - 0.40, xi + 0.40], [bv, bv],
                color='#7f8c8d', lw=1.6, ls='--', zorder=5)

    # 模型柱狀
    for mi, (name, vals, color) in enumerate(models):
        offset = (mi - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      color=color, alpha=0.88,
                      edgecolor='white', linewidth=0.8,
                      label=name, zorder=4)
        for bar, v, bv in zip(bars, vals, bl):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.35, f'{v:.1f}%',
                    ha='center', va='bottom',
                    fontsize=7.2, color=color,
                    fontweight='bold', zorder=6)
            if v > bv + 1.5:
                ax.annotate('▲', xy=(bar.get_x() + bar.get_width() / 2, v + 2.2),
                            ha='center', fontsize=6, color=color, zorder=7)

    bl_patch = mpatches.Patch(color='#7f8c8d', label='── 永遠猜漲基準線 (實際漲天%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(38, 88)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylabel('預測準確度 (%)', fontsize=12, labelpad=10)
    ax.set_xlabel('測試期間', fontsize=12, labelpad=10)
    ax.set_title(
        '台積電 (2330.TW)  量化預測模型 — 六期間消融實驗（2021–2026）\n'
        '演算法: RF + XGBoost VotingClassifier  |  訓練窗格: 滾動 5 年',
        fontsize=14, fontweight='bold', pad=18,
    )
    ax.grid(axis='y', color='#dfe6e9', linestyle='--', linewidth=0.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    handles, llabels = ax.get_legend_handles_labels()
    ax.legend(handles + [bl_patch], llabels + [bl_patch.get_label()],
              loc='upper right', fontsize=9.5, ncol=2,
              framealpha=0.92, edgecolor='#dfe6e9')

    # 最佳模型標示（每組右上角）
    best_col = df['最佳模型'].tolist() if '最佳模型' in df.columns else []
    for xi, bm in zip(x, best_col):
        ax.text(xi, 87.0, f'最佳:{bm[:2]}', ha='center', va='bottom',
                fontsize=6.5, color='#2d3436',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#b2bec3', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, 'ablation_bar_chart.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f'✅ 消融實驗柱狀圖 → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 圖二：NLP 資料覆蓋率時間趨勢線
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_trend():
    df = pd.read_csv(
        os.path.join(os.path.dirname(OUT_DIR), 'output', 'TSMC_matrix_2000-2026.csv'),
        parse_dates=['Date'],
    )

    NEUTRAL = 0.5
    df['year'] = df['Date'].dt.year

    rows = []
    for yr, sub in df.groupby('year'):
        ptt  = (sub['PTT_Raw_Score']  != NEUTRAL).mean() * 100
        news = (sub['News_Raw_Score'] != NEUTRAL).mean() * 100
        rows.append({'year': yr, 'ptt': ptt, 'news': news,
                     'combined': (ptt + news) / 2, 'n': len(sub)})
    cov = pd.DataFrame(rows)

    KEY_YEARS = [2008, 2011, 2015, 2018, 2020, 2022, 2025]
    EVENTS = {
        2008: '金融\n海嘯', 2011: '歐債\n危機', 2015: '半導體\n寒冬',
        2018: '貿易\n戰', 2020: 'COVID', 2022: '激進\n升息', 2025: 'AI\n狂熱',
    }

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#f8f9fa')

    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.8, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for ax in (ax1, ax2):
        ax.set_facecolor('#f8f9fa')

    yrs = cov['year'].values

    # ── 上圖：覆蓋率趨勢線 ────────────────────────────────────────────────────
    # 填色區域
    ax1.fill_between(yrs, 0, cov['news'].values,
                     alpha=0.18, color='#27ae60', label='_nolegend_')
    ax1.fill_between(yrs, 0, cov['ptt'].values,
                     alpha=0.22, color='#e67e22', label='_nolegend_')

    # 新聞覆蓋率線
    ax1.plot(yrs, cov['news'].values,
             color='#27ae60', lw=2.8, marker='o', ms=7,
             markerfacecolor='white', markeredgewidth=2,
             label='新聞情緒覆蓋率 (Taiwan News)')
    # PTT覆蓋率線
    ax1.plot(yrs, cov['ptt'].values,
             color='#e67e22', lw=2.8, marker='s', ms=7,
             markerfacecolor='white', markeredgewidth=2,
             label='PTT 情緒覆蓋率')
    # 合計均值
    ax1.plot(yrs, cov['combined'].values,
             color='#8e44ad', lw=1.8, ls=':', marker='^', ms=6,
             markerfacecolor='white', markeredgewidth=1.5,
             label='平均覆蓋率 (兩源均值)')

    # 數值標籤（僅非零）
    for _, row in cov.iterrows():
        if row['news'] > 3:
            ax1.text(row['year'], row['news'] + 1.8, f"{row['news']:.0f}%",
                     ha='center', fontsize=7.5, color='#27ae60', fontweight='bold')
        if row['ptt'] > 3:
            ax1.text(row['year'], row['ptt'] + 1.8, f"{row['ptt']:.0f}%",
                     ha='center', fontsize=7.5, color='#e67e22', fontweight='bold')

    # 關鍵年份垂直標注
    for yr in KEY_YEARS:
        ax1.axvline(yr, color='#636e72', lw=0.9, ls='--', alpha=0.5)

    # 階段分區說明
    phases = [
        (2000, 2012, '#e8f4f8', '無 NLP 資料期\n(技術指標為主)'),
        (2013, 2018, '#e8f8ef', '新聞萌芽期\n(新聞初步覆蓋)'),
        (2019, 2021, '#fff8e8', 'NLP 成長期\n(PTT + 新聞)'),
        (2022, 2026, '#f0ebf8', 'NLP 成熟期\n(99%新聞覆蓋)'),
    ]
    for x0, x1, fc, txt in phases:
        ax1.axvspan(x0 - 0.4, x1 + 0.4, alpha=0.35, color=fc, zorder=0)
        ax1.text((x0 + x1) / 2, 92, txt,
                 ha='center', va='top', fontsize=8.5,
                 color='#2d3436', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=fc,
                           edgecolor='#b2bec3', alpha=0.9))

    ax1.set_ylim(-5, 108)
    ax1.set_xlim(1999.5, 2026.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.set_ylabel('情緒資料覆蓋率 (%)', fontsize=12, labelpad=10)
    ax1.set_title(
        '台積電量化研究 — NLP 情緒資料覆蓋率時間趨勢\n'
        '涵蓋 6,633 個交易日  (2000-03-02 ～ 2026-04-15)',
        fontsize=14, fontweight='bold', pad=16,
    )
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.92,
               edgecolor='#dfe6e9', ncol=3)
    ax1.grid(axis='y', color='#dfe6e9', linestyle='--', linewidth=0.7)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(labelbottom=False)

    # ── 下圖：每年交易日數 & 關鍵事件標注 ────────────────────────────────────
    bar_colors = ['#c0392b' if yr in KEY_YEARS else '#bdc3c7' for yr in yrs]
    ax2.bar(yrs, cov['n'].values, color=bar_colors, alpha=0.75,
            edgecolor='white', linewidth=0.5)

    for yr in KEY_YEARS:
        sub_n = cov[cov['year'] == yr]['n'].values
        if len(sub_n):
            ax2.text(yr, sub_n[0] + 3, EVENTS.get(yr, ''),
                     ha='center', va='bottom', fontsize=7.5,
                     color='#c0392b', fontweight='bold')

    ax2.set_ylabel('交易日數', fontsize=10, labelpad=10)
    ax2.set_xlabel('年份', fontsize=12, labelpad=8)
    ax2.set_ylim(0, 310)
    ax2.set_xticks(range(2000, 2027))
    ax2.set_xticklabels([str(y) if y % 2 == 0 else '' for y in range(2000, 2027)],
                         fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', color='#dfe6e9', linestyle='--', linewidth=0.5)

    # 圖例說明
    red_patch  = mpatches.Patch(color='#c0392b', alpha=0.75, label='消融實驗測試年份')
    grey_patch = mpatches.Patch(color='#bdc3c7', alpha=0.75, label='一般年份')
    ax2.legend(handles=[red_patch, grey_patch], loc='upper right',
               fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'coverage_trend_chart.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f'✅ 覆蓋率趨勢圖 → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 圖三：13期間合併長條圖（7個關鍵歷史年 + 6個近期滾動期間）
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_bar():
    """
    合併 ablation_results.csv (7大歷史關鍵年) 與
         best_model_2021_2026.csv (6個近期滾動期間)
    畫出 13期間 × 5模型 分組柱狀圖，覆蓋並取代舊版 ablation_bar_chart.png。
    """
    # ── 讀取並統一格式 ────────────────────────────────────────────────────────
    abl  = pd.read_csv(os.path.join(OUT_DIR, 'ablation_results.csv'))
    best = pd.read_csv(os.path.join(OUT_DIR, 'best_model_2021_2026.csv'))

    MODEL_COLS = ['① 純NLP情緒版(%)', '② 純技術大盤版(%)',
                  '③ 大盤+新聞版(%)',  '④ 大盤+PTT版(%)',
                  '⑤ 終極完全體(%)']

    # 7 個歷史年 — 標籤格式「年份\n情境」
    hist_rows = []
    for _, r in abl.iterrows():
        hist_rows.append({
            'label':     f"{int(r['Year'])}\n{r['情境']}",
            'group':     'hist',
            '永遠猜漲(%)': r['永遠猜漲(%)'],
            **{c: r[c] for c in MODEL_COLS},
        })

    # 6 個近期期間 — 標籤格式「期間」（換行）
    recent_rows = []
    for _, r in best.iterrows():
        label = r['期間'].replace('全年', '\n全年').replace('-', '\n', 1) \
                if '/' in r['期間'] else r['期間'].replace('全年', '\n全年')
        recent_rows.append({
            'label':     label,
            'group':     'recent',
            '永遠猜漸(%)': r['永遠猜漲(%)'],
            '永遠猜漲(%)': r['永遠猜漲(%)'],
            **{c: r[c] for c in MODEL_COLS},
        })

    all_rows = hist_rows + recent_rows
    labels   = [r['label']       for r in all_rows]
    bl       = np.array([r['永遠猜漲(%)'] for r in all_rows], dtype=float)
    groups   = [r['group']       for r in all_rows]

    models = [
        ('① 純NLP情緒版',  '① 純NLP情緒版(%)',  '#e74c3c'),
        ('② 純技術大盤版', '② 純技術大盤版(%)', '#2980b9'),
        ('③ 大盤+新聞版',  '③ 大盤+新聞版(%)',  '#27ae60'),
        ('④ 大盤+PTT版',   '④ 大盤+PTT版(%)',   '#e67e22'),
        ('⑤ 終極完全體',   '⑤ 終極完全體(%)',   '#8e44ad'),
    ]

    n_total  = len(all_rows)
    n_models = len(models)
    bar_w    = 0.13
    x        = np.arange(n_total)

    fig, ax = plt.subplots(figsize=(26, 9))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # ── 區域分隔線（歷史 vs 近期）────────────────────────────────────────────
    split = len(hist_rows) - 0.5
    ax.axvline(split, color='#2d3436', lw=2.0, ls='--', alpha=0.6, zorder=6)
    ax.text(split - 3.3, 86.5, '◀ 7 大歷史關鍵年（壓力測試）',
            ha='center', fontsize=9, color='#636e72', style='italic')
    ax.text(split + 3.3, 86.5, '6 個近期滾動期間（2021–2026）▶',
            ha='center', fontsize=9, color='#636e72', style='italic')

    # ── 區域底色 ──────────────────────────────────────────────────────────────
    ax.axvspan(-0.5, split, alpha=0.04, color='#3498db', zorder=0)
    ax.axvspan(split, n_total - 0.5, alpha=0.04, color='#e67e22', zorder=0)

    # ── 永遠猜漲虛線（每組一條）─────────────────────────────────────────────
    for xi, bv in zip(x, bl):
        ax.plot([xi - 0.42, xi + 0.42], [bv, bv],
                color='#7f8c8d', lw=1.8, ls='--', zorder=5)

    # ── 五模型柱狀 ────────────────────────────────────────────────────────────
    for mi, (name, col, color) in enumerate(models):
        offset = (mi - n_models / 2 + 0.5) * bar_w
        vals   = np.array([r.get(col, np.nan) for r in all_rows], dtype=float)
        bars   = ax.bar(x + offset, vals, bar_w,
                        color=color, alpha=0.87,
                        edgecolor='white', linewidth=0.7,
                        label=name, zorder=4)
        for bar, v, bv in zip(bars, vals, bl):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.35, f'{v:.1f}%',
                    ha='center', va='bottom',
                    fontsize=6.2, color=color,
                    fontweight='bold', zorder=6)
            # ▲ 超越永遠猜漲超過 10% 才標
            if v > bv + 10:
                ax.annotate('▲',
                    xy=(bar.get_x() + bar.get_width() / 2, v + 2.3),
                    ha='center', fontsize=5.5, color=color, zorder=7)

    # ── 裝飾 ──────────────────────────────────────────────────────────────────
    bl_patch = mpatches.Patch(color='#7f8c8d',
                               label='── 永遠猜漲基準線 (實際漲天%)')
    handles, llabels = ax.get_legend_handles_labels()
    ax.legend(handles + [bl_patch], llabels + [bl_patch.get_label()],
              loc='upper right', fontsize=8.5, ncol=3,
              framealpha=0.92, edgecolor='#dfe6e9')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, fontweight='bold')
    ax.set_ylim(35, 90)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylabel('預測準確度 (%)', fontsize=12, labelpad=10)
    ax.set_xlabel('測試期間', fontsize=12, labelpad=10)
    ax.set_title(
        '台積電 (2330.TW) 量化預測模型 — 完整消融實驗總覽\n'
        '7 大歷史關鍵壓力測試年  +  6 個近期滾動期間（2021–2026）'
        '  |  演算法: RF + XGBoost VotingClassifier',
        fontsize=13, fontweight='bold', pad=16,
    )
    ax.grid(axis='y', color='#dfe6e9', linestyle='--', linewidth=0.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # ── NLP 品質標示（歷史年）────────────────────────────────────────────────
    quality_hist = {0: '✗無', 1: '✗無', 2: '★★中', 3: '★★高',
                    4: '★★★高', 5: '★★高', 6: '★★★高'}
    for xi, (r, q) in enumerate(zip(all_rows, [quality_hist.get(i,'') for i in range(n_total)])):
        if r['group'] == 'hist' and q:
            ax.text(xi, 89.2, f'NLP:{q}', ha='center', va='bottom',
                    fontsize=6.5, color='#636e72',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#b2bec3', alpha=0.82))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, 'ablation_bar_chart.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f'✅ 合併消融長條圖（13期間）→ {path}')


if __name__ == '__main__':
    plot_combined_bar()      # 取代並覆蓋舊版 ablation_bar_chart.png
    plot_coverage_trend()
    print('\n圖表已儲存至 output/ 目錄。')
