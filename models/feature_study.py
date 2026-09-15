"""
特徵重要度 & 最佳化研究 — 5模型 × 6期間 = 30 組
================================================
目標: 針對 2021–2025 及 2025/4–2026/4 共六個期間，
      在 5 種消融模型下各自計算特徵重要度比重。

輸出:
  output/feature_importance_30.csv   — 30 組 (5模型×6期間) 特徵重要度
  output/best_model_2021_2026.csv    — 六期間消融結果 + 永遠猜漲
  output/feature_importance_heatmap/ — 每個模型一張熱力圖 (5張)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

warnings.filterwarnings('ignore')

# ── 字型 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'Heiti TC', 'PingFang TC',
    'STHeiti', 'Microsoft JhengHei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

# ── 特徵定義（與 prediction_model.py 完全一致）──────────────────────────────
TECH  = ['RSI_14', 'MACD', 'MACD_Signal',
         'Ret_5D', 'Ret_10D', 'Ret_20D', 'Ret_42D',
         'Vol_42D', 'Position_42D', 'Volume']
MACRO = ['VIX_Close', 'SOX_Ret', 'ADR_Ret', 'FX_Ret']
NEWS  = ['News_Raw_Score', 'News_Volume']
PTT_F = ['PTT_Raw_Score', 'PTT_Volume']
ALL_FEATURES = TECH + MACRO + NEWS + PTT_F

ABLATION_SETS = {
    '① 純NLP情緒版':  NEWS + PTT_F,
    '② 純技術大盤版':  TECH + MACRO,
    '③ 大盤+新聞版':   TECH + MACRO + NEWS,
    '④ 大盤+PTT版':    TECH + MACRO + PTT_F,
    '⑤ 終極完全體':    TECH + MACRO + NEWS + PTT_F,
}

# ── 六個測試期間定義 ──────────────────────────────────────────────────────────
# (label, train_start, train_end, test_start, test_end)
import datetime as dt

PERIODS = [
    ('2021全年',
     dt.date(2016,1,1), dt.date(2020,12,31),
     dt.date(2021,1,1), dt.date(2021,12,31)),
    ('2022全年',
     dt.date(2017,1,1), dt.date(2021,12,31),
     dt.date(2022,1,1), dt.date(2022,12,31)),
    ('2023全年',
     dt.date(2018,1,1), dt.date(2022,12,31),
     dt.date(2023,1,1), dt.date(2023,12,31)),
    ('2024全年',
     dt.date(2019,1,1), dt.date(2023,12,31),
     dt.date(2024,1,1), dt.date(2024,12,31)),
    ('2025全年',
     dt.date(2020,1,1), dt.date(2024,12,31),
     dt.date(2025,1,1), dt.date(2025,12,31)),
    ('2025/04-2026/04',
     dt.date(2020,4,1), dt.date(2025,3,31),
     dt.date(2025,4,1), dt.date(2026,4,30)),
]

# ── 模型建構 ──────────────────────────────────────────────────────────────────
def _build_clf():
    rf  = RandomForestClassifier(**RF_PARAMS)
    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    return VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf)],
        voting='soft', weights=[1, 1],
    )


# ── 單次訓練 + 特徵重要度 ─────────────────────────────────────────────────────
def _train_and_importance(train: pd.DataFrame,
                          test: pd.DataFrame,
                          features: list) -> tuple[float, float, dict]:
    """
    回傳 (accuracy%, always_up%, {feature: importance_score})
    importance = RF 與 XGB 特徵重要度的平均，再正規化為加總 1。
    """
    avail = [f for f in features if f in train.columns and f in test.columns]
    TARGET = 'Target_Binary'
    tr = train[avail + [TARGET]].dropna()
    te = test[avail + [TARGET]].dropna()

    if len(tr) < 60 or len(te) < 10:
        return np.nan, np.nan, {}

    X_tr, y_tr = tr[avail].values, tr[TARGET].values
    X_te, y_te = te[avail].values, te[TARGET].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = _build_clf()
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    acc      = accuracy_score(y_te, y_pred) * 100
    always_up = y_te.mean() * 100

    # 從 RF 和 XGB 各取特徵重要度，平均後正規化
    rf_imp  = clf.estimators_[0].feature_importances_
    xgb_imp = clf.estimators_[1].feature_importances_
    avg_imp = (rf_imp + xgb_imp) / 2
    total   = avg_imp.sum()
    norm_imp = avg_imp / total if total > 0 else avg_imp

    imp_dict = {feat: float(imp) for feat, imp in zip(avail, norm_imp)}
    return acc, always_up, imp_dict


# ── 主研究流程 ────────────────────────────────────────────────────────────────
def run_feature_study(matrix_path: str):
    print("載入矩陣...")
    df = pd.read_csv(matrix_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  {len(df)} 筆  ({df.Date.min().date()} ~ {df.Date.max().date()})")

    importance_rows = []  # 30 組 (5模型 × 6期間) 特徵重要度
    best_model_rows = []  # 6期間消融彙總

    for (label, tr_s, tr_e, te_s, te_e) in PERIODS:
        print(f"\n{'='*65}")
        print(f"  期間: {label}  |  訓練: {tr_s}~{tr_e}  |  測試: {te_s}~{te_e}")
        print(f"{'='*65}")

        train = df[(df['Date'].dt.date >= tr_s) &
                   (df['Date'].dt.date <= tr_e)].copy()
        test  = df[(df['Date'].dt.date >= te_s) &
                   (df['Date'].dt.date <= te_e)].copy()
        print(f"  訓練集: {len(train)} 天  測試集: {len(test)} 天")

        always_up = None
        best_acc, best_name = -1, ''

        best_model_row = {
            '期間': label,
            '測試開始': str(te_s), '測試結束': str(te_e),
            '測試天數': len(test),
        }

        # ── 5 種消融模型逐一訓練 ─────────────────────────────────────────────
        for model_name, feats in ABLATION_SETS.items():
            acc, up_pct, imp = _train_and_importance(train, test, feats)

            # 第一個有效模型決定 always_up（所有模型測試集一致）
            if always_up is None and not np.isnan(up_pct):
                always_up = up_pct

            # 最佳模型追蹤
            if not np.isnan(acc) and acc > best_acc:
                best_acc, best_name = acc, model_name

            best_model_row[f'{model_name}(%)'] = round(acc, 1) if not np.isnan(acc) else np.nan

            # ── 特徵重要度列 ─────────────────────────────────────────────────
            # 不在本模型特徵集的欄位填 0（維持全量欄位一致）
            imp_row = {
                '期間': label,
                '測試開始': str(te_s), '測試結束': str(te_e),
                '測試天數': len(test),
                '模型': model_name,
                '準確度(%)': round(acc, 1) if not np.isnan(acc) else np.nan,
                '永遠猜漲(%)': round(up_pct, 1) if not np.isnan(up_pct) else np.nan,
                'Alpha(%)': round(acc - up_pct, 1) if not np.isnan(acc) else np.nan,
                '使用特徵數': len(feats),
            }
            for feat in ALL_FEATURES:
                imp_row[f'IMP_{feat}(%)'] = round(imp.get(feat, 0) * 100, 2)

            importance_rows.append(imp_row)

            # 控制台列印
            acc_str = f"{acc:.1f}%" if not np.isnan(acc) else "N/A"
            marker  = " ★最佳" if model_name == best_name else ""
            print(f"\n  [{model_name}]  準確度: {acc_str}{marker}")
            if imp:
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                for rank, (feat, score) in enumerate(sorted_imp[:5], 1):
                    bar = '█' * int(score * 150)
                    print(f"    #{rank} {feat:<20} {score*100:5.1f}%  {bar}")
                if len(sorted_imp) > 5:
                    others = sum(v for _, v in sorted_imp[5:])
                    print(f"    ... 其餘 {len(sorted_imp)-5} 個特徵合計 {others*100:.1f}%")

        always_up = always_up or 0.0
        best_model_row['永遠猜漲(%)']      = round(always_up, 1)
        best_model_row['最佳模型']          = best_name
        best_model_row['最佳準確度(%)']     = round(best_acc, 1)
        best_model_row['Alpha(最佳-猜漲)'] = round(best_acc - always_up, 1)
        best_model_rows.append(best_model_row)

    # ── 儲存結果 ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    heatmap_dir = os.path.join(OUTPUT_DIR, 'feature_importance_heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)

    imp_df  = pd.DataFrame(importance_rows)
    best_df = pd.DataFrame(best_model_rows)

    imp_path  = os.path.join(OUTPUT_DIR, 'feature_importance_30.csv')
    best_path = os.path.join(OUTPUT_DIR, 'best_model_2021_2026.csv')
    imp_df.to_csv(imp_path,  index=False, encoding='utf-8-sig')
    best_df.to_csv(best_path, index=False, encoding='utf-8-sig')

    # ── 彙總列印 ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("【六期間 × 五模型 消融結果彙總】")
    print('='*70)
    cols = ['期間', '永遠猜漲(%)',
            '① 純NLP情緒版(%)', '② 純技術大盤版(%)',
            '③ 大盤+新聞版(%)', '④ 大盤+PTT版(%)',
            '⑤ 終極完全體(%)', '最佳模型', '最佳準確度(%)', 'Alpha(最佳-猜漲)']
    pd.set_option('display.width', 200)
    print(best_df[cols].to_string(index=False))

    print(f"\n✅ 30組特徵重要度 → {imp_path}")
    print(f"✅ 消融彙總       → {best_path}")

    # ── 繪製熱力圖（每個模型一張，共 5 張）──────────────────────────────────
    for model_name in ABLATION_SETS:
        sub = imp_df[imp_df['模型'] == model_name]
        _plot_importance_heatmap(sub, model_name, heatmap_dir)

    # 另外畫一張「最佳準確度對應模型」的比較圖
    _plot_best_accuracy_summary(best_df)

    return imp_df, best_df


# ── 熱力圖（單一模型，橫軸=6期間，縱軸=該模型使用的特徵）────────────────────
def _plot_importance_heatmap(imp_df: pd.DataFrame,
                              model_name: str,
                              out_dir: str):
    feat_cols = [c for c in imp_df.columns if c.startswith('IMP_') and c.endswith('(%)')]
    # 只保留該模型實際有使用的特徵（重要度 > 0 的）
    nonzero_mask = imp_df[feat_cols].values.max(axis=0) > 0
    feat_cols    = [c for c, nz in zip(feat_cols, nonzero_mask) if nz]
    feats        = [c.replace('IMP_', '').replace('(%)', '') for c in feat_cols]
    periods      = imp_df['期間'].tolist()
    acc_vals     = imp_df['準確度(%)'].tolist()
    up_vals      = imp_df['永遠猜漲(%)'].tolist()

    data = imp_df[feat_cols].values.T   # (n_features, n_periods)

    # 依全期平均排序（高在上）
    avg_imp = data.mean(axis=1)
    order   = np.argsort(avg_imp)[::-1]
    data_s  = data[order]
    feats_s = [feats[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, max(7, len(feats_s) * 0.55 + 2)))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    vmax = max(data_s.max(), 1.0)
    im   = ax.imshow(data_s, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)

    for i in range(len(feats_s)):
        for j in range(len(periods)):
            val = data_s[i, j]
            if val == 0:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=8, color='#b2bec3')
            else:
                txt_col = 'white' if val > vmax * 0.65 else '#2d3436'
                ax.text(j, i, f'{val:.1f}%',
                        ha='center', va='center', fontsize=8.5,
                        color=txt_col, fontweight='bold')

    # X 軸：期間 + 準確度標注
    x_labels = [f"{p}\n準確={a:.1f}%\n猜漲={u:.1f}%"
                 for p, a, u in zip(periods, acc_vals, up_vals)]
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_yticks(range(len(feats_s)))
    ax.set_yticklabels(feats_s, fontsize=9)

    # 特徵分組色條（左側）
    group_colors = {'TECH': '#3498db', 'MACRO': '#2ecc71',
                    'NEWS': '#e67e22', 'PTT':   '#e74c3c'}
    def _group(f):
        if f in TECH:  return 'TECH'
        if f in MACRO: return 'MACRO'
        if f in NEWS:  return 'NEWS'
        return 'PTT'

    for i, feat in enumerate(feats_s):
        ax.add_patch(plt.Rectangle((-0.5 - 0.3, i - 0.5), 0.25, 1,
                                    color=group_colors[_group(feat)],
                                    clip_on=False))

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=k)
                        for k, c in group_colors.items()],
              loc='upper right', fontsize=8, framealpha=0.9,
              bbox_to_anchor=(1.19, 1.02))

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('特徵重要度 (%)', fontsize=9)

    safe_name = model_name.replace('/', '_').replace(' ', '_')
    ax.set_title(
        f'特徵重要度熱力圖 — {model_name}\n'
        f'(RF + XGB 平均  |  訓練窗格: 5年滾動  |  六測試期間)',
        fontsize=11, fontweight='bold', pad=14,
    )
    ax.set_xlabel('測試期間', fontsize=10, labelpad=8)
    ax.set_ylabel('特徵名稱', fontsize=10, labelpad=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f'heatmap_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"✅ 熱力圖 [{model_name}] → {path}")


# ── 最佳準確度彙總比較圖 ──────────────────────────────────────────────────────
def _plot_best_accuracy_summary(best_df: pd.DataFrame):
    periods  = best_df['期間'].tolist()
    x        = np.arange(len(periods))
    bar_w    = 0.13

    model_colors = {
        '① 純NLP情緒版':  '#e74c3c',
        '② 純技術大盤版':  '#3498db',
        '③ 大盤+新聞版':   '#2ecc71',
        '④ 大盤+PTT版':    '#f39c12',
        '⑤ 終極完全體':    '#9b59b6',
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # 永遠猜漲虛線
    up_vals = best_df['永遠猜漲(%)'].values
    for xi, uv in zip(x, up_vals):
        ax.plot([xi - 0.38, xi + 0.38], [uv, uv],
                color='#636e72', lw=1.8, ls='--', zorder=5)

    # 五模型柱狀
    names = list(model_colors.keys())
    for mi, name in enumerate(names):
        col_key = f'{name}(%)'
        if col_key not in best_df.columns:
            continue
        vals   = best_df[col_key].values
        offset = (mi - len(names) / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        color=model_colors[name], alpha=0.87,
                        edgecolor='white', linewidth=0.7,
                        label=name, zorder=4)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.4, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=6.8,
                        color=model_colors[name], fontweight='bold')

    # 最佳模型標星
    for xi, row in zip(x, best_df.itertuples()):
        ax.text(xi, row._10 + 2.2, '★',    # 最佳準確度(%)
                ha='center', va='bottom', fontsize=11,
                color='#2d3436')

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    dash_line = Line2D([0], [0], color='#636e72', lw=1.8, ls='--',
                        label='永遠猜漲基準線')
    ax.legend(handles + [dash_line], labels + ['永遠猜漲基準線'],
              loc='upper right', fontsize=8.5, ncol=2, framealpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=10, fontweight='bold')
    ax.set_ylim(38, 85)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylabel('預測準確度 (%)', fontsize=11)
    ax.set_title(
        '台積電量化模型 — 六期間 × 五消融模型準確度比較\n'
        '★ = 該期間最佳模型  |  虛線 = 永遠猜漲基準',
        fontsize=12, fontweight='bold', pad=14,
    )
    ax.grid(axis='y', color='#dfe6e9', ls='--', lw=0.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'feature_importance_heatmap',
                        'summary_accuracy_6periods.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"✅ 準確度彙總圖 → {path}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    matrix_path = os.path.join(OUTPUT_DIR, 'TSMC_matrix_2000-2026.csv')
    run_feature_study(matrix_path)
