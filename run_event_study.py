import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# 確保輸出資料夾存在
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/tables', exist_ok=True)

# ==========================================
# 1. 載入資料並生成領先 (Lead) 與滯後 (Lag) 變數
# ==========================================
df = pd.read_csv('data/processed/clean_panel.csv')
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

# 篩選活躍研發樣本
df = df[df['rd_exp'] > 0].copy()

# 生成微觀控制變數滯後一期 (避免當期聯立偏誤)
for v in ['size', 'cash_ratio', 'tangibility', 'leverage']:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

# 生成交乘項的領先與滯後序列 (tau = -2 到 +3)
# Lead: shift(-k) 代表事前 k 期衝擊；Lag: shift(k) 代表事後 k 期衝擊
df['inter_lead2'] = df.groupby('firm_id')['inter_tight_findep'].shift(-2)
# inter_lead1 (tau = -1) 作為基準參考組，在回歸中省略以避免完全共線性
df['inter_lag0']  = df['inter_tight_findep']
df['inter_lag1']  = df.groupby('firm_id')['inter_tight_findep'].shift(1)
df['inter_lag2']  = df.groupby('firm_id')['inter_tight_findep'].shift(2)
df['inter_lag3']  = df.groupby('firm_id')['inter_tight_findep'].shift(3)

# 刪除因平移產生的缺失值
event_vars = ['inter_lead2', 'inter_lag0', 'inter_lag1', 'inter_lag2', 'inter_lag3']
control_vars = ['size_lag1', 'cash_ratio_lag1', 'tangibility_lag1', 'leverage_lag1']
df_reg = df.dropna(subset=event_vars + control_vars).copy()
df_reg = df_reg.set_index(['firm_id', 'year'])

# ==========================================
# 2. 估計動態事件研究迴歸模型
# ==========================================
formula = f"""
log_rd ~ 1 + {' + '.join(event_vars)} + {' + '.join(control_vars)} 
         + EntityEffects + TimeEffects
"""

model = PanelOLS.from_formula(formula, data=df_reg, drop_absorbed=True)
results = model.fit(cov_type='clustered', cluster_entity=True)
print(results.summary)

with open('output/tables/table8_event_study_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 動態事件研究迴歸結果 ===\n")
    f.write(results.summary.as_text())

# ==========================================
# 3. 提取係數與 95% 信心區間
# ==========================================
# 定義事件時間軸 tau: -2, -1 (基準), 0, 1, 2, 3
tau_labels = [-2, -1, 0, 1, 2, 3]

# 提取估計係數與標準誤 (tau = -1 基準點設為 0)
coefs = [
    results.params['inter_lead2'],
    0.0,  # Reference period (tau = -1)
    results.params['inter_lag0'],
    results.params['inter_lag1'],
    results.params['inter_lag2'],
    results.params['inter_lag3']
]

stderrs = [
    results.std_errors['inter_lead2'],
    0.0,  # Reference period
    results.std_errors['inter_lag0'],
    results.std_errors['inter_lag1'],
    results.std_errors['inter_lag2'],
    results.std_errors['inter_lag3']
]

# 計算 95% 信心區間 (1.96 * SE)
ci_lower = [c - 1.96 * s for c, s in zip(coefs, stderrs)]
ci_upper = [c + 1.96 * s for c, s in zip(coefs, stderrs)]

# ==========================================
# 4. 繪製學術標準事件研究圖 (Figure 1)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# 繪製估計點與誤差線
ax.errorbar(tau_labels, coefs, yerr=[np.array(coefs) - np.array(ci_lower), np.array(ci_upper) - np.array(coefs)],
            fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=2, capsize=4, capthick=1.5,
            markersize=7, label=r'Dynamic Point Estimates ($\hat{\beta}_\tau$ with 95% CI)')

# 繪製趨勢連結虛線
ax.plot(tau_labels, coefs, color='#1f77b4', linestyle='--', alpha=0.7)

# 基準參考線
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
ax.axvline(-0.5, color='crimson', linestyle=':', linewidth=1.5, label=r'Shock Timing ($\tau = 0$)')

# 圖表美化與標籤
ax.set_title('Figure 1: Dynamic Impact of Monetary Tightening on Corporate R&D', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel(r'Relative Time to Monetary Shock ($\tau$ in Years)', fontsize=11, fontweight='bold')
ax.set_ylabel(r'Marginal Effect on $\ln(\text{R\&D}_{i,t})$', fontsize=11, fontweight='bold')
ax.set_xticks(tau_labels)
ax.set_xticklabels([r'$\tau - 2$', r'$\tau - 1$', r'$\tau = 0$', r'$\tau + 1$', r'$\tau + 2$', r'$\tau + 3$'])
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='best')

plt.tight_layout()
plt.savefig('output/figures/figure1_event_study.png')
print("Figure 1 動態事件研究圖已成功輸出至 output/figures/figure1_event_study.png")
