import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import os

os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/tables', exist_ok=True)

# ==========================================
# 1. 載入資料與構建領先/滯後序列
# ==========================================
df = pd.read_csv('data/processed/clean_panel.csv')
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

# 篩選活躍研發樣本
df = df[df['rd_exp'] > 0].copy()

# 微觀控制變數取滯後一期 (避免聯立偏誤)
for v in ['size', 'cash_ratio', 'tangibility', 'leverage']:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

# 生成交乘項序列 (tau = -2 到 +3)
df['inter_lead2'] = df.groupby('firm_id')['inter_tight_findep'].shift(-2)
df['inter_lag0']  = df['inter_tight_findep']
df['inter_lag1']  = df.groupby('firm_id')['inter_tight_findep'].shift(1)
df['inter_lag2']  = df.groupby('firm_id')['inter_tight_findep'].shift(2)
df['inter_lag3']  = df.groupby('firm_id')['inter_tight_findep'].shift(3)

event_vars = ['inter_lead2', 'inter_lag0', 'inter_lag1', 'inter_lag2', 'inter_lag3']
control_vars = ['size_lag1', 'cash_ratio_lag1', 'tangibility_lag1', 'leverage_lag1']

# 刪除缺失值
df_clean = df.dropna(subset=event_vars + control_vars).copy()

# ==========================================
# 2. 定義迴歸與係數提取函數
# ==========================================
def run_event_study(sub_df):
    sub_df = sub_df.set_index(['firm_id', 'year'])
    formula = f"""
    log_rd ~ 1 + {' + '.join(event_vars)} + {' + '.join(control_vars)} 
             + EntityEffects + TimeEffects
    """
    model = PanelOLS.from_formula(formula, data=sub_df, drop_absorbed=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 提取 tau = -2, -1 (基準0), 0, 1, 2, 3
    coefs = [
        res.params['inter_lead2'],
        0.0,
        res.params['inter_lag0'],
        res.params['inter_lag1'],
        res.params['inter_lag2'],
        res.params['inter_lag3']
    ]
    stderrs = [
        res.std_errors['inter_lead2'],
        0.0,
        res.std_errors['inter_lag0'],
        res.std_errors['inter_lag1'],
        res.std_errors['inter_lag2'],
        res.std_errors['inter_lag3']
    ]
    return np.array(coefs), np.array(stderrs)

# ==========================================
# 3. 依規模中位數分組估計
# ==========================================
median_size = df_clean['size_lag1'].median()
df_small = df_clean[df_clean['size_lag1'] <= median_size].copy()
df_large = df_clean[df_clean['size_lag1'] > median_size].copy()

coef_s, se_s = run_event_study(df_small)
coef_l, se_l = run_event_study(df_large)

tau_labels = [-2, -1, 0, 1, 2, 3]

# ==========================================
# 4. 繪製雙子圖 (Panel A & Panel B)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=300)

# 子圖 A: 中小型企業組 (受約束組)
ax1.errorbar(tau_labels, coef_s, yerr=1.96 * se_s, fmt='o', color='#c0392b',
             ecolor='#c0392b', elinewidth=2, capsize=4, capthick=1.5, markersize=7,
             label=r'Estimates ($\hat{\beta}_\tau$ with 95\% CI)')
ax1.plot(tau_labels, coef_s, color='#c0392b', linestyle='--', alpha=0.7)
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
ax1.axvline(-0.5, color='grey', linestyle=':', linewidth=1.5, label='Shock Timing')
ax1.set_title('(a) Small & Medium Firms (Financially Constrained)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel(r'Relative Time to Monetary Shock ($\tau$ in Years)', fontsize=10, fontweight='bold')
ax1.set_ylabel(r'Marginal Effect on $\ln(\text{R\&D}_{i,t})$', fontsize=10, fontweight='bold')
ax1.set_xticks(tau_labels)
ax1.set_xticklabels([r'$\tau-2$', r'$\tau-1$', r'$\tau=0$', r'$\tau+1$', r'$\tau+2$', r'$\tau+3$'])
ax1.legend(frameon=True, facecolor='white', loc='upper left')

# 子圖 B: 大型企業組 (深口袋組)
ax2.errorbar(tau_labels, coef_l, yerr=1.96 * se_l, fmt='o', color='#2980b9',
             ecolor='#2980b9', elinewidth=2, capsize=4, capthick=1.5, markersize=7,
             label=r'Estimates ($\hat{\beta}_\tau$ with 95\% CI)')
ax2.plot(tau_labels, coef_l, color='#2980b9', linestyle='--', alpha=0.7)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
ax2.axvline(-0.5, color='grey', linestyle=':', linewidth=1.5, label='Shock Timing')
ax2.set_title('(b) Large Firms (Deep Pockets / Unconstrained)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel(r'Relative Time to Monetary Shock ($\tau$ in Years)', fontsize=10, fontweight='bold')
ax2.set_xticks(tau_labels)
ax2.set_xticklabels([r'$\tau-2$', r'$\tau-1$', r'$\tau=0$', r'$\tau+1$', r'$\tau+2$', r'$\tau+3$'])
ax2.legend(frameon=True, facecolor='white', loc='upper left')

# 整體圖表大標題
plt.suptitle('Figure 1: Heterogeneous Dynamic Impact of Monetary Tightening on Corporate R&D',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/figures/figure1_dual_panel_event_study.png', bbox_inches='tight')
print("Dual panel figure saved.")
