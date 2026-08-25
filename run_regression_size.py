import pandas as pd
from linearmodels.panel import PanelOLS

# 1. 載入清理後的面板資料
df = pd.read_csv('data/processed/clean_panel.csv')

# 2. 排序以建立正確的時間滯後序列
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

# 3. 建立滯後變數
control_vars = ['size', 'cash_ratio', 'tangibility', 'leverage']
for v in control_vars:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)
    df[f'{v}_lag2'] = df.groupby('firm_id')[v].shift(2)

# 建立當期、滯後一期、滯後二期的核心交乘項
df['inter_lag0'] = df['inter_tight_findep']
df['inter_lag1'] = df.groupby('firm_id')['inter_tight_findep'].shift(1)
df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)

# 4. 篩選活躍研發子樣本
df_innovative = df[df['rd_exp'] > 0].dropna(subset=[f'{v}_lag2' for v in control_vars] + ['inter_lag2']).copy()

# ==========================================
# 依資產規模中位數切分：中小企業組 vs 大型企業組
# ==========================================
median_size = df_innovative['size_lag2'].median()
df_small = df_innovative[df_innovative['size_lag2'] <= median_size].set_index(['firm_id', 'year'])
df_large = df_innovative[df_innovative['size_lag2'] > median_size].set_index(['firm_id', 'year'])

formula_lag2 = """
log_rd ~ 1 + inter_lag2 + size_lag2 + cash_ratio_lag2 + tangibility_lag2 + leverage_lag2 
         + EntityEffects + TimeEffects
"""

res_small = PanelOLS.from_formula(formula_lag2, data=df_small, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
res_large = PanelOLS.from_formula(formula_lag2, data=df_large, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

print("=== 中小型企業組（真實受信用約束者）===")
print(res_small.summary)

print("=== 大型企業組（深口袋逆勢創新者）===")
print(res_large.summary)

with open('output/tables/table6_size_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 中小型企業組（真實受信用約束者）===\n")
    f.write(res_small.summary.as_text())
    f.write("\n\n=== 大型企業組（深口袋逆勢創新者）===\n")
    f.write(res_large.summary.as_text())
