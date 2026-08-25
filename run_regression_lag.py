import pandas as pd
from linearmodels.panel import PanelOLS

# 1. 載入清理後的面板資料
df = pd.read_csv('data/processed/clean_panel.csv')

# 2. 排序以建立正確的時間滯後序列
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

# 3. 建立滯後變數 (避免聯立偏誤並捕捉研發黏滯性)
control_vars = ['size', 'cash_ratio', 'tangibility', 'leverage']
for v in control_vars:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

# 建立當期、滯後一期、滯後二期的核心交乘項
df['inter_lag0'] = df['inter_tight_findep']
df['inter_lag1'] = df.groupby('firm_id')['inter_tight_findep'].shift(1)
df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)

# 4. 篩選活躍研發子樣本 (排除長期零研發企業的干擾)
df_innovative = df[df['rd_exp'] > 0].dropna(subset=[f'{v}_lag1' for v in control_vars] + ['inter_lag1']).copy()

# 5. 設定雙重面板索引
df_reg = df_innovative.set_index(['firm_id', 'year'])

# 6. 估計滯後一期基準模型 (標準學術規格)
formula_lag1 = """
log_rd ~ 1 + inter_lag1 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
         + EntityEffects + TimeEffects
"""

model_lag1 = PanelOLS.from_formula(formula_lag1, data=df_reg, drop_absorbed=True)
results_lag1 = model_lag1.fit(cov_type='clustered', cluster_entity=True)

print("=== 滯後一期 (Lag 1) 實證迴歸結果 ===")
print(results_lag1.summary)

with open('output/tables/table3_lag_results.txt', 'w') as f:
    f.write(results_lag1.summary.as_text())
