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
# 新增：構建滯後二期的三重交乘項 (測試流動性緩衝定理)
# ==========================================
df_innovative['inter_triple_lag2'] = df_innovative['inter_lag2'] * df_innovative['cash_ratio_lag2']

# 5. 設定雙重面板索引
df_reg = df_innovative.set_index(['firm_id', 'year'])

# 6. 估計流動性調節模型 (包含三重交乘項)
formula_triple = """
log_rd ~ 1 + inter_lag2 + inter_triple_lag2 
         + size_lag2 + cash_ratio_lag2 + tangibility_lag2 + leverage_lag2 
         + EntityEffects + TimeEffects
"""

model_triple = PanelOLS.from_formula(formula_triple, data=df_reg, drop_absorbed=True)
results_triple = model_triple.fit(cov_type='clustered', cluster_entity=True)

print("=== 流動性緩衝定理 (Triple Interaction) 實證迴歸結果 ===")
print(results_triple.summary)

with open('output/tables/table5_triple_results.txt', 'w') as f:
    f.write(results_triple.summary.as_text())
