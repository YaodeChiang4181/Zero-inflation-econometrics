import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from linearmodels.panel import PanelOLS

# ==========================================
# 1. 載入面板與構建加成率指標
# ==========================================
df = pd.read_csv('data/processed/clean_panel.csv')
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

# 進行 1% 雙尾縮尾 (處理極端值)
df['gross_margin'] = winsorize(df['gross_margin'], limits=[0.01, 0.01])
df['op_margin'] = winsorize(df['op_margin'], limits=[0.01, 0.01])

# 生成滯後微觀控制變數
control_vars = ['size', 'cash_ratio', 'tangibility', 'leverage']
for v in control_vars:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

# 生成滯後二期核心交乘項與三重交乘項
df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)
df['cash_ratio_lag2'] = df.groupby('firm_id')['cash_ratio'].shift(2)
df['inter_triple_lag2'] = df['inter_lag2'] * df['cash_ratio_lag2']

# 篩選完整樣本並設定雙重索引
clean_df = df.dropna(subset=['gross_margin', 'op_margin', 'inter_lag2', 'inter_triple_lag2'] + 
                             [f'{v}_lag1' for v in control_vars]).copy()
df_reg = clean_df.set_index(['firm_id', 'year'])

# ==========================================
# 2. 估計加成率衝擊模型 (Kimball Variable Elasticity of Demand)
# ==========================================
# 規格 (1): 營業毛利率基準模型
f_gm_base = """
gross_margin ~ 1 + inter_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
               + EntityEffects + TimeEffects
"""
res_gm_base = PanelOLS.from_formula(f_gm_base, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

# 規格 (2): 營業毛利率 + 流動性調節模型 (驗證政策替代定理)
f_gm_triple = """
gross_margin ~ 1 + inter_lag2 + inter_triple_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
                + EntityEffects + TimeEffects
"""
res_gm_triple = PanelOLS.from_formula(f_gm_triple, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

# 規格 (3): 營業利益率基準模型
f_op_base = """
op_margin ~ 1 + inter_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
              + EntityEffects + TimeEffects
"""
res_op_base = PanelOLS.from_formula(f_op_base, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

print("=== (1) 營業毛利率 (Gross Margin) 基準結果 ===")
print(res_gm_base.summary)

print("\n=== (2) 營業毛利率 (Gross Margin) 流動性調節結果 ===")
print(res_gm_triple.summary)

print("\n=== (3) 營業利益率 (Operating Margin) 基準結果 ===")
print(res_op_base.summary)

with open('output/tables/table9_kimball_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== (1) 營業毛利率 (Gross Margin) 基準結果 ===\n")
    f.write(res_gm_base.summary.as_text())
    f.write("\n\n=== (2) 營業毛利率 (Gross Margin) 流動性調節結果 ===\n")
    f.write(res_gm_triple.summary.as_text())
    f.write("\n\n=== (3) 營業利益率 (Operating Margin) 基準結果 ===\n")
    f.write(res_op_base.summary.as_text())
