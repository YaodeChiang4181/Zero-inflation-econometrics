import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv('data/processed/clean_panel.csv')

# 建立產業×年份、國家/市場×年份組合標籤
df['ind_year'] = df['industry'] + "_" + df['year'].astype(str)

# 設定雙重面板索引 (firm_id, year)
df_reg = df.set_index(['firm_id', 'year'])

# 基準規格：包含企業固定效應與年份固定效應，企業層級叢聚標準誤
formula = """
log_rd ~ 1 + inter_tight_findep + size + cash_ratio + tangibility + leverage 
         + EntityEffects 
         + TimeEffects
"""

# 註：如果使用產業×年份固定效應 (ind_year)，將會完全吸收掉交乘項 (inter_tight_findep)，導致共線性錯誤 (rank deficiency)
model = PanelOLS.from_formula(formula, data=df_reg, drop_absorbed=True)
results = model.fit(cov_type='clustered', cluster_entity=True)

# 印出迴歸報表
print(results.summary)

# 匯出迴歸係數表
with open('output/tables/table2_baseline_results.txt', 'w') as f:
    f.write(results.summary.as_text())
