import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

# ==========================================
# 1. 載入資料與變數建構
# ==========================================
df = pd.read_csv('data/processed/clean_panel.csv')
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

df['gross_margin'] = winsorize(df['gross_margin'], limits=[0.01, 0.01])
df['op_margin'] = winsorize(df['op_margin'], limits=[0.01, 0.01])

control_vars = ['size', 'cash_ratio', 'tangibility', 'leverage']
for v in control_vars:
    df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)

# ==========================================
# 載入美國聯邦資金市場利率並建立工具變數
# ==========================================
df_us_macro = pd.read_csv('data/raw/新增美國聯邦資金市場利率.csv', encoding='cp950')
df_us_macro.columns = df_us_macro.columns.str.strip()
us_ffr_name = df_us_macro['名稱'].unique()[-1]
df_ffr = df_us_macro[df_us_macro['名稱'] == us_ffr_name].copy()
df_ffr['year'] = pd.to_datetime(df_ffr['年月'].astype(str), errors='coerce').dt.year
df_ffr['us_rate'] = pd.to_numeric(df_ffr['數值'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
df_ffr = df_ffr.dropna(subset=['year', 'us_rate']).sort_values('year').drop_duplicates('year')
df_ffr['delta_ffr'] = df_ffr['us_rate'].diff()

df = df.merge(df_ffr[['year', 'delta_ffr']], on='year', how='left')
df['delta_ffr_lag2'] = df.groupby('firm_id')['delta_ffr'].shift(2)
df['inter_us_lag2'] = df['delta_ffr_lag2'] * df['findep_s']

# ==========================================
# 準備樣本
# ==========================================
clean_df = df.dropna(subset=['gross_margin', 'op_margin', 'inter_lag2', 'inter_us_lag2'] + 
                             [f'{v}_lag1' for v in control_vars]).copy()
df_reg = clean_df.set_index(['firm_id', 'year'])

# ==========================================
# 1. 營業毛利率的 2SLS 工具變數估計
# ==========================================
f_gm_iv = """
gross_margin ~ 1 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
               + [inter_lag2 ~ inter_us_lag2]
"""
clusters = df_reg.index.get_level_values('firm_id')
res_gm_iv = IV2SLS.from_formula(f_gm_iv, data=df_reg).fit(cov_type='clustered', clusters=clusters)
print("=== (1) 營業毛利率 2SLS 工具變數結果 ===")
print(res_gm_iv.summary)

# ==========================================
# 2. 營業毛利率的規模分組 (中小企業 vs. 大型企業)
# ==========================================
median_size = clean_df['size_lag1'].median()
df_small = clean_df[clean_df['size_lag1'] <= median_size].set_index(['firm_id', 'year'])
df_large = clean_df[clean_df['size_lag1'] > median_size].set_index(['firm_id', 'year'])

f_gm_panel = """
gross_margin ~ 1 + inter_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
               + EntityEffects + TimeEffects
"""
res_gm_small = PanelOLS.from_formula(f_gm_panel, data=df_small, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
res_gm_large = PanelOLS.from_formula(f_gm_panel, data=df_large, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

print("\n=== (2a) 中小型企業組 (毛利率受衝擊組) ===")
print(res_gm_small.summary)

print("\n=== (2b) 大型企業組 (毛利率定價防禦組) ===")
print(res_gm_large.summary)

with open('output/tables/table10_kimball_extended_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== (1) 營業毛利率 2SLS 工具變數結果 ===\n")
    f.write(res_gm_iv.summary.as_text())
    f.write("\n\n=== (2a) 中小型企業組 (毛利率受衝擊組) ===\n")
    f.write(res_gm_small.summary.as_text())
    f.write("\n\n=== (2b) 大型企業組 (毛利率定價防禦組) ===\n")
    f.write(res_gm_large.summary.as_text())
