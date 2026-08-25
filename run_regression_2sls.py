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

df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)

# ==========================================
# 載入美國聯邦資金市場利率並建立工具變數
# ==========================================
# 讀取新的巨觀資料
df_us_macro = pd.read_csv('data/raw/新增美國聯邦資金市場利率.csv', encoding='cp950')
df_us_macro.columns = df_us_macro.columns.str.strip()

# 取出美國聯邦資金利率 (最後一個指標)
us_ffr_name = df_us_macro['名稱'].unique()[-1]
df_ffr = df_us_macro[df_us_macro['名稱'] == us_ffr_name].copy()

# 整理年份與數值
df_ffr['year'] = pd.to_datetime(df_ffr['年月'].astype(str), errors='coerce').dt.year
df_ffr['us_rate'] = pd.to_numeric(df_ffr['數值'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
df_ffr = df_ffr.dropna(subset=['year', 'us_rate']).sort_values('year').drop_duplicates('year')

# 計算 US FFR 變動量 (delta_ffr)
df_ffr['delta_ffr'] = df_ffr['us_rate'].diff()

# 合併回企業資料面板
df = df.merge(df_ffr[['year', 'delta_ffr']], on='year', how='left')
df['delta_ffr_lag2'] = df.groupby('firm_id')['delta_ffr'].shift(2)

# 核心內生變數: inter_lag2 (TW) | 外生工具變數: inter_us_lag2 (US FFR * FinDep)
df['inter_us_lag2'] = df['delta_ffr_lag2'] * df['findep_s']

# 4. 篩選活躍研發子樣本
df_innovative = df[df['rd_exp'] > 0].dropna(subset=[f'{v}_lag2' for v in control_vars] + ['inter_lag2', 'inter_us_lag2']).copy()

# 5. 設定雙重面板索引
df_reg = df_innovative.set_index(['firm_id', 'year'])

# ==========================================
# 估計 2SLS 工具變數模型 (手動兩階段以支援雙固定效應)
# ==========================================
print("=== 第一階段 (First Stage) ===")
formula_stage1 = """
inter_lag2 ~ 1 + inter_us_lag2 + size_lag2 + cash_ratio_lag2 + tangibility_lag2 + leverage_lag2 
             + EntityEffects + TimeEffects
"""
res_stage1 = PanelOLS.from_formula(formula_stage1, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
print(res_stage1.summary)

# 取得第一階段預測值
df_reg['inter_lag2_hat'] = res_stage1.predict().fitted_values

print("\n=== 第二階段 (Second Stage) ===")
formula_stage2 = """
log_rd ~ 1 + inter_lag2_hat + size_lag2 + cash_ratio_lag2 + tangibility_lag2 + leverage_lag2 
         + EntityEffects + TimeEffects
"""
res_stage2 = PanelOLS.from_formula(formula_stage2, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
print(res_stage2.summary)

with open('output/tables/table7_2sls_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 第一階段 (First Stage) ===\n")
    f.write(res_stage1.summary.as_text())
    f.write("\n\n=== 第二階段 (Second Stage) ===\n")
    f.write(res_stage2.summary.as_text())
