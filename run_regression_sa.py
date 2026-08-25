import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from linearmodels.panel import PanelOLS

# ==========================================================
# 1. 載入資料、清洗上市日期並計算企業掛牌年齡 (Listing Age)
# ==========================================================
df = pd.read_csv('data/processed/clean_panel.csv')

# 載入 TEJ Company DB 匯出的基本資料
df_basic = pd.read_csv('data/raw/公司基本資料.csv', encoding='cp950')
df_basic.columns = df_basic.columns.str.strip()

# 找出 firm_id 的欄位名稱
firm_col = None
for col in ['公司', '公司代碼', '代號']:
    if col in df_basic.columns:
        firm_col = col
        break

# 找出上市日期的欄位名稱
date_col = None
for col in ['最近上市日', '上市日期', 'TSE上市日']:
    if col in df_basic.columns:
        date_col = col
        break
if date_col is None:
    # 預設第五個欄位通常是最近上市日
    date_col = df_basic.columns[4]

df_basic['firm_id'] = df_basic[firm_col].astype(str).str.strip()
df['firm_id'] = df['firm_id'].astype(str).str.strip()

df_basic['listing_date'] = pd.to_datetime(df_basic[date_col], errors='coerce')
df_basic['listing_year'] = df_basic['listing_date'].dt.year

df = df.merge(df_basic[['firm_id', 'listing_year']], on='firm_id', how='left')

df['firm_age'] = df['year'] - df['listing_year']
df['firm_age'] = df['firm_age'].fillna(df['firm_age'].median())
df['firm_age'] = df['firm_age'].clip(lower=1, upper=37)

# ==========================================================
# 2. 構建 Hadlock & Pierce (2010) SA Index (微觀融資約束指標)
# ==========================================================
df['log_assets_capped'] = winsorize(df['size'], limits=[0.01, 0.01])
df['sa_index'] = -0.737 * df['log_assets_capped'] + 0.043 * (df['log_assets_capped']**2) - 0.040 * df['firm_age']

df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)
df['sa_index_lag2'] = df.groupby('firm_id')['sa_index'].shift(2)

# 先將 monetary_tightening shift 2 變成 tight_lag2
df['tight_lag2'] = df.groupby('firm_id')['monetary_tightening'].shift(2)
df['inter_sa_lag2'] = df['tight_lag2'] * df['sa_index_lag2']
# 同時也產生 inter_lag2 用來做基準模型的控制項
df['inter_lag2'] = df.groupby('firm_id')['inter_tight_findep'].shift(2)

ctrl_vars = ['size_lag1', 'cash_ratio_lag1', 'tangibility_lag1', 'leverage_lag1']
for v in ['size', 'cash_ratio', 'tangibility', 'leverage']:
    if f'{v}_lag1' not in df.columns:
        df[f'{v}_lag1'] = df.groupby('firm_id')[v].shift(1)

# 針對 margin 做雙尾縮尾
if 'gross_margin' in df.columns:
    df['gross_margin'] = winsorize(df['gross_margin'], limits=[0.01, 0.01])

clean_df = df.dropna(subset=['gross_margin', 'log_rd', 'inter_sa_lag2', 'sa_index_lag2', 'inter_lag2'] + ctrl_vars).copy()
df_reg = clean_df.set_index(['firm_id', 'year'])

# ==========================================================
# 3. 估計模型 (Table 7: Firm-Level Financial Constraints)
# ==========================================================
f_sa_margin = """
gross_margin ~ 1 + inter_sa_lag2 + sa_index_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
               + EntityEffects + TimeEffects
"""
res_sa_margin = PanelOLS.from_formula(f_sa_margin, data=df_reg, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

f_sa_rd = """
log_rd ~ 1 + inter_sa_lag2 + sa_index_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
         + EntityEffects + TimeEffects
"""
res_sa_rd = PanelOLS.from_formula(f_sa_rd, data=df_reg[df_reg['rd_exp'] > 0], drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

median_sa = clean_df['sa_index_lag2'].median()
df_high_sa = clean_df[clean_df['sa_index_lag2'] > median_sa].set_index(['firm_id', 'year']) 
df_low_sa  = clean_df[clean_df['sa_index_lag2'] <= median_sa].set_index(['firm_id', 'year']) 

f_base_margin = """
gross_margin ~ 1 + inter_lag2 + size_lag1 + cash_ratio_lag1 + tangibility_lag1 + leverage_lag1 
               + EntityEffects + TimeEffects
"""
res_high_sa = PanelOLS.from_formula(f_base_margin, data=df_high_sa, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
res_low_sa  = PanelOLS.from_formula(f_base_margin, data=df_low_sa, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)

print("=== (1) 微觀 SA Index 對營業毛利率之交乘結果 ===")
print(res_sa_margin.summary)
print("\n=== (2) 微觀 SA Index 對研發支出之交乘結果 ===")
print(res_sa_rd.summary)
print("\n=== (3a) 高融資約束組 (High SA Firms: 小型/年輕) ===")
print(res_high_sa.summary)
print("\n=== (3b) 低融資約束組 (Low SA Firms: 大型/成熟) ===")
print(res_low_sa.summary)

with open('output/tables/table11_sa_index_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== (1) 微觀 SA Index 對營業毛利率之交乘結果 ===\n")
    f.write(res_sa_margin.summary.as_text())
    f.write("\n\n=== (2) 微觀 SA Index 對研發支出之交乘結果 ===\n")
    f.write(res_sa_rd.summary.as_text())
    f.write("\n\n=== (3a) 高融資約束組 (High SA Firms: 小型/年輕) ===\n")
    f.write(res_high_sa.summary.as_text())
    f.write("\n\n=== (3b) 低融資約束組 (Low SA Firms: 大型/成熟) ===\n")
    f.write(res_low_sa.summary.as_text())
