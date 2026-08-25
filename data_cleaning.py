import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# ==========================================
# 0. 環境路徑設定
# ==========================================
RAW_FIRM_PATH = 'data/raw/raw_firm_panel.csv'
RAW_MACRO_PATH = 'data/raw/raw_macro_rate.csv'
OUTPUT_PATH = 'data/processed/clean_panel.csv'

os.makedirs('data/processed', exist_ok=True)
os.makedirs('output/tables', exist_ok=True)

# ==========================================
# 1. 讀取並標準化微觀企業財報
# ==========================================
print("[Step 1/5] 正在讀取並標準化 TEJ 財報資料...")
try:
    df_firm = pd.read_csv(RAW_FIRM_PATH, encoding='utf-8-sig', dtype={'公司代碼': str, '代號': str})
except UnicodeDecodeError:
    df_firm = pd.read_csv(RAW_FIRM_PATH, encoding='cp950', dtype={'公司代碼': str, '代號': str})

# 去除所有欄位名稱的頭尾空白
df_firm.columns = df_firm.columns.str.strip()

# 欄位名稱對照表（依 TEJ 實際欄位名稱自動適配）
rename_dict = {
    '公司代碼': 'firm_id', '代號': 'firm_id',
    '公司簡稱': 'firm_name', '名稱': 'firm_name',
    '年月日': 'date', '年/月': 'date',
    'TSE產業別': 'industry', 'TEJ產業名': 'industry', 'TEJ主產業代碼': 'industry',
    '研究發展費用': 'rd_exp', '研究發展費': 'rd_exp',
    '購置不動產廠房設備': 'capx', '購置不動產廠房設備（含預付）－CFI': 'capx',
    '來自營運之現金流量': 'oancf',
    '資產總額': 'assets',
    '不動產廠房及設備淨額': 'ppe', '不動產廠房及設備': 'ppe',
    '現金及約當現金': 'cash',
    '負債總額': 'debt',
    '期末市值': 'mkt_cap', '市值': 'mkt_cap', '季底普通股市值': 'mkt_cap'
}
df_firm = df_firm.rename(columns={k: v for k, v in rename_dict.items() if k in df_firm.columns})

# 清理數值欄位中的逗號與缺失字元
numeric_cols = ['rd_exp', 'capx', 'oancf', 'assets', 'ppe', 'cash', 'debt', 'mkt_cap']
for col in numeric_cols:
    if col in df_firm.columns:
        df_firm[col] = df_firm[col].astype(str).str.replace(',', '').str.strip()
        df_firm[col] = pd.to_numeric(df_firm[col], errors='coerce')

# TEJ 的投資現金流 (CFI) 資本支出為負值 (流出)，取絕對值轉換為正向規模
if 'capx' in df_firm.columns:
    df_firm['capx'] = df_firm['capx'].abs()

# 提取西元年份 (YYYY)
df_firm['year'] = pd.to_datetime(df_firm['date'].astype(str), errors='coerce').dt.year

# 暫時解決缺少「上市別」的問題：過濾長度恰好為 4 碼的 firm_id（大致對應上市上櫃公司）
df_firm = df_firm[df_firm['firm_id'].astype(str).str.len() == 4]

df_firm = df_firm.dropna(subset=['firm_id', 'year', 'industry'])
df_firm['year'] = df_firm['year'].astype(int)

# ==========================================
# 2. 樣本篩選（排除金融業與異常樣本）
# ==========================================
print("[Step 2/5] 正在進行製造業樣本篩選...")
exclude_ind = ['金融保險業', '金融業', '建材營造', '建材營造業', '油電燃氣業', '公用事業', '證券業']
df_clean = df_firm[~df_firm['industry'].isin(exclude_ind)].copy()

# 排除資產或資本支出小於等於 0 的不合規樣本
df_clean = df_clean[(df_clean['assets'] > 0) & (df_clean['capx'] > 0)].copy()

# ==========================================
# 3. 計算 Rajan-Zingales 外部融資依賴度 (FinDep)
# ==========================================
print("[Step 3/5] 正在計算各產業 Rajan-Zingales FinDep 指標...")
# 個別企業年度融資缺口：(資本支出 - 營運現金流) / 資本支出
df_clean['firm_findep'] = (df_clean['capx'] - df_clean['oancf']) / df_clean['capx']

# 取各產業在整個樣本期間的中位數 (Median)
findep_table = df_clean.groupby('industry')['firm_findep'].median().reset_index()
findep_table.rename(columns={'firm_findep': 'findep_s'}, inplace=True)
findep_table = findep_table.sort_values(by='findep_s', ascending=False)

# 合併回主表
df_clean = df_clean.merge(findep_table, on='industry', how='left')

# 匯出產業融資依賴度清單
findep_table.to_csv('output/tables/table_findep_ranking.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 4. 構建微觀控制變數與 1% 雙尾縮尾
# ==========================================
print("[Step 4/5] 構建微觀比率變數與執行 Winsorization...")
# 研發支出取自然對數（加 1 處理零值）
df_clean['log_rd'] = np.log(np.maximum(df_clean['rd_exp'].fillna(0), 0) + 1)
df_clean['size'] = np.log(df_clean['assets'])
df_clean['cash_ratio'] = df_clean['cash'] / df_clean['assets']
df_clean['tangibility'] = df_clean['ppe'] / df_clean['assets']
df_clean['leverage'] = df_clean['debt'] / df_clean['assets']

if 'mkt_cap' in df_clean.columns:
    df_clean['tobin_q'] = (df_clean['mkt_cap'] + df_clean['debt']) / df_clean['assets']

# 針對連續變數進行 1% 與 99% 分位數縮尾，排除極端離群值
vars_to_winsorize = ['log_rd', 'size', 'cash_ratio', 'tangibility', 'leverage']
if 'tobin_q' in df_clean.columns:
    vars_to_winsorize.append('tobin_q')

for v in vars_to_winsorize:
    df_clean[v] = winsorize(df_clean[v], limits=[0.01, 0.01])

# ==========================================
# 5. 合併總體貨幣政策衝擊與交乘項
# ==========================================
print("[Step 5/5] 合併宏觀政策利率並生成核心交乘項...")
try:
    df_macro = pd.read_csv(RAW_MACRO_PATH, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_macro = pd.read_csv(RAW_MACRO_PATH, encoding='cp950')
macro_col_map = {'年/月': 'date', '年月': 'date', '年月日': 'date', '數值': 'rate', '中央銀行重貼現率': 'rate'}
df_macro = df_macro.rename(columns={k: v for k, v in macro_col_map.items() if k in df_macro.columns})

df_macro['year'] = pd.to_datetime(df_macro['date'].astype(str), errors='coerce').dt.year

# 確保在跨年份計算時，使用的是同一個總經指標（取檔案中第一個出現的指標名稱）
if '名稱' in df_macro.columns:
    indicator_name = df_macro['名稱'].dropna().iloc[0]
    df_macro = df_macro[df_macro['名稱'] == indicator_name]

df_macro = df_macro.dropna(subset=['year', 'rate']).sort_values('year').drop_duplicates('year')

# 計算政策利率年度變動量 Delta r_t
df_macro['monetary_tightening'] = df_macro['rate'].diff()
df_macro = df_macro[['year', 'monetary_tightening']]

# 合併至微觀面板
df_final = df_clean.merge(df_macro, on='year', how='left')

# 生成 Rajan-Zingales 核心交乘項
df_final['inter_tight_findep'] = df_final['monetary_tightening'] * df_final['findep_s']

# 儲存最終乾淨面板
df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"清洗完成！面板資料已成功輸出至: {OUTPUT_PATH}")
print(f"總觀測值筆數: {len(df_final)}, 涵蓋企業家數: {df_final['firm_id'].nunique()}")
