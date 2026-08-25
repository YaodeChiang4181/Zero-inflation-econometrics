import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/processed/clean_panel.csv')

# 1. 全樣本描述性統計 (Panel A)
vars_list = ['log_rd', 'rd_exp', 'size', 'cash_ratio', 'tangibility', 'leverage']
if 'tobin_q' in df.columns:
    vars_list.append('tobin_q')

summary_a = df[vars_list].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
summary_a.columns = ['N', 'Mean', 'SD', 'Min', 'Median', 'Max']
summary_a.to_csv('output/tables/table1_panel_a.csv')

# 2. 高低 FinDep 分組差異檢定 (Panel B)
median_findep = df['findep_s'].median()
high_group = df[df['findep_s'] > median_findep]
low_group = df[df['findep_s'] <= median_findep]

diff_records = []
for var in vars_list:
    h_mean = high_group[var].mean()
    l_mean = low_group[var].mean()
    t_stat, p_val = stats.ttest_ind(high_group[var].dropna(), low_group[var].dropna())
    diff_records.append({
        'Variable': var,
        'High_FinDep_Mean': h_mean,
        'Low_FinDep_Mean': l_mean,
        'Diff (High-Low)': h_mean - l_mean,
        'p-value': p_val
    })

summary_b = pd.DataFrame(diff_records)
summary_b.to_csv('output/tables/table1_panel_b.csv', index=False)
print("Table 1 描述性統計已產出至 output/tables/")
