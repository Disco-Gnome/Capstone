import pandas as pd
import zipfile

vars = pd.read_csv('vars.csv')

usecols = list(vars['VAR'])

# For testing
with zipfile.ZipFile('csv_pny.zip') as z:
    with z.open('psam_p36.csv') as f:
        pums_data = pd.read_csv(f, usecols=usecols, nrows=200000)

# For Final analysis
# us_files = ['psam_pusa.csv', 'psam_pusb.csv', 'psam_pusc.csv', 'psam_pusd.csv']
# dfs = []
# with zipfile.ZipFile('csv_pus.zip') as z:
#     for name in us_files:
#         with z.open(name) as f:
#             df_part = pd.read_csv(f, usecols=usecols)
#             dfs.append(df_part)
# pums_data = pd.concat(dfs, ignore_index=True)

# Check resource usage
print(round(pums_data.memory_usage(deep=True).sum() / 1024 ** 2, 4), "MB")
#%% Infer NOC
# Filter for children
child_codes = [3, 4, 5]
children = pums_data[pums_data['RELSHIPP'].isin(child_codes)]

# Count children per household
child_counts = children.groupby('SERIALNO').size().reset_index(name='NOC')

# Merge back to full dataset
pums_data = pums_data.merge(child_counts, on='SERIALNO', how='left')

# Fill households with 0 children
pums_data['NOC'] = pums_data['NOC'].fillna(0).astype(int)
#%%
