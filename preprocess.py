import pandas as pd
import numpy as np
import zipfile

# TODO: Pandas for prototyping, switch to Spark dfs for final

# Data dict: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2021.pdf
vars = pd.read_csv('vars.csv')
usecols = list(vars['VAR'])

# For testing
with zipfile.ZipFile('csv_pny.zip') as z:
    with z.open('psam_p36.csv') as f:
        pums_data = pd.read_csv(f,
                                usecols=usecols,
                                dtype={'Nativity': str, 'OCCP': str, 'INDP': str, 'FOD1P': str, 'RAC1P': str,
                                       'HISP': str, 'SEX': str, 'ESR': str, 'STATE': str, 'RELSHIPP': str, 'SERIALNO': str},
                                nrows=200000)

# For Final analysis
us_files = ['psam_pusa.csv', 'psam_pusb.csv', 'psam_pusc.csv', 'psam_pusd.csv']
dfs = []

# with zipfile.ZipFile('csv_pus.zip') as z:
#     for name in us_files:
#         with z.open(name) as f:
#             df_part = pd.read_csv(f,
#                                   usecols=usecols)
#             dfs.append(df_part)
# pums_data = pd.concat(dfs, ignore_index=True)

# Check resource usage
print("Dataset space in RAM:", round(pums_data.memory_usage(deep=True).sum() / 1024 ** 3, 4), "GB")

#%% Infer NOC
# Filter for children
child_codes = ['25',  # Biological son or daughter
               '26',  # Adopted son or daughter
               '27',  # Stepson or stepdaughter
               '30',  # Grandchild
               '35',  # Foster child
               ]
children = pums_data[pums_data['RELSHIPP'].isin(child_codes)]
# Count children per household
child_counts = children.groupby('SERIALNO').size().reset_index(name='NOC')
# Merge back to full dataset
pums_data = pums_data.merge(child_counts, on='SERIALNO', how='left')
# Fill households with 0 children
pums_data['NOC'] = pums_data['NOC'].fillna(0).astype(int)

#%% Drop unwanted values
# Drop NaN values
pums_data.dropna(subset=['WKWN', 'WKHP', 'ESR', 'PINCP', 'WAGP'], inplace=True)
# Drop less than 50 weeks worked per year
pums_data = pums_data[pums_data['WKWN'] >= 50]
# Drop less than 30 hours worked per week
pums_data = pums_data[pums_data['WKHP'] >= 30]
# Drop unemployed & not in labor force
pums_data = pums_data[~pums_data['ESR'].isin(['3', '6'])]
# Drop 0 or negative total income
pums_data = pums_data[pums_data['PINCP'] > 0]
# Drop 0 or negative salary income
pums_data = pums_data[pums_data['WAGP'] > 0]

#%% Add age-squared
pums_data['AGE-SQUARED'] = pums_data['AGEP'] ** 2

#%% Reduce and combine demographic codes
# Make Hispanic binary
# New mapping: 1==non-hispanic, 2==hispanic
pums_data['HISP'] = np.where(pums_data['HISP'] == '01', 'non-Hispanic', 'Hispanic')

# Reduce race categories. All will be labeled 'non-Hispanic' until we combine race with ethnicity.
race_map = {
    '1': 'White non-Hispanic',  # White
    '2': 'Black non-Hispanic',  # Black
    '6': 'Asian non-Hispanic',  # Asian
    '3': 'Other', '4': 'Other', '5': 'Other', '7': 'Other', '8': 'Other', '9': 'Other'  # All other categories -> 4
}
pums_data['RAC1P'] = pums_data['RAC1P'].replace(race_map)

# Combine race & ethnicity
pums_data.loc[pums_data['HISP'] == 'Hispanic', 'RAC1P'] = 'Hispanic'

# Convert sex code to key
pums_data['SEX'] = np.where(pums_data['SEX'] == '1', 'Male', 'Female')

#%% Create sex-race-ethnicity groups
pums_data['sex-race-ethnicity'] = pums_data['RAC1P'] + " " + pums_data['SEX']

#%% Dissimilarity Indices

#%% Percent with each baccalaureate major in each sex-race group

#%% Dominance analysis

#%% Run Regressions


#%% Kitigawa-Oaxaca-Blinder

