from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import zipfile

# Pandas is used for this PUMS 1% pop dataset. This code is designed to be easily converted to use Spark dfs for use
# on the larger restricted-use PUMS 5% pop dataset.

# Data dict: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2021.pdf
vars = pd.read_csv('vars.csv')
majors_codes_df = pd.read_csv('Majors_mappings.csv')
majors_codes_dict = {row['code']: row['name'] for _, row in majors_codes_df.iterrows()}
# Major groups ref: https://www2.census.gov/library/publications/2012/acs/acs-18.pdf
# see also: https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2021_ACSSubjectDefinitions.pdf
major_groups_dict = {row['name']: row['five_group_category'] for _, row in majors_codes_df.iterrows()}

usecols = list(vars['VAR'])

# PUMS 2023 5-Year: https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/
us_files = ['psam_pusa.csv', 'psam_pusb.csv', 'psam_pusc.csv', 'psam_pusd.csv']
dfs = []

with zipfile.ZipFile('csv_pus.zip') as z:
    for name in us_files:
        with z.open(name) as f:
            df_part = pd.read_csv(f,
                                  usecols=usecols,
                                  dtype={'NATIVITY': str, 'OCCP': str, 'INDP': str, 'FOD1P': str, 'RAC1P': str,
                                         'HISP': str, 'SCHL': str, 'SEX': str, 'ESR': str, 'STATE': str,
                                         'RELSHIPP': str, 'SERIALNO': str},
                                  # nrows=50000  # for testing only
                                  )
            dfs.append(df_part)
pums_data = pd.concat(dfs, ignore_index=True)

# Check resource usage
print("Dataset RAM usage:", round(pums_data.memory_usage(deep=True).sum() / 1024 ** 3, 4), "GB")

# Garbage collect
del(vars, majors_codes_df, usecols, f, z)


#%% Drop unwanted values
# Drop NaN values
pums_data.dropna(subset=['WKWN', 'WKHP', 'ESR', 'WAGP', 'FOD1P'], inplace=True)
# Drop unemployed & not in labor force
pums_data = pums_data[~pums_data['ESR'].isin(['3', '5', '6'])]
# Drop educational attainment not BA/BS or AA/AS
pums_data = pums_data[pums_data['SCHL'].isin(['20', '21'])]
# Drop 0 or negative salary income
pums_data = pums_data[pums_data['WAGP'] > 0]
# Drop less than 30 hours worked per week
pums_data = pums_data[pums_data['WKHP'] >= 30]
# Drop less than 50 weeks worked per year
pums_data = pums_data[pums_data['WKWN'] >= 50]


#%% Inflation-adjust WAGP
# Inflation data from BLS CPI: https://www.bls.gov/cpi/tables/home.htm
pums_data['WAGP_2021'] = pums_data['WAGP'] * (pums_data['ADJINC'] / 1_000_000)
cpi_2021 = 270.9
cpi_2023 = 303.9
adjustment_2021_to_2023 = cpi_2023 / cpi_2021  # ~1.1218
pums_data['WAGP_2023'] = pums_data['WAGP_2021'] * adjustment_2021_to_2023

# Clean & garbage collect
pums_data.drop(['WAGP_2021', 'WAGP'], axis=1, inplace=True)
del(cpi_2021, cpi_2023, adjustment_2021_to_2023)


#%% Add log earnings var
pums_data['log_WAGP'] = np.log(pums_data['WAGP_2023'])


#%% Convert major from codes to names & introduce 5-group categories
pums_data['FOD1P'] = pums_data['FOD1P'].map(majors_codes_dict)

pums_data['FOD1P5'] = pums_data['FOD1P'].map(major_groups_dict)

# Garbage collect
del(majors_codes_dict, major_groups_dict)


#%% Add AGE_SQUARED var
pums_data['AGE_SQUARED'] = pums_data['AGEP'] ** 2


#%% Infer NOC (Number of children)
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
# Make binary for children in household
# ~98% of respondents have 0-1 children, so treating as continuous may not enhance analysis, and could hinder
# interpretability & overcomplicate dominance analysis.
pums_data['NOC'] = np.where(pums_data['NOC'] == 0, 0, 1)
# Convert to string to avoid statistical methods inferring category distance from integer values.
pums_data['NOC'] = pums_data['NOC'].astype(str)
# Drop RELSHIPP, no longer needed
pums_data.drop('RELSHIPP', axis=1, inplace=True)

# Garbage collect
del(child_codes, children, child_counts)


#%% Reduce and combine demographic codes
# Make Hispanic binary
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

# Convert sex code to string
pums_data['SEX'] = np.where(pums_data['SEX'] == '1', 'Male', 'Female')

# Create sex-race-ethnicity groups
pums_data['race_ethnicity_sex'] = pums_data['RAC1P'] + " " + pums_data['SEX']

# Garbage Collect
del(race_map)


#%% Dissimilarity Indices
def dissimilarity_index(ref_dist, group_dist):
    """
    Compute dissimilarity index between 2 distributions. Duncan's D is the proportion of samples in a group that
    would have to change categories in some categorical variable to obtain the same distribution across that variable
    as some reference group.

    :param ref_dist: The reference distribution.
    :type ref_dist: pd.core.series.Series
    :param group_dist: The distribution to compare to the reference.
    :type group_dist: pd.core.series.Series

    :return: Duncan's dissimilarity index.
    :rtype: float64
    """
    # Create a complete set of all unique majors found in either group, so we can compare across the whole list, in case
    # some majors are missing from one or the other group. This ensures both distributions will have the same set of
    # categories and avoids mismatched major indices between groups.
    all_majors = set(ref_dist.index).union(set(group_dist.index))
    ref_aligned = ref_dist.reindex(all_majors, fill_value=0)
    group_aligned = group_dist.reindex(all_majors, fill_value=0)
    # Final step actually calculates D
    return 0.5 * np.sum(np.abs(ref_aligned - group_aligned))

# Calculate reference group major distribution (normalized value counts)
ref_group = pums_data[pums_data['race_ethnicity_sex'] == 'White non-Hispanic Male']
ref_major_dist = ref_group['FOD1P'].value_counts(normalize=True)
# Get group names
groups = pums_data['race_ethnicity_sex'].unique().tolist()

rows = []  # init rows
for group in groups:  # For each race_ethnicity_sex group
    # Get data for group
    group_data = pums_data[pums_data['race_ethnicity_sex'] == group]
    # Calculate group major distribution (normalized value counts)
    group_major_dist = group_data['FOD1P'].value_counts(normalize=True)

    if group == 'White non-Hispanic Male':  # Skip calc for ref group against itself
        D = 0.0
        mean_diff = 0.0
    else:  # Else calc D
        D = dissimilarity_index(ref_major_dist, group_major_dist)
        mean_diff = group_data['WAGP_2023'].mean() - ref_group['WAGP_2023'].mean()

    # Create a list of dicts for each row
    rows.append({
        'race_ethnicity_sex': group,
        'dissimilarity_index': round(D, 4),
        'mean_earnings_diff': round(mean_diff, 2),
        'N': len(group_data)
    })

# Create D table
dissimilarity_table = pd.DataFrame(rows)

# Save table
dissimilarity_table.to_csv("dissimilarity_table.csv", index=False)

# Garbage collect
del(ref_group, ref_major_dist, groups, rows, group, group_data, group_major_dist, D, mean_diff)


#%% Percent with each baccalaureate major in each sex-race group
# Select majors to inspect
major_selection = ["Business Management And Administration", "General Business", "Finance", "Mechanical Engineering",
                   "Electrical Engineering", "Computer Engineering", "Computer Science", "General Education",
                   "Elementary Education", "Nursing", "Economics", "English Language And Literature",
                   "Criminal Justice And Fire Protection"]
# init pivot table
pivot = pd.DataFrame(index=major_selection)
pivot.index.name = "Major"
# Get demographic groups
groups = pums_data['race_ethnicity_sex'].unique()
# Get counts by group
for group in groups:
    group_data = pums_data[pums_data['race_ethnicity_sex'] == group]
    group_total = len(group_data)
    major_counts = group_data['FOD1P'].value_counts(normalize=True).mul(100).reindex(major_selection, fill_value=0)
    pivot[group] = major_counts
# Add N row
pivot.loc['N'] = [len(pums_data[pums_data['race_ethnicity_sex'] == group]) for group in groups]

# Round
pivot = pivot.round(3)

# Save table
pivot.to_csv('major_by_group_table.csv')

# Garbage collect
del(major_selection, groups, group_data, group_total, major_counts)


#%%  Save for dominance analysis
with zipfile.ZipFile('pums_data_for_domin.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
    with archive.open('pums_data_for_domin.csv', mode='w') as file:
        pums_data.to_csv(file, index=False)


#%% Save data for OBD
# Filter data to save for analysis
pums_data = pums_data[['AGEP', 'AGE_SQUARED', 'ESR', 'FOD1P', 'FOD1P5', 'INDP', 'log_WAGP',
                       'NATIVITY', 'NOC', 'OCCP', 'race_ethnicity_sex', 'STATE', 'WKHP', 'WKWN']]

with zipfile.ZipFile('pums_data_for_OBKD.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
    with archive.open('pums_data_for_OBKD.csv', mode='w') as file:
        pums_data.to_csv(file, index=False)

#%%
