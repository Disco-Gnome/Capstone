import pandas as pd
import numpy as np
import zipfile

# TODO: Pandas for prototyping, switch to Spark dfs for final

# Data dict: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2021.pdf
vars = pd.read_csv('vars.csv')
majors_codes_df = pd.read_csv('Major_Codes_Mapping.csv')
majors_codes_dict = {row['code']: row['name'] for _, row in majors_codes_df.iterrows()}
usecols = list(vars['VAR'])

# For testing
with zipfile.ZipFile('csv_pny.zip') as z:
    with z.open('psam_p36.csv') as f:
        pums_data = pd.read_csv(f,
                                usecols=usecols,
                                dtype={'Nativity': str, 'OCCP': str, 'INDP': str, 'FOD1P': str, 'RAC1P': str,
                                       'HISP': str, 'SCHL': str, 'SEX': str, 'ESR': str, 'STATE': str, 'RELSHIPP': str,
                                       'SERIALNO': str},
                                nrows=200000)

# For Final analysis
# us_files = ['psam_pusa.csv', 'psam_pusb.csv', 'psam_pusc.csv', 'psam_pusd.csv']
# dfs = []
#
# with zipfile.ZipFile('csv_pus.zip') as z:
#     for name in us_files:
#         with z.open(name) as f:
#             df_part = pd.read_csv(f,
#                                   usecols=usecols)
#             dfs.append(df_part)
# pums_data = pd.concat(dfs, ignore_index=True)

# Check resource usage
print("Dataset RAM usage:", round(pums_data.memory_usage(deep=True).sum() / 1024 ** 3, 4), "GB")


#%% Drop unwanted values
# Drop NaN values
pums_data.dropna(subset=['WKWN', 'WKHP', 'ESR', 'PINCP', 'WAGP', 'FOD1P'], inplace=True)
# Drop unemployed & not in labor force
pums_data = pums_data[~pums_data['ESR'].isin(['3', '6'])]
# Drop 0 or negative total income
pums_data = pums_data[pums_data['PINCP'] > 0]
# Drop educational attainment not BA/BS or AA/AS
pums_data = pums_data[pums_data['SCHL'].isin(['20', '21'])]
# Drop 0 or negative salary income
pums_data = pums_data[pums_data['WAGP'] > 0]
# Drop less than 30 hours worked per week
pums_data = pums_data[pums_data['WKHP'] >= 30]
# Drop less than 50 weeks worked per year
pums_data = pums_data[pums_data['WKWN'] >= 50]


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


#%% Add age-squared var
pums_data['AGE-SQUARED'] = pums_data['AGEP'] ** 2


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
pums_data['race-ethnicity-sex'] = pums_data['RAC1P'] + " " + pums_data['SEX']

#%% Convert major from codes to names

pums_data['FOD1P'] = pums_data['FOD1P'].map(majors_codes_dict)


#%% Dissimilarity Indices
def dissimilarity_index(ref_dist, group_dist):
    """
    Compute dissimilarity index between 2 distributions.

    :param ref_dist: The reference distribution.
    :type ref_dist: pd.core.series.Series
    :param group_dist: The comparison distribution.
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
ref_group = pums_data[pums_data['race-ethnicity-sex'] == 'White non-Hispanic Male']
ref_major_dist = ref_group['FOD1P'].value_counts(normalize=True)
# Get group names
groups = pums_data['race-ethnicity-sex'].unique().tolist()

rows = []  # init rows
for group in groups:  # For each race-ethnicity-sex group
    # Get data for group
    group_data = pums_data[pums_data['race-ethnicity-sex'] == group]
    # Calculate group major distribution (normalized value counts)
    group_major_dist = group_data['FOD1P'].value_counts(normalize=True)

    if group == 'White non-Hispanic Male':  # Skip calc for ref group against itself
        D = 0.0
        mean_diff = 0.0
    else:  # Else calc D
        D = dissimilarity_index(ref_major_dist, group_major_dist)
        mean_diff = group_data['WAGP'].mean() - ref_group['WAGP'].mean()

    # Create a list of dicts for each row
    rows.append({
        'race-ethnicity-sex': group,
        'dissimilarity_index': round(D, 4),
        'mean_earnings_diff': round(mean_diff, 2),
        'N': len(group_data)
    })

# Create D table
dissimilarity_table = pd.DataFrame(rows)

# Save table
dissimilarity_table.to_csv("dissimilarity_table.csv", index=False)


#%% Percent with each baccalaureate major in each sex-race group
# Select majors to inspect
major_selection = ["Business Management", "General Business", "Finance", "Mechanical Engineering",
                   "Electrical Engineering", "Computer Engineering", "Computer Science", "General Education",
                   "Elementary Education", "Nursing", "Economics", "English Lang & Lit.", "Criminal Justice"]
# init pivot table
pivot = pd.DataFrame(index=major_selection)
pivot.index.name = "Major"
# Get demographic groups
groups = pums_data['race-ethnicity-sex'].unique()
# Get counts by group
for group in groups:
    group_data = pums_data[pums_data['race-ethnicity-sex'] == group]
    group_total = len(group_data)
    major_counts = group_data['FOD1P'].value_counts(normalize=True).mul(100).reindex(major_selection, fill_value=0)
    pivot[group] = major_counts
# Add N row
pivot.loc['N'] = [len(pums_data[pums_data['race-ethnicity-sex'] == group]) for group in groups]

# Round
pivot = pivot.round(3)

# Save table
pivot.to_csv('percent_major_by_group.csv')


#%% Dominance analysis

#%% Run Regressions


#%% Kitigawa-Oaxaca-Blinder

