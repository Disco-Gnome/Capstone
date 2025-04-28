from dominance_analysis import Dominance
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

pums_data = pd.read_csv('pums_data_for_domin.csv',
                        dtype={'ESR': str, 'FOD1P': str, 'HISP': str, 'INDP': str, 'NATIVITY': str, 'OCCP': str,
                               'RAC1P': str, 'SCHL': str, 'SERIALNO': str, 'SEX': str,  'STATE': str
                               },
                        )

#%% Dominance analysis 1: without occupation & industry
# Init dataset for domin
df_for_domin = pums_data[['AGEP', 'AGE_SQUARED', 'HISP', 'log_WAGP', 'NOC', 'SEX', 'STATE', 'WKHP', 'WKWN']]

# Convert categorical variables to category dtype
for col in ['HISP', 'SEX', 'STATE']:
    df_for_domin[col] = df_for_domin[col].astype('category')

# Define groups (raw, unencoded)
group_defs = {
    'sex_parenthood': ['SEX', 'NOC'],
    'ethnicity': ['HISP'],
    'age': ['AGEP', 'AGE_SQUARED'],
    'state': ['STATE'],
    'time_worked': ['WKHP', 'WKWN'],
}

# We'll replace each group of categorical variables with a single **group index** via regression:
group_features = {}
X_temp = df_for_domin.drop(columns='log_WAGP')
y_temp = df_for_domin['log_WAGP']

for group_name, cols in group_defs.items():
    X_group = []

    for col in cols:
        if df_for_domin[col].dtype.name == 'category':
            # If categorical, one-hot encode
            dummies = pd.get_dummies(df_for_domin[col], prefix=col, drop_first=True)
            X_group.append(dummies)
        else:
            # If numeric, use as-is
            X_group.append(df_for_domin[[col]])

    # Concatenate all variables back together
    X_group = pd.concat(X_group, axis=1)

    # Create and run model for group
    model = LinearRegression().fit(X_group, y_temp)
    group_pred = model.predict(X_group)
    group_features[group_name] = group_pred

# Build final DataFrame with one column per group
X_reduced = pd.DataFrame(group_features)
X_reduced['log_WAGP'] = y_temp

# Run dominance analysis with group-level predictors
dominance_reg1 = Dominance(data=X_reduced, target='log_WAGP', top_k=None, objective=1)
dominance_reg1.incremental_rsquare()
dominance_df1 = dominance_reg1.dominance_stats()

# Reset the index so the predictor names become a column
dominance_table1 = dominance_df1.reset_index().rename(
    columns={
        'index': 'Predictor or Set of Predictors',
        'Percentage Relative Importance': 'Standardized Dominance'
    }
)

# Keep only the Std Domins
dominance_table1 = dominance_table1[['Predictor or Set of Predictors', 'Standardized Dominance']]

# Normalize percent to proportion
dominance_table1['Standardized Dominance'] /= 100

# Add TOTAL row
total_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['TOTAL'],
    'Standardized Dominance': [dominance_table1['Standardized Dominance'].sum()]
})

# Add R-sqrd row
r_squared_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['% Variance Explained (R²)'],
    'Standardized Dominance':  dominance_df1['Total Dominance'].sum()
})

# Add N row
n_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['N'],
    'Standardized Dominance': [len(X_reduced)]
})

# Concatenate all
dominance_table1 = pd.concat([dominance_table1, total_row, r_squared_row, n_row], ignore_index=True)

dominance_table1.to_csv('dominance_table_without_OCCPINDP.csv')

# Garbage collect
del(df_for_domin, group_defs, group_features, X_temp, y_temp, group_name, cols, X_group, model, group_pred, group_features, X_reduced, dominance_reg1, dominance_df1)


#%% Dominance analysis 2: with occupation & industry

# Init dataset for domin
df_for_domin = pums_data[['AGEP', 'AGE_SQUARED', 'HISP', 'INDP', 'log_WAGP', 'NOC', 'OCCP', 'SEX', 'STATE', 'WKHP', 'WKWN']]

# One-hot encode categorical vars
# df_for_domin = pd.get_dummies(df_for_domin,
#                               columns=['HISP', 'INDP', 'OCCP',  'SEX', 'STATE'],
#                               drop_first=True)  # Drop first cat to avoid redundant data

# Convert categorical variables to category dtype
for col in ['HISP', 'INDP', 'OCCP', 'SEX', 'STATE']:
    df_for_domin[col] = df_for_domin[col].astype('category')

# Define groups (raw, unencoded)
group_defs = {
    'sex_parenthood': ['SEX', 'NOC'],
    'ethnicity': ['HISP'],
    'age': ['AGEP', 'AGE_SQUARED'],
    'state': ['STATE'],
    'time_worked': ['WKHP', 'WKWN'],
    'occupation_industry': ['OCCP', 'INDP']
}

# We'll replace each group of categorical variables with a single **group index** via regression:
group_features = {}
X_temp = df_for_domin.drop(columns='log_WAGP')
y_temp = df_for_domin['log_WAGP']

for group_name, cols in group_defs.items():
    X_group = []

    for col in cols:
        if df_for_domin[col].dtype.name == 'category':
            # If categorical, one-hot encode
            dummies = pd.get_dummies(df_for_domin[col], prefix=col, drop_first=True)
            X_group.append(dummies)
        else:
            # If numeric, use as-is
            X_group.append(df_for_domin[[col]])

    # Concatenate all variables back together
    X_group = pd.concat(X_group, axis=1)

    # Create and run model for group
    model = LinearRegression().fit(X_group, y_temp)
    group_pred = model.predict(X_group)
    group_features[group_name] = group_pred

# Build final DataFrame with one column per group
X_reduced = pd.DataFrame(group_features)
X_reduced['log_WAGP'] = y_temp

# Run dominance analysis with group-level predictors
dominance_reg2 = Dominance(data=X_reduced, target='log_WAGP', top_k=None, objective=1)
dominance_reg2.incremental_rsquare()
dominance_df2 = dominance_reg2.dominance_stats()

# Reset the index so the predictor names become a column
dominance_table2 = dominance_df2.reset_index().rename(
    columns={
        'index': 'Predictor or Set of Predictors',
        'Percentage Relative Importance': 'Standardized Dominance with OCCP & INDP'
    }
)

# Keep only the Std Domins
dominance_table2 = dominance_table2[['Predictor or Set of Predictors', 'Standardized Dominance with OCCP & INDP']]

# Normalize percent to proportion
dominance_table2['Standardized Dominance with OCCP & INDP'] /= 100

# Add TOTAL row
total_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['TOTAL'],
    'Standardized Dominance with OCCP & INDP': [dominance_table2['Standardized Dominance with OCCP & INDP'].sum()]
})

# Add R-sqrd row
r_squared_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['% Variance Explained (R²)'],
    'Standardized Dominance with OCCP & INDP':  dominance_df2['Total Dominance'].sum()
})

# Add N row
n_row = pd.DataFrame({
    'Predictor or Set of Predictors': ['N'],
    'Standardized Dominance with OCCP & INDP': [len(X_reduced)]
})

# Concatenate all
dominance_table2 = pd.concat([dominance_table2, total_row, r_squared_row, n_row], ignore_index=True)

dominance_table2.to_csv('dominance_table_with_OCCPINDP.csv')

# Garbage collect
del(df_for_domin, group_defs, group_features, X_temp, y_temp, group_name, cols, X_group, model, group_pred, X_reduced, dominance_reg2, dominance_df2)
