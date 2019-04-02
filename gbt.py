import numpy as np
import pandas as pd
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid', font_scale = 1.5, color_codes=True)
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50)

# Because historical transactions dataset is so large, we separated it into batches and
# processed separately. So now we need to start by merging the batches into a single dataframe.
aux_df = None

for i in [0, 1, 2, 3, 4, 5]:
    new_df = pd.read_csv('/home/nikolaevra/datasets/elo/' + 'df_' + str(i) + '.csv',
                         index_col='card_id')

    if i == 1:
        aux_df = new_df
    else:
        aux_df = pd.concat([aux_df, new_df], sort=False)

# Fill all rows that have nans with the man value for that column.
aux_df['std_days_between_purch'] = aux_df['std_days_between_purch'].fillna(aux_df['std_days_between_purch'].mean())
aux_df['std_purchase_amount'] = aux_df['std_purchase_amount'].fillna(aux_df['std_purchase_amount'].mean())
aux_df['std_numerical_1'] = aux_df['std_numerical_1'].fillna(aux_df['std_numerical_1'].mean())
aux_df['std_numerical_2'] = aux_df['std_numerical_2'].fillna(aux_df['std_numerical_2'].mean())

# Because preprocessing was done in batches, we have some repeat ids after
# merging batch dataframes together (i.e. when we split datasets for
# batching some ids appeared in two splits)
aux_df = aux_df.drop(
    aux_df[
        (aux_df.index == 'C_ID_12d7f4104c') &
        (aux_df['num_purchases'] == 10.0)
    ].index,
    axis=0
)
aux_df = aux_df.drop(
    aux_df[
        (aux_df.index == 'C_ID_85ff5e8b39') &
        (aux_df['num_purchases'] == 13.0)
    ].index,
    axis=0
)
aux_df = aux_df.drop(
    aux_df[
        (aux_df.index == 'C_ID_97c510f051') &
        (aux_df['num_purchases'] == 171.0)
    ].index,
    axis=0
)
aux_df = aux_df.drop(
    aux_df[
        (aux_df.index == 'C_ID_9822d11802') &
        (aux_df['num_purchases'] == 31.0)
    ].index,
    axis=0
)

# Load the training dataset.
train_df = pd.read_csv('/home/nikolaevra/datasets/elo/train.csv', index_col='card_id')

# Join training dataset with auxilary preprocessed data.
# Since we want to keep same number of rows as in
# training set, we do a left join.
joined_df = train_df.join(
    aux_df,
    how='left',
    on='card_id'
)

# Shuffle the training dataset.
joined_df = joined_df.sample(frac=1)

# Create dummy variables for the feature columns provided
# in the training dataset.
cat_cols = ['feature_1', 'feature_2', 'feature_3']
for cat in cat_cols:
    dummies = pd.get_dummies(joined_df[cat], prefix=cat)
    joined_df = pd.concat([joined_df, dummies], axis=1)

final_df = joined_df.reset_index(drop=True)

# Drop uneccessary columns.
final_df = final_df.drop(columns=['first_active_month'] + cat_cols)
final_df = final_df.replace(np.inf, 0)

# Drop all of the nan, inf, -inf values.
# final_df = final_df[~final_df.isin([np.isnan, np.inf, -np.inf]).any(1)]
# final_df[np.isnan(final_df).any(1)].shape

# Drop Nan rows because those are ids with no transactions all together.
# Can't do anything about them and there are a too many of
# them (~15,000) to fill rows with mean of each column.
# Don't want to degrade the model.
final_df = final_df.dropna()

# Convert y values into a numpy array.
raw_Y = final_df['target'].values

# Convert X values into numpy array.
cols_to_use = final_df.columns.difference(['target', 'card_id'])
raw_X = final_df[cols_to_use].values

split_ratio = 0.9 # i.e. 90% for training and 10% for test.
split_point = int(split_ratio * raw_X.shape[0])

# Split into train and test sets.
print('Splitting at index:', split_point)
X_train = raw_X[:split_point, :]
y_train = raw_Y[:split_point]

X_test = raw_X[split_point:, :]
y_test = raw_Y[split_point:]

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

fits = []
i = 0

# Fit regression model
for n_est in [500]:
    for max_d in [4, 8]:
        for min_s in [2, 50, 500]:
            params = {
                'n_estimators': n_est,
                'max_depth': max_d,
                'min_samples_split': min_s,
                'learning_rate': 0.01,
                'loss': 'ls'
            }
            # increase learning rate to 0.1
            # Find optimum number of trees for this lr
            # Increase number of samples for split

            clf = ensemble.GradientBoostingRegressor(**params)
            clf.fit(X_train, y_train)
            fits.append(clf)

            mse_train = mean_squared_error(y_train, clf.predict(X_train))
            mse_test = mean_squared_error(y_test, clf.predict(X_test))

            print("==================================")
            print(i, n_est, max_d, min_s)
            print("Train RMSE: %.4f" % np.sqrt(mse_train))
            print("Test RMSE: %.4f" % np.sqrt(mse_test))
            i += 1



