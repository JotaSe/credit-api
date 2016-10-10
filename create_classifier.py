import pandas as pd
# Create a dataframe with train data
df = pd.read_csv("data/cs-training.csv",
                     sep=',',
                     header=0)
# Fill missing data
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df=df.replace(0,df['age'].median())

data = df.drop(
    df.columns[0],
    axis=1)
###
### Convert Data Into List Of Dict Records
###
data = data.to_dict(orient='records')

# Feature extraction

###
### Separate Target and Outcome Features
###
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
vec = DictVectorizer()

df_data = vec.fit_transform(data).toarray()
feature_names = vec.get_feature_names()
df_data = DataFrame(
    df_data,
columns=feature_names)

outcome_feature = df_data['SeriousDlqin2yrs']
target_features = df_data.drop('SeriousDlqin2yrs', axis=1)

###
### Generate Training and Testing Set
###
from sklearn import cross_validation

"""
    X_1: independent (target) variables for first data set
    Y_1: dependent (outcome) variable for first data set
    X_2: independent (target) variables for the second data set
    Y_2: dependent (outcome) variable for the second data set
"""
X_1, X_2, Y_1, Y_2 = cross_validation.train_test_split(
target_features, outcome_feature, test_size=0.5, random_state=0)

# Define classifier

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
clf_rf.fit(X_1,Y_1)

# Save classifier

from sklearn.externals import joblib
joblib.dump(clf_rf, 'model/rf.pkl')
