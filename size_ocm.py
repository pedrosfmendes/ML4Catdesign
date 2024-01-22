import pandas as pd
import random
import pickle
import numpy as np
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
import sys


def split_data(size, data,state):
    data = data.sample(n=size,random_state=state)
    return data
    
    
def run_model_ocm(data, data_test):
    # defining label and features
    #predict = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y"]
    #only predict C2 yield for this work
    predict = "C2y"
    categorical_columns = ["M1", "M2", "M3", "Support_ID"]
    numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
    features = categorical_columns + numerical_columns
    # make feature and label arrays
    X = np.array(data.drop(predict, 1))
    y = np.array(data[predict])
    
    X_test = np.array(data_test.drop(predict, 1))
    y_test = np.array(data_test[predict])
    
    # standardizing the feature values
    X = preprocessing.StandardScaler().fit_transform(X)
    y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
    y = np.ravel(y.reshape(-1, 1))
    
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
    y_test = preprocessing.StandardScaler().fit_transform(y_test.reshape(-1, 1))
    y_test = np.ravel(y_test.reshape(-1, 1))

    train_scores, val_scores, test_scores,test_predictions = [],[],[],[]
    impurity_imp,impurity_perm = [], []
    # loop that constructs models for different dataset splits, chooses the best model for a given dataset split 
    # and saves the respective feature importances to the dataframe
    number_of_splits = 10
    random_states = [21, 50, 15]
    for k in range(number_of_splits):
        x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=k)
        x_test = X_test
        Validation_score = 0.0
        for random_state in random_states:
            #importances, importances_bis = [], []
            forest = RandomForestRegressor(n_estimators=100, random_state=random_state)
            forest.fit(x_train, y_train)
            if (forest.score(x_validate, y_validate) > Validation_score):
                train_score = forest.score(x_train, y_train)
                test_score = forest.score(x_test, y_test)
                val_score = forest.score(x_validate, y_validate)
                _test_predictions = forest.predict(x_test)
                importances = forest.feature_importances_.tolist()
                Validation_score = val_score
                result = permutation_importance(forest, x_test, y_test, n_repeats=5)
                importances_bis = result.importances_mean.tolist()

        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)
        test_predictions.append(np.array(_test_predictions))
        impurity_imp.append(np.array(importances))
        impurity_perm.append(np.array(importances_bis))
    
    test_predictions = np.array(test_predictions)
            
    df_results = pd.DataFrame()
    df_results['train_score'] = train_scores
    df_results['val_scores'] = val_scores
    df_results['test_scores'] = test_scores
    
    impurity_imp = np.array(impurity_imp)
    impurity_perm = np.array(impurity_perm)
    for i, f in enumerate(features):
        df_results[f'impurity_{f}'] = impurity_imp[:,i]
        df_results[f'permutation_{f}'] = impurity_perm[:,i]
        
    return df_results, test_predictions


#define directory to save everything
directory = os.path.join('.', 'results', 'size_ocm')
if not os.path.exists(directory):
    os.makedirs(directory)

#read and convert original dataset
df_ocm = pd.read_csv('OCM-NguyenEtAl.csv')
le = preprocessing.LabelEncoder()
df_ocm["M1"] = le.fit_transform(df_ocm["M1"])
df_ocm["M2"] = le.fit_transform(df_ocm["M2"])
df_ocm["M3"] = le.fit_transform(df_ocm["M3"])
df_ocm["Support_ID"] = le.fit_transform(df_ocm["Support_ID"])
# calculate the M1 mol feature as it is missing in the original dataset
df_ocm["M2_mol%"] = df_ocm["M2_mol%"] + 1e-7 # to avoid divide by zero errors
df_ocm["M3_mol%"] = df_ocm["M3_mol%"] + 1e-7 # to avoid divide by zero errors
df_ocm["M1_mol"] = df_ocm["M1_mol%"]/100. * (df_ocm["M2_mol"] + df_ocm["M3_mol"]) / (1.-df_ocm["M1_mol%"]/100.)
predict = ["C2y"]
categorical_columns = ["M1", "M2", "M3", "Support_ID"]
numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
columns = categorical_columns + numerical_columns + predict
data = df_ocm[columns]
data = data.sample(frac=1)
    
with open(os.path.join(directory, 'ocm_all.pickle'), 'wb') as f:
    pickle.dump(data, f)    
    
#create 3 test sets of 1000 datapoints (+-10% of max training+validation) and save in pickle file, but leave enough data for 10k training
data_test = [data.sample(n=1000,random_state=i) for i in range(0,3)]
with open(os.path.join(directory, 'ocm_test_1k.pickle'), 'wb') as f:
    pickle.dump(data_test, f)    
    
sizes = [100, 300, 500, 1000, 5000, 10000]
all_results = dict()
for size in sizes:
    for j, _data_test in enumerate(data_test):
        # take 3 random samples of dataset for validation
        k = 3
        for i in range(k):
            _data = pd.concat([data, _data_test, _data_test]).drop_duplicates(keep=False)
            _data = split_data(size, _data, i)
            print(f'start {(size, j, i)}')
            df_results, test_predictions = run_model_ocm(_data, _data_test)
            print(f'done {(size, j, i)}')
            all_results[(size, j, i)] = (df_results, test_predictions)

            with open(os.path.join(directory,'results_ocm_size.pickle'), 'wb') as f:
                pickle.dump(all_results, f)
