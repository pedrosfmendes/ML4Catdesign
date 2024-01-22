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
    
    
def run_model_ocm(data):
    # defining label and features
    #predict = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y"]
    #only predict C2 yield for this work
    predict = "C2y"
    #categorical_columns = ["M1", "M2", "M3", "Support_ID"]
    #numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
    #features = categorical_columns + numerical_columns
    # make feature and label arrays
    X = np.array(data.drop(predict, 1))
    y = np.array(data[predict])
    
    #X_test = np.array(data_test.drop(predict, 1))
    #y_test = np.array(data_test[predict])
    
    # standardizing the feature values
    X = preprocessing.StandardScaler().fit_transform(X)
    y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
    y = np.ravel(y.reshape(-1, 1))
    
    #X_test = preprocessing.StandardScaler().fit_transform(X_test)
    #y_test = preprocessing.StandardScaler().fit_transform(y_test.reshape(-1, 1))
    #y_test = np.ravel(y_test.reshape(-1, 1))

    train_scores, val_scores, test_scores,test_predictions = [],[],[],[]
    impurity_imp,impurity_perm = [], []
    # loop that constructs models for different dataset splits, chooses the best model for a given dataset split 
    # and saves the respective feature importances to the dataframe
    number_of_splits = 10
    random_states = [0, 42, 10]
    for k in range(number_of_splits):
        x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=k)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.1, random_state=k)
        #x_test = X_test
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
    for i, f in enumerate(data.drop(predict, 1).columns):
        df_results[f'impurity_{f}'] = impurity_imp[:,i]
        df_results[f'permutation_{f}'] = impurity_perm[:,i]
        
    return df_results, test_predictions


#define directory to save everything
directory = os.path.join('.', 'features_ocm')
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

ordered_columns = ['CT', 'M1_mol', 'M3_mol', 'CH4_flow', 'M3','O2_flow', 'M2', 'Ar_flow','Support_ID','M1']
with open(os.path.join(directory, 'ocm_all.pickle'), 'wb') as f:
    pickle.dump(data, f)    


all_results = dict()


for o, o_name in enumerate(ordered_columns):
        _data = data.drop(ordered_columns[:o+1], axis=1)
        print(f'start {(o)}, {ordered_columns[:o+1]}')
        df_results, test_predictions = run_model_ocm(_data)
        print(f'done {(o)}')
        all_results[(o)] = (df_results, test_predictions)
        print(f'done {df_results.columns}')
        with open(os.path.join(directory,'results_ocm_features.pickle'), 'wb') as f:
            pickle.dump(all_results, f)
