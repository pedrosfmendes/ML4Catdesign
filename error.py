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

    
def run_model(data, true_data):
    # defining label and features
    predict = 'yield'
    features = ['metal', 'acid', 'gamma', 'epsilon']

    # make feature and label arrays
    X = np.array(data.drop(predict, 1))
    y = np.array(data[predict])
    X_true = np.array(true_data.drop(predict, 1))
    y_true = np.array(true_data[predict]) 
       
    # standardizing the feature values
    X = preprocessing.StandardScaler().fit_transform(X)
    y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
    y = np.ravel(y.reshape(-1, 1))
    X_true = preprocessing.StandardScaler().fit_transform(X_true)
    y_true = preprocessing.StandardScaler().fit_transform(y_true.reshape(-1, 1))
    y_true = np.ravel(y_true.reshape(-1, 1))
    
    train_scores, val_scores, test_scores, test_predictions = [],[],[],[]
    train_scores_true, val_scores_true, test_scores_true, train_scores_small = [],[],[],[]
    impurity_imp,impurity_perm = [], []
    # loop that constructs models for different dataset splits, chooses the best model for a given dataset split 
    # and saves the respective feature importances to the dataframe
    number_of_splits = 10
    random_states = [0, 42, 10]
    for k in range(number_of_splits):
        x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=k)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.1, random_state=k)
        x_train_small, rest1, y_train_small, rest2 = sklearn.model_selection.train_test_split(x_train, y_train, train_size=0.1, random_state=k)
        x_traint, x_validatet, y_traint, y_validatet = sklearn.model_selection.train_test_split(X_true, y_true, test_size=0.1, random_state=k)
        x_traint, x_testt, y_traint, y_testt = sklearn.model_selection.train_test_split(x_traint, y_traint, test_size=0.1, random_state=k)
        Validation_score = 0

        for random_state in random_states:
            forest = RandomForestRegressor(n_estimators=100, random_state=random_state)
            forest.fit(x_train, y_train)

            if (forest.score(x_validate, y_validate) > Validation_score):
                train_score = forest.score(x_train, y_train)
                test_score = forest.score(x_test, y_test)
                val_score = forest.score(x_validate, y_validate)
                train_score_true = forest.score(x_train, y_traint)
                train_score_small = forest.score(x_train_small, y_train_small)
                test_score_true = forest.score(x_test, y_testt)
                val_score_true = forest.score(x_validate, y_validatet)
                _test_predictions = forest.predict(x_test)

                importances = forest.feature_importances_.tolist()
                Validation_score = val_score

                result = permutation_importance(forest, x_test, y_test, n_repeats=5)
                importances_bis = result.importances_mean.tolist()

        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)
        train_scores_true.append(train_score_true)
        train_scores_small.append(train_score_small)
        val_scores_true.append(val_score_true)
        test_scores_true.append(test_score_true)
        test_predictions.append(np.array(_test_predictions))
        impurity_imp.append(np.array(importances))
        impurity_perm.append(np.array(importances_bis))
    
    test_predictions =np.array(test_predictions)
    
    df_results = pd.DataFrame()
    df_results['train_score'] = train_scores
    df_results['val_scores'] = val_scores
    df_results['test_scores'] = test_scores
    df_results['train_score_true'] = train_scores_true
    df_results['train_score_small'] = train_scores_small
    df_results['val_scores_true'] = val_scores_true
    df_results['test_scores_true'] = test_scores_true
    
    impurity_imp = np.array(impurity_imp)
    df_results['impurity_metal'] = impurity_imp[:,0]
    df_results['impurity_acid'] = impurity_imp[:,1]
    df_results['impurity_gamma'] = impurity_imp[:,2]
    df_results['impurity_epsilon'] = impurity_imp[:,3]

    impurity_perm = np.array(impurity_perm)
    df_results['permutation_metal'] = impurity_perm[:,0]
    df_results['permutation_acid'] = impurity_perm[:,1]
    df_results['permutation_gamma'] = impurity_perm[:,2]
    df_results['permutation_epsilon'] = impurity_perm[:,3]
    return df_results, test_predictions


#define directory to save everything
directory = os.path.join('.', 'results', 'error')
if not os.path.exists(directory):
    os.makedirs(directory)

#read previous dataset, take a random sample of 100k datapoints
with open(os.path.join('results', 'size', 'SyntheticDataset_3M.pickle'), 'rb') as f:
    temp = pickle.load(f)
    
data_initial = temp.sample(n=100000, random_state=0)
#data_initial['original_yield'] = data_initial['yield'].values

with open(os.path.join(directory, 'Error_100k_original.pickle'), 'wb') as f:
     pickle.dump(data_initial, f) 

#add error to metal and acid concentrations and yield
error_percent = [0., 5.,10.,15.,20.,25.]
data = dict()
data_true = dict()
for c in ['yield']:
    for e in error_percent:
        noise = np.random.uniform(1-e/100,1+e/100,len(data_initial[c]))
        df_temp = data_initial.copy()
        df_temp[c] = data_initial[c].values * noise
        data[(c,e)] = df_temp
        data_true[(c,e)] = data_initial

with open(os.path.join(directory, 'Error_100k.pickle'), 'wb') as f:
     pickle.dump(data, f)          

all_results = dict()

for key in data.keys():
    df_results = run_model(data[key], data_true[key])
    all_results[key] = df_results
    
    with open(os.path.join(directory,'results_error.pickle'), 'wb') as f:
        pickle.dump(all_results, f)
        
