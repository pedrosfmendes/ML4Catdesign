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

def create_dataset(datapoints, seed):
    # defining the boundary values of the features
    metal_conc_min = 0.3
    metal_conc_max = 70
    metal_conc_delta = metal_conc_max - metal_conc_min

    acid_conc_min = 114
    acid_conc_max = 1458
    acid_conc_delta = acid_conc_max - acid_conc_min

    gamma_min = 0.2
    gamma_max = 0.84
    gamma_delta = gamma_max - gamma_min

    epsilon_min = 37/100000
    epsilon_max = 45/10000
    epsilon_delta = epsilon_max - epsilon_min
    
    random.seed(seed)

    # pandas dataframe to which datapoints are added
    df = pd.DataFrame()
    m,a,g,e,y = [],[],[],[],[]

    # loop that generates the different datapoints in the synthetic dataset, this example has orthogonal features
    for i in range(datapoints):

        metal_random = random.random()
        acid_random = random.random()
        gamma_random = random.random()
        epsilon_random = random.random()

        Metal_conc = metal_conc_min + metal_random * metal_conc_delta
        Acid_conc = acid_conc_min + acid_random * acid_conc_delta
        Gamma = gamma_min + gamma_random * gamma_delta
        Epsilon = epsilon_min + epsilon_random * epsilon_delta

        Yield = Gamma / (1 + Epsilon * (Acid_conc / Metal_conc)) # based on the mechanism of a hydrocracking reaction

        m.append(Metal_conc)
        a.append(Acid_conc)
        g.append(Gamma)
        e.append(Epsilon)
        y.append(Yield)

    df['metal'] = m
    df['acid'] = a
    df['gamma'] = g
    df['epsilon'] = e
    df['yield'] = y
    
    return df


def split_data(size, data,state):
    data = data.sample(n=size,random_state=state)
    return data
    
    
def run_model(data, data_test):
    # defining label and features
    predict = 'yield'
    features = ['metal', 'acid', 'gamma', 'epsilon']

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
    
    train_scores, val_scores, test_scores, test_predictions = [],[],[],[]
    impurity_imp,impurity_perm = [], []
    # loop that constructs models for different dataset splits, chooses the best model for a given dataset split 
    # and saves the respective feature importances to the dataframe
    number_of_splits = 10
    random_states = [0, 42, 10]
    for k in range(number_of_splits):
        x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        x_test = X_test
        Validation_score = 0

        for random_state in random_states:
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
    
    test_predictions =np.array(test_predictions)
    
    df_results = pd.DataFrame()
    df_results['train_score'] = train_scores
    df_results['val_scores'] = val_scores
    df_results['test_scores'] = test_scores
    
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
directory = os.path.join('.', 'results', 'size')
if not os.path.exists(directory):
    os.makedirs(directory)

#create dataset of 3000 000 datapoints and save in pickle file
#make big enough such that 5 random samples can be taken for the largest size of 500k    
data = create_dataset(3000000, 5)
with open(os.path.join(directory, 'SyntheticDataset_3M.pickle'), 'wb') as f:
    pickle.dump(data, f)


#create 3 test sets of 50000 datapoints (10% of max training+validation) and save in pickle file
data_test = [data.sample(n=50000,random_state=i) for i in range(0,3)]
with open(os.path.join(directory, 'SyntheticDataset_test_50k.pickle'), 'wb') as f:
    pickle.dump(data_test, f)    

#remove test data from training data
for d in data_test:
    data = pd.concat([data, d, d]).drop_duplicates(keep=False)
  
    
sizes = [100, 300, 500, 1000, 5000, 10000, 50000, 100000, 500000]

all_results = dict()
for size in sizes:
    for j, _data_test in enumerate(data_test):
        # take 3 random samples of dataset for validation
        k = 3
        for i in range(k):
            _data = pd.concat([data, _data_test, _data_test]).drop_duplicates(keep=False)
            _data = split_data(size, _data, i)
            df_results, test_predictions = run_model(_data, _data_test)
            all_results[(size, j, i)] = (df_results, test_predictions)

            with open(os.path.join(directory,'results_size.pickle'), 'wb') as f:
                pickle.dump(all_results, f)
