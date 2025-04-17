import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def best_RF(X, y, verbose = True):
    
    """This function finds the optimal hyperparameters for a Random Forest
    classifier using RandomizedSearchCV
    -------
    Parameters:
    -------
    X : Features
    y : Labels
    random_state : (int) Random seed
    verbose : if True, returns the best model
    -------
    Returns:
    -------
    best_RFmodel : (sklearn.RandomForestClassifier) The best RF model found
    by RandomizedSearchCV.
    
    """

    #Hyperparameter grid to be tested
    params = {'n_estimators': 'n_estimators': list(range(100, 1000)),
    'max_depth': list(range(1, 13)),
    'min_samples_split': [2, 3, 4]}
    
    #Randomized search on hyper parameters
    search = RandomizedSearchCV(RandomForestClassifier(random_state=42), params, cv=5, n_iter = 50, n_jobs=-1)
    #Fit the search to the data
    search.fit(X, y)

    #Find the best classifier
    best_RFmodel = search.best_estimator_

    return best_RFmodel
