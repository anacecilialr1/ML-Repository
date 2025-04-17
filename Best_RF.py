import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def best_RF(X, y, random_state = 42, verbose = True):
    
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
    params = {'n_estimators': range(100, 600, 100),
    'max_depth': range(5, 20),
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]}
    
    #Randomized search on hyper parameters
    search = RandomizedSearchCV(RandomForestClassifier(random_state=random_state), params, cv=5, n_iter = 70, n_jobs=-1)
    #Fit the search to the data
    search.fit(X, y)

    #Find the best classifier
    best_RFmodel = search.best_estimator_

    return best_RFmodel
