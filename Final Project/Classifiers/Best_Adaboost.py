from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def train_best_adaboost(X_train, y_train, test_size=0.5, random_state=423454, verbose=True):
    """
    Trains an AdaBoost classifier with hyperparameter tuning using RandomizedSearchCV.

    Parameters
    ----------
    X : Features
    y : Classifications
    test_size : Proportion of dataset to include in the test split.
    random_state : int
        Random seed
    verbose : 
        If True, prints the best parameters and score.

    Returns
    -------
    best_model : sklearn.AdaBoostClassifier
        The best AdaBoost model found by RandomizedSearchCV.
    """

    # Base model
    clf = AdaBoostClassifier(random_state=random_state)

    # Hyperparameter grid
    params = {
        "n_estimators": [50, 100, 150, 200,250,300],
        "learning_rate": [0.1, 0.3, 0.5, 0.8, 1.0]
    }

    # Randomized search
    search = RandomizedSearchCV(estimator=clf,param_distributions=params,n_iter = 15,
        scoring='accuracy',cv=5,random_state=random_state,n_jobs=-1)

    # Fit search
    search.fit(X_train, y_train)

    # Best model
    best_model = search.best_estimator_

    return best_model
