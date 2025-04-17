from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, reciprocal

def best_svm(X, y, random_state=42, verbose=True):
    """
    Trains an SVM classifier with hyperparameter tuning using RandomizedSearchCV.

    Parameters
    ----------
    X : features.
    y : target labels.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        If True, prints best parameters and score.

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        The best pipeline (scaler + SVM) found by RandomizedSearchCV.
    """

    # SVM
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC())
    ])

    # Hyperparameter space
    param_distributions = {
        "svc__C": uniform(1, 10),
        "svc__gamma": reciprocal(0.001, 0.1),
        "svc__kernel": ["linear", "rbf", "poly", "sigmoid"],
        "svc__degree": [2, 3, 4],
        "svc__coef0": [0, 1]
    }

    # Randomized search + cross-validation
    search = RandomizedSearchCV(
        classifier,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1_weighted",
        cv=3,
        random_state=random_state,
        verbose=1 if verbose else 0,
        n_jobs=-1
    )

    # Fit the model on the training set
    search.fit(X, y)

    if verbose:
        print("Best Parameters:", search.best_params_)
        print("Best Score (f1_weighted):", search.best_score_)
    model = search.best_estimator_
    model.set_params(random_state=random_state)

    return model
