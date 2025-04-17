import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin

class GMMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=2, covariance_type='full', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store unique classes
        self.classes_ = unique_labels(y)
        self.gmms_ = {}
        self.priors_ = {}
        
        # Fit one GMM per class
        for cl in self.classes_:
            X_cl = X[y == cl]
            gmm = GaussianMixture(n_components=self.n_components,
                                  covariance_type=self.covariance_type,
                                  random_state=self.random_state)
            gmm.fit(X_cl)
            self.gmms_[cl] = gmm
            self.priors_[cl] = float(X_cl.shape[0]) / X.shape[0]
        return self

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        # Number of samples and classes
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))
        
        # Evaluate probability for each class
        for idx, cl in enumerate(self.classes_):
            # score_samples returns log likelihoods
            log_prob = self.gmms_[cl].score_samples(X)
            # Multiply by class prior and take exponential for density
            probs[:, idx] = np.exp(log_prob) * self.priors_[cl]
        
        # Normalize across classes so that row sums equal 1
        probs_sum = np.sum(probs, axis=1, keepdims=True)
        probs /= probs_sum
        return probs

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        # Use predict_proba to get final predictions
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
def best_GMM(random_state=42):
    return GMMClassifier(n_components=14, covariance_type='full', random_state=random_state)