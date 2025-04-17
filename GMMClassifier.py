import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin


class GMMClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each component has its own diagonal covariance matrix.
        - 'spherical': each component has its own single variance.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(self, n_components=2, covariance_type='full', random_state=None, priors = {'star':0.99939, 'quasar': 0.00047, 'galaxy':0.00014}, prob_scale = 2):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.prob_scale = prob_scale
        self.priors = priors

    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store unique classes
        self.classes_ = unique_labels(y)
        self.gmms_ = {}
        # self.priors_ = {}
        
        # Fit one GMM per class
        for cl in self.classes_:
            X_cl = X[y == cl]
            gmm = GaussianMixture(n_components=self.n_components,
                                  covariance_type=self.covariance_type,
                                  random_state=self.random_state)
            gmm.fit(X_cl)
            self.gmms_[cl] = gmm
            # self.priors_[cl] = float(self.frac[cl]) / sum(self.frac.values())
        return self

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X,reset=False)
        # Number of samples and classes
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        # Evaluate probability for each class
        for idx, cl in enumerate(self.classes_):
            # score_samples returns log likelihoods
            log_prob = self.gmms_[cl].score_samples(X)
            # Multiply by class prior and take exponential for density
            log_probs[:, idx] = (log_prob + np.log(self.priors[cl]))/self.prob_scale
        
        # Normalize across classes so that row sums equal 1
        probs = np.exp(log_probs)
        probs_sum = np.sum(probs, axis=1, keepdims=True)

        probs_norm = probs/probs_sum
    

        return probs_norm

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)
        # Use predict_proba to get final predictions
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]



def best_GMM(random_state=42):
    return GMMClassifier(n_components=14, covariance_type='full', prob_scale = 3, random_state=random_state)