from GMMClassifier import GMMCLassifier


def best_GMM(random_state=42):
    return GMMClassifier(n_components=14, covariance_type='full', prob_scale = 3, random_state=random_state)