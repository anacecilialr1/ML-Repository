from GMMClassifier import GMMCLassifier


def GMM(random_state):
    return GMMClassifier(n_components=14, covariance_type='full', prob_scale = 3, random_state=random_state)