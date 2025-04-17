import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl

mpl.rcParams.update({
    'font.size':       14,   # base font size for text
    'axes.titlesize':  16,   # title
    'axes.labelsize':  14,   # x/y label
    'xtick.labelsize': 12,   # tick labels
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def plot_confusion(y_test, y_pred, labels, title, corrected = False, savefig = False, count = {'star':1301314924, 'quasar': 621523, 'galaxy':172752}, namefig = "Metrics/Training_ConfusionMatrix.pdf"):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.

    title: str
        Title of the figure.

    corrected: True or False
        If "True", using value from count to correct the confusion matrix.

    savefig: True or False
        If "True", saving the figure as "confusion.pdf"

    count: dict, default={'star':1301314924, 'quasar': 621523, 'galaxy':172752}
        The true number of objects in each class
        
    Example:
    --------
    from plot_confusion import plot_confusion

    classifier = GMMClassifier(n_components=18, covariance_type='full', prob_scale = 3, random_state=42)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    labels = classifier.classes_
    
    plot_confusion(y_test,y_pred, labels = labels, title = 'Gaussian Mixture Model', corrected = True, savefig = True)
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    if corrected:
        for i, cl in enumerate(labels):
            cm[i,:] = cm[i,:]*(count[cl]/np.sum(cm[i,:]))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels
                                 )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(values_format="d", colorbar = False, ax=ax)
    
    if ax.images:
        ax.images[0].set_visible(False)
    for txt in disp.ax_.texts:
        txt.set_color("black")
    
    ax.set_title(title)

    if savefig:
        plt.savefig(namefig, format="pdf")

    plt.show()
    
    
def plot_feature_importances(perm_importance_result, feat_name, savefig = False, namefig = 'Metrics/Training_Feature_importance.pdf'):
    """bar plot the feature importance"""
    fig, ax = plt.subplots(figsize = (8,7))
    indices = perm_importance_result["importances_mean"].argsort()
    plt.barh(
        range(len(indices)),
        perm_importance_result["importances_mean"][indices],
        xerr=perm_importance_result["importances_std"][indices],
    )
    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])
    ax.set_title("Permutation Importances on selected subset of features")
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    if savefig:
        plt.savefig(namefig, format="pdf")