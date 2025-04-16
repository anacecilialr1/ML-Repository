import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def plot_confusion(y_test, y_pred, labels, title, savefig = False, count = {'star':1301314924, 'quasar': 621523, 'galaxy':172752}):
    """
    Example:
    
    from plot_confusion import plot_confusion

    classifier = GMMClassifier(n_components=18, covariance_type='full', prob_scale = 3, random_state=42)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    labels = classifier.classes_
    
    plot_confusion(y_test,y_pred, labels = labels, title = 'Gaussian Mixture Model', savefig = True)
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

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

    if savefig == True:
        plt.savefig('confusion.pdf')

    plt.show()