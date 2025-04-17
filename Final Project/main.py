# Core Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.table import Table, MaskedColumn

# Scikit-learn
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Best model imports
from Classifiers.Best_Adaboost import train_best_adaboost
from Classifiers.Best_RF import best_RF
from Classifiers.GMMClassifier import best_GMM
from Classifiers.Best_SVM import train_best_svm

# Utilities
from Utilities.plot_confusion import plot_confusion, plot_feature_importances
import Utilities.query as query


def train_stacking_classifier(fits_input_path,random_state = 42,verbose=False):
    """
    Trains a stacking classifier using the best AdaBoost, Decision Tree, and SVM models.

    Parameters
    ----------
    X : Features
    y : Classifications
    test_size : 
        Proportion of data to use as the test set.
    random_state : 
        Random seed.
    verbose : True or False
        Whether to print evaluation results such as confusion matrix, feature importance
        and classification report.

    Returns
    -------
    stacking_clf : 
        Trained stacking classifier.
    """
    
     ## Trainning 
     
    # Check if the file exists
    # If not, it uses the query to generate it
    if not os.path.exists(fits_input_path):
        print(f"File '{fits_input_path}' not found. Querying from Gaia to generate it...")
        data_training = query.query('training', savetable=True)
        fits_path = "GaiaData/TRAININGData_15kstars_15kgalaxies_15kquasars.fits"
        
    # If it exists then it directly uses it to continue the analysis
    else:
        fits_path = fits_input_path

    data_training = Table.read(fits_path, format = "fits").to_pandas()

    # 2. Define features and label
    features = ["parallax", "sinb", "pm", "uwe", "phot_g_mean_mag", "bp_g", "g_rp", "relvarg"]
    X_train = data_training[features].values
    y_train = data_training["classification"].values
    
    
    ## Small interlude to make y_train usable

    # Decode bytes → str
    y_training = np.array([label.decode() if isinstance(label, bytes) else label for label in y_training])
    ###### End of interlude

    # Get best models
    clf_ab = train_best_adaboost(X_train, y_train, random_state = random_state)
    clf_rf = best_RF(X_train, y_train,random_state = random_state)
    clf_svm = train_best_svm(X_train, y_train,random_state = random_state)
    clf_gmm = best_GMM(X_train,y_train, random_state = random_state)

    
    # Define base learners with pipelines where needed
    estimators = [
        ("adaboost", clf_ab),
        ("rf", clf_rf),
        ("gmm", clf_gmm),
        ("svm", clf_svm) 
    ]

    # Final estimator for the stacking classifier
    final_estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    # Build stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1
    )

    # Train stacking classifier
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate
    if verbose:
        
        ## Classification report
        y_pred = stacking_clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        ## Customized Confusion matrix
        
        labels = stacking_clf.classes_
        plot_confusion(y_test,y_pred, labels = labels, title = 'Confusion Matrix', corrected = True, savefig = True)
        
        ## Customized feature importance
        
        print("Printing feature importance... (This may take some time but Everything else is ready :) )")
        
        stacking_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", stacking_clf)
        ])
        stacking_pipeline.fit(X_train, y_train)

        perm_importance_result = permutation_importance(
            stacking_pipeline, X_test, y_test, n_repeats=10, random_state=42
        )
        # Plot
        plot_feature_importances(perm_importance_result, np.array(features),savefig = True)
        
    return stacking_clf


def classify_dataset(stacking_clf, fits_input_path, output_path,verbose = True):
    """
    Loads a FITS file, applies a trained stacking classifier, and saves a new FITS file with predictions.

    Parameters
    ----------
    stacking_clf : 
        The fitted ensemble model.

    fits_input_path : str
        Path to the input FITS file.
        
    output_path : str, optional
        Path to save the new FITS file with predicted classifications.
        
    verbose: Bolean
        Print or not the metrics
    """

    # Testing
    
    # Check if the file exists
    # If not, it uses the query to generate it
    if not os.path.exists(fits_input_path):
        print(f"File '{fits_input_path}' not found. Querying from Gaia to generate it...")
        data_testing = query.query('testing', savetable=True)
        fits_path = "GaiaData/TESTINGData_150kstars_150kgalaxies_150kquasars.fits"  
        
    # If it exists then it directly uses it to continue the analysis
    else:
        fits_path = fits_input_path

    data_testing = Table.read(fits_path, format = "fits").to_pandas()

    # 2. Define features and label
    features = ["parallax", "sinb", "pm", "uwe", "phot_g_mean_mag", "bp_g", "g_rp", "relvarg"]

    X_testing = data_testing[features].values
    y_testing = data_testing["classification"].values
    
    
    ## Small interlude to make y_tsting usable
    # Decode bytes → str
    y_testing = np.array([label.decode() if isinstance(label, bytes) else label for label in y_testing])
    ###### End of interlude

    # Predict with stacking classifier
    predictions = stacking_clf.predict(X_testing)
    
    # Add predictions to fits file
    data_testing["predicted_class"] = predictions
    
    # Save
    
    data_testing = Table.from_pandas(data_testing)
    data_testing.write(output_path, format="fits", overwrite=True)
    
    if verbose:
        
        y_pred = predictions
        ## Customized Confusion matrix
        labels = stacking_clf.classes_
        plot_confusion(y_testing,y_pred, labels = labels, title = 'Confusion Matrix', corrected = True, savefig = True,namefig = 'Metrics/TestingDataset_confusionmatrix.pdf')


if __name__ == "__main__":

    # Paths
    training_path = "GaiaData/TRAININGData_15kstars_15kgalaxies_15kquasars.fits"
    testing_path = "GaiaData/TESTINGData_150kstars_150kgalaxies_150kquasars.fits"
    output_path = "ClassifiedData/classified_dataset.fits"
    
    
    # Train classifier
    stacking_clf = train_stacking_classifier(training_path, verbose=False)
    
    # Testing Classifier
    classify_dataset(stacking_clf=stacking_clf,fits_input_path=testing_path,
        output_path=output_path,verbose=True)