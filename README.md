# Gaia Stacking Classifier

Machine Learning final project which implements a **Stacking classifier** for classifying astronomical objects (stars, galaxies, and quasars) using data from the **Gaia DR3 survey**. It uses multiple machine learning models (Random Forest, AdaBoost, SVM, and GMM) and combines them in an ensemble.

---

## Models Used

**Random Forest**  
**AdaBoost**  
**Support Vector Machine (SVM)**  
**Gaussian Mixture Model (GMM)**  
Combined via **StackingClassifier** with Logistic Regression as the final estimator.

---

Project Structure

```
.
├── Final_Project.py              # Main script with training and prediction functions
├── Classifiers/
│   ├── Best_Adaboost.py         # AdaBoost with hyperparameter tuning
│   ├── Best_RF.py               # Random Forest with tuning
│   ├── Best_SVM.py              # Support Vector Machine setup
│   └── GMMClassifier.py         # Gaussian Mixture Model setup
├── Utilities/
│   └── plot_confusion.py        # Custom plotting for confusion matrix and feature importance
├── GaiaData/
│   ├── TRAININGData_*.fits      # FITS files used for training
│   └── TESTINGData_*.fits       # FITS files used for testing
└── ClassifiedData/
    └── classified_dataset.fits  # Output file with predicted classes
```

---

## Installation

Install the dependencies:

```bash
pip install numpy pandas matplotlib astropy scikit-learn
```
---

## Usage

1. Place your training and testing `.fits` files in the `GaiaData/` folder.
2. Run the main script:

```bash
python Final_Project.py
```

It will:
- Train the stacking classifier using the training FITS file.
- Predict the class of objects in the test file.
- Save the classified dataset as a new `.fits` file in `ClassifiedData/`.
- Plot evaluation metrics like the confusion matrix and feature importance.

---

## Features Used

The classifier uses the following features from the Gaia catalog:

- `parallax`
- `sinb`
- `pm`
- `uwe`
- `phot_g_mean_mag`
- `bp_g`
- `g_rp`
- `relvarg`

---

## Output

- `classified_dataset.fits`: A FITS file containing original Gaia data + predicted class.
- `/Metrics/TestingDataset_confusionmatrix.pdf`: Confusion matrix visualization.
- `feature_importances.pdf`: Permutation importance of the features.

---


## Authors

**Anaïs Antonini**  
**Ana Cecila Luis Ramírez**  
**Andrés Villares**  
**Nhu-Tin Mai**  

Master program in Astrophysics and Space Science-MASS
Université Côte d'Azur
