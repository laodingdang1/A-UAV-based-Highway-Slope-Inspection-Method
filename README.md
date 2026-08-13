# A UAV-based Highway Slope Inspection Method

This repository contains the implementation of a multimodal UAV-based highway slope inspection method. The method uses aligned RGB, thermal infrared (TIR), and color-encoded DEM images for patch-level classification.

## Code Description

### `01_extract_dinov2_features.ipynb`

Extracts image features from the RGB, TIR, and DEM patches using a frozen DINOv2 ViT-S/14 model.

- Reads image paths and labels from `data/manifest.csv`.
- Converts all three input products to RGB-compatible images.
- Applies DINOv2 image preprocessing.
- Extracts one 384-dimensional feature vector for each modality.
- Saves the features to `outputs/dinov2_features.npz`.

### `02_train_xgboost_base_classifiers.ipynb`

Trains an independent XGBoost classifier for each modality and generates class probabilities.

- Loads the DINOv2 features produced by the first notebook.
- Uses five outer folds and five inner folds.
- Generates leakage-safe out-of-fold probabilities for the training samples.
- Averages the five inner-model predictions for each held-out sample.
- Saves the probability files to `outputs/base_probabilities/`.

### `03_decision_level_fusion.ipynb`

Performs decision-level fusion using the RGB, TIR, and DEM probability outputs.

- Constructs a 36-dimensional representation containing probabilities, log-odds, prediction margins, and intermodal differences.
- Standardizes the representation using the outer-training data only.
- Trains an L2-regularized logistic regression meta-learner.
- Produces the final five-class predictions and evaluation metrics.
- Saves the results to `outputs/fusion/`.

## Running the Code

Run the notebooks in the following order:

```text
01_extract_dinov2_features.ipynb
02_train_xgboost_base_classifiers.ipynb
03_decision_level_fusion.ipynb
```

The notebooks use relative paths. Prepare the input data under `data/`, then execute every cell from top to bottom.

## Main Dependencies

```text
Python
PyTorch
torchvision
NumPy
Pillow
pandas
scikit-learn
XGBoost
joblib
Jupyter
```
