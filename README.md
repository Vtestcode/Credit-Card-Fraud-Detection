# Credit Card Fraud Detection

A machine learning notebook project for detecting fraudulent credit card transactions using the Kaggle `mlg-ulb/creditcardfraud` dataset.

## Project Structure

- `cc.ipynb`: Main notebook for data loading, preprocessing, training, and evaluation.

## What This Notebook Covers

- Downloads dataset using `kagglehub`
- Loads and explores transaction data with `pandas`
- Handles class imbalance analysis (fraud vs non-fraud)
- Scales transaction amount with `StandardScaler`
- Splits data into train/test sets
- Trains baseline models:
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
- Evaluates models with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix visualization
- Applies oversampling with `SMOTE`
- Retrains Random Forest on resampled data and compares performance

## Requirements

Use Python 3.10+ (tested here with Python 3.14).

Install dependencies:

```powershell
py -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub
```

## How to Run

1. Open `cc.ipynb` in Jupyter Notebook or VS Code.
2. Select the same Python interpreter where dependencies were installed.
3. Run cells top-to-bottom without skipping.
4. If you install packages mid-session, restart the kernel and run all cells again.

## Expected Data Path

The notebook downloads data via `kagglehub` and reads:

- `creditcard.csv` from your local KaggleHub cache

In this environment, it resolved to:

`C:\Users\Visha\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3\creditcard.csv`

If your machine uses a different user/profile path, update the `pd.read_csv(...)` path or use the `path` returned by `kagglehub.dataset_download(...)`.

## Common Errors and Quick Fixes

- `ModuleNotFoundError: No module named 'sklearn'`
  - Install: `py -m pip install scikit-learn`
- `KeyError` for a column
  - Check columns first: `print(dataframe.columns.tolist())`
  - Use exact column spelling/case.
- `NameError: plot_confusion_matrix is not defined`
  - Either define the helper function first, or use `ConfusionMatrixDisplay` from scikit-learn directly.
- `NameError: y_predict is not defined`
  - Run the prediction cell before evaluation: `y_predict = model.predict(test_X)`.
- `NameError: rf_model is not defined`
  - Train/assign your Random Forest model variable before calling `predict` or `predict_proba`.

## Notes

- This dataset is highly imbalanced; prioritize precision/recall and confusion matrix over accuracy alone.
- Keep variable names consistent across notebook cells to avoid `NameError` after reordering execution.

## Next Improvements

- Add ROC-AUC and PR-AUC plots
- Add cross-validation
- Add threshold tuning for fraud probability
- Export trained model and preprocessing pipeline
