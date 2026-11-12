# Explainable Multi-Model Machine Learning for Osteoporosis Risk Prediction

Research prototype for **preliminary osteoporosis risk screening**, not a clinical diagnostic tool or replacement for DXA.

## Overview

This project compares six trained classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost and TabNet. It uses demographic, anthropometric, nutritional, physical-activity, lifestyle and medical-history inputs, with seven engineered features. SHAP and TabNet feature importance provide model explanations.

## Results

| Metric | Best reported model | Score |
|---|---|---:|
| ROC-AUC | XGBoost | 0.816 |
| Accuracy | LightGBM | 0.932 |
| Average precision | LightGBM | 0.213 |
| Balanced accuracy | Logistic Regression | 0.728 |
| Sensitivity | Logistic Regression / TabNet | 0.659 |
| F1-score | CatBoost | 0.302 |

Results are from the project's held-out test evaluation, not a clinical validation.

## Run the Streamlit application

From this folder, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

The application loads the already-trained models from `saved_models/`; **you do not need to retrain TabNet**. Python/package compatibility may need adjustment to match the original training environment.

## Notebook

`Osteoporosis_merge.ipynb` documents the model development and evaluation workflow. It includes time-consuming training and tuning cells. For a quick review, use the saved-model loading and evaluation sections instead of running all notebook cells.

## Files

- `app.py`: Streamlit patient-input and prediction interface
- `saved_models/`: six saved models, preprocessors and thresholds
- `osteoporosis_clean_dataset.csv`: prepared analysis dataset
- `Osteoporosis_merge.ipynb`: full experiment notebook
- `*.png`: selected final result visualizations

The raw NHANES `.xpt` source files and temporary logs were omitted from this streamlined GitHub package; the notebook loads the prepared CSV.

## Disclaimer

Predictions indicate modeled risk only. They must not be used to diagnose osteoporosis, replace clinical judgment, or substitute for DXA assessment.
