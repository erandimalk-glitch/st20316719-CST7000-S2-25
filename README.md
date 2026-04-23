# st20316719-CST7000-S2-25
Automatic Product Category Classification for Fashion E-Commerce Platforms

# Product Classification on the Atlas Fashion Dataset

A comparative study of machine learning and deep learning models for automatic product category classification, built as the full implementation supporting an MSc dissertation of the same title.

## Overview

This notebook builds a fashion product pair-matching pipeline on the [Atlas Fashion Dataset](https://github.com/erandimalk-glitch/st20316719-CST7000-S2-25). It synthesises positive and negative product pairs from a single-product image dataset, engineers TF-IDF and categorical features, then trains and evaluates nine classifiers — six traditional ML models and three neural networks — producing the exact metrics reported in Chapter 4 of the dissertation.

## Repository Structure

```
st20316719-CST7000-S2-25/
├── Product_Classification_Atlas.ipynb   # Main notebook (this file)
├── README.md
└── atlas_assets/
    ├── atlas_dataset.json               # Primary Atlas fashion dataset
    └── zvsn_data.json                   # Zoomed vs. normal image labels
```

## Dataset

The Atlas Fashion Dataset (`atlas_dataset.json`) contains single-product fashion images annotated with a three-token category hierarchy: `gender / wear / category`. An optional image-quality filter (`zvsn_data.json`) drops rows whose image is flagged as `zoomed`, keeping only normal-perspective product shots.

Since Atlas is a single-product dataset (no natural left/right pairs), the notebook synthesises 50,000 pairs:

- **Positive pairs (label=1):** two distinct products sharing the same `gender / wear / category` triple — 25,000 pairs
- **Negative pairs (label=0):** two products from different categories, sampled with inverse-frequency weighting to avoid the Women/Ethnic majority dominating — 25,000 pairs

## Pipeline

```
Load Atlas JSON
    → Image-quality filter (drop zoomed)
    → Pair synthesis (25k positive + 25k negative)
    → Feature engineering (TF-IDF + label-encoded categoricals)
    → Train / val / test split (60 / 20 / 20)
    → Train 9 classifiers
    → Evaluate & compare
```

### Feature Engineering

- TF-IDF on product titles (1–2 grams, sublinear TF, top 20,000 features) fitted on the union of left and right titles
- Label-encoded `gender`, `wear`, and `category` for both sides of each pair
- All features concatenated into a single sparse matrix

## Models

### Traditional ML

| Model | Notes |
|---|---|
| Random Forest | 200 estimators, max depth 20 |
| K-Nearest Neighbours | k=5 |
| Linear SVM | C=1.0 |
| Multinomial Naive Bayes | Works because TF-IDF features are non-negative |
| Decision Tree | Max depth 20, min samples leaf 4 |
| XGBoost | 400 estimators, early stopping, histogram method |

### Neural Networks

| Model | Architecture |
|---|---|
| Feed-forward NN | Sparse mini-batch streaming to avoid materialising the full dense matrix; standard dense layers + sigmoid output |
| Siamese NN | Twin-tower weight-sharing architecture on left/right TF-IDF vectors; concatenated tower output → sigmoid match probability |
| LSTM | Left + right titles joined with `[SEP]` token; embedding → LSTM → dense head; max sequence length 32 |

All neural network cells are self-bootstrapping — they re-import TensorFlow on kernel restart without needing to re-run the full imports cell.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost >= 2.0
imbalanced-learn
tensorflow (optional — required for NN/Siamese/LSTM cells)
```

On Google Colab all packages except `imbalanced-learn` are pre-installed. The setup cell installs any missing packages automatically.

## Running the Notebook

### Google Colab (recommended)

1. Upload `atlas_assets/` to your Google Drive under `MyDrive/Research/`
2. Open the notebook in Colab
3. Run all cells — Drive will mount automatically and the setup cell handles any missing packages

### Local

1. Clone the repo and `cd` into it
2. Place `atlas_assets/` in the same directory as the notebook
3. Install requirements: `pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn tensorflow`
4. Run all cells — the notebook detects it is not in Colab and adjusts paths automatically

## Outputs

The final comparison cell produces:

- A styled DataFrame ranking all nine models by weighted F1 score
- A bar chart comparing F1 scores across models
- `results_atlas.csv` saved to the working directory with full precision/recall/F1/accuracy for each model

