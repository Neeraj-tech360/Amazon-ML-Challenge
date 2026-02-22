# 🛒 Amazon ML Challenge 2025 – Smart Product Pricing

## 📌 Overview

E-commerce pricing depends on a complex mix of textual attributes (brand, pack size, specifications) and visual cues (packaging, perceived quality).
This project builds a **multimodal machine learning pipeline** to predict product prices using both **text and image data**.

**Objective:** Predict continuous product prices.
**Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)

Dataset Link: [https://www.kaggle.com/datasets/infiniper/amazon-ml-challenge-2025-dataset](https://www.kaggle.com/datasets/infiniper/amazon-ml-challenge-2025-dataset)

Repository Link: [https://github.com/Infiniper/Amazon-ML-Challenge-2025.git](https://github.com/Infiniper/Amazon-ML-Challenge-2025.git)

---

## 📊 Dataset

| Split | Size   | Description                      |
| ----- | ------ | -------------------------------- |
| Train | 75,000 | Product details + images + price |
| Test  | 75,000 | Product details (no price)       |

### Data Fields

* **sample_id** – Unique product identifier
* **catalog_content** – Title + description + item pack quantity
* **image_link** – Public URL of product image
* **price** – Continuous float value (target variable, train only)

---

## 🧠 Multimodal Architecture

### 1️⃣ Raw Inputs

* Text: `catalog_content`
* Image: `image_link`
* Target: `price`

### 2️⃣ Feature Engineering

Unstructured data is converted into numerical embeddings:

* **Regex Numeric Extraction** → word count, char length, extracted numeric values & units
* **S-BERT (Text Embeddings)** → 384-dimensional semantic vector
* **CLIP (Image Embeddings)** → 512-dimensional visual feature vector

### 3️⃣ Feature Stacking

All features are concatenated into a unified matrix (~900 features per product).

### 4️⃣ Regression Model

A **LightGBM Regressor** learns non-linear relationships between stacked features and price.

---

## ⚙️ Training Strategy

### 🔹 Log Target Transformation

To handle skewed price distribution:

```
log_price = ln(1 + price)
```

Predictions are inverse-transformed after inference.

### 🔹 Stratified K-Fold (for Regression)

Log-prices are binned into quantiles to maintain balanced price distribution across folds.

### 🔹 Custom SMAPE Metric

SMAPE is implemented directly inside LightGBM for evaluation consistency.

### 🔹 Ensemble Bagging

Multiple models trained across different seeds and averaged to reduce variance.

---

## 💡 Design Decisions

* **Why Multimodal?** Text misses visual cues; images miss quantity/scale. Fusion removes blind spots.
* **Why S-BERT & CLIP?** Transfer learning provides rich features without heavy training cost.
* **Why LightGBM?** Tree-based models outperform deep networks on structured embedding features.

---

## 📈 Evaluation Metric

```
SMAPE = (1/n) * Σ |Actual - Predicted| / ((|Actual| + |Predicted|)/2) * 100%
```

* Range: 0% – 200%
* Lower is better

---

## 📂 Repository Structure

```
├── dataset/                     # Sample CSVs and dataset info
├── src/
│   ├── utils.py                 # Image downloading utilities
│   └── example.ipynb            # Starter notebook
├── Other_Notebooks/
│   ├── light-gbm/               # LightGBM pipeline & feature extraction
│   └── SVM/                     # Alternative approach
├── Results on test data/        # Prediction outputs
├── main_notebook.ipynb          # Final multimodal pipeline
├── download_dataset.py          # Dataset download script
└── calculate_smape.py           # Local SMAPE evaluation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```
git clone https://github.com/Infiniper/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025
```

### 2️⃣ Download Dataset

Download from Kaggle and place:

```
dataset/train.csv
dataset/test.csv
```

### 3️⃣ Run Pipeline

Open:

```
main_notebook.ipynb
```

This notebook covers end-to-end:

* Data preprocessing
* Text & image embedding generation
* Feature stacking
* LightGBM training
* SMAPE evaluation

---

## 📌 Notes

* No external price lookup was used.
* Model size constraints follow competition guidelines.
* Designed for reproducibility and modular experimentation.

---
