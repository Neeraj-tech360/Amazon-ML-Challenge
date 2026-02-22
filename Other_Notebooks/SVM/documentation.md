# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** SVM  
**Team Members:** Shashwat Pandey, Vishwajeet Singh, Md Yusuf Murtaza  
**Submission Date:** 12th October 2025

---

## 1. Executive Summary
Our solution addresses the product pricing challenge by creating a rich, hybrid feature set. We leverage pre-trained deep learning models—Sentence-Transformers for text and CLIP for images—to generate powerful semantic embeddings. This data is fused with engineered tabular features and fed into a LightGBM model, achieving high predictive accuracy with remarkable computational efficiency.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We treated this as a multimodal regression task. Key insights from EDA were the heavily skewed price distribution, which necessitated a log transformation, and the presence of crucial features like product Value and Unit embedded within the unstructured catalog_content.

#### Key Observations:
1. **Price Skewness:** The price distribution was heavily right-skewed, making a log transformation essential for model stability and performance.
2. **Embedded Features:** The unstructured catalog_content field contained critical, high-signal features like product Value and Unit (e.g., "12 Ounce"), which directly correlate with price.
3. **Data Quality:** A portion of the image links were broken, necessitating a robust pipeline that could handle missing visual data gracefully.

### 2.2 Solution Strategy
Our strategy generates a powerful feature set by converting unstructured images and text into semantic embeddings using pre-trained models. These are then combined with parsed tabular features and fed into a gradient boosting regressor.

**Approach Type:** Hybrid Feature Fusion

**Core Innovation:** Our key innovation is fusing rich, pre-trained deep learning embeddings with high-signal, engineered tabular features. This hybrid approach allows a computationally inexpensive LightGBM model to achieve performance that rivals more complex, end-to-end architectures.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our architecture is a hybrid, feature-fusion pipeline. It independently processes text and image data through pre-trained deep learning models to generate fixed-size embeddings. These embeddings are then concatenated with simple, hand-crafted features into a single vector, which is fed into a LightGBM model for the final price prediction.


### 3.2 Model Components

**Text Processing Pipeline:**
- **Model type:** Sentence-Transformers (all-MiniLM-L6-v2).
- **Preprocessing steps:** Filled missing text with empty strings; the model's tokenizer handled all other processing.
- **Key parameters:** A 384-dimensional embedding vector per product.

**Image Processing Pipeline:**
- **Model type:** OpenAI CLIP (openai/clip-vit-base-patch32).
- **Preprocessing steps:** Images were resized and normalized via the standard CLIP processor. A blank white image was used for missing or broken links.
- **Key parameters:** A 512-dimensional embedding vector per product.

**Tabular Feature Pipeline**
- **Feature engineering:** Manually extracted Value, Unit, char_len, and word_count from catalog_content using regex and string operations.
- **Preprocessing:** Missing numerical values were imputed with zero.
- **Output:** A 4-dimensional numerical feature vector.

**Final Regression Model:**
- **Model type:** LightGBM Regressor (lgb.LGBMRegressor).
- **Input:** A concatenated feature vector of 900 dimensions (384 + 512 + 4).
- **Target:** The price column, transformed using np.log1p to stabilize variance.


---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** [your best validation SMAPE]
- **Other Metrics:** [MAE, RMSE, R² if calculated]


## 5. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*


### B. Additional Results
*Include any additional charts, graphs, or detailed results*

---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.