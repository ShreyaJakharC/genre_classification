# Music Genre Classification Capstone

## Overview
For my Numerical Computing capstone, I tackled a music genre prediction challenge: given audio and metadata features for 5,000 tracks, build a model that accurately assigns each song to one of ten genres. I cleaned and preprocessed the data, engineered a robust pipeline, and leveraged XGBoost to deliver strong multi-class performance.

## Features
- **Data Cleaning & Imputation**  
  - Replaced “?” with `NaN`, dropped rows missing critical identifiers (`track_name`, `artist_name`), and imputed numeric features with the median to guard against outliers.
- **Feature Engineering & Encoding**  
  - Standardized numerical features (`popularity`, `danceability`, `loudness`, etc.) with `StandardScaler`.  
  - One-hot encoded categorical fields (`key`, `mode`) and label-encoded the target `music_genre`.
- **Pipeline Architecture**  
  - Built a single `sklearn` pipeline to chain preprocessing and modeling, ensuring reproducibility and preventing data leakage.
- **XGBoost Classifier**  
  - Trained an XGBoost model tuned for multi-class classification, balancing speed and accuracy on mixed numeric/categorical data.

## Tech & Tools
- **Language:** Python 3  
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook (`analysis.ipynb`)  
- **Version Control:** Git & GitHub  

## Results & Key Takeaways
- **Overall Accuracy:** 58% on the held-out test set  
- **Confusion Matrix Insights:**  
  - Genres 1, 3, 5, and 7 were reliably identified (e.g., 441 correct for genre 3).  
  - Common confusions between genre 0 ↔ 9 and genre 6 ↔ 9 highlighted feature overlap.
- **ROC-AUC Performance:**  
  - Per-class AUC ranged from 0.86 (genre 0) to 0.98 (genres 1 & 3).  
  - Macro-averaged AUC of 0.93 demonstrates strong discriminative power across all genres.

## Skills Gained
Data Wrangling, Preprocessing Pipelines, Modeling with XGBoost, Performance Evaluation, and Visualization.

## Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
