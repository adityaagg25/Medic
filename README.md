# Brain Signal Analysis for Attention Identification

This project focuses on identifying attention levels using brain signal data. The data was sourced from Kaggle and processed using machine learning techniques to classify attention into high, medium, and low levels based on predicted values.

## Dataset

- **Dataset Name:** `feature_raw.csv`
- The dataset was sourced from Kaggle.
- Data preprocessing steps included:
  - **Standardization** of the dataset.
  - **Outlier removal** using the Interquartile Range (IQR) method.

## Data Processing

1. **Outlier Removal & Standardization**:
   - We applied the IQR method to remove outliers and standardized the data to ensure it followed a consistent scale for model training.

2. **Train-Test Split**:
   - The dataset was split into:
     - **80% for training**
     - **20% for testing**

## Models Applied

We used the following machine learning models to train and test the dataset:

1. **Random Forest**
2. **XGBoost**
3. **Decision Tree**

### Model Performance:

| Model           | MSE (Train) | R-squared (Train) | MSE (Test) | R-squared (Test) |
|-----------------|-------------|-------------------|------------|------------------|
| Random Forest   | 0.000119    | 0.994527          | 0.001782   | 0.934577         |
| XGBoost         | 0.000029    | 0.998661          | 0.000720   | 0.973577         |
| Decision Tree   | 0.000000    | 1.000000          | 0.001326   | 0.951297         |

### Best Performing Model:
- **XGBoost** performed the best, as seen from its lowest **MSE** and highest **R²** score for both training and test sets. It shows strong predictive capabilities and generalizes well to unseen data.

## Attention Categorization

We used the **NLTK library** to categorize attention levels based on predicted values from the models:

- **High Attention**: Predicted values between **0.7 - 1**
- **Medium Attention**: Predicted values between **0.4 - 0.7**
- **Low Attention**: Predicted values below **0.4**

### Results:
Based on the scatter plots, the model's performance on the test data appears to match its performance on the training data. Both datasets show a similar distribution of predicted attention values across different sentiment categories, indicating that the model generalizes well.

## Conclusion

- XGBoost emerged as the best model for attention prediction based on brain signal data.
- The model's generalization capabilities were confirmed through consistent performance across both training and testing data.

## How to Run

1. Clone the repository.
2. Install necessary dependencies (NLTK, scikit-learn, etc.).
3. Run the code to replicate the results and categorize attention levels.

