# 🏠 Kaggle House Prices Prediction

A machine learning solution for the Kaggle House Prices: Advanced Regression Techniques competition, predicting residential home prices in Ames, Iowa.

## 📋 Description

This repository contains a comprehensive solution for predicting house prices using regression techniques. The implementation leverages multiple machine learning models in an ensemble approach to achieve high prediction accuracy.

The project addresses the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), which challenges participants to predict the sale price of homes in Ames, Iowa based on 79 explanatory variables covering almost every aspect of residential properties.

## 🔍 Features

- **Extensive Data Analysis**: Includes correlation analysis, outlier detection, and feature importance evaluation
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature engineering
- **Model Ensemble**: Combines multiple regression models for improved prediction accuracy:
  - Lasso Regression
  - Ridge Regression
  - Gradient Boosting Regressor
  - Elastic Net
- **Model Stacking**: Implements a stacked model approach with cross-validation
- **Feature Engineering**: Creates new features to improve model performance
- **Outlier Handling**: Identifies and addresses outliers in the dataset

## 🛠️ Setup Guide

### Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
   ```

## 📊 Usage

1. Ensure the dataset files (`train.csv` and `test.csv`) are in the repository root directory
2. Run the main script:
   ```bash
   python housesPredictPrices.py
   ```
3. The script will generate predictions in `test_set_prediction.csv`

## 🧮 Implementation Details

### Data Preprocessing

- **Missing Value Handling**: Implements sophisticated strategies for handling missing values based on domain knowledge
- **Feature Transformation**: Applies log transformation to skewed numerical features
- **Categorical Encoding**: Converts categorical variables to numerical using ordinal encoding and one-hot encoding

### Model Architecture

The solution employs a stacked ensemble approach:

1. **Base Models**:
   - Lasso Regression with alpha tuning
   - Ridge Regression
   - Gradient Boosting Regressor
   - Elastic Net

2. **Stacking Layer**:
   - Uses a Lasso model to combine predictions from base models
   - Implements k-fold cross-validation (k=20) for robust evaluation

### Feature Engineering

- Creates a `TotalSF` feature combining basement, first floor, and second floor square footage
- Develops a `Total_Home_Quality` feature from quality and condition ratings
- Bins year-related features to capture temporal patterns
- Generates squared terms for important numerical features

## 📈 Results

The implementation achieves competitive results on the Kaggle leaderboard by:
- Effectively handling the complex relationships between features
- Addressing outliers and skewed distributions
- Leveraging the strengths of multiple regression algorithms
- Using cross-validation to ensure model robustness

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Ames Housing Dataset Documentation](http://jse.amstat.org/v19n3/decock.pdf)
