# Microplastic Risk Analysis Dashboard

An end-to-end Machine Learning Dashboard built with Streamlit for analyzing, preprocessing, training models, and predicting microplastic risks.

## Features
- **Exploratory Data Analysis**: Visualize data distributions and handle missing values.
- **Preprocessing**: Automatic handling of categorical variables (One-Hot Encoding) and numeric scaling.
- **Feature Selection**: Identify the most important features using statistical tests and Random Forests.
- **Multi-Target Modeling**: Simultaneously train, evaluate, and compare models predicting `Risk_Type` and `Risk_Level`. Handles imbalanced classes using SMOTE.
- **Prediction & Inference**: Upload new datasets and generate predictions using your trained models.

## Installation

Ensure you have Python 3.8+ installed.

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

Start the Streamlit server by running:
```bash
streamlit run app.py
```

This will automatically open the dashboard in your default web browser.

## Workflow
1. Start at the **Home** tab to upload your dataset (`.csv` or `.xlsx`).
2. Move through **Preprocessing** and **Feature Selection**.
3. Go to the **Modeling** tab to train models (you can compare multiple targets like `Risk_Type` and `Risk_Level`).
4. Evaluate models in the **Cross Validation** tab.
5. Go to **Prediction & Inference** to generate predictions on unseen data using your best trained models.
