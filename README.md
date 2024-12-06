# Project: Exploratory Data Analysis and Classification in mixed data using NLP, Encoding methods, Feature Selection, Cross-Validation, Grid Search as well as Data Integration,  SQL Optimization

## Overview

This project focuses on processing a dataset containing company-related information through data integration, exploratory data analysis (EDA), and model development. The workflow is modular, with each task handled by a dedicated Python script.

---

## Folder Structure

- **`1. SQL Queries/`**: Contains SQL queries related to Task 1.
- **`2. Data Integration/`**: Includes the code for data integration, corresponding to Task 2.
- **`3. Exploratory Data Analysis/`**: Contains the code for performing exploratory data analysis for Task 3.
- **`4. Model Development/`**: Contains code related to the model development process for Task 4.
- **`5. Evaluation/`**: Contains code for evaluating the model in Task 5.
- **`6. Next Steps/`**: Includes the code for the next steps to be taken.
- **`db/`**: Contains the script to download the database and is the folder where the SQLite database will be stored.
- **`tools/`**: Utility scripts for encoding, text representation, text preprocessing, and SQLite data reading.
- **`report/`** : Contains the project description and the project report.

---

## How to Run the Code

### 1. Download the Data

Run `db/download.py` to download the SQL database, which is used for Task 1 queries.

### 2. Data Integration and Database Insertion (Task 2)

Run `2. Data Integration/data_integration.py` to process the initial dataset and create the `CompanyData` table in the SQLite database.

### 3. Exploratory Data Analysis (Task 3)

To perform the changes after conducting EDA and creating the `CompanyDataProcessed` table, run the `3. Exploratory Data Analysis/eda_changes.py` script. This script handles the EDA modifications and creates the processed data table.

### 4. Full EDA Code and Plots

The `3. Exploratory Data Analysis/all_eda.py` script contains the full code for the exploratory data analysis, including both the data modifications and the plots, as described in the `report.pdf`. The sequence of execution follows the report’s structure.

### 5. Plots for EDA Questions

The `questions.py` script contains the plots for the "questions about the data" section in the EDA analysis. This script does not include any prior code such as loading data or importing libraries, so you will need to run it in an IPython console (or Jupyter notebook) after executing the necessary data integration and EDA steps.

```bash
# In an IPython console or Jupyter notebook, run:
%run questions.py
```

### 6. Model Development (Task 4)

The **`4. Model Development/identify_best_model.py`** script identifies the optimal text representation, encoding method, feature selection technique, and classifier.

The **`4. Model Development/validate_best_model.py`** script validates the model identified in **`identify_best_model.py`** through cross-validation.

The **`4. Model Development/gridsearch.py`** script performs a grid search to find the best hyperparameters for the model configuration determined in **`identify_best_model.py`**.

### 7. Model Evaluation (Task 5)

The **`5. Evaluation/evaluate_best_model.py`** script evaluates the performance of the final best model.

The **`5. Evaluation/format_results.py`** script formats the results from **`4. Model Development/identify_best_model.py`** into an Excel file.

### 8. Next Steps

The **`6. Next Steps/identify_best_model_gridsearch.py`** script finds the best text representation, encoding method, feature selection, and classifier, similar to **`4. Model Development/identify_best_model.py`**, but incorporates grid search and cross-validation.

The **`6. Next Steps/LLM.py`** script trains and evaluates the DistilBERT model.

---

# Dependencies

Install required packages using requirements.txt:
```bash
pip install -r requirements.txt
```
