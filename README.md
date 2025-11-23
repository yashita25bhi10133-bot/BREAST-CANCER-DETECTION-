# BREAST-CANCER-DETECTION-
# Breast Cancer Wisconsin (Diagnostic) Classification

## Project Title
**Predicting Malignancy: A Machine Learning Approach to Breast Cancer Classification**

## Overview of the Project
This project implements a machine Learning model to classify breast mass tissue as either **Malignant (Cancerous)** or **Benign (Non-Cancerous)**. The model is trained on the Breast Cancer Wisconsin (Diagnostic) Dataset (`brca.csv`), which contains 30 features computed from digitized images of fine needle aspirate (FNA) samples. The primary goal is to build a highly accurate and sensitive classifier to aid in diagnostic prediction.

## Features (Functional Modules)

The project is structured into three major functional modules:

1.  **Data Preprocessing and Standardization:** Cleans the raw data, encodes the categorical target variable ('M' and 'B' to 1 and 0), handles the train/test split, and applies **StandardScaler** to normalize the feature distribution.
2.  **Model Training and Hyperparameter Tuning:** Trains a powerful **Support Vector Machine (SVM)** classifier (or your chosen model) and uses techniques like **GridSearchCV** to find the optimal hyperparameters for maximum performance, particularly for maximizing Recall.
3.  **Performance Evaluation and Reporting:** Generates key classification metrics—**Accuracy, Precision, Recall, F1-Score**—and outputs a **Confusion Matrix** to analyze the model's predictive performance and identify critical errors (False Negatives).

## Technologies/Tools Used
* **Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (sklearn)
* **Development Environment:** Jupyter Notebook / Visual Studio Code

## Steps to Install & Run the Project

1.  **Prerequisites:** Ensure Python 3.x is installed on your system.
2.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [PROJECT_FOLDER_NAME]
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
4.  **Run the Script/Notebook:**
    * **If using a Python script:** `python main_classification.py`
    * **If using the Jupyter Notebook:** Open `brca.ipynb` in Jupyter/Colab and run all cells sequentially.

## Instructions for Testing
To verify the system's output:
1.  Run the main file/notebook as described above.
2.  Review the **Confusion Matrix** output. The key test is to ensure the number of **False Negatives** (Malignant cases predicted as Benign) is minimized.
3.  The final output should display the model's metrics, validating that the overall **Accuracy is above 90%** and **Recall (Sensitivity) is maximized**.

---

## 2. `statement.md` (Scope and Requirements)

This file formally defines the project's purpose, scope, and target audience.

```markdown
# 🎯 Project Statement: Breast Cancer Classification

## Problem Statement
Breast cancer is one of the most common cancers among women globally. Early and accurate diagnosis is crucial for effective treatment and patient prognosis. The current diagnostic process can involve manual analysis of cell samples, which is time-consuming and subject to human error. The problem addressed by this project is to **develop a reliable and automated machine learning system** that can classify a tumor based on its measured cell characteristics, thereby assisting medical professionals in making faster and more consistent diagnostic decisions.

## Scope of the Project

* **In Scope:**
    * Implementing data cleansing and feature scaling (Standardization).
    * Training a binary classification model (e.g., SVM, Logistic Regression, Random Forest).
    * Evaluating the model using classification metrics (Accuracy, Precision, Recall).
    * Focusing on maximizing **Recall (Sensitivity)** for the Malignant class (Class 1) to minimize dangerous False Negatives.
* **Out of Scope:**
    * Deployment as a production-ready web application.
    * Integration with hospital systems or real-time medical imaging hardware.
    * Advanced image processing of the raw cell images (the project starts with the extracted features).

## Target Users
1.  **Medical Researchers:** To quickly test new classification algorithms against a standardized dataset.
2.  **Data Science Students:** To learn and practice binary classification techniques on a real-world health dataset.
3.  **Clinicians/Diagnostic Labs (Conceptually):** As a proof-of-concept tool to provide a second opinion on FNA analysis results.

## High-Level Features
1.  **Data Ingestion & Transformation:** Accepts the `brca.csv` file and prepares it for ML training.
2.  **Predictive Modeling:** Trains a classification algorithm to learn the mapping from 30 features to the 'Benign' or 'Malignant' outcome.
3.  **Metric Reporting:** Provides an automated report of the model's performance on unseen test data.

---

**Next Steps:** You should now focus on **improving your model's performance (specifically Recall)** and then writing the **detailed Project Report (PDF)** based on the structure provided in your `vityarthi.pdf` document.
