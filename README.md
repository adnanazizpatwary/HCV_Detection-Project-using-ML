# HCV_Detection-using-ML
## **Overview**
This repository contains the implementation of a predictive model for Hepatitis C classification using various machine learning techniques, including:

- Logistic Regression
- SVM(Support vector machine)
- Decision Tree Classifier

- Random Forest Classifier

- K-Nearest Neighbors (KNN)

- XGBoost

- AdaBoost

- Graph Neural Networks (GNN) using PyTorch Geometric

The dataset used is the **Egyptian Hepatitis C Dataset from Kaggle**, with feature selection applied via **Sequential Feature Selection (SFS) and Backward Feature Selection**.

## **Dataset**

The dataset consists of multiple features related to patient attributes and Hepatitis C status. Preprocessing techniques include:

- Handling missing values

- Normalization/Scaling

- Applying **SMOTE** for class balancing

- Feature selection via **SFS/Backward feature selection**

## **Model Implementation**

The repository includes implementations of different classifiers and a **Graph Neural Network (GNN)** built using **PyTorch Geometric** with:

- k-Nearest Neighbors (KNN) adjacency matrix

- Two **GCNConv** layers

- Cross-entropy loss function for classification
  
## **Hyperparameter Tuning**

Hyperparameter tuning is applied using:

- **GridSearchCV/Randomizedsearchcv** for optimizing traditional machine learning models

- **Optuna** for tuning XGBoost and GNN parameters

 ## **Explainable AI (XAI)**
In this project, Explainable AI (XAI) techniques were applied to interpret and explain the models' predictions.
- **SHAP** (Shapley Additive Explanations): Used to visualize the impact of each feature on the model's output. SHAP values help explain the model’s predictions in a way that is both locally and globally interpretable.
  
By using SHAP, you can visualize feature importance and better understand how the models are making decisions. In the notebook, you will find examples of SHAP summary plots, dependence plots, and feature importance visualizations.

## **Repository Contents**

- final.ipynb: Jupyter Notebook containing the entire workflow, including preprocessing, feature selection, model training, evaluation, and comparisons.

- data/: (Not included, but dataset details provided)

- models/: Contains saved trained models (if applicable)

- requirements.txt: List of dependencies required to run the notebook

## **Installation & Usage**

**Requirements**

Install dependencies using:

!pip install -r requirements.txt

## **Results & Evaluation**

- Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

- The best-performing model achieved an accuracy of 99%, with an F1-score of 99%.

- XGBoost is selected based on classification performance on the test set.
