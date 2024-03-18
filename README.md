

# Project Motivation
1. The main objective of this project is to create a method that can be used to predict a prognosis given a set of symptoms.
Such a tool could be valuable in the medical field where time is of the essence and precision is literally a matter of life and death

# Libraries Employed
The project is made up of 2 parts, a Extractio, Transformation and Loading part and a Data Analysis part. 
Combined, the 2 parts used the following libraries:
* import networkx as nx
* import matplotlib.pyplot as plt
* import pandas as pd
* from sklearn.feature_selection import mutual_info_regression
* import numpy as np
* import seaborn as sns
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LogisticRegression
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
* from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 
* import tabulate

# Data, Wrangling and Cleaning
The data for this mission is called "Disease Prediction Data" and is sourced from https://www.kaggle.com/datasets/marslinoedward/disease-prediction-data (Edward, M. (Owner). (2024, Jan)). The data comprises 134 columns in total. 132 of these provide symptom data, 1 column comprises 42 diseases and is therefore the target variable and the last column is an unnamed variable. 
The data is originally split into the training and test sets.
Part of the cleaning process involved removing the unnamed variable and isolating the explanatory inputs from the response variable.
 
# Data Analysis
This portion began with analyzing the input data to check if the response variable was balanced and to understand the explanatory variables in terms of relative presence/absence of the symptoms among the cases studied.

# Train and Test Logistic Model
Two models were fit to the data, a logistic regression and a decision tree model. After evaluating their respective Area Under the Curve scores, the models were "upgraded" by use of 5--fold cross validation. In the end, 4 models were fit to the data and their performances were comparatively analysed based on accuracy, precision, recall and f1 scores. 

# Results Interpretation
All 4 models were relatively performant on the out-of-sample testing data. However, their AUCs were relative poor. This could be because the symptoms explaining the diseases are likely highly informative and discriminatory, allowing each of the models to make precise predictions. On the other hand, the low AUCs migh be indicitive of each models' inability to discriminate across the given classes/diseases. As a result, it might be worthwile to seek more realistic data and re-evaluate all models presented herein.

# Conclusion
According to the model, it is likely that if a logistic regression were fit to data comprising symptoms of known ailments then the underlying disease could potentially be well predicted given sufficiently adequate and realistic inputs. 

# Supporting Material
The entirity of the project can be found on Github in repo: https://github.com/BrianMekiSCA/Project-Data-Scientist-Capstone-V2
The reader can also find a write-up of the project at https://medium.com/@brian.meki/life-or-death-46db705b0285
A collection of files is accompanies this project. These include:
* A ETL Python script that concentrates on extraction, transformation and loading of the project's data
* A ML pipeline Python Script that specifically fits data to various models, valuates models and discusses results
* Various plot and tabular outputs many of which serve as attachements to my blogpost
* Copies of training and test Disease Prediction Data in CSV format,
* Plot outputs showing proportions of diseases in the responce variable, proprtions of presence/absence of symptoms from the evaluation of the model are included as well.
