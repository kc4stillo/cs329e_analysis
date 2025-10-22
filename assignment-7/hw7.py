# %% [markdown]
# # Homework 7
# 
# ## Your Name Here (or your names here if you are pair programming)
# 
# Student Name: Kyle Castillo
# 
# Student UT EID: kmc5794
# 
# ---
# 
# Partner Name: Andre Sae
# 
# Partner UT EID: as226576
# 
# ---
# 
# Date Created: 10/22/2025
# 
# Date Last Modified:
# 
# ---
# 
# Totoal Points 20. 
# 
# 
# 
# ## Supprt Vector Machine 

# %%
# Standard Headers
# You are welcome to add additional headers here if you wish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Enable inline mode for matplotlib so that Jupyter displays graphs
%matplotlib inline

# %% [markdown]
# # Your allowed to use only the above libraries that are imported. No other libs should be used in this assignment. 

# %% [markdown]
# ## Heart Dataset 
# 
# In this Assignment we will work with some patients dataset. 
# 
# We have access to 303 patients data. The features are listed below. 

# %%
# Your code here
heart_df = pd.read_csv("Heart.csv")
heart_df

# %% [markdown]
# **Age:** The person’s age in years
# 
# **Sex:** The person’s sex (1 = male, 0 = female)
# 
# **ChestPain:** chest pain type
# 
# * Value 0: asymptomatic
# * Value 1: atypical angina
# * Value 2: non-anginal pain
# * Value 3: typical angina
# 
# **RestBP:** The person’s resting blood pressure (mm Hg on admission to the hospital)
# 
# **Chol:** The person’s cholesterol measurement in mg/dl
# 
# **Fbs:** The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# restecg: resting electrocardiographic results
# 
# * Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
# * Value 1: normal
# * Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 
# **RestECG:** The person’s maximum heart rate achieved
# 
# **MaxHR:** Exercise induced angina (1 = yes; 0 = no)
# 
# **Oldpeak:** ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot. See more here)
# 
# **Slope:** the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping
# 
# * 0: downsloping; 
# * 1: flat; 
# * 2: upsloping
# 
# **Ca:** The number of major vessels (0–3)
# 
# **Thal:** A blood disorder called thalassemia Value 0: NULL (dropped from the dataset previously
# 
# * Value 1: fixed defect (no blood flow in some part of the heart)
# * Value 2: normal blood flow
# * Value 3: reversible defect (a blood flow is observed but it is not normal)
# 
# **Target:** Heart disease (1 = no, 0= yes)

# %% [markdown]
# # Task - 1 Implement SVM using libraries (4 points)
# We want to use **Suppert Vector Machine** to perdict if the patients will have heart problems or not. The column "Target" in our datasets includes data about heart diseases. If the patient had heart disease we have a 1 and if not a zero. 
# 
# Prepare your data set for predicting heart disease ("Target" column) out of 3 features:
# 
# * Age of the patient (Column **"Age"**)
# * Gender of the patient (male or female - Column **"Sex"**)
# * Cholestrol level of the patient (Column **"Chol"**) 
# 
# 
# Split your data into 80% traning data and 20% test data, and implement Support Vector Machine using Scikit-Learn. 
# 
# 
# 
# 

# %%
features = heart_df[["Age", "Sex", "Chol"]]
labels = (heart_df["Target"] == "Yes").astype(int)

# train/test split
split = .8

train = np.arange(0, round(features.shape[0] * split) + 1)
test = np.arange(train[-1] + 1, features.shape[0])

train_x = features.iloc[train]
train_y = labels.iloc[train]

test_x = features.iloc[test]
test_y = labels.iloc[test]

# %%
from sklearn import svm # type: ignore

svm_ =svm.SVC()
trained_svm=svm_.fit(train_x, train_y)
y_pred = trained_svm.predict(test_x)

# %% [markdown]
# # Task 2 - (4 points)
# 
# Cacluate the accuracy, Precision, Recall and F1 score of your **SVM** implementaion from Task 1. 
# Print the results. 
# 
# You may use library methods for this task if you choose to.
# 

# %%
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(test_y, y_pred)
class_report = classification_report(test_y, y_pred)

print(f"confusion matrix:\n{conf_matrix}\n")
print(f"classification report:\n{class_report}")

# %% [markdown]
# # Task 3 - Implement SVM without using libraries  - (4 points)
# 
# Implement SVM from scratch using Hinge Loss function and Gradient Descent. 
# Try to produce the same result as you get from the libraries. 
# 
# 
# * Do as many iterations as needed 
# * Do maximum **100 iterations**
# * Use a very small learning rate for checking your GD implementation. 
# * Your are allowed to use your choice of learning rate, like using 0.0001, 0.001 or 0.01 or 0.1 or higher. 
# * Visualize your costs. 
# * No need to add an y-intercept in this task. 
# * You can use libraries to report accuracy, Precision, Recall and F1. 
# 
# 

# %% [markdown]
# # Task 4 - Compare SVM results with Logistic Regression - (4 points)
# 
# Which model performs better here? Compare your results wit the logistic regression. You can use libraries for this task, it is not necessary to implement logistic regression from sratch.
# 

# %%
# Add your code Here! 

# %% [markdown]
# # Task 5 - Apply a kernel function to improve SVM performance (4 points)
# 
# Use the Scikit-learn librariy and apply a kernel function to improve the SVM performance. Check if this is possible. 
# 

# %%
# Add your code Here! 

# %%


# %%



