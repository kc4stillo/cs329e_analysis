# %% [markdown]
# # Homework 5 - Logistic Regression

# %% [markdown]
# ## Your Name Here
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
# Date Created: 9/30/25
# 
# Date Last Modified: 9/30/25
# 
# ---
# 
# Totoal Points 20. 

# %%
# Standard Headers
# You are welcome to add additional headers here if you wish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression


# Enable inline mode for matplotlib so that Jupyter displays graphs
%matplotlib inline

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
# # Task - 1 (5 points)
# We want to use logistic regerssion to perdict if the patients will have heart problems or not. The column "Target" in our datasets includes data about heart diseases. If the patient had heart disease we have a 1 and if not a zero. 
# 
# Prepare your data set for predicting heart disease ("Target" column) out of 3 features:
# 
# * Age of the patient (Column **"Age"**)
# * Gender of the patient (male or female - Column **"Sex"**)
# * Cholestrol level of the patient (Column **"Chol"**) 
# 
# 
# Split your data into 80% traning data and 20% test data. 

# %%
X = pd.DataFrame(data={
    'Age': heart_df['Age'],
    'Gender': heart_df['Sex'],
    'Cholestrol': heart_df['Chol']
})

y = heart_df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %% [markdown]
# # Task 2 - (10 points)
# 
# Generate a logistic regression model using your training data. 
# 
# Print out the accuracy if your model. 

# %%
model = LogisticRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score: " + str(accuracy_score(y_test, y_pred)))
# %% [markdown]
# # Task 3 - (5 points)
# 
# 
# Generate the classification report for your Logistic regresion model and interpret your results regarding precision, recall and f1-score. 
# 

# %%
print(classification_report(y_test, y_pred))

#%%[markdown]
# Class “No”
# - Precision = 0.81: When the model predicts “No,” it is correct 81% of the time.
# - Recall = 0.74: The model correctly identifies 74% of all true “No” cases.
# - F1 = 0.78:  A good balance between precision and recall for “No.”
# </br>
# Class “Yes”
# - Precision = 0.69: When the model predicts “Yes,” it is correct 69% of the time.
# - Recall = 0.77: The model correctly identifies 77% of all true “Yes” cases.
# - F1 = 0.73: Slightly lower than “No,” but still balanced.
# </br>
# Overall performance
# - Accuracy = 0.75 → The model is correct on 75% of the predictions overall.
# - Macro avg = 0.75 → On average, the model performs about equally well on both classes, though “No” is predicted slightly more precisely and “Yes” is predicted with better recall.
# - Weighted avg = 0.76 → This accounts for the class imbalance (35 “No” vs 26 “Yes”), showing the model maintains stable performance across both classes.

# %% [markdown]
# # Task 4 - Optional Task (2 extra points)
# Which other feature of your data can you use to improve your prediction? Build other models and improve your prediction. 

X = pd.DataFrame(data={
    'Age': heart_df['Age'],
    'Gender': heart_df['Sex'],
    'Cholestrol': heart_df['Chol'],
    "RestECG" : heart_df["RestECG"]
})

y = heart_df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score: " + str(accuracy_score(y_test, y_pred)))

#%% [markdown]
# Adding the column of `ExAng`, tracking the person's maximum heartrate, improved our accuracy up to `0.7868852459016393`.