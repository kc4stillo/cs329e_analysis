# %% [markdown]
# # Assignment 8 - k-Nearest Neighbors (kNNs)
# (20 points)
# 
# ### Add your name(s) and EIDs below
# - Student Name:
# - Student UT EID:
# - Partner Name:
# - Partner UT EID:
# 
# 
# # k-Nearest Neighbors
# For this assignment, we are going explore one new classification technique: k nearest neighbors.
# 
# We are using a different version of the Melbourne housing data set from earlier in the semester, split into training and testing sets for you. Our goal is to predict the housing type as one of three possible categories:
# 
#   - 'h' house
#   - 'u' duplex
#   - 't' townhouse
# 
# At the end of this homework, you will understand how to build and use a kNN model, and improve your data cleaning and data preparation skills. 

# %%
# These are the libraries you will use for this assignment.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import calendar
%matplotlib inline

# %%
# Start off by loading the training dataset.
df_melb = pd.read_csv('melb_data_train.csv')

# %% [markdown]
# ## Q1 
# 
# **Fix our "Date" column to be numeric**: If we inspect our dataframe `df_melb` using the `dtypes` property, we see that the column `Date` is an `object`.  However, we think this column might contain useful information, so your goal is to convert it to [Unix time](https://en.wikipedia.org/wiki/Unix_time).
# 
# Unix time is the number of secconds since a fixed time known as the "Unix epoch", which is midnight on January 1st, 1970. For example, the Unix time for March 10th, 2023 is 1,678,474,369 seconds.
# 
# - **Use only the libraries imported above** imported libraries to create a new column `UnixTime`. 
#     - Be careful, the date strings in the file might have some non-uniform formatting that you have to fix first.  
# - Print out the min and max epoch time to check your work.  
# - Drop the original `Date` column. 
# 
# The Python [reference for time](https://docs.python.org/3/library/time.html) can help you with your conversion to Unix time.
# 
# (**3 points**)

# %%
# For reference, here are the data types of each column.
df_melb.dtypes

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
def standardize_date(date_string):
    """Standardize a date string to a standard format.

    Rules:
    - You can assume the input string is of the form day/month/year.
    - Fixed date strings should be of the form DD/MM/YYYY. If a day is
      one digit, append zeros.
    - If the input string's year is two digits (e.g. 02), assume
      the year is in the 2000s (e.g. 2002).
    """
    fixed_date_string = ...
    return fixed_date_string

def replace_date_with_unix(df):
    """Given a Melbourne dataset dataframe, replace the Date column
    with a UnixTime column.

    Hint: Call standardize_date within this function.
    """
    # Standardize the date column.
    ...
    # Create the UnixTime column
    ...
    # Drop the date column.
    df = ...
    return df

# %%
df_melb_q1 = replace_date_with_unix(df_melb)

# Print the cleaned UnixTime values.
print('Min UnixTime:', df_melb_q1['UnixTime'].min())
print('Max UnixTime:', df_melb_q1['UnixTime'].max())

# %% [markdown]
# <!-- END QUESTION -->
# 
# ## Q2 
# 
# **Use imputation to fill in missing values**: kNN doesn't work when some attributes are not present, so we must fill in all the missing values in `df_melb` with something. As a simple estimate, we will fill in missing values with the **mean** of that value/column.
# 
# What we're trying to classify ('h'ome/'d'u'plex/'t'ownhouse), also knonw as the **target**, is store in the `Type` column. We define a variable `target_col` which lets you automatically infer which column is the target. During imputation, we should skip this target column.
# 
# - Use `df_melb_q1`, i.e. the result from Q1.
# - Save the mean of each column in a dictionary `dict_imputation`. Keys are an attribute's column name, and values are that attribute's mean.
# - Use `dict_imputation` to imputate the missing values in `df_melb_q1`.
# - Store the imputated dataframe in `df_melb_q2`.
# 
# (**3 points**)

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
def build_imputation_dict(df, target_col):
    """Collect the mean values of each column, excluding NaN values
    and the target column.
    """
    dict_imputation = {}

    # Get the mean value of each column.
    ...
    return dict_imputation

def imputate(df, dict_imputation, target_col):
    """Imputate a dataframe, replacing missing values with those
    given in dict_imputation. Do not imputate target_col."""
    df = ...
    ...
    return df

# %%
# Define the target column as a string
target_col = ...

# Collect imputation values
dict_imputation = build_imputation_dict(df_melb_q1, target_col)

# Imputate the dataframe
df_melb_q2 = imputate(df_melb_q1, dict_imputation, target_col)

# %%
# Check your results
dict_imputation

# %%
# Check your results
df_melb_q2.head()

# %% [markdown]
# <!-- END QUESTION -->
# 
# ## Q3
# 
# **Normalize all attributes to be between [0,1]**: Normalize all the attribute columns in `df_melb_q2` so they have a value between zero and one (inclusive). 
# 
# To do this, we will build a dictionary `dict_normalize`, with column names for keys and (min, max) tuples for values, which are the min (resp. max) value found in the dataframe for that column. Just like in Q2, we do not normalize the target column.
# 
# After creating `dict_normalize`, we will use it to normalize each column and generate a new dataframe, `df_melb_q3`. The resulting dataframe is now your model that you can use to classify new data points.
# 
# - Use `df_melb_q2`, i.e. the result from Q2.
# - Save the minimum and maximum values of each column in a dictionary `dict_normalize`. Keys are an attribute's column name, and values are a (min, amx) tuple for that column,
# - Use `dict_normalize` to normalize the missing values in `df_melb_q2`.
# - Store the imputated dataframe in `df_melb_q3`.
# 
# (**3 points**)

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
def build_normalization_dict(df, target_col):
    """Collect the (min, max) values of each column, except the
    target column.
    """
    dict_normalize = {}

    # Get the min and max values of each column.
    ...
    return dict_normalize

def normalize(df, dict_normalize, target_col):
    """Normalize a dataframe, setting all values to the range [0, 1]
    using (min, max) values in dict_normalize. Do not normalize target_col."""
    df = ...
    ...
    return df

# %%
# Define the target column as a string
target_col = ...

# Collect normalization values
dict_normalize = build_normalization_dict(df_melb_q2, target_col)

# Normalize the dataframe
df_melb_q3 = normalize(df_melb_q2, dict_normalize, target_col)

# %%
# Check your results
dict_normalize

# %%
# Check your results
df_melb_q3.head()

# %% [markdown]
# <!-- END QUESTION -->
# 
# ## Q4 
# 
# **Prepare the test data**: Load in `melb_data_test.csv` and repeat the steps in Q1, Q2, and Q3 (unix time, imputation, and normalization).
# 
# (**1 point**)

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
# Load the test dataframe
df_melb_test = ...

# Clean the dates, add unix time
df_melb_test = replace_date_with_unix(df_melb_test)

# Imputate the dataframe
target_col = ...
dict_imputation_test = ...
df_melb_test = ...

# Normalize the dataframe
dict_normalize_test = ...
df_melb_test = ...

# %%
# Check your results
df_melb_test.head()

# %% [markdown]
# <!-- END QUESTION -->
# 
# ## Q5
# 
# **Write the kNN classifier function**: Your function, `predict_knn` will take in the following four parameters:
# - Training dataframe `df_train`
# - Hyperparameter `k`
# - Testing sample `test_sample` (one row of the DataFrame, we can generate it using `iloc` or `iterrows`).
# - Target column string `target_col`, which defines the variable we want to predict.
# 
# It will predict which class `test_sample` belongs to, based on the `k` nearest neighbors to the sample.
# 
# - We assume `df_train` is normalized/imputated, contains all attributes, and also contains the target column.
# - Likewise, we assume `test_sample` is normalized/imputated and contains all attributes. (But, it does not have to have the target column).
# 
# *Hint*: To find the distance between the test sample and any element of the training dataset, you may use the [L2 norm function from numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).
# 
# (**5 points**)

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
def predict_knn(df_train: pd.DataFrame, k: int, 
                test_sample: pd.Series, target_col: str):
    """Use the k-nearest neighbors algorithm to predict the class of a test-sample,
    given a training set.
    
    Parameters:
        df_train:    DataFrame of training samples
        k:           Number of neighbors to consider
        test_sample: Single evaluation sample
        target_col:  Name of the target variable (column)

    Returns:
        prediction: Predicted class of the test sample using kNN.
    """
    ...
    prediction = ...
    return prediction

# %% [markdown]
# <!-- END QUESTION -->
# 
# ## Q6 
# 
# **Compute the accuracy using different k values**: For each value of $k$ in the set $\{1,3,13,25,50,100\}$, compute the kNN prediction for each oberservation in the test set, and the overall accuracy of the classifier.  Plot the accuracy as a function of $k$.
# 
# - Use your imputed, normalize training dataframe (`df_melb_q3`).
# - Use your imputed, normalized testing dataframe (`df_melb_test`).
# - Have an outer loop over the k-values, and an inner loop computing the prediction for each testing sample under that k-value.
# 
# Which value of $k$ would you choose? Why?
# 
# (This can take a while to run; probably at least 5-10 minutes.)
# 
# (**5 points - 3 for implementation, 1 for plot, 1 for description**).
# 
# (**This question will be manually graded.**)

# %% [markdown]
# <!-- BEGIN QUESTION -->
# 
# 

# %%
# Sweep over the k-values. Place your accuracies for each k-value in acc_k.
poss_k = [1, 3, 13, 25, 50, 100]
acc_k = []

# Your code goes below.
...

# %%
# Plot your accuracies for each k-value.
...

# %% [markdown]
# <!-- END QUESTION -->
# 
# 


