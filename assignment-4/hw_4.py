# %% [markdown]
# # Homework 4
# 
# ## Kyle Castillo and Andre Sae
# 
# ## Linear Regression with Gradient Descent

# %% [markdown]
# Your task in this assignment is to implement Multiple Linear Regression. 
# 
# We will use the New York City Taxi trip reports in the Year 2013. 
# The dataset was released under the FOIL (The Freedom of Information Law) and made public by Chris Whong (\url{https://chriswhong.com/open-data/foil_nyc_taxi/}).
# 
# 
# 
# # Taxi Data Set
# The data set itself is a simple text file. Each taxi trip report is a different line in the file. Among other things, each trip report 
# includes the starting point, the drop-off point, corresponding timestamps, and information related to the payment. The data are reported 
# by the time that the trip ended, i.e., upon arriving in the order of the drop-off timestamps. 
# The attributes present on each line of the file are, in order:
# 
# 
# 
# | index | **Attribute** | **Description** |
# | --- | --- | --- |
# | 0 | medallion           |  an md5sum of the identifier of the taxi - vehicle bound (Taxi ID)  | 
# | 1 | hack license       |  an md5sum of the identifier for the taxi license (Driver ID)  |  
# | 2 | pickup datetime    | time when the passenger(s) were picked up  |  
# | 3 | dropoff datetime   | time when the passenger(s) were dropped off  | 
# | 4 | trip time in secs | duration of the trip  |  
# | 5 | trip distance | trip distance in miles  |  
# | 6 | pickup longitude | longitude coordinate of the pickup location  |  
# | 7 | pickup latitude | latitude coordinate of the pickup location  |  
# | 8 | dropoff longitude | longitude coordinate of the drop-off location   |  
# | 9 | dropoff latitude | latitude coordinate of the drop-off location  | 
# | 10 | payment type | the payment method -credit card or cash  |  
# | 11 | fare amount | fare amount in dollars  |  
# | 12 | surcharge | surcharge in dollars  |  
# | 13 | mta tax | tax in dollars  |  
# | 14 | tip amount | tip in dollars  |  
# | 15 | tolls amount | bridge and tunnel tolls in dollars  |  
# | 16 | total amount | total paid amount in dollars  |  
# 
# 
# 
# 
# The data files are in comma separated values (CSV) format. Example lines from the file are:
# 
# 07290D3599E7A0D62097A346EFCC1FB5,E7750A37CAB07D0DFF0AF7E3573AC141,\\
# 2013-01-01,00:00:00,2013-01-01 00:02:00,120,0.44,-73.956528,40.716976,-73.962440,\\
# 40.715008,CSH,3.50,0.50,0.50,0.00,0.00,4.50
# 
# 22D70BF00EEB0ADC83BA8177BB861991,3FF2709163DE7036FCAA4E5A3324E4BF,\\
# 2013-01-01,00:02:00,2013-01-01 00:02:00,0,0.00,0.000000,0.000000,0.000000,0.000000,\\
# CSH,27.00,0.00,0.50,0.00,0.00,27.50
# 
# 0EC22AAF491A8BD91F279350C2B010FD,778C92B26AE78A9EBDF96B49C67E4007,\\
# 2013-01-01,00:01:00,2013-01-01 00:03:00,120,0.71,-73.973145,40.752827,-73.965897\\
# 73.965897,40.760445,CSH,4.00,0.50,0.50,0.00,0.00,5.00
# 
# 

# %%
# Standard Headers
# You are welcome to add additional headers here if you wish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

# Enable inline mode for matplotlib so that Jupyter displays graphs
%matplotlib inline

# %%
df = pd.read_csv('nyc-taxi-data.csv', header = None)
df

# %% [markdown]
# ## Task - 1
# The dataset is a real-world dataset and many records are incorrect and wrong. Your task is first to describe at least 4 data clean up tasks and implement them on this data set.
# 
# For example, you can define to remove lines with the following property, if a taxi trip (one of the data rows) has a travel distance less than 1 mile and total amount of more than 20 dollar, then it is an wrong record. 
# 
# Another example is that you would expect to have a float number (e.g., index 12 to 16) and then you got a String in that position. You can remove such lines from your data. 
# 
# Perform the following cleanup tasks:
# 1. Add column headers based on the description above (no need to describe).
# 2. Drop rows with a travel distance < 1 mile and a total fare > $20 (no need to describe). Both conditions need to be met.
# 3. Your own task (please describe your task in a comment/cell).
# 4. Your own task (please describe your task in a comment/cell).
# 
# **(4 points)**

# %%
df.columns = ["medallion", "hack license", "pickup datetime", "dropoff datetime", "trip time in secs", "trip distance", "pickup longitude", "pickup latitude", "dropoff longitude", "dropoff latitude", "payment type", "fare amount", "surcharge", "mta tax", "tip amount", "tolls amount", "total amount"]

# %%
df.shape

# indexes to drop
indexes_to_drop = df[(df["trip distance"] <= 1) & (df["fare amount"] > 20)].index
df = df.drop(index = indexes_to_drop)

df.reset_index(drop=True, inplace=True)
# %%
# here i converted several features into the appropriate datatypes, allowing us to do a more in depth analysis.

# before
print(df.dtypes)

df["payment type"] = df["payment type"].astype("category")
df['pickup datetime'] = pd.to_datetime(df["pickup datetime"])
df['dropoff datetime'] = pd.to_datetime(df["dropoff datetime"])

# after
print(df.dtypes)

# %%
# removed all rows where pickup/dropoff coordinates are 0
indexes_to_drop = df[(df["pickup longitude"] == 0) | (df["pickup latitude"] == 0) | (df["dropoff longitude"] == 0) | (df["dropoff latitude"] == 0)].index
df = df.drop(index=indexes_to_drop)

# %% [markdown]
# ## Task - 2
# Use the cleaned dataset from task 1.
# Find out the correlations between trip distance, travel time and fare amount. Visualize the correlations in form of Scatterplot matrix. 
# Describe the correlations between them. **(4 points)**

# %%
print(df[["trip distance", "trip time in secs", "fare amount"]].corr())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df["trip distance"], df["trip time in secs"], df["fare amount"])
ax.set_xlabel('Trip Distance (miles)')
ax.set_ylabel('Trip Time (sec)')
ax.set_zlabel('Fare Amount (USD)')

#%% [markdown]
# As seen by the correlation output and 3d plot, these 3 features are closely correlated. 
# - Trip distance has the strongest correlation with fare amount (r=.9389), but also has a moderately strong correlation with trip time in seconds (r = .7743) 
# - Trip time is correlated strongly with fare amount (r=.8481)

# %% [markdown]
# ## Task - 3
# Use the cleaned dataset from task 1.
# We want to find a simple line to our data (distance, money). 
# We want to use trip distance, and fare amount. Use the **Scikit-learn** library to fit a line into the data. **(2 points)**
# 
# * Provide the Regression Cofficients of your model
# * Write down your linear regression equation. 
# * Use your model to predict the fare amount for a 3 miles trip. 
# 
# A visualization of the model is not required, but it would be nice to have and good practice.

# %%

lr = linear_model.LinearRegression()
lr_fit = lr.fit(df["trip distance"].to_numpy().reshape(-1,1), df["fare amount"].to_numpy().reshape(-1,1))

predicted = lr_fit.predict([[3]])

print(f"coefficient: {lr.coef_}")
print(f"y-intercept: {lr.intercept_}")

sns.regplot(data=df, x="trip distance", y ="fare amount")

# %% [markdown]
# Equation: 
# Fare Amount = 3.75478895 + Distance * 2.85937406

# %% [markdown]
# ## Task - 4
# Fit a Multiple Linear Regression hyperplane into this data. We are interested to predict total amount out of trip distance, tavel time. **(4 points)**
# Use the **Scikit-learn** library.
# 
# * Provide the Regression Cofficients of your model
# * Write down your Multiple Linear Regression equation. 
# * Use your model to predict the total amount for a 3 miles trip and 6 min travel time. 
# 
# A visualization of the model is not required, but it would be nice to have and good practice.

# %%
# Your code here

# %% [markdown]
# ## Task - 5 - Gradient Descent
# Implement the gradient descent optimization to find the optimal parameters for our Simple Linear Regression model of task 3. **(6 points)**
# 
# * Define and set your learning rate (start with a very small number and increase it if your GC works)
# * Instantiate all coefficients from zero
# * Run maximum 400 interations. You can stop if your cost converge with a precision of 0.01 
# * Print and visualize the optimization costs
# * Provide the Regression Coefficients of your model after stop or convergance. 
# 
# Please note that a visualization **is** required, unlike in Task 3 and 4.

# %%
# Your code here


