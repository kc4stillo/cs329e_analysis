# %% [markdown]
# ### Homework 3
# 
# ### Kyle Castillo, Andre Sae
# 
# ### Normal and t distributions
# ### t-test
# 

# %% [markdown]
# An experiment was conducted to determine the effect of children participating in a given meal preparation on calorie intake for that meal. Data are recorded below. 
# 
# Save the data to a format that can be read into python. Read the data in for analysis. Data is provided in two separted CSV files. 
# 
# * Use python to calculate the quantities and generate the visual summaries requested below. You will lose points if you are not utilizing python.
# 
# * You can use scipy libary or other libraries to do your tests or you can implement them from scratch in python 
# 
# 

# %%
# Standard Headers
# You are welcome to add additional headers here if you wish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Enable inline mode for matplotlib so that Jupyter displays graphs
%matplotlib inline

# %% [markdown]
# # Question - 1
# Summarize the data by whether children participated in the meal preparation or not. Use an appropriately labelled table to show the results. Also include a graphical presentation that shows the distribution of calories for participants vs. non-participants. Describe the shape of each distribution and comment on the similarity (or lack thereof) between the distributions in each population. **(2 points)**
# 
# Be aware that there is not one specific way the graph needs to look. Experiment with different types of graphs, and different parameters for the graph type. Your goal is to present the data as readable as possible. 

# %%
part = pd.read_csv("participants.csv")
non_part = pd.read_csv("nonparticipants.csv")

fig, ax = plt.subplots()

ax.hist(part, bins=8, alpha=0.5, label="Participants")     # orange
ax.hist(non_part, bins=8, alpha=0.5, label="Non-Participants")  # blue

ax.set_title("Participants vs Non-Partipants")
ax.legend()

plt.show()

# %%[markdown]

# The distribution of each dataset is similar and float around the mean, but the distrubition isn't convincingly normal.

# %% [markdown]
# # Question - 2 
# 
# Does the mean calorie consumption for those who participated in the meal preparation differ from **425**? Formally test at the $\alpha = 0.05$ level using the 5 steps outlined in the module. **(6 points)**
# 

# %%
t_stat, p_value = scipy.stats.ttest_ind(part.to_numpy(), 425)

print(f't test statistic = {t_stat}')
print(f'p value = {p_value}')

#%% [markdown]

# The mean of 425 does indeed differ from the mean calories of participants. This result was concluded after conducting a t-test using the participants dataset vs the suggested mean.

# %% [markdown]
# # Question -3 
# Calculate a **90%** confidence interval for the mean calorie intake for participants in the meal preparation. Interpret the confidence interval. **(6 points)**

mean, std = np.mean(part), np.std(part)

lower, upper = scipy.stats.norm.interval(.9, loc = mean, scale = std / np.sqrt(part.shape[0]))

print(f"lower_bound: {lower}, upper bound: {upper}")

#%% [markdown]

# Given 25 sample draw, we are 90% confident that the mean is within 370.912 and 449.246.

# %% [markdown]
# # Question 4 
# Formally test whether or not participants consumed 
# more calories than non-participants at the $\alpha = 0.05$ level using the 5 steps procedure for hypothesis tests. **(6 points )**

# %%
t_stat, p_val = scipy.stats.ttest_ind(
    part,
    non_part,
    alternative='less',
    equal_var=False
)

alpha = 0.05

print("p-val: " + str(p_val))

if p_val < alpha: 
    print("The null hypothesis is rejected because the p-val from the two sample t-test is smaller than tha alpha.")
elif p_val >= alpha:
    print("The null hypothesis was not rejected because the p-val from the two sample t-test was the same or larger than the two sample t-test alpha value.")
# %% [markdown]
# The null hypothesis was not rejected because the p-val from the two sample t-test was the same or larger than the two sample t-test alpha value.
