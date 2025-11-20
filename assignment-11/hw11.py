# %% [markdown]
# ## C S 329E HW 11
# 
# # Add YOUR NAMES HERE!

# %% [markdown]
# ## C S 329E HW 11
# 
# # Hierarchical Clustering and Cluster Evaluation
# 
# ## Your name here (and your partner's name if you are working in a pair)
# 
# As we are well into November, let us consider presidential elections of years past and do some exploratory grouping of states based on the 2008,2012, and 2016 presidential elections results.  The data we have is from the [The American Presidency Project](https://www.presidency.ucsb.edu/statistics/elections) and split into two files:
# 
#   - `votes_by_state.csv` => has the raw number of ballots cast for each of the top presidential candidates from 2008-2012 by US State (and the District of Columbia)
#   - `republican_percentage_by_state.csv` => has the percentage of votes that were for the Republican candidate for 2008, 2012, and 2016 by US State (and the District of Columbia)
#   
# Our job is to group together states using different hierarchical agglomerative methodologies and compare the results.  You can imagine how finding states that tend to vote similarly to each other might be useful for election predictions, or for identifying swing states as part of a political campaign to invest in.

# %%
# Do not modify import block, all necessary imports are included

# Our regular libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline

# This is for our hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# For our k-means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# %% [markdown]
# ## Q1 - Using Different Proximity Functions
# 
# There is another scientific python library we haven't used much, SciPy, which has [library for hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage) and for [plotting dendrograms](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram). 
# 
# Use these libraries to plot the hierarchical clusters formed from the raw vote count (`votes_by_state.csv`) using three different proximity measures:
#   - Ward
#   - Min
#   - Max
#   
# The x-axis of the dendrogram should be labeled using the state name (and not the index of the state in the data frame).  The title should reflect what proximity measure you used to create the clusters.  In this use case only the number of votes cast are attributes, and the State names are the labels on that row of data.
# 
# Note, to make the plot legible you will have to change the plot size.  I found `figsize=(12,8)` worked well for me, but your mileage may vary depending on your setup. 

# %%
# Load the data, save off the state names into another series and just leave the attributes.
df_X = pd.read_csv('votes_by_state.csv')
s_states = df_X['State']
df_X = df_X.drop(columns=['State'])

# %%
# Plot the dendrogram for the Ward proximity measure
plt.figure(figsize=(12, 8))
ward = linkage(df_X, method='ward')

dendrogram(ward, labels=s_states.values, leaf_rotation=90)
plt.title("ward linkage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()


# %%
# Plot the dendrogram for the min proximity measure
plt.figure(figsize=(12, 8))
min = linkage(df_X, method='single')

dendrogram(min, labels=s_states.values, leaf_rotation=90)
plt.title("min linkage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()

# %%
# Plot the dendrogram for the max proximity measure
plt.figure(figsize=(12, 8))
max = linkage(df_X, method='complete')

dendrogram(max, labels=s_states.values, leaf_rotation=90)
plt.title("max linkage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()

# %% [markdown]
# ## Q2 - Using Different Proximity Functions (part 2)
# 
# What is happening?  Even if you aren't a big politics wonk, you might find it strange that New York is closer to Texas than it is to California in all of those graphs!! You may think to yourself, "Self, how could this be?" You also might notice that Montana, Wyoming, and Alaska all tend to cluster together in all of these scenarios. 
# 
# At this point, your data science brain might be noticing that the _population_ of that state seems to be more important than which party _won_ the state.  You decide to engineer the data such that you reduce the dimensions down from 7 attributes, to 3 attributes, where each column is the _percentage_ of votes that were cast to the Republican candidate, and a 4th attribute indicating the range of percentages across the 3 elections.  I went ahead and did this for you, and created the file `republican_percentage_by_state.csv`. 
# 
# Plot the hierarchical clusters formed from percentage of ballots cast by Republicans (`republican_percentage_by_state.csv`) using three different proximity measures:
#   - Ward
#   - Min
#   - Max
#   
# The x-axis of the dendrogram should be labeled using the state name (and not the index of the state in the data frame).  The title should reflect what proximity measure you used to create the clusters.  In this use case only the number of votes cast are attributes, and the State names are the labels on that row of data.
# 
# Note, to make the plot legible you will have to change the plot size.  I found `figsize=(12,8)` worked well for me, but your mileage may vary depending on your setup. 

# %%
# Load the data, save off the state names into another series and just leave the attributes.
df_X = pd.read_csv('republican_percentage_by_state.csv')
s_states = df_X['State']
df_X = df_X.drop(columns=['State'])

# %%
# Plot the dendrogram for the Ward proximity measure
plt.figure(figsize=(12, 8))
ward = linkage(df_X, method='ward')

dendrogram(ward, labels=s_states.values, leaf_rotation=90)
plt.title("ward linkage, republican percentage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()

# %%
# Plot the dendrogram for the min proximity measure
plt.figure(figsize=(12, 8))
min = linkage(df_X, method='single')

dendrogram(min, labels=s_states.values, leaf_rotation=90)
plt.title("min linkagem Republican percentage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()

# %%
# Plot the dendrogram for the max proximity measure
# Plot the dendrogram for the max proximity measure (Complete Linkage)

plt.figure(figsize=(12, 8))
max = linkage(df_X, method='complete')

dendrogram(max, labels=s_states.values, leaf_rotation=90)
plt.title("complete linkage, republican percentage")
plt.xlabel("state")
plt.ylabel("distance")
plt.show()

# %% [markdown]
# ## Q3 Calculate the Cophenetic Correlation Coefficient (CPCC)
# 
# Using the data from question 2 (creating hierarchical clusters on the percentage of Republican votes), calculate the Cophenetic Correlation Coefficient for each of the three methods.  Display the results, and add a markdown cell explaining how to interpret this metric. You may use the scipy [cophenet function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html) with the output from  [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) function as the `Y` parameter. 

# %%


# %%
# Show the results
(cpcc_ward,cpcc_min,cpcc_max)

# %% [markdown]
# _explanation_

# %% [markdown]
# ## Q4 Find new Clusters Based on K-Means
# 
# A political wonk has come to visit you in your data science dungeon, and looks over your shoulder.  They like what they see, and especially like the patterns from the Ward proximity diagram, and their team is going to come up with 4 different strategies based on the clustering that you have done.  But, your hierarchical clustering didn't come up with a natural 4 clusters! No problem, you decide to use k-means to come up with the 4 clusters.  You choose these points to initialize your cluster centers based on the Ward proximity graph in your hierarchical clusters:
# 
#   - Montana
#   - Arkansas
#   - Massachusetts
#   - Minnesota
# 
# Pass `random_state = 23` into the [KMeans function](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
# 
# Print out your centroids before you pass them to K-Means. Print out the States that are in each k-means cluster.

# %%
# Find the centroids
init = ["Montana", "Arkansas", "Massachusetts", "Minnesota"]
centroids = df_X[s_states.isin(init)].values

# %%
# Show the centroids
centroids

# %%
# Compute the k-means clusters and show the listing of the States in each cluster
kmeans = KMeans(
    n_clusters=4,
    init=centroids,
    n_init=1,
    random_state=23
)

kmeans.fit(df_X)
labels = kmeans.labels_

clusters = {}

for i, state in enumerate(s_states):
    curr_cluster = list(labels)[i]
    key = str(curr_cluster)

    if key not in clusters:
        clusters[key] = []

    clusters[key].append(state)

clusters

# %% [markdown]
# ## Q5 Visualize the Silhouette Coefficients for Each Cluster
# 
# Using the [Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html) implementation in sci-kit learn, calculate the silhouette coefficient for each of the States in our data frame using the clustering from Q4, and create a bar graph, similar to the [left plot in this link](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
# 
# Use the Silhouette Coefficient plot to identify clusters that have outliers.  Describe in a markdown cells how the visualization helped you identify the outlier points.  

# %%


# %% [markdown]
# _description of the graph, and how you used it to detect outliers, and what the outliers you found are (tell me the states)_


