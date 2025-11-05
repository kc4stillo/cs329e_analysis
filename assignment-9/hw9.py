# %%
# Kyle Castillo
# Andre Sae

# %% [markdown]
# ## CS 329E 
# 
# # Ensemble Methods and Skewed Data
# 
# For this week's homework we are going explore two ensemble methods:
# 
#   - AdaBoost, and
#   - Random Forests
#   
# Along with applying different KPIs (key performance indicators) that are more appropriate to highly skewed data sets. 
# 
# The dataset contains transactions made by credit cards in September 2013 by european cardholders.
# This dataset presents transactions that occurred in two days, where we have 237 frauds out of 142,167 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.17% of all transactions.
# 
# It contains only numerical input variables which are the result of a [PCA transformation](https://en.wikipedia.org/wiki/Principal_component_analysis). Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount in Euros. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# At the end of this homework, I expect you to understand how to train and use ensemble classifiers, how to characterize model performance with ROC curves, and be familiar with the difference between accuracy, true positive rate, and positive predictive value. 

# %%
# These are the libraries you will use for this assignment, you may not import anything else
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve, auc, RocCurveDisplay, recall_score, precision_score, roc_curve

# This is the credit card data provided, we'll use sklearn methods to do cross validation
# to estimate error
df_cc = pd.read_csv("cc.csv")

# %% [markdown]
# ## Q1 Parition the data for cross validation
# 
# Load the data, and split the data set into $X$ (the feature dataframe, `df_X`) and $y$ (the target series `s_y`). Define our partitions.  
# 
# We know this is a _super_ skewed data set, so we worry about our target class being underrepresented in a random k-fold selection. With this in mind, we use a [stratifed k-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), since it will preserve our class balance in our experiements. Use $k=3$, . Instantiate an instance of the `StratifiedKFold` class, and use the generator `split` to populate the test and train dictonaries:
#    - `d_train_df_X` : key is the fold number, value is the attribute training dataframe at that fold
#    - `d_test_df_X`  : key is the fold number, value is the attribute test dataframe at that fold
#    - `d_train_s_y`  : key is the fold number, value is the target training series at that fold
#    - `d_train_s_y`  : key is the fold number, value is the target test series at that fold

# %%
df_X = df_cc.drop(columns=["Class"])
s_y = df_cc["Class"]

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)

# %%
d_train_df_X = dict()
d_test_df_X = dict()
d_train_s_y = dict()
d_test_s_y = dict()

for fold, (train_index, test_index) in enumerate(skf.split(df_X, s_y)):
    d_train_df_X[fold] = df_X.iloc[train_index]
    d_test_df_X[fold] = df_X.iloc[test_index]
    d_train_s_y[fold] = s_y[train_index]
    d_test_s_y[fold] = s_y[test_index]

# %%
# Look at the test data and verify that the target training is equally distributed as possible
for key in d_test_s_y.keys():
    print(d_test_s_y[key].value_counts())

# %% [markdown]
# ## Section 1 - AdaBoost

# %% [markdown]
# # Q2 Test the Performance of AdaBoost
# 
# When we talked about AdaBoost in class, we used a collection of "Decision Stumps". In this assignment, we will use the implementation of [AdaBoost in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).  As you browse the documentation, you will notice that the default base esimator in this implentation is a `DecisionTreeClassifier(max_depth=1)` (our friend the decision stump). 
# 
# After you fit an AdaBoost model, you can call the method `predict` to get a class prediction, or you can call `predict_proba` to get the probability of being in the class `0` or the class `1`. These probabilities are used when creating ROC curves. 
# 
# Loop over the $k$ folds using the dictionaries from the first problem, and for each fold calculate the accuracy, TPR, the PPV, and the FPR.  Plot the ROC curve for each fold. You may use the [plot roc curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html) from Scikit-learn.  There is a great example in the documentation [on plotting ROC curves in cross validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html) that is helpful.  
# 
# When creating your AdaBoost classifier, please use the following parameters: 
# `AdaBoostClassifier(n_estimators=25, random_state=23)`
# 
# Save the predictions from the 3rd fold into a variable called `y_hat_ab` for use in a future problem.
# 

# %%
k = 3
acc_ab = np.zeros(k)
tpr_ab = np.zeros(k)
ppv_ab = np.zeros(k)
fpr_ab = np.zeros(k)

# %%
for i in range(k):
    ada = AdaBoostClassifier(n_estimators=25, random_state=23)
    ada.fit(d_train_df_X[i], d_train_s_y[i])
    print(f"fitted fold: {i}")

    y_pred = ada.predict(d_test_df_X[i]) 
    y_prob = ada.predict_proba(d_test_df_X[i])[:,1]
    y_true = d_test_s_y[i]
    print(f"predicted fold: {i}")

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    acc_ab[i] = (tp + tn) / (tp +fn + fp + tn)
    tpr_ab[i] = tp / (tp + fn)
    ppv_ab[i] = tp / (tp + fp)
    fpr_ab[i] = fp / (fp + tn)
    print(f"calculated metrics fold: {i}")

    disp = RocCurveDisplay.from_predictions(y_true, y_prob)
    disp.ax_.plot([0, 1], [0, 1])
    disp.ax_.set_title('ROC Curve')
    plt.show()
    print(f"plotted roc_auc for fold: {i}")

y_hat_ab = y_pred

# %%
print(
    "The min, mean, and max TPR are: {:.2f}, {:.2f}, and {:.2f}".format(
        tpr_ab.min(), tpr_ab.mean(), tpr_ab.max()
    )
)
print(
    "The min, mean, and max PPV are: {:.2f}, {:.2f}, and {:.2f}".format(
        ppv_ab.min(), ppv_ab.mean(), ppv_ab.max()
    )
)
print(
    "The min, mean, and max ACC are: {:.2f}, {:.2f}, and {:.2f}".format(
        acc_ab.min(), acc_ab.mean(), acc_ab.max()
    )
)

# %% [markdown]
# ## Q3 Test the Performance of Random Forests
# 
# Now, let's try another ensemble method: Random Forests, again using the [Scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 
# 
# Following our book, we will build complete trees, with no pruning.  That means every leaf in the tree will be completelely pure, and if you exam an individual Decision Tree it would be overtrained to our training set.  While building the decision trees, at every internal node, we select $p$ attributes at random, and then find the best split that minimizes impurtity.  The value, $p$, is a hyperparamter of the Random Forest and corresponds to the `max_features` parameter in the Random Forest Class. 
# 
# After you fit an RandomForest model, you can call the method `predict` to get a class prediction, or you can call `predict_proba` to get the probability of being in the class `0` or the class `1`. These probabilities are used when creating ROC curves. 
# 
# Loop over the $k$ folds using the dictionaries from the first problem, and for each fold calculate the accuracy, TPR, the PPV, and the FPR.  Plot the ROC curve for each fold. You may use the [plot roc curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html) from Scikit-learn. There is a great example in the documentation [on plotting ROC curves in cross validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html) that is helpful.  
# 
# When creating your Random Forest classifier, please use the following parameters: 
# `RandomForestClassifier(criterion="entropy", max_features="sqrt", random_state=23)`
# 
# Save the predictions from the 3rd fold into a variable called `y_hat_rf` for use in a future problem.

# %%
k = 3
acc_rf = np.zeros(k)
tpr_rf = np.zeros(k)
ppv_rf = np.zeros(k)
fpr_rf = np.zeros(k)

# %%
# your code here

# %%
print(
    "The min, mean, and max TPR are: {:.2f}, {:.2f}, and {:.2f}".format(
        tpr_rf.min(), tpr_rf.mean(), tpr_rf.max()
    )
)
print(
    "The min, mean, and max PPV are: {:.2f}, {:.2f}, and {:.2f}".format(
        ppv_rf.min(), ppv_rf.mean(), ppv_rf.max()
    )
)
print(
    "The min, mean, and max ACC are: {:.2f}, {:.2f}, and {:.2f}".format(
        acc_rf.min(), acc_rf.mean(), acc_rf.max()
    )
)

# %% [markdown]
# # Q4 Calculate the Cost of Fraud 
# 
# In the above problems, we saved the predictions of the 3rd fold into the variables `y_hat_ab` and `y_hat_rf` for the AdaBoost and RandomForest models respectively. 
# 
# Now, Mr. Bank Man wants you to tell him how much money he is going to save if he deploys either of these fraud algorithms to the real-time payment processing system.  Assume that there is not a currently deployed fraud detection algorithm.  
# 
# For every fraudulent transaction that is not predicted as fraudulent the bank looses twice that much money.  So, a fradulent charge for €10 is undectected, it costs the bank €20.  Also, if a charge is predicted as fradulent, but wasn't, it costs the bank a flat fee of €3 in customer service support to communicate with the customer, and mark the possible fraud as a normal transaction. 
# 
# Using the 3rd fold test sample, calculate how much money Mr Bank Man will save with each algorithm, and make a recommendation of which algorithm to deploy to production.

# %%


# %% [markdown]
# Mr Bank man will save more money, if we deploy the <> algorithm! 


