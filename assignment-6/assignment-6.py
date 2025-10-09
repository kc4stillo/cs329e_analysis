# %% [markdown]
# # Homework 6  - Logistic Regression without using any libraries. 
#  
# Student Name: Kyle Castillo
# 
# Student UT EID: kmc5794
# 
# ---
# 
# Partner Name: Andre Sae
# 
# Partner UT EID:
# 
# ---
# 
# Date Created: 10/6/2025
# 
# Date Last Modified:
# 
# ---
# 
# Totoal Points 20. 
# 
# 

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
# # Task 1 - (4 points)
# We want to use logistic regerssion to predict if a patient will have heart problems or not. The column "Target" in our datasets includes data about heart disease. If the patient had heart disease, the patient's "Target" value equals 1. Otherwise, "Target" equals 0.
# 
# Prepare your data set for predicting heart disease ("Target" column) by using 3 features:
# 
# * Age of the patient (Column **"Age"**)
# * Gender of the patient (male or female - Column **"Sex"**)
# * Cholestrol level of the patient (Column **"Chol"**) 
# 
# Split your data into 80% traning data and 20% test data ***without*** using any libraries other than the ones imported above. You must do it manually.
# 
# * Do a maximum of **100 iterations**
# * Use a very small learning rate for checking your GD implementation. 
# * Your are allowed to use your choice of learning rate, like using 0.0001, 0.001 or 0.01 or 0.1 or higher/lower. 
# * **Visualize your error/costs over the iterations with a plot**.
# * No need to add an y-intercept in this task. 
# 
# (**4 points** - 3 points for code, 1 point for cost visualization)

# %%
# selecting features
features = heart_df[["Age", "Sex", "Chol"]]
labels = (heart_df["Target"] == "Yes").astype(int)

# train/test split
split = .8

train = np.arange(0, round(features.shape[0] * split) + 1)
test = np.arange(train[-1] + 1, features.shape[0])

train_x = features.loc[train]
train_y = labels.loc[train]

test_x = features.loc[test]
test_y = labels.loc[test]

# %%
def sigmoid(w , x):
    z = x.dot(w)
    return (1   /    (1 + (np.exp(-z))) ).reshape(-1,1)

def cost(w, x, y):
    a = sigmoid(w,x)
    # a = np.clip(sigmoid(w, x), 1e-12, 1 - 1e-12)
    cost_val = (-1/len(y)) * np.sum((y *   np.log(a)) + ((1 - y) * (np.log(1-a))))
    return cost_val

def grad_descent(w,x,y,alpha=.1):
    # learning_rule
    a = sigmoid(w,x)
    gradient = (x.T.dot(a - y)) / len(y)

    # descending to new w
    new_w =  w.reshape(-1, 1) - alpha * gradient
    return new_w

# %%
w = np.zeros(train_x.shape[1]).reshape(3,)
x = train_x.to_numpy().reshape(-1,3)
y = train_y.to_numpy().reshape(-1,1)

cost_hist = np.array([])

iterations = 100
alpha = .0001

for i in range(iterations):
    cost_val = cost(w,x,y)
    cost_hist = np.append(cost_hist,cost_val)
    w = grad_descent(w,x,y,alpha)

    print(f"new cost: {cost_val}")

# %%
x = np.arange(1,101)
y = cost_hist

plt.plot(x,y)

plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("iterations vs cost")

plt.show()

# %% [markdown]
# # Task 2 - (4 points)
# 
# Cacluate the Accuracy, Precision, Recall and F1 score of your logistic regression implementaion on the testing set. 
# Print the results. 
# 
# You may use equations shown in lecture/slides/examples.
# 
# Calcuate the accuracy, precision, recall and F1 score of your logistic regression implementaion on the testing set. 
# Print the results. (**4 points**)
# 

# %%
x = test_x.to_numpy()
y_true = test_y.to_numpy()

y_prob = sigmoid(w, x)
y_pred = [1 if i >= .5 else 0  for i in y_prob]

results = pd.DataFrame({"y_true": y_true.flatten(),
    "y_pred": np.array(y_pred)})

results["tp"] = ((results["y_pred"] == 1) & (results["y_true"] == 1)).astype(int)
results["fp"] = ((results["y_pred"] == 1) & (results["y_true"] == 0)).astype(int)

results["tn"] = ((results["y_pred"] == 0) & (results["y_true"] == 0)).astype(int)
results["fn"] = ((results["y_pred"] == 0) & (results["y_true"] == 1)).astype(int)

accuracy = np.mean(results["y_pred"] == results["y_true"])
precision = results["tp"].sum() / (results["tp"].sum() + results["fp"].sum())
recall = results["tp"].sum() / (results["tp"].sum() + results["fn"].sum())
f1 = 2 * ((precision * recall) / (precision + recall))

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# %% [markdown]
# # Task 3 - (4 points)
# 
# Add a y-intercept and repeat the above tasks. Do you see any differences after adding the y-intercept?
# 
# (4 points - 2 for code, 1 for cost visualizaiton, 1 for description.)

# %%
def sigmoid_w_bias(b, w , x):
    z = x.dot(w) + b
    return (1   /    (1 + (np.exp((-1 * z)))) ).reshape(-1,1)

def cost_w_bias(b, w, x, y):
    # a = sigmoid_w_bias(b, w,x)
    a = np.clip(sigmoid_w_bias(b, w, x), 1e-12, 1 - 1e-12)
    cost_val = (-1/len(y)) * np.sum((y *   np.log(a)) + ((1 - y) * (np.log(1-a))))
    return cost_val

def grad_descent_w_bias(b, w,x,y,alpha=.1):
    # learning_rule
    a = sigmoid_w_bias(b, w,x)
    error = (a - y)
    w_gradient = (x.T.dot(error)) / len(y)
    b_gradient = np.mean(error)

    # descending to new w
    new_w = w - alpha * w_gradient
    new_b = b - alpha * b_gradient
    return new_b, new_w

# %%
x_unscaled = train_x.to_numpy()
mean = np.mean(x_unscaled, axis=0)
std = np.std(x_unscaled, axis=0)

x = (x_unscaled - mean) / std
b = 0
w = np.zeros((train_x.shape[1], 1))
y = train_y.to_numpy().reshape(-1,1)

cost_hist = np.array([])

iterations = 100
alpha = .1

for i in range(iterations):
    cost_val = cost_w_bias(b,w,x,y)
    cost_hist = np.append(cost_hist,cost_val)
    b, w = grad_descent_w_bias(b,w,x,y,alpha)

    print(f"new cost: {cost_val}")

# %%
x = np.arange(1,iterations+1)
y = cost_hist

plt.plot(x,y)

plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("iterations vs cost")

plt.show()

# %%
x_test_unscaled = test_x.to_numpy()
y_true = test_y.to_numpy().reshape(-1,1)

x_test_scaled = (x_test_unscaled - mean) / std

y_prob = sigmoid_w_bias(b, w, x_test_scaled)
y_pred = (y_prob >= 0.5).astype(int)

results = pd.DataFrame({
    "y_true": y_true.flatten(),
    "y_pred": y_pred.flatten()
})

results["tp"] = ((results["y_pred"] == 1) & (results["y_true"] == 1)).astype(int)
results["fp"] = ((results["y_pred"] == 1) & (results["y_true"] == 0)).astype(int)

results["tn"] = ((results["y_pred"] == 0) & (results["y_true"] == 0)).astype(int)
results["fn"] = ((results["y_pred"] == 0) & (results["y_true"] == 1)).astype(int)

accuracy = np.mean(results["y_pred"] == results["y_true"])
precision = results["tp"].sum() / (results["tp"].sum() + results["fp"].sum())
recall = results["tp"].sum() / (results["tp"].sum() + results["fn"].sum())
f1 = 2 * ((precision * recall) / (precision + recall))

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

#%% [markdown]
# Scaling the features set each column to mean 0 and standard deviation 1, and adding a y-intercept let the model adjust the decision boundary. This kept the sigmoid inputs in a good range and helped the model learn properly. As a result, predictions were more balanced and accuracy and recall improved compared to the unscaled and no y-intercept version.

# %% [markdown]
# # Task 4 - Implement the Bold Driver   - (4 points)
# 
# Implement the bold driver into your gradient descent implementation, which lets us have a dynamic learning rate. Visualize the costs and print the accuracy/etc. metrics as before. Do not use a y-intercept this time.
# 
# Add a stop condition that stop the GD when the cost is not changing more than 0.001 between iterations. 
# Describe the results. Did you stop earlier than 100 iterations?
# (4 points - 2 for code, 1 for cost visualizaiton, 1 for description.)
# 
# 

# %%
w = np.zeros(train_x.shape[1]).reshape(3,)
x = train_x.to_numpy().reshape(-1,3)
y = train_y.to_numpy().reshape(-1,1)

cost_hist = np.array([])
prev_weight = None

iterations = 100
alpha = .0001
final_iterations = iterations

for i in range(iterations):
    cost_val = cost(w,x,y)
    w = grad_descent(w,x,y,alpha)

    if i != 0: 
        prev_cost_val = cost_hist[len(cost_hist) - 1]
        if prev_cost_val > cost_val: 
            alpha -= alpha * 0.025
        elif prev_cost_val + (10 ** -10) < cost_val:
            w = prev_weight
            alpha = alpha * 0.5
            
        if abs(cost_val - cost_hist[len(cost_hist) - 2]) < 0.001:
            final_iterations = i
            cost_hist = np.append(cost_hist,cost_val)
            print("Stopped at iteration # " + str(final_iterations))
            break
        

    cost_hist = np.append(cost_hist,cost_val)
    prev_weight = w 

    print(f"new cost: {cost_val}")
    
y_pred = np.matmul(w.T, x.T)
y_pred = np.transpose(y_pred)
y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1

y_pred

# %%
x = np.arange(1,len(cost_hist)+1)
y = cost_hist

plt.plot(x,y)

plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("iterations vs cost")

# %% [markdown]
# # Task 5 - Implement the L2 norm regularization.  - (4 points)
# 
# 
# 
# Modify your Cost and gradient to implement the l2 norm regularization. Repeat the steps taken in prior tasks and describe your result.
# 
#  * Use a y-intercept.
#  * Do a maximum of 100 iterations as before and report your accuracy, precision, recall and F1 score.
#  * Optional: You can stop earlier, if the cost is not changing more than 0.001 between iterations.
#  * Optional: You can use the bold driver, if you want. But a bold driver is not required to perform L2 norm regularizaiton.
# 
# (4 points - 2 for code, 1 for cost visualizaiton, 1 for description.)

# %%
def sigmoid_w_bias(b, w , x):
    z = x.dot(w) + b
    return (1   /    (1 + (np.exp((-1 * z)))) ).reshape(-1,1)

def cost_w_bias_l2(b, w, x, y, lam):
    # a = sigmoid_w_bias(b, w,x)
    a = np.clip(sigmoid_w_bias(b, w, x), 1e-12, 1 - 1e-12)
    loss = (-1/len(y)) * np.sum((y *   np.log(a)) + ((1 - y) * (np.log(1-a))))
    l2_penalty = (lam / (2 * len(y))) * np.sum(np.square(w))
    cost_val = loss + l2_penalty
    return cost_val

def grad_descent_w_bias_l2(b, w,x,y,lam,alpha=.1, ):
    # learning_rule
    a = sigmoid_w_bias(b, w, x)
    error = a - y

    w_gradient = (x.T.dot(error)) / len(y) + (lam / len(y)) * w
    b_gradient = np.mean(error)

    new_w = w - alpha * w_gradient
    new_b = b - alpha * b_gradient
    return new_b, new_w

# %%
x_unscaled = train_x.to_numpy()
mean = np.mean(x_unscaled, axis=0)
std = np.std(x_unscaled, axis=0)

x = (x_unscaled - mean) / std
b = 0
w = np.zeros((train_x.shape[1], 1))
y = train_y.to_numpy().reshape(-1,1)

cost_hist = np.array([])

iterations = 100
lam= 10
alpha = .1

for i in range(iterations):
    cost_val = cost_w_bias_l2(b,w,x,y,lam)
    cost_hist = np.append(cost_hist,cost_val)
    b, w = grad_descent_w_bias_l2(b,w,x,y,lam,alpha)

    print(f"new cost: {cost_val}")

# %%
x = np.arange(1,iterations+1)
y = cost_hist

plt.plot(x,y)

plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("iterations vs cost")

plt.show()

# %%
x_test_unscaled = test_x.to_numpy()
y_true = test_y.to_numpy().reshape(-1,1)

x_test_scaled = (x_test_unscaled - mean) / std

y_prob = sigmoid_w_bias(b, w, x_test_scaled)
y_pred = (y_prob >= 0.5).astype(int)

results = pd.DataFrame({"y_true": y_true.flatten(),
    "y_pred": np.array(y_pred).flatten()})

results["tp"] = ((results["y_pred"] == 1) & (results["y_true"] == 1)).astype(int)
results["fp"] = ((results["y_pred"] == 1) & (results["y_true"] == 0)).astype(int)

results["tn"] = ((results["y_pred"] == 0) & (results["y_true"] == 0)).astype(int)
results["fn"] = ((results["y_pred"] == 0) & (results["y_true"] == 1)).astype(int)

accuracy = np.mean(results["y_pred"] == results["y_true"])
precision = results["tp"].sum() / (results["tp"].sum() + results["fp"].sum())
recall = results["tp"].sum() / (results["tp"].sum() + results["fn"].sum())
f1 = 2 * ((precision * recall) / (precision + recall))

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# w's no lambda
# array([[0.49151999],
#        [0.61648032],
#        [0.21707886]])


# w's with lambda
# array([[0.49891395],
#       [0.62617623],
#       [0.22044241]])

# %%
# When lambda was added, the weights got a little smaller because regularization pushes them toward zero. This makes the model a bit less flexible, so the training error went up slightly. The tradeoff is that it can help the model generalize better on new data.