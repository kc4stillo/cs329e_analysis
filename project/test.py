# %% [markdown]
# #### Semantic Analysis and Visualization of Tweets from Public Topics
# - Andre Sae & Kyle Castillo
# - November 6, 2025
# - CS 329E - Elements of Data Visualization
#
# %% [markdown]
# **Research Question and Learning Model**
#
# **Research Question:**
# > How do tweets regarding different public topics (film, sports, and public policy) differ in language and content, and can we visualize meaningful clusters using tweet semantics?
#
# **Learning Model / Approach:**
# - Type: Unsupervised learning / clustering + visualization.
# - Method:
#   - Use SentenceTransformer semantic embeddings of tweet text.
#   - Apply t-SNE to project embeddings into 2D for visualization.
#   - Use KMeans to cluster tweets in embedding space (3 clusters, one per topic).
# - Goal: Identify whether clusters correspond to topics based on tweet semantics.
#
# We treat the topics as **hidden** during clustering (unsupervised), then use them
# afterward for evaluation and interpretation.

# %% [markdown]
# **Current Expectations About the Results**
#
# - Tweets about the same topic (Eddington film, Arch Manning, gun violence) should cluster
#   together in the embedding space due to similar vocabulary and context.
# - We expect roughly **three main clusters**, one per topic, with some overlap
#   where language is generic or ambiguous.
# - Clusters may not be perfectly separated, but we anticipate **moderate separation**
#   visible in the 2D t-SNE visualization.

# %% [markdown]
# **How to Evaluate the Project**
#
# - **Quantitative evaluation:**
#   - Fit KMeans on an unlabeled training subset (unsupervised).
#   - For each cluster, assign it to the topic that appears most frequently
#     in that cluster (majority vote) using the training labels.
#   - Use this cluster→topic mapping to predict topics on a held-out test set.
#   - Compute:
#     - Accuracy
#     - Confusion matrix
#     - Per-class precision/recall (classification report)
#
# - **Visual inspection:**
#   - t-SNE plot colored by true topic.
#   - t-SNE plot colored by KMeans cluster (train set).
#   - t-SNE plot of the test set highlighting **correct vs incorrect** predictions.
#
# This combines unsupervised clustering with supervised labels used only for
# post-hoc evaluation and interpretation.

# %% [markdown]
# **Expected Model Performance**
#
# - SentenceTransformer embeddings should place semantically similar tweets close together.
# - KMeans should find clusters that **roughly align** with the three topics.
# - We expect:
#   - Above-chance accuracy in mapping clusters to topics.
#   - Some confusion between topics that share similar language (e.g., news-style tweets,
#     generic political language).
# - t-SNE is used **only as a visualization tool**, not as a predictive model.

# %%
### 1. Import Libraries
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (Optional) for nicer confusion matrix display
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# %% [markdown]
# **What Is Your Dataset About?**
#
# The dataset contains tweets about three different public topics:
#
# - **Eddington (film)**
# - **Arch Manning (athlete)**
# - **Gun Violence (public policy)**
#
# Each CSV includes:
# - `tweet` – the tweet text
# - `date` – posting date/time
# - `replies`, `reposts`, `likes`, `bookmarks`, `views` – engagement metrics
# - `query` – the topic label (one of the three above)
#
# For this project we:
# - Sample a **balanced subset** of tweets per topic.
# - Use one portion for **unsupervised clustering** (training).
# - Hold out the rest for **evaluation** (test set).

# %%
### 2. Load CSVs

df_ed = pd.read_csv("ed_cleaned_v2.csv")
df_arch = pd.read_csv("arch_cleaned_v2.csv")
df_gun = pd.read_csv("gun_cleaned_v2.csv")

print("Eddington head:\n", df_ed.head(), "\n")
print("Arch Manning head:\n", df_arch.head(), "\n")
print("Gun violence head:\n", df_gun.head(), "\n")

# %% [markdown]
# **Data Sampling and Split**
#
# - For each topic, we:
#   - Randomly sample **2,000 tweets** (assuming enough data is available).
#   - Use **1,500** for training (unlabeled for the clustering algorithm).
#   - Use **500** for testing (held-out labeled data for evaluation).
# - This keeps the dataset **balanced across topics** and separates
#   model fitting (unsupervised) from evaluation.

# %%
### 3. Balanced Sampling and Train/Test Split

RANDOM_STATE = 42
N_PER_TOPIC_TOTAL = 2000
N_TRAIN_PER_TOPIC = 1500
N_TEST_PER_TOPIC = N_PER_TOPIC_TOTAL - N_TRAIN_PER_TOPIC


def sample_and_split_topic(df_topic, label_name):
    # Ensure we don't request more rows than available
    n_total = min(N_PER_TOPIC_TOTAL, len(df_topic))
    n_train = min(
        N_TRAIN_PER_TOPIC, n_total - max(1, n_total // 4)
    )  # keep some for test
    n_test = n_total - n_train

    df_sample = df_topic.sample(n=n_total, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    df_train_topic = df_sample.iloc[:n_train].copy()
    df_test_topic = df_sample.iloc[n_train:].copy()

    df_train_topic["topic"] = label_name
    df_test_topic["topic"] = label_name

    return df_train_topic, df_test_topic


train_ed, test_ed = sample_and_split_topic(df_ed, "Eddington (film)")
train_arch, test_arch = sample_and_split_topic(df_arch, "Arch Manning")
train_gun, test_gun = sample_and_split_topic(df_gun, "Gun Violence")

df_train = pd.concat([train_ed, train_arch, train_gun], ignore_index=True)
df_test = pd.concat([test_ed, test_arch, test_gun], ignore_index=True)

df_train["split"] = "train"
df_test["split"] = "test"

print("Train size:", df_train.shape)
print("Test size:", df_test.shape)
print("Train counts by topic:\n", df_train["topic"].value_counts(), "\n")
print("Test counts by topic:\n", df_test["topic"].value_counts(), "\n")

# %% [markdown]
# **Data Cleaning and Reduction**
#
# - **Filtering**:
#   - Kept only tweets from the three selected topics.
# - **Sampling**:
#   - For each topic, we use up to ~2,000 tweets:
#     - ~1,500 for training (unsupervised clustering).
#     - Remaining (~500) for evaluation.
# - **Cleaning**:
#   - Remove URLs, mentions, hashtags, punctuation.
#   - Lowercase text and normalize whitespace.
# - **Dimensionality Reduction**:
#   - Convert cleaned text into high-dimensional semantic embeddings.
#   - Project embeddings into 2D using t-SNE for visualization.

# %%
### 4. Text Cleaning


def clean_tweet(text: str) -> str:
    """
    Cleans raw tweet text by removing URLs, mentions, hashtags,
    non-alphanumeric characters, and extra whitespace.
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # @mentions
    text = re.sub(r"#\S+", "", text)  # hashtags (remove tag token)
    text = re.sub(r"[^a-z0-9\s]", "", text)  # punctuation / non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text


df_train["clean_tweet"] = df_train["tweet"].apply(clean_tweet)
df_test["clean_tweet"] = df_test["tweet"].apply(clean_tweet)

# Convert date column to datetime, if present
if "date" in df_train.columns:
    df_train["date"] = pd.to_datetime(df_train["date"], errors="coerce")
if "date" in df_test.columns:
    df_test["date"] = pd.to_datetime(df_test["date"], errors="coerce")

# %% [markdown]
# **Model Description**
#
# We use the following pipeline:
#
# 1. **SentenceTransformer embeddings**
#    - Model: `"all-MiniLM-L6-v2"`
#    - Each tweet → a dense vector encoding semantic meaning.
# 2. **t-SNE for Visualization**
#    - Project all embeddings (train + test) into 2D.
#    - Fit t-SNE **once on all data** so that train and test live in the same 2D space.
# 3. **KMeans Clustering**
#    - Fit KMeans on the **training embeddings** (unsupervised).
#    - Use 3 clusters (one per topic).
#    - Map each cluster to the majority topic among its training tweets.
#    - Predict topics for the test set using this mapping.
#
# This provides both a **quantitative evaluation** and a **visual 2D embedding**.

# %%
### 5. Embedding with SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

train_texts = df_train["clean_tweet"].tolist()
test_texts = df_test["clean_tweet"].tolist()

train_embeddings = model.encode(train_texts, show_progress_bar=True)
test_embeddings = model.encode(test_texts, show_progress_bar=True)

train_embeddings = np.array(train_embeddings)
test_embeddings = np.array(test_embeddings)

# %%
### 6. t-SNE on ALL embeddings (fit once)

all_embeddings = np.vstack([train_embeddings, test_embeddings])

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=50,  # slightly higher for a few thousand points
    init="random",
    learning_rate="auto",
)
all_tsne = tsne.fit_transform(all_embeddings)

tsne_train = all_tsne[: len(df_train)]
tsne_test = all_tsne[len(df_train) :]

df_train["tsne_1"] = tsne_train[:, 0]
df_train["tsne_2"] = tsne_train[:, 1]

df_test["tsne_1"] = tsne_test[:, 0]
df_test["tsne_2"] = tsne_test[:, 1]

# %%
### 7. KMeans Clustering on Training Embeddings

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)

train_clusters = kmeans.fit_predict(train_embeddings)
df_train["cluster"] = train_clusters

print("Training cluster counts:\n", df_train["cluster"].value_counts(), "\n")

# Predict cluster assignments for the test set
test_clusters = kmeans.predict(test_embeddings)
df_test["cluster"] = test_clusters

# %% [markdown]
# **Cluster → Topic Mapping (Majority Vote)**
#
# After clustering, we want to interpret each cluster as a topic.
#
# - For each cluster, look at the training tweets assigned to that cluster.
# - Find the **most frequent topic** in that cluster.
# - Assign that topic as the "label" for the cluster.
# - Use this mapping to produce predicted topics for both train and test sets.
#
# This lets us compute accuracy and other metrics even though the clustering
# itself is unsupervised.

# %%
### 8. Map Clusters to Topics via Majority Vote (Using Training Data)

cluster_to_topic = {}

for cluster_id in sorted(df_train["cluster"].unique()):
    subset = df_train[df_train["cluster"] == cluster_id]
    # Most common topic in this cluster
    if subset["topic"].nunique() == 0:
        # Fallback if something goes wrong
        cluster_to_topic[cluster_id] = "Unknown"
    else:
        majority_topic = subset["topic"].value_counts().idxmax()
        cluster_to_topic[cluster_id] = majority_topic

print("Cluster → Topic mapping:")
for c, t in cluster_to_topic.items():
    print(f"  Cluster {c} → {t}")

# Apply mapping to get predicted topics
df_train["pred_topic"] = df_train["cluster"].map(cluster_to_topic)
df_test["pred_topic"] = df_test["cluster"].map(cluster_to_topic)

# %%
### 9. Quantitative Evaluation

topics = sorted(df_train["topic"].unique())

train_acc = accuracy_score(df_train["topic"], df_train["pred_topic"])
test_acc = accuracy_score(df_test["topic"], df_test["pred_topic"])

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy:     {test_acc:.3f}\n")

print("Classification report (test set):")
print(classification_report(df_test["topic"], df_test["pred_topic"], labels=topics))

cm = confusion_matrix(df_test["topic"], df_test["pred_topic"], labels=topics)
cm_df = pd.DataFrame(
    cm, index=[f"True: {t}" for t in topics], columns=[f"Pred: {t}" for t in topics]
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

# %%
### 10A. Visualize KMeans Clusters on t-SNE (Training Set)

plt.figure(figsize=(12, 8))
cluster_colors = ["#fe218b", "#21b0fe", "#fed700"]  # one color per cluster

for c in sorted(df_train["cluster"].unique()):
    subset = df_train[df_train["cluster"] == c]
    plt.scatter(
        subset["tsne_1"],
        subset["tsne_2"],
        color=cluster_colors[c % len(cluster_colors)],
        label=f"Cluster {c}",
        alpha=0.7,
        s=40,
    )

plt.title("t-SNE Visualization (Training) Colored by KMeans Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
### 10B. Visualize True Topics on t-SNE (Test Set)

plt.figure(figsize=(12, 8))
topic_colors = {
    "Eddington (film)": "#fe218b",
    "Arch Manning": "#fed700",
    "Gun Violence": "#21b0fe",
}

for topic in topics:
    subset = df_test[df_test["topic"] == topic]
    plt.scatter(
        subset["tsne_1"],
        subset["tsne_2"],
        label=topic,
        alpha=0.7,
        s=40,
        color=topic_colors.get(topic, "#333333"),
    )

plt.title("t-SNE Visualization of Test Tweet Embeddings by True Topic")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
### 10C. Visualize Correct vs Incorrect Predictions on t-SNE (Test Set)

df_test["correct"] = df_test["topic"] == df_test["pred_topic"]

plt.figure(figsize=(12, 8))
correct_subset = df_test[df_test["correct"]]
incorrect_subset = df_test[~df_test["correct"]]

plt.scatter(
    correct_subset["tsne_1"],
    correct_subset["tsne_2"],
    alpha=0.6,
    s=40,
    label="Correct",
)

plt.scatter(
    incorrect_subset["tsne_1"],
    incorrect_subset["tsne_2"],
    alpha=0.9,
    s=60,
    label="Incorrect",
    marker="x",
)

plt.title("t-SNE (Test Set): Correct vs Incorrect Topic Predictions")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 11. Clusterability Metrics: Hopkins Statistic & Similarity Matrix

# %%
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------
# 1. Hopkins Statistic (Measures cluster tendency)
# --------------------------------------------------------------

def hopkins_statistic(X, sampling_ratio=0.05, random_state=42):
    """
    Compute the Hopkins statistic for a dataset X.
    X must be a numpy array (n_samples x n_features).
    Values near 0 → highly clusterable
    Values ~0.5 → random/unclustered
    Values near 1 → uniformly distributed (anti-cluster structure)
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    m = int(np.ceil(sampling_ratio * n_samples))  # subsample size

    # Randomly sample m rows from X
    sample_indices = np.random.choice(n_samples, m, replace=False)
    X_sample = X[sample_indices]

    # Generate m random points from the feature range
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_random = X_min + np.random.rand(m, X.shape[1]) * (X_max - X_min)

    # Fit NN on the dataset
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)

    # u = distances from synthetic points to nearest neighbor in X
    u, _ = nbrs.kneighbors(X_random, return_distance=True)

    # w = distances from sampled real points to nearest neighbor in X (excluding itself)
    w, _ = nbrs.kneighbors(X_sample, return_distance=True)

    H = u.sum() / (u.sum() + w.sum())
    return float(H)


# Compute Hopkins Statistic for training embeddings
hopkins_value = hopkins_statistic(train_embeddings)
print(f"Hopkins Statistic (train set): {hopkins_value:.4f}")


# --------------------------------------------------------------
# 2. Cosine Similarity Matrix (Training Set)
# --------------------------------------------------------------

# Compute full cosine similarity matrix
similarity_matrix = cosine_similarity(train_embeddings)

print("\nCosine similarity matrix shape:", similarity_matrix.shape)

# Heatmap (OPTIONAL — shows structure but can be large)
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix[:300, :300], cmap="viridis")  # show first 300x300 to avoid lag
plt.title("Cosine Similarity Matrix (First 300 Training Tweets)")
plt.xlabel("Tweet Index")
plt.ylabel("Tweet Index")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Interpretations**
#
# - The **KMeans clustering** in embedding space, when mapped to topics via majority vote,
#   achieves non-trivial accuracy on the held-out test set.
# - The **t-SNE visualization** of the training set colored by clusters shows that
#   many tweets assigned to the same cluster occupy similar regions in 2D, suggesting
#   that embeddings capture meaningful structure.
# - The **t-SNE visualization of the test set** colored by true topic shows that:
#   - Tweets about **Arch Manning (sports)** often form a distinct region, reflecting
#     sports-specific language (teams, games, stats).
#   - Tweets about **Gun Violence (public policy)** tend to cluster in another region,
#     often containing political and policy-related vocabulary.
#   - Tweets about **Eddington (film)** form their own region, but may overlap with
#     other topics when discussing broader themes (awards, reviews, emotions).
# - The **correct vs incorrect** t-SNE plot reveals that misclassified points usually
#   lie in overlapping regions where language is more generic or could plausibly refer
#   to multiple topics.
#
# Overall, the combination of SentenceTransformer embeddings, KMeans clustering,
# and t-SNE visualization provides evidence that:
#
# - Tweet semantics differ across these three topics.
# - Unsupervised clustering can recover much of this structure.
# - Visualization helps us understand where the model is confident and where it struggles.

# %% [markdown]
# **Ideas to Improve Results**
#
# - **Increase dataset size**: Use more tweets per topic if available.
# - **Richer features**:
#   - Incorporate metadata (e.g., engagement counts, presence of links or media).
#   - Include bigram/phrase-level information or topic modeling.
# - **Alternative clustering**:
#   - Try algorithms like DBSCAN or HDBSCAN that can model non-spherical clusters.
# - **Hyperparameter tuning**:
#   - Experiment with different embedding models.
#   - Tune t-SNE hyperparameters (perplexity, learning rate) and KMeans settings.
#
# These extensions could yield tighter clusters, better separation, and more nuanced
# insights into how language differs across public conversations on film, sports,
# and public policy.

# %% [markdown]
# **References**
#
# - scikit-learn TSNE Documentation:
#   - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# - scikit-learn KMeans Documentation:
#   - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# - GeeksforGeeks. *t-SNE Algorithm in Machine Learning*:
#   - https://www.geeksforgeeks.org/t-sne-algorithm-in-machine-learning/
# - SentenceTransformer Documentation:
#   - https://www.sbert.net/
# - We lightly used Generative AI to reword ideas for better clarity and flow.
