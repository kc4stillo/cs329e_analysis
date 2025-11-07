# %%
# Andre Sae
# Kyle Castillo

# %%
### 1. Import Libraries
import re

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# %%
### 2. Load and Prepare Data

# Define authors and sample size
TARGET_AUTHORS = ["BarackObama", "Cristiano", "instagram"]
SAMPLE_SIZE = 2000

# Load CSV
df_raw = pd.read_csv("tweets_raw.csv")

# Filter authors, select relevant columns, sample per author
df = (
    df_raw[df_raw["author"].isin(TARGET_AUTHORS)]
    .loc[:, ["author", "content", "date_time", "number_of_likes", "number_of_shares"]]
    .groupby("author", group_keys=False)
    .apply(lambda x: x.sample(n=SAMPLE_SIZE, replace=True))
    .reset_index(drop=True)
)

# %%
### 3. Text Cleaning


def clean_tweet(text):
    """Cleans raw tweet text by removing URLs, mentions, hashtags, and punctuation."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


# Apply cleaning function
df["content_cleaned"] = df["content"].apply(clean_tweet)

# Convert date_time column to datetime format
df["date_time"] = pd.to_datetime(
    df["date_time"], format="%d/%m/%Y %H:%M", errors="coerce"
)

# Drop any rows where date parsing failed
df = df.dropna(subset=["date_time"]).reset_index(drop=True)

# %%
### 4. Feature Engineering: Sentence Embeddings & t-SNE

# Embed tweet content using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["content_cleaned"].tolist(), show_progress_bar=True)

# Reduce embeddings to 2D using t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,  # Perplexity < number of samples
)
features_2d = tsne.fit_transform(embeddings)

# Add 2D features back to DataFrame
df["tsne_1"] = features_2d[:, 0]
df["tsne_2"] = features_2d[:, 1]

# %%
### 5. Visualize Results

plt.figure(figsize=(10, 8))

for author in df["author"].unique():
    subset = df[df["author"] == author]
    plt.scatter(
        subset["tsne_1"],
        subset["tsne_2"],
        label=author,
        alpha=0.7,
        s=50,  # marker size
    )

plt.title("t-SNE Visualization of Tweet Embeddings by Author")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# %%
