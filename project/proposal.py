# %%
# Andre Sae
# Kyle Castillo

# %%
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

df = pd.read_csv("tweets_raw.csv")
df["author"].value_counts()

df = df[
    (df["author"] == "Barack Obama")
    | (df["author"] == "katyperry")
    | (df["author"] == "instagram")
]
df = df[["author", "content", "date_time", "number_of_likes", "number_of_shares"]]


# %%
def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["content"] = df["content"].apply(clean_tweet)
df["date_time"] = pd.to_datetime(df["date_time"], format="%d/%m/%Y %H:%M")
df = df.reset_index(drop=True)

# %%
vec = TfidfVectorizer(max_features=1000)
X = vec.fit_transform(df["content"]).toarray()

tsne = TSNE(n_components=2, random_state=42)
feat_2 = tsne.fit_transform(X)

df["f1"] = feat_2[:, 0]
df["f2"] = feat_2[:, 1]
