import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Load results and embeddings
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
df = pd.DataFrame(papers)
abstracts = [p["abstract"] for p in papers]
embeddings = np.load("paper_embeddings.npy")
print(f"Loaded {len(df)} papers")

# Fit BERTopic with stopword removal
print("Fitting BERTopic... (this may take a few minutes)")
vectorizer = CountVectorizer(stop_words="english", min_df=2)
topic_model = BERTopic(language="english", vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)
topics, _ = topic_model.fit_transform(abstracts, embeddings)
df["topic_id"] = topics

# Topic info
topic_info = topic_model.get_topic_info()
print("\nTop 10 topics:")
print(topic_info.head(10))

# Mean alignment score per topic (excluding Topic -1)
topic_alignment = df.groupby("topic_id")["alignment_score"].mean().reset_index()
topic_alignment = topic_alignment.sort_values("alignment_score")
topic_alignment = topic_alignment[topic_alignment["topic_id"] != -1]

print("\n5 most misaligned topics:")
for _, row in topic_alignment.head(5).iterrows():
    topic_words = topic_model.get_topic(int(row["topic_id"]))
    words = [w[0] for w in topic_words[:5]] if topic_words else []
    print(f"  Topic {int(row['topic_id'])} [{row['alignment_score']:.3f}]: {', '.join(words)}")

print("\n5 most aligned topics:")
for _, row in topic_alignment.tail(5).iterrows():
    topic_words = topic_model.get_topic(int(row["topic_id"]))
    words = [w[0] for w in topic_words[:5]] if topic_words else []
    print(f"  Topic {int(row['topic_id'])} [{row['alignment_score']:.3f}]: {', '.join(words)}")

# Visualize top 15 topics by alignment score
top_topics = topic_alignment.tail(15)
plt.figure(figsize=(10, 6))
plt.barh(
    [f"Topic {int(t)}" for t in top_topics["topic_id"]],
    top_topics["alignment_score"],
    color="steelblue", edgecolor="black"
)
plt.xlabel("Mean Alignment Score")
plt.title("Mean Alignment Score by Topic (BERTopic)")
plt.tight_layout()
plt.savefig("topic_alignment.png", dpi=200)
plt.show()
print("topic_alignment.png saved")

# Save results with topic assignments
df.to_csv("results_with_topics.csv", index=False)
print("results_with_topics.csv saved")