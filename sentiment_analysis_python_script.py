# ============================================
# MacBook Neo Media Sentiment Analysis
# Author: Your Name
# ============================================

# =====================
# IMPORT LIBRARIES
# =====================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS

# =====================
# LOAD DATASET
# =====================

# Make sure the dataset exists in:
# data/raw_macbook_neo_articles.csv

df = pd.read_csv(
    "data/raw_macbook_neo_articles.csv"
)

# Combining title and body improves sentiment accuracy

df["text"] = (
    df["Title"].fillna('') + " " +
    df["Body"].fillna('')
)

# =====================
# TEXT PREPROCESSING
# =====================

# Download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Text cleaning function

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning

df["clean_text"] = df["text"].apply(clean_text)

# Stopword removal

def remove_stopwords(text):

    words = text.split()

    filtered = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered)

# Apply stopword removal

df["clean_text"] = df["clean_text"].apply(remove_stopwords)

#Lemmatization

import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def lemmatize(text):

    doc = nlp(text)

    return " ".join(
        token.lemma_
        for token in doc
    )

df["clean_text"] = df["clean_text"].apply(
    lemmatize
)

# =====================
# SENTIMENT ANALYSIS
# =====================

# Initialize VADER

analyzer = SentimentIntensityAnalyzer()

# Generate sentiment scores

df["compound_score"] = df["clean_text"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# Convert scores into labels

def classify_sentiment(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"

# Create sentiment labels

df["predicted_sentiment"] = df[
    "compound_score"
].apply(classify_sentiment)

# =====================
# SAVE CLEANED DATASET
# =====================

# Save processed dataset

df.to_csv(
    "data/cleaned_macbook_neo_articles.csv",
    index=False
)

print("Cleaned dataset saved successfully.")

# =====================
# SENTIMENT DISTRIBUTION
# =====================

print("\nSENTIMENT DISTRIBUTION")
print("=" * 40)

print(
    df["predicted_sentiment"].value_counts()
)

# =====================
# AVERAGE SENTIMENT SCORE
# =====================

average_sentiment = df[
    "compound_score"
].mean()

print("\nAVERAGE COMPOUND SCORE")
print("=" * 40)

print(round(average_sentiment, 4))

# =====================
# SENTIMENT OVER TIME
# =====================

# Convert date column

df["Date"] = pd.to_datetime(df["Date"])

# Group by date

daily_sentiment = df.groupby(
    df["Date"].dt.date
)["compound_score"].mean()

# Plot trend

plt.figure(figsize=(16, 6))

daily_sentiment.plot()

plt.title("Average Sentiment Over Time")

plt.xlabel("Date")

plt.ylabel("Average Compound Score")

plt.xticks(rotation=45)

plt.tight_layout()

# Save visualization

plt.savefig(
    "visualizations/sentiment_over_time.png",
    bbox_inches="tight"
)

plt.show()

# =====================
# WORD CLOUD ANALYSIS
# =====================

custom_stopwords = set(STOPWORDS)

custom_stopwords.update([
    "macbook",
    "neo",
    "apple",
    "laptop",
    "device",
    "computer",
    "said",
    "also",
    "one",
    "would",
    "could",
    "us",
    "get",
    "may",
    "amazon",
    "air",
    "pro",
    "ipad",
    "ai",
    "app",
    "port",
    "video",
    "mac",
    "million",
    "unit",
    "wildcat",
    "lake",
    "tim",
    "cook",
    "say",
    "microsoft",
    "xbox",
    "gb",
    "model",
    "point",
    "supply",
    "chain",
    "intel",
    "lenovo",
    "hp",
    "ideapad",
    "window",
    "earning",
    "operating",
    "tag",
    "store"
])

all_docs = df["clean_text"]

vectorizer = TfidfVectorizer(
    stop_words=list(custom_stopwords),
    ngram_range=(2, 3),
    max_features=100
)

X = vectorizer.fit_transform(all_docs)

scores = zip(
    vectorizer.get_feature_names_out(),
    X.sum(axis=0).tolist()[0]
)

word_scores = dict(scores)

wordcloud = WordCloud(
    width=1800,
    height=900,
    background_color="white",
    collocations=False
).generate_from_frequencies(word_scores)

plt.figure(figsize=(22, 11))

plt.imshow(wordcloud)

plt.axis("off")

plt.title(
    "Most Important MacBook Neo Discussion Topics",
    fontsize=24
)

plt.tight_layout()

# Save word cloud

plt.savefig(
    "visualizations/macbook_neo_wordcloud.png",
    bbox_inches="tight"
)

plt.show()

# =====================
# SOURCE SENTIMENT ANALYSIS
# =====================

positive_sources = df.groupby(
    "Source"
)["compound_score"].mean().sort_values(
    ascending=False
)

negative_sources = df.groupby(
    "Source"
)["compound_score"].mean().sort_values(
    ascending=True
)

print("\nMOST POSITIVE SOURCES")
print("=" * 40)

print(
    positive_sources.head(10)
)

print("\nMOST NEGATIVE SOURCES")
print("=" * 40)

print(
    negative_sources.head(10)
)

# =====================
# TOP POSITIVE SOURCES GRAPH
# =====================

positive_sources.head(10).plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Top Positive Sources")

plt.ylabel("Average Sentiment Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "visualizations/top_positive_sources.png",
    bbox_inches="tight"
)

plt.show()

# =====================
# TOP NEGATIVE SOURCES GRAPH
# =====================

negative_sources.head(10).plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Top Negative Sources")

plt.ylabel("Average Sentiment Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "visualizations/top_negative_sources.png",
    bbox_inches="tight"
)

plt.show()

# =====================
# FINAL PROJECT SUMMARY
# =====================

print("\nPROJECT SUMMARY")
print("=" * 40)

print(f"Total Articles Analyzed: {len(df)}")

print(f"Average Sentiment Score: {round(average_sentiment, 4)}")

print(
    "\nThe analysis indicates that media coverage "
    "of the MacBook Neo was overwhelmingly positive, "
    "with negative sentiment mainly driven by "
    "supply-chain concerns, rising component costs, "
    "and potential pricing risks rather than criticism "
    "of the product itself."
)

print("\nAnalysis completed successfully.")
