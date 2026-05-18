# MacBook Neo Media Sentiment Analysis

## Project Overview

This project analyzes media sentiment surrounding the **Apple MacBook Neo** using Natural Language Processing (NLP) techniques. The goal of the project is to understand how news outlets and technology media reacted to the product launch by analyzing article sentiment, discussion topics, and source-level media behavior.

The analysis was performed using article data collected from the NewsAPI.ai (Event Registry) API between April 2026 and May 2026.

---

## Why This Project Matters

Public and media sentiment can significantly influence product perception, consumer behavior, and brand reputation. By analyzing media coverage through NLP, companies and analysts can better understand:

- Overall public reception
- Positive and negative discussion trends
- Key concerns raised by media outlets
- Market risks and opportunities

This project demonstrates how sentiment analysis can be applied to real-world business and technology reporting.

---

## Dataset Information

The dataset contains news articles related to the MacBook Neo with the following features:

| Feature | Description |
|---|---|
| Date | Publication date |
| Source | News source or publisher |
| Title | Article headline |
| Sentiment | Original API sentiment |
| Body | Full article content |
| URL | Article link |

A total of **321 relevant articles** were collected after filtering duplicates and irrelevant results.

---

## Technologies Used

- Python
- Pandas
- NLTK
- spaCy
- VADER Sentiment Analysis
- Scikit-learn
- Matplotlib
- WordCloud
- NewsAPI.ai / Event Registry API

---

## Methodology

### 1. Data Collection

Articles were scraped using the Event Registry API with:
- Exact keyword matching
- English-language filtering
- Duplicate removal
- Date filtering
- Relevance filtering

### 2. Text Preprocessing

The text preprocessing pipeline included:
- Lowercasing
- URL removal
- Punctuation removal
- Stopword removal
- Lemmatization using spaCy

### 3. Sentiment Analysis

Sentiment scoring was performed using the **VADER Sentiment Analyzer**, which generated compound sentiment scores for each article.

Articles were classified into:
- Positive
- Negative
- Neutral

### 4. Insight Generation

The project analyzed:
- Overall sentiment distribution
- Sentiment trends over time
- Source-level sentiment behavior
- Keyword/topic extraction using TF-IDF
- Word cloud visualization

---

## Key Findings

### Overall Sentiment

The analysis revealed overwhelmingly positive media coverage of the MacBook Neo:

| Sentiment | Count |
|---|---|
| Positive | 306 |
| Negative | 12 |
| Neutral | 3 |

The average compound sentiment score was approximately **0.857**, indicating strongly positive media perception overall.

### Positive Media Themes

Positive coverage commonly highlighted:
- Affordable pricing
- Premium Apple design
- Strong battery life
- AI-powered features
- Accessibility for first-time Apple users
- Strong competition against Chromebooks and Windows laptops

### Negative Media Themes

Negative sentiment was primarily driven by:
- Supply shortages
- Rising RAM and chip prices
- Delivery delays
- Concerns over potential price increases

Importantly, most negative coverage focused on operational and supply-chain concerns rather than criticism of the product itself.

### Source-Level Insights

Consumer-focused tech media and shopping-oriented websites tended to publish highly positive coverage, while analytical and business-focused publications adopted a more cautious tone by discussing pricing sustainability and supply risks.

---

## Visualizations

### Sentiment Over Time

Tracks average sentiment changes throughout the analysis period.

<img width="1590" height="590" alt="average_sentiment_overtime" src="https://github.com/user-attachments/assets/683a659a-bdc3-4b4d-803f-594ddc0b19f1" />

---

### Word Cloud of Main Discussion Topics

Displays the most important discussion topics extracted using TF-IDF keyword analysis.

<img width="2091" height="1089" alt="word_cloud" src="https://github.com/user-attachments/assets/38bec2a0-be18-41c4-b780-e810e1eb0da2" />

---

### Top Positive Sources

Shows the media outlets with the highest average sentiment scores.

<img width="567" height="611" alt="Top_positive_sources" src="https://github.com/user-attachments/assets/4a9168a9-5ac4-4ea3-bc60-cfb1e3f1221e" />

---

### Top Negative Sources

Shows the media outlets with the lowest average sentiment scores.

<img width="567" height="1014" alt="Top_negative_sourcespng" src="https://github.com/user-attachments/assets/5ce8b0a2-b39c-4fa3-afb3-1396015e08aa" />

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Running the Project

Run the sentiment analysis script:

```bash
python sentiment_analysis.py
```

---

## Repository Structure

```text
macbook-neo-sentiment-analysis/
│
├── macbook_neo_articles_data.csv
│
├── notebooks/
│   └── 01_data_collection.ipynb
|   └── 02_preprocessing_sentiment_analysis_visualization.ipynb
│
├── visualizations/
│   ├── avg_sentiment_over_time.png
│   ├── macbook_neo_wordcloud.png
│   ├── top_positive_sources.png
│   └── top_negative_sources.png
│
├── sentiment_analysis.py
├── requirements.txt
└── README.md
```

---

## Conclusion

The sentiment analysis indicates that media coverage of the MacBook Neo was overwhelmingly positive, with most articles emphasizing affordability, premium design, strong performance, and accessibility to new Apple users. Negative sentiment was largely associated with supply-chain concerns and pricing risks rather than dissatisfaction with the product itself. Overall, the findings suggest that the MacBook Neo was perceived as a highly successful and disruptive entry into the budget laptop market.

---

## Limitations

Although the data collection period was configured to begin on March 11, 2026, the earliest relevant articles retrieved from the API only started appearing in April 2026. This may have been caused by several factors, including limited historical article availability, low keyword usage frequency, NewsAPI.ai indexing limitations, or the possibility that the phrase “MacBook Neo” was not widely used in media coverage before April 2026.

Additionally, the analysis relied on VADER sentiment analysis, which is a lexicon-based NLP model. While effective for media and social-text sentiment classification, it may not fully capture sarcasm, contextual nuance, or highly technical language found in certain technology articles. The dataset was also limited to English-language articles, meaning that media sentiment from non-English sources was not included in the analysis.

Finally, the project focused primarily on media sentiment rather than direct consumer sentiment from social media platforms, forums, or customer reviews, which may provide different perspectives on product perception.

---

## Future Research

Future research could expand the analysis by incorporating additional data sources such as Reddit, X (Twitter), YouTube comments, online forums, or customer reviews to better capture public sentiment beyond news media coverage. A larger multilingual dataset could also provide a more global understanding of media perception across different regions and languages.

More advanced NLP models such as BERT, RoBERTa, or FinBERT could be explored to improve sentiment classification accuracy and better understand contextual meaning in technology-related articles. Topic modeling techniques such as LDA (Latent Dirichlet Allocation) could also be applied to identify emerging discussion themes automatically.

Additionally, future studies could compare sentiment trends between competing products such as MacBook Air, Chromebook Plus devices, or Windows laptops to better understand the MacBook Neo’s positioning within the broader laptop market.
