import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('amazon_product_data.csv')  # Replace with your actual dataset

# 1. Data Preprocessing and Exploration
print("1. Data Preprocessing and Exploration")
print(df.head())
print(df.info())

# Handle missing values
df = df.dropna()

# 2. Feature Engineering
print("\n2. Feature Engineering")

# Text feature extraction
tfidf = TfidfVectorizer(max_features=1000)
product_features = tfidf.fit_transform(df['about_product'])

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['review_content'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x > 0 else 'negative')

# 3. Sentiment Analysis Model
print("\n3. Sentiment Analysis Model")
X = tfidf.fit_transform(df['review_content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print("Sentiment Analysis Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Product Recommendation System
print("\n4. Product Recommendation System")
# Using cosine similarity for a simple recommendation system
cosine_sim = cosine_similarity(product_features)

def get_recommendations(product_id, cosine_sim=cosine_sim):
    idx = df.index[df['product_id'] == product_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar products
    product_indices = [i[0] for i in sim_scores]
    return df['product_name'].iloc[product_indices]

# Example recommendation
print(get_recommendations(df['product_id'].iloc[0]))

# 5. Pricing Analysis
print("\n5. Pricing Analysis")
# Analyze relationship between sentiment, rating, and price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sentiment_scores', y='discounted_price', hue='rating')
plt.title('Sentiment vs Price vs Rating')
plt.show()

# Simple price optimization model
from sklearn.linear_model import LinearRegression

X = df[['rating', 'sentiment_scores']]
y = df['discount_percentage']

model = LinearRegression()
model.fit(X, y)

print("Coefficients:")
print("Rating:", model.coef_[0])
print("Sentiment Score:", model.coef_[1])

# 6. Insights and Recommendations
print("\n6. Insights and Recommendations")
print("1. Sentiment Analysis can help identify products that need improvement.")
print("2. The recommendation system can be used to suggest related products to customers.")
print("3. Pricing strategy should consider both product ratings and customer sentiment.")
print("4. Further analysis is needed to optimize discount percentages based on product features and customer feedback.")

# Optional: Save results to a file
df[['product_id', 'product_name', 'sentiment', 'rating', 'discounted_price']].to_csv('analysis_results.csv', index=False)
print("\nResults saved to 'analysis_results.csv'")