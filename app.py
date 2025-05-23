import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Download NLTK data
nltk.download('vader_lexicon')

# Load data
data_path = '../data/'
movies = pd.read_csv(os.path.join(data_path, 'movie.csv'))
ratings = pd.read_csv(os.path.join(data_path, 'rating.csv'), nrows=1000000)  # Load 1M ratings
tags = pd.read_csv(os.path.join(data_path, 'tag.csv'))

# Preprocess data
movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]
tags = tags[tags['movieId'].isin(ratings['movieId'].unique())]
movies['genres'] = movies['genres'].replace('|', ' ', regex=True)
movies['genres'] = movies['genres'].replace('(no genres listed)', '')
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')
movies['content'] = movies['genres'] + ' ' + movies['tag']
ratings = ratings.merge(movies[['movieId', 'title', 'content']], on='movieId', how='left')

# Collaborative filtering setup
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=100, random_state=42)
svd.fit(trainset)

# Content-based filtering setup
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Sentiment analysis setup
sia = SentimentIntensityAnalyzer()
tags['sentiment'] = tags['tag'].apply(lambda x: sia.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0.0)
movie_sentiment = tags.groupby('movieId')['sentiment'].mean().reset_index()

# Collaborative filtering function
def get_collaborative_recommendations(user_id, n=5):
    movie_ids = movies['movieId'].unique()
    predictions = [svd.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movie_ids = [pred.iid for pred in predictions[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'content']]

# Content-based filtering function
def get_content_recommendations(title, n=5):
    idx = movies[movies['title'] == title].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movieId', 'title', 'content']]

# Sentiment filtering function
def filter_by_sentiment(recommendations, min_sentiment=0.1):
    recs_with_sentiment = recommendations.merge(movie_sentiment, on='movieId', how='left')
    recs_with_sentiment['sentiment'] = recs_with_sentiment['sentiment'].fillna(0)
    return recs_with_sentiment[recs_with_sentiment['sentiment'] >= min_sentiment]

# Hybrid recommendation function
def get_hybrid_recommendations(user_id, movie_title, n=5, use_sentiment=True):
    collab_recs = get_collaborative_recommendations(user_id, n=10)
    content_recs = get_content_recommendations(movie_title, n=10)
    combined = pd.concat([collab_recs, content_recs]).drop_duplicates(subset=['movieId'])
    if use_sentiment:
        combined = filter_by_sentiment(combined)
    return combined.head(n)

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Get personalized movie recommendations based on your preferences!")

# User input
user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max(), value=1)
movie_title = st.selectbox("Select a Movie", movies['title'].sort_values())
use_sentiment = st.checkbox("Filter by Positive Sentiment", value=True)

# Get recommendations
if st.button("Get Recommendations"):
    recs = get_hybrid_recommendations(user_id, movie_title, use_sentiment=use_sentiment)
    st.subheader(f"Top 5 Recommendations for User {user_id} and Movie {movie_title}")
    if recs.empty:
        st.write("No recommendations found.")
    else:
        st.dataframe(recs[['title', 'content']])