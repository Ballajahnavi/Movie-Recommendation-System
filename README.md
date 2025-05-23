# Movie-Recommendation-System

A hybrid movie recommendation system using collaborative and content-based filtering with the MovieLens 20M dataset. Built with Python, Pandas, Scikit-learn, Surprise, NLTK, and Streamlit.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download MovieLens 20M dataset from Kaggle and place files in `data/`.
4. Run the Jupyter notebook: `jupyter notebook notebooks/movie_recommendation_20m.ipynb`
5. Launch Streamlit app: `streamlit run src/app.py`

## Dataset
- MovieLens 20M: 20 million ratings, movie genres, user tags, and tag relevance scores.

## Features
- Collaborative filtering using SVD.
- Content-based filtering using genres and tags.
- Sentiment-based filtering using tag sentiment.
- Interactive Streamlit UI for recommendations.
