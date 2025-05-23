#🎬 Movie Recommendation System  

This is a content-based movie recommender system built using the MovieLens 20M Dataset. It utilizes TF-IDF vectorization on movie genres and taglines to suggest similar movies. A simple Streamlit web interface is included for interactive recommendations.

---

🚀 Features  
- Recommends movies based on genres and tags  
- Clean and intuitive Streamlit interface  
- Fast and lightweight – does not rely on user ratings  
- Scalable to large datasets like MovieLens 20M  

---

📁 Dataset  
Source: [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) by GroupLens

Download and place the following files in your project directory:  
- `movies.csv`  
- `tags.csv`  

---

🛠 Installation  
Clone the repository or download the project files.

Install required Python packages:

```bash
pip install streamlit pandas scikit-learn
```
Download the dataset:

Visit https://grouplens.org/datasets/movielens/20m/

Download and extract the zip file

Place movies.csv and tags.csv in the same directory as app.py

---

▶️ Run the App

```bash
Copy
Edit
streamlit run app.py
```
Then open the provided localhost URL in your browser.

---

🧠 How It Works

Loads and merges movie metadata (genres, tags)

Aggregates tags and genres per movie into a single feature

Converts the combined text into numerical vectors using TF-IDF

Computes cosine similarity between movies

Returns the top N similar movies based on a selected title

---

📦 Project Structure

```bash
Copy
Edit
├── app.py                # Streamlit app
├── movies.csv            # Movie metadata
├── tags.csv              # Movie tags
├── README.md             # Project README
```
vbnet
Copy
Edit
