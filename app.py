from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset with specified dtypes and handling NaN
data = pd.read_csv('K:/Kura/merged_dataset.csv', dtype={'review': str, 'sentiment': str}, low_memory=False)

# Fill NaN values in 'review' with an empty string
data['review'] = data['review'].fillna('')

# Prepare the TF-IDF vectorizer on the 'review' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['review'])

# Calculate cosine similarity for the sentiment-based recommendation
cosine_sim = cosine_similarity(tfidf_matrix)

# Create a movie ratings matrix for collaborative filtering
ratings_matrix = data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Function to get movie recommendations based on rating
def get_rating_based_recommendations(movie_id, num_recommendations=5):
    similar_scores = ratings_matrix.corrwith(ratings_matrix[movie_id])
    similar_scores = similar_scores.sort_values(ascending=False)
    return similar_scores.index[1:num_recommendations + 1].tolist()

# Function to get movie recommendations based on sentiment
def get_sentiment_based_recommendations(movie_id, num_recommendations=5):
    idx = data[data['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    return data['movieId'].iloc[movie_indices].tolist()

# Function for hybrid recommendations
def get_hybrid_recommendations(movie_id, num_recommendations=5):
    rating_recommendations = get_rating_based_recommendations(movie_id, num_recommendations)
    sentiment_recommendations = get_sentiment_based_recommendations(movie_id, num_recommendations)
    return list(set(rating_recommendations) | set(sentiment_recommendations))[:num_recommendations]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_rating = int(request.form['rating'])
        user_sentiment = request.form['sentiment']
        movie_id = int(request.form['movieId'])  # Example movie ID

        if request.form['recommendation_type'] == 'rating':
            recommendations = get_rating_based_recommendations(movie_id)
        elif request.form['recommendation_type'] == 'sentiment':
            recommendations = get_sentiment_based_recommendations(movie_id)
        else:
            recommendations = get_hybrid_recommendations(movie_id)

        recommended_movies = data[data['movieId'].isin(recommendations)]
        return render_template('index.html', movies=recommended_movies)

    return render_template('index.html', movies=None)

if __name__ == '__main__':
    app.run(debug=True)
