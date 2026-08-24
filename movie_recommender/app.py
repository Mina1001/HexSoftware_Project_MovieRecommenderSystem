# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load saved movie database and similarity matrix
with open('movies.pkl', 'rb') as f:
    df = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

def get_recommendations(movie_title, top_n=5):
    # Find index of movie
    if movie_title not in df['title'].values:
        return []

    idx = df[df['title'] == movie_title].index[0]

    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort movies based on similarity score (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top_n most similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n+1]

    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            'title': df.iloc[i]['title'],
            'genre': df.iloc[i]['genre'],
            'overview': df.iloc[i]['overview'],
            'match': f"{round(score * 100, 1)}%"
        })

    return recommendations

@app.route('/')
def home():
    movie_list = df['title'].tolist()
    return render_template('index.html', movie_list=movie_list, selected_movie=None, recommendations=[])

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie_title')
    movie_list = df['title'].tolist()
    
    recommendations = get_recommendations(selected_movie, top_n=4)

    return render_template(
        'index.html', 
        movie_list=movie_list, 
        selected_movie=selected_movie, 
        recommendations=recommendations
    )

if __name__ == '__main__':
    app.run(debug=True)