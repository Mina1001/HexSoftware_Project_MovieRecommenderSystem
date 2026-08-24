# train_model.py
# Content-Based Movie Recommender System using TF-IDF & Cosine Similarity

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📦 Building Curated Movie Dataset...")

# Real famous movies with genres and overviews
movies_data = [
    {"title": "Inception", "genre": "Sci-Fi Action Thriller", "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
    {"title": "The Dark Knight", "genre": "Action Crime Drama", "overview": "When the menace known as the Joker wreaks havoc and chaos on Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
    {"title": "Interstellar", "genre": "Sci-Fi Drama Adventure", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival as Earth becomes uninhabitable."},
    {"title": "The Matrix", "genre": "Sci-Fi Action", "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
    {"title": "Avatar", "genre": "Sci-Fi Action Adventure", "overview": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."},
    {"title": "Oppenheimer", "genre": "Biography Drama History", "overview": "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II."},
    {"title": "Titanic", "genre": "Drama Romance", "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."},
    {"title": "The Avengers", "genre": "Action Sci-Fi Adventure", "overview": "Earth's mightiest heroes must come together and learn to fight as a team if they are to stop the mischievous Loki and his alien army from enslaving humanity."},
    {"title": "Pulp Fiction", "genre": "Crime Drama", "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    {"title": "The Shawshank Redemption", "genre": "Drama Crime", "overview": "Over the course of several years, two convicts form a friendship, seeking solace and eventual redemption through basic compassion."},
    {"title": "Gladiator", "genre": "Action Adventure Drama", "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."},
    {"title": "Forrest Gump", "genre": "Drama Romance", "overview": "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75."},
    {"title": "Jurassic Park", "genre": "Sci-Fi Adventure Action", "overview": "A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting four children after a power failure causes the park's cloned dinosaurs to run loose."},
    {"title": "Spider-Man: Into the Spider-Verse", "genre": "Animation Action Sci-Fi", "overview": "Teen Miles Morales becomes the Spider-Man of his universe and must join with five spider-powered individuals from other dimensions to stop a threat for all realities."},
    {"title": "La La Land", "genre": "Comedy Drama Music Romance", "overview": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."},
    {"title": "The Godfather", "genre": "Crime Drama", "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "Whiplash", "genre": "Drama Music", "overview": "A promising young drummer enlists at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."},
    {"title": "Se7en", "genre": "Crime Drama Mystery Thriller", "overview": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."},
    {"title": "Spirited Away", "genre": "Animation Adventure Fantasy", "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches and spirits, a world where humans are changed into beasts."},
    {"title": "Black Panther", "genre": "Action Sci-Fi Adventure", "overview": "T'Challa, heir to the hidden kingdom of Wakanda, must step forward to lead his people into a new era and must confront a challenger from his country's past."}
]

df = pd.DataFrame(movies_data)

# Combine Genre + Overview to form feature text
df['combined_features'] = df['genre'] + " " + df['overview']

print("🧠 Vectorizing Features using TF-IDF...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

print("📐 Calculating Cosine Similarity Matrix...")
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save movies dataframe & similarity matrix
with open('movies.pkl', 'wb') as f:
    pickle.dump(df, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

print("="*50)
print("🏆 Movie Recommender Model Trained Successfully!")
print(f"Total Movies Indexed: {len(df)}")
print("="*50)