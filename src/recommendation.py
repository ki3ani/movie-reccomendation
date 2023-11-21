import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def recommend_movies(user_name, data, average_ratings, favorite_genres):
    # Step 1: Prepare data for recommendation
    user_id = get_user_id(user_name, data)
    user_movie_matrix = create_user_movie_matrix(data)

    # Step 2: Content-Based Recommendation
    recommended_movies = get_content_based_recommendations(user_id, user_movie_matrix, average_ratings, favorite_genres, data)

    # Step 3: Display recommendations
    print(f"\nHello {user_name}! Here are personalized movie recommendations for you:")
    print(recommended_movies)

def get_user_id(user_name, data):
    label_encoder = LabelEncoder()
    data['User_ID'] = label_encoder.fit_transform(data['User'])
    user_id = label_encoder.transform([user_name])[0]
    return user_id

def create_user_movie_matrix(data):
    user_movie_matrix = data.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    return user_movie_matrix

def get_content_based_recommendations(user_id, user_movie_matrix, average_ratings, favorite_genres, data):
    # Calculate cosine similarity between the user's favorite genre and other movies
    user_favorite_genre = favorite_genres[data[data['User_ID'] == user_id]['User'].iloc[0]]
    user_favorite_genre_vector = create_genre_vector(user_favorite_genre, data)
    movie_similarity = cosine_similarity(user_movie_matrix.T, [user_favorite_genre_vector])

    # Get movie IDs sorted by similarity
    similar_movie_ids = movie_similarity.flatten().argsort()[::-1]

    # Exclude movies the user has already rated
    user_rated_movies = data[data['User_ID'] == user_id]['Movie_ID'].tolist()
    recommended_movies = [movie for movie in similar_movie_ids if movie not in user_rated_movies]

    # Map movie IDs back to movie names
    recommended_movie_names = data[data['Movie_ID'].isin(recommended_movies)]['Movie'].unique()

    return recommended_movie_names

def create_genre_vector(genre, data):
    # Create a binary vector indicating whether each movie belongs to the given genre
    label_encoder = LabelEncoder()
    data['Genre_ID'] = label_encoder.fit_transform(data['Movie'])
    genre_vector = (data[data['Genre_ID'] == label_encoder.transform([genre])[0]]['Movie_ID']).apply(lambda x: 1 if x else 0)
    
    return genre_vector.values.reshape(1, -1)
