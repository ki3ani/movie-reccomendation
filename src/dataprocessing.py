import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_and_analyze_data(data):
    # Load the data
    data = read_data('data/data.txt')

    # Clean the data
    data = clean_data(data)

    # Explore insights
    average_ratings = get_average_ratings(data)
    favorite_genres = get_favorite_genres(data)

    # Return cleaned data and insights
    return data, average_ratings, favorite_genres

def read_data(file_path):
    # Read data from a text file
    # Assuming that the data is separated by commas
    data = pd.read_csv(file_path, delimiter=',')

    return data

def clean_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Convert 'Rating' to numeric, handle errors
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

    # Handle anomalies in 'Rating' column
    data['Rating'] = data['Rating'].apply(clean_rating)

    # Encode user and movie names to numeric IDs
    label_encoder = LabelEncoder()
    data['User_ID'] = label_encoder.fit_transform(data['User'])
    data['Movie_ID'] = label_encoder.fit_transform(data['Movie'])

    return data

def clean_rating(rating):
    try:
        return float(rating)
    except (ValueError, TypeError):
        # Handle anomalies like '5x', 'Five', etc.
        if 'x' in str(rating):
            return float(rating[:-1])
        elif rating.lower() == 'five':
            return 5.0
        else:
            return None

def get_average_ratings(data):
    # Calculate average ratings for each movie
    average_ratings = data.groupby('Movie')['Rating'].mean()
    return average_ratings

def get_favorite_genres(data):
    # Calculate average ratings for each user
    average_user_ratings = data.groupby('User')['Rating'].mean()

    # Identify favorite genres for each user
    favorite_genres = {}
    user_movie_matrix = data.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    
    for user, ratings in user_movie_matrix.iterrows():
        favorite_movie_id = ratings.idxmax()
        favorite_genre = data[data['Movie_ID'] == favorite_movie_id]['Movie'].iloc[0]
        favorite_genres[user] = favorite_genre

    return favorite_genres
