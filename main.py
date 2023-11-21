# main.py

from src.dataprocessing import clean_and_analyze_data
from src.recommendation import recommend_movies
from src.userinterface import get_user_input

def main():
    # Step 1: Data Cleaning and Analysis
    data, average_ratings, favorite_genres = clean_and_analyze_data("data/data.txt")

    # Step 2: Recommendation Algorithm (if needed)

    # Step 3: User Interface
    user_name = get_user_input()
    recommend_movies(user_name, data, average_ratings, favorite_genres)

if __name__ == "__main__":
    main()
