import helper
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load data from CSV
df = helper.retrieve_data_merged()

# Show all ratings for a specific user (for testing purposes only).
print(df.loc[(df["user_id"] == 4) & (~df["rating"].isna()), ["user_id", "movie_title", "rating"]].sort_values(by="rating", ascending=False))


# Select relevant columns for user profiles and movie genres
user_profile = df.groupby('user_id')[['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                      'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                      'War', 'Western']].mean()

# Convert NaN values to 0
user_profile.fillna(0, inplace=True)
print(print(user_profile.loc[4]))

# TF-IDF vectorization
items_df = pd.read_csv('../movie_dataset/u.item.csv', delimiter=';')

# Compute cosine similarity between user profiles and movie genres
def recommend_movies(user_id, top_n=10):
    # Get the user's profile
    user_profile_vec = user_profile.loc[user_id].values.reshape(1, -1)
    
    # Compute cosine similarity between the user's profile and all movie genres
    cosine_similarities = cosine_similarity(user_profile_vec, items_df[['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                                                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                                                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                                                  'War', 'Western']].values)
    
    # Get indices of movies sorted by similarity score
    similar_movie_indices = cosine_similarities.argsort()[0][::-1]
    
    # Exclude movies the user has already rated
    rated_movies = df.loc[df['user_id'] == user_id, 'item_id'].values
    
    recommendations = []
    for movie_index in similar_movie_indices:
        if movie_index + 1 not in rated_movies:
            similarity_score = cosine_similarities[0, movie_index]
            movie_title = df.loc[movie_index, 'movie_title']
            recommendations.append((movie_index, movie_title, similarity_score))
            
            if len(recommendations) == top_n:
                break
    
    return recommendations

# Example: Recommend movies for user 1
user_id = 4
recommendations = recommend_movies(user_id)
print(f"Top 5 movie recommendations for user {user_id}:")
for movie_index, movie, score in recommendations:
    print(f"{movie_index} {movie}: Similarity score = {score:.2f}")
