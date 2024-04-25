import helper
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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
print(user_profile.loc[4])

# Min-Max normalization (values are automatically determined).
scaler = MinMaxScaler()
user_profile_normalized = scaler.fit_transform(user_profile)

# Convert back to DataFrame
user_profile_normalized = pd.DataFrame(user_profile_normalized, columns=user_profile.columns, index=user_profile.index)

# TF-IDF vectorization
items_df = items_df = helper.retrieve_items_data()

def recommend_movies(user_id, top_n=1682):
    # Get the user's profile
    user_profile_vec = user_profile.loc[user_id].values.reshape(1, -1)
    
    # Compute cosine similarity between the user's profile and all movies based on genre
    cosine_similarities = cosine_similarity(user_profile_vec, items_df[['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                                                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                                                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                                                  'War', 'Western']].values)
    
    print("all similartities", cosine_similarities[0][0])
    
    # Get indices of movies sorted by similarity score
    similar_movie_indices_enum = enumerate(cosine_similarities[0], start = 1)  # Start enumeration from index 0
    
    # Exclude movies the user has already rated
    rated_movies = df.loc[df['user_id'] == user_id, 'item_id'].values
    
    recommendations = []
    for movie_index, similarity_score in similar_movie_indices_enum:
        original_movie_index = movie_index  # Adjust movie index to align with original .csv row numbers
        if original_movie_index in rated_movies:
            movie_title = items_df.loc[original_movie_index-1, 'movie_title']
            recommendations.append((original_movie_index, movie_title, similarity_score))
            
            if len(recommendations) == top_n:
                break

    sorted_recommendations = sorted(recommendations, key=lambda x: x[-1], reverse=True)
    
    return sorted_recommendations

# Example: Recommend movies for user 1
user_id = 4
recommendations = recommend_movies(user_id)

print(f"Top 5 movie recommendations for user {user_id}:")
for movie_index, movie, score in recommendations:
    print(f"{movie_index} {movie}: Similarity score = {score:.2f}")