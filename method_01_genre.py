import helper
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies_genre(user_profiles, to_recommend_df, items_df):
    recommendations = []
    suggested_ratings = []

    for index, row in tqdm(to_recommend_df.iterrows(), total=len(to_recommend_df), desc="Processing rows for method 01"):
        # Get the user's profile
        user_profile_vec = user_profiles.loc[row['user_id']].values.reshape(1, -1)
        
        # Compute cosine similarity between the user's profile and all movies based on genre
        cos_similarity_value = cosine_similarity(user_profile_vec, items_df.loc[items_df['item_id'] == row['item_id'], ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                                                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                                                        'War', 'Western']].values)
        
        movie_title = items_df.loc[items_df['item_id'] == row['item_id'], 'movie_title'].values
        recommendations.append((row['user_id'], row['item_id'], movie_title[0], cos_similarity_value[0][0]))
        suggested_ratings.append(helper.map_similarity_to_rating(cos_similarity_value[0][0]))
    
    return recommendations, suggested_ratings