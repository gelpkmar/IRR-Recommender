import helper
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def recommend_user_based_collaborative_filtering(to_recommend_df, items_df, rate_seen=False):
    recommendations = []
    suggested_ratings = []
    # Load data
    merged_df = helper.retrieve_data_merged()
    
    for index, row in tqdm(to_recommend_df.iterrows(), total=len(to_recommend_df), desc="Processing rows for method 04"):
        user_id = row['user_id']
        item_id = row['item_id']
        movie_title = items_df.loc[items_df['item_id'] == row['item_id'], 'movie_title'].values
        predicted_rating = 0
        item_ratings = 0

        user_movie_ratings = merged_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        user_specific_ratings_data = user_movie_ratings.loc[user_id]
        
        # Initialize and fit Nearest Neighbors model
        k = 20  # Number of neighbors to consider
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        nn_model.fit(user_movie_ratings)

        if item_id in user_movie_ratings and (user_specific_ratings_data[item_id] == 0.0 or rate_seen):
            # Find nearest neighbors for active user
            _, indices = nn_model.kneighbors(user_movie_ratings.loc[user_id].values.reshape(1, -1), n_neighbors=k+1)
            neighbor_user_ids = user_movie_ratings.iloc[indices[0][1:]]  # Exclude the active user itself

            item_ratings = neighbor_user_ids[item_id]

            item_ratings = item_ratings.mean()
            predicted_rating += item_ratings

        
        recommendations.append((user_id, item_id, movie_title[0], item_ratings))
        suggested_ratings.append(predicted_rating)

    return recommendations, suggested_ratings