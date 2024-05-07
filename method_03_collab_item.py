import helper
from tqdm import tqdm

def recommend_item_based_collaborative_filtering(items_similarity_matrix, to_recommend_df, items_df):
    recommendations = []
    suggested_ratings = []
    merged_df = helper.retrieve_data_merged()

    for index, row in tqdm(to_recommend_df.iterrows(), total=len(to_recommend_df), desc="Processing rows for method 03"):
        user_id = row['user_id']
        item_id = row['item_id']
        
        # Get user's rated items and corresponding ratings
        rated_items_ids = merged_df.loc[merged_df['user_id'] == user_id, 'item_id'].values
        
        # Initialize similarity score for the current item
        item_similarity_score = 0
        
        # Calculate similarity score between the current item and each rated item
        for rated_item in rated_items_ids:
            # Accumulate similarity scores multiplied with rating of the similar item
            # (e.g., if an item is very similar to another item that was rated with a score of 5, its similarity value is multiplied by 5).
            item_similarity_score += (items_similarity_matrix[rated_item][item_id] * merged_df.loc[(merged_df['user_id'] == user_id) & (merged_df['item_id'] == rated_item), 'rating'].values)
        
        # Calculate average similarity score
        if len(rated_items_ids) > 0:
            average_similarity_score = item_similarity_score / len(rated_items_ids)
        else:
            average_similarity_score = 0
        
        movie_title = items_df.loc[items_df['item_id'] == row['item_id'], 'movie_title'].values
        
        # Store the recommendation and its similarity score
        recommendations.append((user_id, item_id, movie_title[0], average_similarity_score))
        suggested_ratings.append(helper.map_similarity_to_rating(average_similarity_score[0]))
        # similarity_scores.append(average_similarity_score)
    
    # Sort recommendations based on similarity score in descending order
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    return recommendations, suggested_ratings

