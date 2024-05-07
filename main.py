import helper, evaluate, method_01_genre, method_02_content_extended, method_03_collab_item, method_04_collab_user

# Global VARS
_TEST_SET = '../movie_dataset/selected_ratings.csv'
# _TEST_SET = '../movie_dataset/test.csv'
_USER_PROFILES_NORMALIZED = helper.prepare_user_profiles()
_USER_PROFILES_EXTENDED_NORMALIZED = helper.prepare_user_profiles(extended=True)
_TO_RECOMMEND_DF = helper.retrieve_to_recommend_data(test_set=_TEST_SET, delim=',')
_ITEMS_DF = helper.load_data()[2]
_ITEMS_EXTENDED_DF = helper.load_data()[3]
_ITEMS_SIMILARITY_MATRIX = helper.prepare_item_item_similarity_index()
_RATE_ALREADY_SEEN = True

# Execute Method 01 - Content-Based (Genre only) Recommendation
recommendations, suggested_ratings = method_01_genre.recommend_movies_genre(
    _USER_PROFILES_NORMALIZED,
    _TO_RECOMMEND_DF,
    _ITEMS_DF,
    )
# Execute Proposed Rating Evaluation
## sorted_ratings_df = helper.retrieve_data_merged.loc[(_MERGED_DF["user_id"] == 4) & (~_MERGED_DF["rating"].isna()), ["item_id", "rating"]].sort_values(by="item_id", ascending=True)

## Print evaluation conclusion MAE and RMSE scores for the above method.
evaluate.return_evaluations(recommendations, suggested_ratings, "Method 01")

## Export results to .csv
helper.export_results(recommendations, suggested_ratings, "method_01.csv")

# Execute Method 02 - Content-Based (extended movie data) Recommendation
recommendations, suggested_ratings = method_02_content_extended.recommend_movies_extended(
    _USER_PROFILES_EXTENDED_NORMALIZED,
    _TO_RECOMMEND_DF,
    _ITEMS_EXTENDED_DF,
    )
# Execute Proposed Rating Evaluation
# sorted_ratings_df = helper.retrieve_data_merged.loc[(_MERGED_DF["user_id"] == 4) & (~_MERGED_DF["rating"].isna()), ["item_id", "rating"]].sort_values(by="item_id", ascending=True)

# Print evaluation conclusion MAE and RMSE scores for the above method.
evaluate.return_evaluations(recommendations, suggested_ratings, "Method 02")

# Export results to .csv
helper.export_results(recommendations, suggested_ratings, "method_02.csv")

# Execute Method 03 - Item-based collaborative Filtering Recommender
recommendations, suggested_ratings = method_03_collab_item.recommend_item_based_collaborative_filtering(
    _ITEMS_SIMILARITY_MATRIX,
    _TO_RECOMMEND_DF,
    _ITEMS_DF
    )

## Print evaluation conclusion MAE and RMSE scores for the above method.
evaluate.return_evaluations(recommendations, suggested_ratings, "Method 03")

## Export results to .csv
helper.export_results(recommendations, suggested_ratings, "method_03.csv")

# Execute Method 04 - User-based collaborative Filtering Recommender
recommendations, suggested_ratings = method_04_collab_user.recommend_user_based_collaborative_filtering(
    _TO_RECOMMEND_DF,
    _ITEMS_DF,
    _RATE_ALREADY_SEEN
    )

## Print evaluation conclusion MAE and RMSE scores for the above method.
evaluate.return_evaluations(recommendations, suggested_ratings, "Method 04")

## Export results to .csv
helper.export_results(recommendations, suggested_ratings, "method_04.csv")