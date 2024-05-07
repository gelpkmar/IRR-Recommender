from sklearn.metrics import mean_absolute_error, mean_squared_error
import math, helper
import numpy as np

# Calculate MAE
def calculate_mae(actual_ratings, predicted_ratings):
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    return mae

# Calculate RMSE
def calculate_mse(actual_ratings, predicted_ratings):
    rmse = math.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse

def return_evaluations(recommendations, suggested_ratings, method_name):

    pred_ratings_list = []
    actual_ratings_list = []
    ratings_df = helper.load_data()[1]

    for (user_id, movie_index, movie, score), rating in zip(recommendations, suggested_ratings):
        exists = (ratings_df['item_id'] == movie_index) & (ratings_df['user_id'] == user_id)
        if exists.any():
            actual_rating = ratings_df.loc[exists, 'rating'].iloc[0]
            pred_ratings_list.append(rating)
            actual_ratings_list.append(actual_rating)
        else:
            pass

    if len(actual_ratings_list) > 0:


        print(f"""
        {method_name}:
        #########################
        #                       #
        #       Evaluation      #
        #                       #
        #########################
        """)
        print(f"Mean Absolute Error (MAE): {calculate_mae(actual_ratings_list, pred_ratings_list):.2f}")
        print(f"Mean Squared Error (MSE): {calculate_mse(actual_ratings_list, pred_ratings_list):.2f}")
        print(f"Root Mean Squared Error (RMSE): {np.sqrt(calculate_mse(actual_ratings_list, pred_ratings_list)):.2f}")

    else:
        print(f"No evaluation possible for {method_name}. Probably missing reference values.")
