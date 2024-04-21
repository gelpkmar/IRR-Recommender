import pandas as pd

# Load movie data from TSV files
def load_data():
    users_df = pd.read_csv('../movie_dataset/u.user.csv', delimiter=';')
    ratings_df = pd.read_csv('../movie_dataset/u.ratings.csv', delimiter=';')
    items_df = pd.read_csv('../movie_dataset/u.item.csv', delimiter=';')
    return users_df, ratings_df, items_df

# Merge movie data with ratings data using movie ID as the key
def retrieve_data_merged():
    result_dfs = load_data()
    users_df = result_dfs[0]
    ratings_df = result_dfs[1]
    items_df = result_dfs[2]
    merged_df_temp = pd.merge(users_df, ratings_df, on='user_id')
    merged_df = pd.merge(merged_df_temp, items_df, on='item_id', how='left')
    return merged_df