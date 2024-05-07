import helper
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def recommend_movies_extended(user_profiles, to_recommend_df, items_df):
    recommendations = []
    suggested_ratings = []

    # print(user_profiles)

    for index, row in tqdm(to_recommend_df.iterrows(), total=len(to_recommend_df), desc="Processing rows for method 02"):
        # # Get the user's profile
        user_id = row['user_id']
        item_id = row['item_id']
        movie_title = items_df.loc[items_df['item_id'] == item_id, 'movie_title'].values
        missed_entries = {}

        genres_to_retrieve = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy','Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

        # # Numerical features (genres)
        # genre_similarity = cosine_similarity(user_profiles.loc[user_id, genres_to_retrieve].values.reshape(1, -1),
        #                                         items_df.loc[items_df['item_id'] == item_id, ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        #                                             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        #                                             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
        #                                             'War', 'Western']].values.reshape(1, -1))[0][0]
        
        user_profile_vec = user_profiles.loc[row['user_id'], genres_to_retrieve].values.reshape(1, -1)
        items_df.fillna(value=0, inplace=True)  # Replace NaN with 0

        
        # Compute cosine similarity between the user's profile and all movies based on genre
        cos_similarity_value = cosine_similarity(user_profile_vec, items_df.loc[items_df['item_id'] == row['item_id'], ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                                                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                                                        'War', 'Western']].values)
        
        if len(cos_similarity_value) == 0:
            missed_entries[user_id] = [item_id]
        else:
            cos_similarity_value = cos_similarity_value[0][0]
        
        # Text features (summary)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        item_summary = helper.preprocess(items_df.loc[items_df['item_id'] == item_id, 'Summary'].values[0])
        item_cast = helper.preprocess(items_df.loc[items_df['item_id'] == item_id, 'Cast'].values[0])
        item_director = helper.preprocess(items_df.loc[items_df['item_id'] == item_id, 'Director'].values[0])
        
        user_director = user_profiles.loc[user_id, 'Director']
        user_cast = user_profiles.loc[user_id, 'Cast']
        user_profile_summary = user_profiles.loc[user_id, 'Summary']
        if user_profile_summary != [] and item_summary != []:
            tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(item_summary), ' '.join(user_profile_summary)])

            summary_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        else:
            summary_similarity = 0.0


        if user_cast != [] and item_cast != []:
            tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(item_cast), ' '.join(user_cast)])

            cast_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        else:
            cast_similarity = 0.0


        if user_cast != [] and item_cast != []:
            tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(item_director), ' '.join(user_director)])

            director_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        else:
            director_similarity = 0.0

        # Calculating the overall item-user similarity with equal weights.
        overall_similarity = (0.25 * cos_similarity_value) + (0.25 * summary_similarity) + (0.25 * cast_similarity) + (0.25 * director_similarity)
        
        recommendations.append((row['user_id'], row['item_id'], movie_title[0], overall_similarity))
        suggested_ratings.append(helper.map_similarity_to_rating(overall_similarity))

    if len(missed_entries.keys()) > 0:
        print("The following entries were not evaluated due to some error:")
        for key, value in missed_entries:
            print(f"User {key}: Item {value}")

    return recommendations, suggested_ratings
