import helper
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Define a function to lemmatize tokens
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

# Define a custom tokenizer that combines tokenization and lemmatization
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    return lemmatize_tokens(tokens)

def recommend_movies_extended(user_profiles, to_recommend_df, items_df):
    recommendations = []
    suggested_ratings = []

    # print(user_profiles)

    for index, row in tqdm(to_recommend_df.iterrows(), total=len(to_recommend_df), desc="Processing rows for method 02"):
        # # Get the user's profile
        user_id = row['user_id']
        item_id = row['item_id']
        movie_title = items_df.loc[items_df['item_id'] == item_id, 'movie_title'].values
        user_profile_vec = user_profiles.loc[user_id].values.reshape(1, -1)

        genres_to_retrieve = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy','Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western']

        # print(user_profiles.loc[4])

        # Numerical features (genres)
        numerical_similarity = cosine_similarity(user_profiles.loc[user_id, genres_to_retrieve].values.reshape(1, -1),
                                                items_df.loc[items_df['item_id'] == item_id, ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                                    'War', 'Western']].values.reshape(1, -1))[0][0]
        
        # Text features (summary)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        item_summary = helper.preprocess(items_df.loc[items_df['item_id'] == item_id, 'Summary'].values[0])
        user_profile_summary = user_profiles.loc[user_id, 'Summary']
        if user_profile_summary != [] and item_summary != []:
            tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(item_summary), ' '.join(user_profile_summary)])

            text_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        else:
            text_similarity = 0.0

        # # Categorical features (cast, director)
        # user_cast = set(user_profile_vec['Cast'].split())
        # item_cast = set(items_df.loc[items_df['item_id'] == row['item_id'], 'Cast'].split())
        # cast_similarity = jaccard_score(user_cast, item_cast)

        # user_director = set(user_profile_vec['Director'].split())
        # item_director = set(items_df.loc[items_df['item_id'] == row['item_id'], 'Director'].split())
        # director_similarity = jaccard_score(user_director, item_director)

        # Combine similarity scores with weights
        # overall_similarity = (0.6 * numerical_similarity) + (0.3 * text_similarity) + (0.1 * (cast_similarity + director_similarity) / 2)

        overall_similarity = (0.5 * numerical_similarity) + (0.5 * text_similarity)
        
        recommendations.append((row['user_id'], row['item_id'], movie_title[0], overall_similarity))
        suggested_ratings.append(helper.map_similarity_to_rating(overall_similarity))

    return recommendations, suggested_ratings
