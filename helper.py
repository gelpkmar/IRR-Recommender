import nltk, string
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Load movie data from TSV files
def load_data():
    users_df = pd.read_csv('../movie_dataset/u.user.csv', delimiter=';')
    ratings_df = pd.read_csv('../movie_dataset/u.ratings.csv', delimiter=';')
    items_df = pd.read_csv('../movie_dataset/u.item.csv', delimiter=';')
    items_detailed_df = pd.read_csv('../movie_dataset/items.csv', delimiter=',')
    return users_df, ratings_df, items_df, items_detailed_df

# Merge movie data with ratings data using movie ID as the key
def retrieve_data_merged():
    result_dfs = load_data()
    users_df = result_dfs[0]
    ratings_df = result_dfs[1]
    items_df = result_dfs[3]
    merged_df_temp = pd.merge(users_df, ratings_df, on='user_id')
    merged_df = pd.merge(merged_df_temp, items_df, on='item_id', how='left')
    # print(merged_df)
    return merged_df

def retrieve_to_recommend_data(user_id=None, test_set=None, delim=','):
    result_df = pd.DataFrame()
    if test_set != None:
        result_df = pd.read_csv(test_set, delimiter = delim)
    else:
        print("""No test set chosen! 
              Please provide a .csv file in the main.py global var '_TEST_SET' 
              containing user-item pairs to compute the recommended ratings.
              """)

    return result_df[:2000]

def preprocess(text):
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        cleaned_tokens = [token for token in lemmatized_tokens if token not in string.punctuation]
        return cleaned_tokens
    else:
        return []

def prepare_user_profiles(extended=False):
    # Load data from CSV
    df = retrieve_data_merged()

    rating_weights = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    # Normalize numerical features
    numerical_features = df[['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                            'War', 'Western']]
    
    # Convert NaN values to 0
    numerical_features.fillna(0)

    # Calculate weighted average of numerical features for each user
    weighted_numerical_features = numerical_features.mul(df['rating'].map(rating_weights), axis=0)
    user_profile_numerical = weighted_numerical_features.groupby(df['user_id']).mean()
    
    if extended:
        # Initialize the WordNetLemmatizer
        # Combine numerical, text, and categorical features
        processed_summaries = df.groupby('user_id')['Summary'].apply(lambda x: x.apply(preprocess))
        processed_summaries.fillna(0)

        processed_cast = df.groupby('user_id')['Cast'].apply(lambda x: x.apply(preprocess))
        processed_cast.fillna(0)

        processed_director = df.groupby('user_id')['Director'].apply(lambda x: x.apply(preprocess))
        processed_director.fillna(0)

        user_profile_numerical.reset_index(drop=True, inplace=True)
        processed_summaries.reset_index(drop=True, inplace=True)
        processed_cast.reset_index(drop=True, inplace=True)
        processed_director.reset_index(drop=True, inplace=True)

        user_profile = pd.concat([user_profile_numerical, processed_summaries, processed_cast, processed_director], axis=1)


        return user_profile

    return user_profile_numerical

def map_similarity_to_rating(similarity_value):
    # Scale the similarity value from 0 to 1 to the range 1 to 5 using linear mapping
    min_similarity = 0
    max_similarity = 1
    min_rating = 1
    max_rating = 5

    # Calculate the mapped rating
    mapped_rating = min_rating + (max_rating - min_rating) * (similarity_value - min_similarity) / (max_similarity - min_similarity)
    return round(mapped_rating)

def prepare_item_item_similarity_index():
    # Load data from CSV
    movie_data = load_data()[3]

    # Select relevant features
    genre_features = movie_data[['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                'War', 'Western']]

    summary_features = movie_data['Summary']  # Assuming 'Summary' column contains textual information
    director_features = movie_data['Director']
    cast_features = movie_data['Cast']


    # Normalize the data (if necessary)
    # Normalize genre features using StandardScaler
    scaler = MinMaxScaler()
    normalized_genre_features = scaler.fit_transform(genre_features)

    # Preprocess textual data to handle missing values
    summary_features.fillna('', inplace=True)  # Replace NaN with empty string
    director_features.fillna('', inplace=True)
    cast_features.fillna('', inplace=True)

    # Process textual information using TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    summary_matrix = vectorizer.fit_transform(summary_features)
    cast_matrix = vectorizer.fit_transform(cast_features)
    director_matrix = vectorizer.fit_transform(director_features)

    # Calculate similarity based on genre profile
    genre_similarity_matrix = cosine_similarity(normalized_genre_features)

    # Calculate similarity based on textual information
    textual_summary_similarity_matrix = cosine_similarity(summary_matrix)
    textual_cast_similarity_matrix = cosine_similarity(cast_matrix)
    textual_director_similarity_matrix = cosine_similarity(director_matrix)

    # Combine similarity scores
    combined_similarity_matrix = 0.25 * genre_similarity_matrix + 0.25 * textual_summary_similarity_matrix + 0.25 * textual_cast_similarity_matrix + 0.25 * textual_director_similarity_matrix

    return combined_similarity_matrix

def export_results(recommendations, suggested_ratings, destination_path):
    if len(recommendations) < 1 or len(suggested_ratings) < 1:
        with open(f"../rating_results/{destination_path}", 'w', encoding='utf-8') as f:
            # Write header
            f.write("Unfortunately no ratings could be calculated.")
            f.write("Please try a different test set, user_id or data collection.")
    else:
        with open(f"../rating_results/{destination_path}", 'w', encoding='utf-8') as f:
            # Write header
            f.write("user_id,item_id,predicted_rating\n")
            # Write results
            for (user_id, movie_index, movie, score), rating in zip(recommendations, suggested_ratings):
                f.write(f"{user_id},{movie_index},{rating}\n")

        print(f"Results have been exported to ../rating_results/{destination_path}")