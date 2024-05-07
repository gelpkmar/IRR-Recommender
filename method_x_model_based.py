import helper
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# Merge ratings with movie data
data = helper.retrieve_data_merged()

# Feature Engineering
# Assuming movie features include genres, summaries, cast, director, etc.
# Perform text preprocessing on summaries
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
summary_features = vectorizer.fit_transform(data['Summary'].fillna(''))
cast_features = vectorizer.fit_transform(data['Cast'].fillna(''))
director_features = vectorizer.fit_transform(data['Director'].fillna(''))


# Combine features
movie_features = pd.concat([data[['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                  'War', 'Western']], pd.DataFrame(summary_features.toarray()), 
                                  pd.DataFrame(cast_features.toarray()), 
                                pd.DataFrame(director_features.toarray())], axis=1)

movie_features.columns = movie_features.columns.astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(movie_features, data['rating'], test_size=0.2, random_state=42)

# Train Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"""
Model-Based Recommendation:
#########################
#                       #
#       Evaluation      #
#                       #
#########################
""")

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Recommendation Generation
# For each user, predict ratings for unseen items using the trained model
# Sort items based on predicted ratings and recommend top-rated items to the user
# Implement recommendation generation logic based on predicted ratings
