# IRR Assignment02 - Movie Recommender System

|| |
|-----------|--------------|
|**Author:**| Marlon Gelpke|
|**Matriculation Number:**|15-532-849|
|**Date of Submission:**| 08.05.2024|

The submission to this project contains the following files:
```
.
└── IRR_assignment01_Marlon_Gelpke
    ├── code.zip
    ├── method_results/
    │   ├── method_01.csv
    │   ├── method_02.csv
    │   ├── method_03.csv
    │   ├── method_04.csv
    └── IRR_assignment02_approach.pdf
````

## 1. Introduction
This document describes the approach and setup of the Movie Recommender System that was developed in scope of assignment 02 of the University of Zurich Computer Science course "Introduction to Retrieval and Recommendation".

The exact requirements for this task can be accessed for authorized users at the following link: [Exercise 2 Movie Recommender System.pdf](https://lms.uzh.ch/auth/1%3A1%3A1067538524%3A2%3A0%3Aserv%3Ax%3A_csrf%3Af0459af1-67aa-4665-876d-2d149b5cdae1/Exercise%202%20Movie%20Recommender%20System.pdf).

The following materials were supplied to us:
- 5 documents in .csv format providing information about movies, users and users' ratings of movies.
- test.csv: New combination of userID's and movieID's of not yet performed ratings (i.e. movies not seen by a respective user)
- items.csv: Detailed information on movies incl. link to IMDB database and movie summary, cast, etc.

The goal was to follow the instructions and provide a solution in code for the following recommendation methods:

1. Content-based recommender algorithm based on movie genre
2. Content-based recommender algorithm based on more information than just movie genre
3. Item-based collaborative filtering recommender algorithm
4. User-based collaborative filtering recommender algorithm

## 2. Approach
The below describes the technical approach and outlines the design decisions taken to solve this project.

### 2.1 Technical Implementation
The technical execution of this project relies heavily on the following third party libraries:
- scikit-learn - TFIDF vectorization, cosine similarity, score normalization
- pandas - reading of the `.csv` files
- json - writing and reading of the inverse document index
- NLTK - all natural language processing (e.g., tokenization, lemmatization, calculation of similarity, etc.)

The project is spread over nine different Python files. A quick overview and explanation of the different files:
- `method_##_xyz.py` files: Files used to implement the individual solutions fo the methods outlined in chapter 1.
- `method_x_model_based.py`: ML-based algorithm to evaluate the manually implemented methode 01-04.
- `main.py`: Definition of global variables and main file to import all others.
- `helper.py`: Definition of helper functions ranging from loading of pandas data frames over text preprocessing to preparation of user profiles.
- `evaluate.py`: Program to evaluate the calculated movie ratings.
- `offline_preprocessing.py`: Script used to run the preprocessing functions to create the inverse document index to facilitate the simple full-text-search.

### 2.2 Recommendation Methods
Below the overview of a query process can be seen in a flow diagram. The steps in the individual methods are numbered and explained in more detail beneath the flow diagram incl. important technical design decisions.

<img src="data/IRR02_Pipeline.drawio.png" alt="IRR02 Pipeline" width="60%">

>Some performed actions in the code are considered trivial and therefore either not explained or not explained extensively.

>Similarity values were calculated using Cosine Similarity only to make them comparable and due to the fact that it returned the most stable results and the time restriction of this project prevented a more thorough study / conparison of other similarity functions use in code.

#### 2.2.0 Preprocessing and preparation
This stage fulfills a crucial role in preprocessing and preparing the available data so it can be used by the individual recommendation methods (1-4) described below.

##### 2.2.0.1 Offline Preprocessing
>Code to be found in `offline_preprocessing.py`

The largest part of the offline preprocessing is the harmonization of the data files to use common delimeters and to add the respective headers describing the column names as indicated in the README file. The result are three files with `.csv` suffix containing user information, movie item information and information on individual movie ratings.

>In this particular project, the common delimeter ";" was chosen for the output files. This was done as a work around to avoid the fact that some URL's contain commas within them.

##### 2.2.0.2 Creation of "Test Set"
>Code to be found in `offline_preprocessing.py` & `helper.py`

The word "Test Set" means the set of user-item combinations on which recommendations should be computed and presented. The result is a `.csv` file with a defined delimeter with 2 columns: `item_id` & `user_id` (see short snippet below).

```Text
user_id;item_id
1;1
1;2
1;3
1;4
... (additional line omitted on purpose)
```

The `The helper.retrieve_to_recommend_data()` function loads the user-items pairs as a Pandas DataFrame. user-items pairs are retrieved from the previously created test set `.csv`.

>In this project, the available data from the ratings.csv (first 2000 rows) was chosen to compute recommendations and to be able to evaluate the proposed ratings (see more below in chapter 3.2).

##### 2.2.0.3 Creation of user profiles normal
>Code to be found in `helper.py`

When the `extended` argument is set to its defaul value `False`, user profiles are created for every user taking into consideration the genre information only. The values in the user profiles for the respective genres are weighted in a linear manner with the following scale (key = user rating, value = factor to be used in multiplication): `rating_weights = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}`. for each user, a grouped entry is created with the average value of the weighted numerical features for each genre.

The result is a realistic profile per user based on the genre information of previously liked movies.

>For future use, ratings could be normalized to catch possible use of varying rating scales.

##### 2.2.0.4 Creation of user profiles extended
>Code to be found in `helper.py`

When the `extended` argument is set to `True`, user profiles are created for every user taking into consideration the the movie cast, director and summary information additionally to the existing genre information. For this textual information, word tokenization and lemmatization are applied as defined in the `helper.preprocess()` function, grouped by `user_id` and then concatenated together to make one large Pandas DataFrame out of the 4 individuals.


##### 2.2.0.5 Creation of Item-Item Similarity Matrix
>Code to be found in `helper.py`

The Item-Item Similarity Matrix is created to show the similarity between items based on genre, cast, movie summary, and director information taken from the available `.csv`files. With help of scikit-learn `TfidfVectorizer` for each type of information an item-item matrix is created.

The cosine similarity is calculated between the respective matrix and itself as can be seen from below's code snippet taken out of `helper.py`'s `prepare_item_item_similarity_index()` function:

```Python
# Calculate similarity based on genre profile
genre_similarity_matrix = cosine_similarity(normalized_genre_features)

# Calculate similarity based on textual information
textual_summary_similarity_matrix = cosine_similarity(summary_matrix)
textual_cast_similarity_matrix = cosine_similarity(cast_matrix)
textual_director_similarity_matrix = cosine_similarity(director_matrix)
```

To combine the individuals into one large item-item matrix, they are added using an equal weight of `0.25`.

#### 2.2.1 Method 1: Content-based recommender algorithm based on movie genre
For every row in the previously created test set (see chapter 2.2.0.2), the Cosine Similarity between the item's genre information (taken from the preprocessed items.csv in chapter 2.2.0.1) and the vectorized user profile (see chapters 2.2.0.3 & 4) are calculated as follows:

```Python
# Compute cosine similarity between the user's profile and all movies based on genre
cos_similarity_value = cosine_similarity(user_profile_vec, items_df.loc[items_df['item_id'] == row['item_id'], ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].values)
```

#### 2.2.2 Method 2: Content-based recommender algorithm based on more information than just movie genre
For every row in the previously created test set (see chapter 2.2.0.2), the Cosine Similarity is calculated between the extended user profile (see chapter 2.2.0.4) and the respective preprocessed summary, cast or director information.

The overall similarity is calculated as follows:
```Python
# Calculating the overall item-user similarity with equal weights.
overall_similarity = (0.25 * genre_similarity) + (0.25 * summary_similarity) + (0.25 * cast_similarity) + (0.25 * director_similarity)
```

#### 2.2.3 Method 3: Item-based collaborative filtering recommender algorithm
For every row in the previously created test set (see chapter 2.2.0.2) the previously rated items (of that user) are retrieved and their similarity to the "to be rated item" is calculated weigthed using the score given (e.g., if an item is very similar to another item that was rated with a score of 5, its similarity value is multiplied by 5):

```Python
# Initialize similarity score for the current item
item_similarity_score = 0
        
# Calculate similarity score between the current item and each rated item
for rated_item in rated_items_ids:
    # Accumulate similarity scores multiplied with rating of the similar item
    item_similarity_score += (items_similarity_matrix[rated_item][item_id] * merged_df.loc[(merged_df['user_id'] == user_id) & (merged_df['item_id'] == rated_item), 'rating'].values)
```

The average rating of the accumulated similarity score is returned as a recommended rating for the not yet seen movie.

#### 2.2.4 Method 4: User-based collaborative filtering recommender algorithm
For every row in the previously created test set (see chapter 2.2.0.2) the nearest neighbor cosine similarity ist being calculate to determine recommended ratings from similar user's previous ratings. As the number of neighbors 20 was deemed appropriate as the analysis of the MovieLens dataset (as presented in IRR class) had shown.

## 3. Testing and Interpretation of Recommendation Methods Performance

### 3.1 Calling the Program
When trying this recommendation algorithm it is important to understand how this program can or should be called.
The `main.py` file defines the most important environment variables that can be used to alter the behavior of the program:

- `_TEST_SET`: This is where the `.csv` file with the user_id / item_id combinations for which recommendations shall be calculated must be defined using a valid path. It is important thath the `.csv` file has the format shown in chapter 2.2.0.2 (delimeters may change but must be defined in `_TO_RECOMMEND_DF` below).
- `_USER_PROFILES_NORMALIZED`: Function call to create genre-based user profiles (only added for future extensibility)
- `_USER_PROFILES_EXTENDED_NORMALIZED`: Function call to create extended user profiles (only added for future extensibility)
- `_TO_RECOMMEND_DF` = Function call to create .csv file for which user_id/ item_id combinations recommendations shall be computed (delimeter may be changed here)
- `_ITEMS_DF`: Place to define which items data shall be used (only added for future extensibility)
- `_ITEMS_EXTENDED_DF`: Place to define which items data shall be used (only added for future extensibility)
- `_ITEMS_SIMILARITY_MATRIX`: Function call to create item-item matrix (only added for future extensibility)
- `_RATE_ALREADY_SEEN`: Either `True` or `False` to define whether already-rated items shall be rated again (for testing purposes).

>Additionally, after each execution of a recommendation method, a call to an evaluation method is done (see code below). This line will not return an evaluation if for the recommended user-movie pair, no rating can be found in `u.ratings.csv`. Instead a warning is issued. This could be improved in the future.

### 3.2 Evaluation of algorithm performance
The evaluation of the correctness and relevance of the methods results was done on a subset of already rated movies from the provided file `u.ratings.csv`. A test set was created as described previously in chapter 2.2.0.2.

#### 3.2.1 Mapping Similarity values to movie ratings
>Code can be found in `helper.py`

In order to evaluate the performance, the calculated similarity values must be converted into valid movie ratings (from 1 to 5). This is doen through the `helper.map_similarity_to_rating()` function.
Similarity values are considered to range from 0 to 1 (practice has found that no completely opposite items, and therefore no negative values for cosine similarity, exist in this example) and minimum rating is considered to be 1 (as 0 is considered unrated / irrelevant), while maximum rating is 5.

The mapping is done with the following formula:
```Python
mapped_rating = min_rating + (max_rating - min_rating) * (similarity_value - min_similarity) / (max_similarity - min_similarity)
```

#### 3.2.1 Performance Metrics and Outcome
>Code to be found in `evaluate.py`

With the similarity value mapped to a rating from 1 to 5, the algorithm performance can be calculated using a subset of the available `u.ratings.csv` file (the first 2000 rows).
For evaluation the assignment defined that RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) must be used. Additionally, MSE (RMSE without root) was chosen as an evaluation metric.

The results are as follows:
- Method 1:
    - Mean Absolute Error (MAE): 1.23
    - Mean Squared Error (MSE): 1.56
    - Root Mean Squared Error (RMSE): 1.25
- Method 2:
    - Mean Absolute Error (MAE): 2.21
    - Mean Squared Error (MSE): 2.51
    - Root Mean Squared Error (RMSE): 1.58
- Method 3:
    - Mean Absolute Error (MAE): 1.85
    - Mean Squared Error (MSE): 2.12
    - Root Mean Squared Error (RMSE): 1.46
- Method 4:
    - Mean Absolute Error (MAE): 1.74
    - Mean Squared Error (MSE): 2.08
    - Root Mean Squared Error (RMSE): 1.44

Judging from the above results, it seems that the simple genre-based method is the most reliable and correct. It seems that adding additional information, such as movie summary, cast, and director doesn't improve similarity ratings, but rather the opposite. This could have multiple causes, and would have to be analyzed / improved further.

Out of curiosity and for further evaluation a model-based method was implemented (see `method_x_model_based.py`). In this example, scikit-learns `RandomForestRegressor` model was used to evaluate the performance with the following outcome:
- Mean Absolute Error (MAE): 0.82
- Mean Squared Error (RMSE): 1.06
- Root Mean Squared Error (RMSE): 1.03

To conclude, 

## 4 Known Bugs

### 4.1 Method 2
When running method 2 on the whole test set defined in `test.csv`, an error is thrown at item 9420/9430: `ValueError: Input contains NaN.`


