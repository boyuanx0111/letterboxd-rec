import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

#RUN FROM ROOT DIR

model = load_model('models/two_tower.keras')

user_scaler = joblib.load('models/scaler_users.pkl')
movie_scaler = joblib.load('models/scaler_movies.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')

movie_catalogue = pd.read_parquet('data/movie_catalogue.parquet')

metadata_cols = [
    "movieId",
    "title_clean",
    "Release_Year"
]
X_movies = movie_catalogue.drop(
    columns=metadata_cols
)

X_movies.columns = ["movie_" + col for col in X_movies.columns]

def score_movies(X_user):
    """
    Score movies for a given user using the two-tower model.
    
    Parameters:
    - user_features: 2D array of shape (1, num_user_features) numpy array of user features.
    - movie_features: A 2D numpy array of movie features.
    
    Returns:
    - 2D array of shape (1, num_movies) numpy array of predicted scores for each movie.
    """
    # Scale user features
    X_user = user_scaler.transform(X_user)
    
    # Scale movie features
    X_movie = movie_scaler.transform(X_movies)

    # Repeat the user once for every movie
    X_user_matrix = np.repeat(
        X_user,
        len(X_movie),
        axis=0
    )
    
    # Predict scores
    predictions = model.predict([X_user_matrix, X_movie])
    
    # Inverse scale the predictions
    predictions_unscaled = y_scaler.inverse_transform(predictions)
    
    return predictions_unscaled.flatten()

def recommend_movies(X_user, top_n, watched_movie_ids):
    """
    Recommend top N movies for a given user based on predicted ratings.
    Parameters: 
    - X_user: 2D array of shape (1, num_user_features) numpy array of user features.
    - top_n: Number of top recommendations to return.
    - watched_movie_ids: List of movie IDs that the user has already watched.
    """
    # Score all movies
    scores = score_movies(X_user)

    results = movie_catalogue[metadata_cols].copy()

    results['predicted_rating'] = scores
    
    #filter out watched movies
    results = results[~results['movieId'].isin(watched_movie_ids)]

    results = results.sort_values(by='predicted_rating', ascending=False)
    
    # Return the top N movies
    return results.head(top_n)

#Test with temp user
#Prediction
test_user = pd.read_csv("data/test_user.csv")

#manually set now, but will be parsed auto from use input
watched_movie_ids = [7460, 2141, 2908, 185, 36, 2064, 165, 44889, 2009, 2959, 2174, 337, 1094, 1271, 1193, 4993, 3157, 30793, 50872, 48593, 788, 2710, 2137, 2291, 1411, 917, 2746, 364, 1073, 971, 2080, 24, 5218, 1994, 55442, 141, 208, 5816, 480, 47099, 2761, 150, 1232, 1345, 473, 8360, 5617, 4896, 2138, 6881, 45208, 1721, 3793, 5445, 1307, 1097, 1380, 2362, 2161, 318, 53894, 2123, 8228, 587, 2139, 38600, 6502, 2082, 2081, 1037, 4017, 1409, 47999, 3418, 442, 3969, 5308, 56152, 1240, 586, 37240, 5669, 56782, 1923, 53121, 1270, 1682, 48394, 594, 2142, 1015, 34, 48780, 45431, 508, 203, 235, 2617, 377, 2871, 597, 2085, 2399, 1393, 4306, 2797, 1265, 1206, 454, 56757, 2571, 593, 1921, 4326, 344, 750, 1282, 3243, 8950, 356, 50, 4878, 4027, 1214, 3844, 19, 2762, 509, 3147, 1033, 223, 919, 7153, 3401, 62, 2313, 1028, 253, 410, 3504, 2804, 589, 4018, 26680, 902, 1276, 6934, 2140, 968, 45210, 1784, 1291, 7361, 380, 3672, 6539, 55814, 5505, 1997, 1198, 1231, 4039, 49932, 231, 588, 1012, 296, 4935, 592, 37729, 8582, 26258, 266, 48696, 6373, 923, 6909, 45928, 2297, 45517, 5444, 44761, 65, 247, 714, 4963, 104, 7020, 4973, 54259, 56367, 2431, 3676, 317, 1207, 95, 56171, 542, 110, 367, 3863, 912, 2076, 2455, 2000, 48414, 783, 780, 8507, 590, 8784, 1259, 45720, 1104, 5958, 31, 5992, 59615, 2067, 6223, 3949, 3751, 1580, 1258, 2115, 52328, 595, 53460, 551, 56174, 1968, 2683, 1151, 6365, 8622, 2028, 8464, 520, 56915, 5349, 1199, 6222, 4008, 2987, 1036, 48698, 55820, 6291, 8961, 2087, 48, 920, 2502, 1021, 3186, 58299, 1032, 3108, 153, 1022, 1029, 3034, 260, 3107, 1, 1961, 4226, 2858, 233, 539, 54272, 6377, 596, 2054, 497, 1298, 55999, 2916, 1441, 2716, 4995, 1620, 3798, 491, 527, 1196, 2846]
recommendations = recommend_movies(test_user, 20, watched_movie_ids)
print("Recommendations: \n", recommendations)

#Previous
ratings = pd.read_csv("data/ml-32m/ratings.csv")

user_ratings = ratings[
    ratings["userId"] == 103752 #test_user_id from 03. 32M-MovieLens_exploration.ipynb
]

user_ratings = user_ratings.merge(
    movie_catalogue[
        ["movieId", "title_clean", "Release_Year"]
    ],
    on="movieId",
    how="left"
)

top_rated = (
    user_ratings
    .sort_values("rating", ascending=False)
    .head(20)
)

print("\nPrevious:")
print(
    top_rated[
        ["movieId", "title_clean", "Release_Year", "rating"]
    ].to_string(index=False)
)

#TODO:
#Proper eval
#User_features transformation to feed into pipeline
#Connect frontend
#Pipeline API call and initial filter data pipeline 