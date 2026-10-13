from typing import cast

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# RUN FROM ROOT DIR

model = load_model("models/two_tower.keras")

user_scaler = joblib.load("models/scaler_users.pkl")
movie_scaler = joblib.load("models/scaler_movies.pkl")
y_scaler = joblib.load("models/scaler_y.pkl")

# from 03
movie_catalogue: pd.DataFrame = pd.read_parquet("data/movie_catalogue.parquet")

metadata_cols = ["movieId", "title_clean", "Release_Year"]
X_movies = movie_catalogue.drop(columns=metadata_cols)

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
    X_user_matrix = np.repeat(X_user, len(X_movie), axis=0)

    # Predict scores
    predictions = model.predict([X_user_matrix, X_movie])

    # Inverse scale the predictions
    predictions_unscaled = y_scaler.inverse_transform(predictions)

    return predictions_unscaled.flatten()


# not part of current pipeline, but can be used to recommend movies for a given user
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

    results = cast(pd.DataFrame, movie_catalogue.loc[:, metadata_cols].copy())

    results["predicted_rating"] = scores

    # filter out watched movies
    # current pipeline not taking from watched movies
    results = results[~results["movieId"].isin(watched_movie_ids)]

    results = cast(pd.DataFrame, results).sort_values(by="predicted_rating", ascending=False)

    # Return the top N movies
    return results.head(top_n)


# #Test with temp user
# #Prediction
# test_user = pd.read_csv("data/test_user.csv")

# #manually set now, but will be parsed auto from user input
# Example watched movie IDs can be passed to recommend_movies when testing.
# recommendations = recommend_movies(test_user, 20, watched_movie_ids)
# print("Recommendations: \n", recommendations)

# Generate predictions for held-out test interactions for NDCG evaluation

# from 03, combined and enriched table
test_data = pd.read_csv("data/test_data.csv")

user_cols = np.load("data/user_cols.npy")
movie_cols = np.load("data/movie_cols.npy")

# get test features
X_user_test = test_data[user_cols]
X_movie_test = test_data[movie_cols]

# scale
X_user_test_scaled = user_scaler.transform(X_user_test)
X_movie_test_scaled = movie_scaler.transform(X_movie_test)

# predict all test interactions
predictions = model.predict([X_user_test_scaled, X_movie_test_scaled])

# convert predictions back to rating scale
predictions = y_scaler.inverse_transform(predictions).flatten()

# loading xgboost features
candidate_preds: pd.DataFrame = test_data.loc[
    :,
    [
        "userId",
        "movieId",
        "rating",
        "user_avg_rating",
        "user_std_rating",
        "user_count_rating",
        "movie_Release_Year",
        "genre_affinity",
        "decade_affinity",
        "embedding_cosine_similarity",
    ],
].copy()

candidate_preds["predicted_rating"] = predictions

# export for NDCG notebook
candidate_preds.to_parquet("data/candidate_preds.parquet", index=False)
print(candidate_preds.head())
print("Predictions:", len(candidate_preds))

# rank and group by user
candidate_preds_ranked = candidate_preds.sort_values(
    ["userId", "predicted_rating"], ascending=[True, False]
).reset_index(drop=True)
candidate_preds_ranked["user_rank"] = (
    candidate_preds_ranked.groupby("userId")["predicted_rating"]
    .rank(method="first", ascending=False)
    .astype(int)
)
candidate_preds_ranked["z_score"] = (
    candidate_preds_ranked["predicted_rating"]
    - candidate_preds_ranked.groupby("userId")["predicted_rating"].transform("mean")
) / candidate_preds_ranked.groupby("userId")["predicted_rating"].transform("std")

print(candidate_preds_ranked)
print(candidate_preds_ranked["user_rank"].describe())
print(candidate_preds_ranked.columns.tolist())

# export for xgboost
candidate_preds_ranked.to_parquet("data/test_candidate_preds_ranked.parquet", index=False)

# TODO:
# XGboost remaining features
# Collab Filtering Hybrid (AP)
