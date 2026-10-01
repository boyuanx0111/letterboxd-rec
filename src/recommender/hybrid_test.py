#test hybrid pipeline 
# tt-nn filters top 300 per user, then xgboost ranks and filters top 10 per user
# because only using seen movies to test ndcg@20 improvements for now, ranks all user's seen movies in test set

import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# process test data to tt-nn input
# movieId, userId, rating, timestamp
filtered_ratings2 = pd.read_csv("data/filtered_ratings2.csv")

# complete movie embeddings with movieId as index
movie_embeddings = pd.read_parquet("data/movie_embeddings.parquet")

user_scaler = joblib.load('models/scaler_users.pkl')
movie_scaler = joblib.load('models/scaler_movies.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')

# from 03, combined and enriched table 
test_data = pd.read_csv("data/test_data.csv")

user_cols = np.load(
    "data/user_cols.npy"
)
movie_cols = np.load(
    "data/movie_cols.npy"
)

# get test features
X_user_test = test_data[user_cols]
X_movie_test = test_data[movie_cols]

# scale 
X_user_test_scaled = user_scaler.transform(
    X_user_test
)
X_movie_test_scaled = movie_scaler.transform(
    X_movie_test
)

# predict all test interactions
predictions = model.predict(
    [
        X_user_test_scaled,
        X_movie_test_scaled
    ]
)

# convert predictions back to rating scale
predictions = y_scaler.inverse_transform(
    predictions
).flatten()




# loading xgboost features 
candidate_preds = test_data[
    ["userId", "movieId", "rating","user_avg_rating", "user_std_rating", "user_count_rating", "movie_Release_Year", "genre_affinity", "decade_affinity"]
].copy()

candidate_preds["predicted_rating"] = predictions

# export for NDCG notebook
candidate_preds.to_parquet(
    "data/candidate_preds.parquet",
    index=False
)
print(candidate_preds.head())
print("Predictions:", len(candidate_preds))

# rank and group by user
candidate_preds_ranked = (
    candidate_preds
    .sort_values(
        ["userId", "predicted_rating"],
        ascending=[True, False]
    )
    .reset_index(drop=True)
)
candidate_preds_ranked["user_rank"] = (
    candidate_preds_ranked
    .groupby("userId")["predicted_rating"]
    .rank(method="first", ascending=False)
    .astype(int)
)
candidate_preds_ranked["z_score"] = (
    candidate_preds_ranked["predicted_rating"] - candidate_preds_ranked.groupby("userId")["predicted_rating"].transform("mean")
) / candidate_preds_ranked.groupby("userId")["predicted_rating"].transform("std")

print(candidate_preds_ranked)
print(candidate_preds_ranked["user_rank"].describe())
print(candidate_preds_ranked.columns.tolist())

# export for xgboost
candidate_preds_ranked.to_parquet(
    "data/test_candidate_preds_ranked.parquet",
    index=False
)