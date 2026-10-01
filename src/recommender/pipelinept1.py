#for xgboost, fixed retrieval pipeline

import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

#RUN FROM ROOT DIR

model = load_model('models/two_tower.keras')

user_scaler = joblib.load('models/scaler_users.pkl')
movie_scaler = joblib.load('models/scaler_movies.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')

# from 03, combined and enriched table 
train_data = pd.read_csv("data/training_data.csv")

user_cols = np.load(
    "data/user_cols.npy"
)
movie_cols = np.load(
    "data/movie_cols.npy"
)

# get train features
X_user_train = train_data[user_cols]
X_movie_train = train_data[movie_cols]

# scale 
X_user_train_scaled = user_scaler.transform(
    X_user_train
)
X_movie_train_scaled = movie_scaler.transform(
    X_movie_train
)


# predict all train interactions
predictions = model.predict(
    [
        X_user_train_scaled,
        X_movie_train_scaled
    ]
)

# convert predictions back to rating scale
predictions = y_scaler.inverse_transform(
    predictions
).flatten()

# loading xgboost features 
candidate_preds = train_data[
    ["userId", "movieId", "rating","user_avg_rating", "user_std_rating", "user_count_rating", "movie_Release_Year", "genre_affinity", "decade_affinity"]
].copy()

candidate_preds["predicted_rating"] = predictions

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
    "data/train_candidate_preds_ranked.parquet",
    index=False
)