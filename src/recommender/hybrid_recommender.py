# hybrid rec pipeline 
# tt-nn filters top 300 per user, then xgboost ranks and filters top 10
# input parsed user ratings (tmdbId, rating) to build user features and embeddings, then score all movies and rank with xgboost

import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# run from this dir
from pipelinept1 import initial_rank
from build_user_features import build_user_features
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import ndcg_score

# preprocess test data to tt-nn input

# PREPROCESS USER INPUT TO THIS (tmdbId, rating) !!
# PASS PARSED HERE 
user_ratings = # user_input parsed into suitable form 

# import movie feature frame
movie_feature_frame = pd.read_parquet("data/movie_feature_frame.parquet")

# complete movie embeddings with tmdbId as index
movie_embeddings = pd.read_parquet("data/movie_embeddings_nn.parquet")

# users feature frame from USER INPUT
user_feature_frame = build_user_features(user_ratings, movie_embeddings)

# user embeddings from filtered sample ratings set 2
user_embeddings = pd.read_parquet("data/user_embeddings_nn.parquet")

user_scaler = joblib.load('models/scaler_users.pkl')
movie_scaler = joblib.load('models/scaler_movies.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')

# create user embeddings from user feature frame using userNN model
user_feature_scaled = user_scaler.transform(user_feature_frame)
userNN = load_model('../models/userNN.keras')
user_embeddings = userNN.predict(user_feature_scaled)
user_embeddings_df = pd.DataFrame(user_embeddings, index=user_feature_frame.index)

# predict
# movie_embeddings - (87585, 32)
# user_embeddings - 
user_vecs = user_embeddings_df.to_numpy()
# create unwatched movie list of tmdbIds for user to extract corresponding movie embeddings
movie_vecs = movie_embeddings[~movie_embeddings.index.isin(user_ratings["tmdbId"])]

# normalise before dot product
user_vecs = tf.keras.ops.normalize(user_vecs, axis=1)
movie_vecs = tf.keras.ops.normalize(movie_vecs, axis=1)

# vectorised dot product to get predictions for all user-movie pairs in the test set
predictions_scaled = np.sum(user_vecs * movie_vecs, axis=1)
# convert predictions back to rating scale
predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

#build candidate_preds dataframe for xgboost input
candidate_preds = pd.DataFrame({
    "tmdbId": user_ratings["tmdbId"],
    "rating": user_ratings["rating"],
    # user_avg rating
    # user_std rating
    # user_count rating
    # movie_Release_Year
    # genre_affinity
    # decade_affinity
    "predicted_rating": predictions
})
candidate_preds = candidate_preds.merge(user_feature_frame[["user_avg_rating", "user_std_rating", "user_count_rating"]], how="left")
candidate_preds = candidate_preds.merge(movie_feature_frame[["movie_Release_Year"]], on="tmdbId", how="left")

# calc genre affinity for each user-movie pair
genre_columns = pd.read_csv("data/genre_columns.csv").squeeze().tolist()
user_genre_cols = [f"user_{genre}" for genre in genre_columns]
movie_genre_cols = [f"movie_{genre}" for genre in genre_columns]
user_genre_values = (
    user_feature_frame
    .loc[user_genre_cols]
    .fillna(0)
    .to_numpy()
)
movie_genre_values = (
    movie_feature_frame
    .loc[candidate_preds["tmdbId"], movie_genre_cols]
    .fillna(0)
    .to_numpy()
)
genre_counts = movie_genre_values.sum(axis=1)
candidate_preds["genre_affinity"] = np.divide(
    (user_genre_values * movie_genre_values).sum(axis=1), 
    genre_counts,
    out=np.full(len(candidate_preds), np.nan),
    where=genre_counts != 0
)

# calc decade affinity for each user-movie pair
movie_decade_cols = [col for col in movie_feature_frame.columns if col.startswith("movie_Decade_")]
user_decade_cols = [col for col in user_feature_frame.columns if col.startswith("user_Decade_")]
user_decade_values = (
    user_feature_frame
    .loc[user_decade_cols]
    .fillna(0)
    .to_numpy()
)
movie_decade_values = (
    movie_feature_frame
    .loc[candidate_preds["tmdbId"], movie_decade_cols]
    .fillna(0)
    .to_numpy()
)
decade_counts = movie_decade_values.sum(axis=1)
candidate_preds["decade_affinity"] = np.divide(
    (user_decade_values * movie_decade_values).sum(axis=1),
    decade_counts,
    out=np.full(len(candidate_preds), np.nan),
    where=decade_counts != 0
)

# build candidate_preds_ranked dataframe for xgboost input
candidate_preds_ranked = initial_rank(candidate_preds)

# loading xgboost features 
xgbmodel = xgb.XGBRanker()
xgbmodel.load_model("models/xgboostranker_model.json")

xgboost_input = candidate_preds_ranked.copy()
feature_cols = xgbmodel.get_booster().feature_names
xgboost_input = xgboost_input[feature_cols]

candidate_preds_ranked["xgb_predicted_rating"] = xgbmodel.predict(xgboost_input)

candidate_preds_ranked["xgb_rank"] = (
    candidate_preds_ranked["xgb_predicted_rating"]
    .rank(method="first", ascending=False)
)

# keep top 10 recs per user from xgboost ranked predictions
xgb_preds_ranked = (
    candidate_preds_ranked[
        candidate_preds_ranked["xgb_rank"] <= 10
    ]
    .sort_values(
        ["xgb_rank"],
        ascending=[True]
    )
    .reset_index(drop=True)
)

# merge tmdbId with movie titles for display
xgb_preds_ranked = xgb_preds_ranked.merge(movie_feature_frame[["tmdbId", "movie_title"]], on="tmdbId", how="left")
print(xgb_preds_ranked[["tmdbId", "movie_title", "xgb_rank"]].head(10))
