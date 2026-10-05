#test hybrid pipeline 
# load in filtered sample ratings set 2
# tt-nn filters top 300 per user, then xgboost ranks and filters top 10 per user
# because only using seen movies to test ndcg@20 improvements for now, ranks all user's seen movies in test set

import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# run from this dir
from pipelinept1 import initial_rank
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import ndcg_score

# preprocess test data to tt-nn input

# filtered sample ratings set 2
# userId,movieId,rating,timestamp
ratings_sample_2 = pd.read_csv("data/filtered_ratings2.csv")
# introduce tmdbIb to ratings_sample_2 for merging with movie embeddings
links = pd.read_csv("data/ml-32m/links.csv")
ratings_sample_2 = ratings_sample_2.merge(
    links[["movieId", "tmdbId"]], on="movieId", how="left"
)

# users feature frame from sample 2
user_feature_frame = pd.read_parquet("data/users_feature_frame2.parquet")
# use build_user_features function when only one user, feature frame is combination of sample 2 users

# user embeddings from filtered sample ratings set 2
user_embeddings = pd.read_parquet("data/user_embeddings_nn.parquet")

# import movie feature frame
movie_feature_frame = pd.read_parquet("data/movie_feature_frame.parquet")

# complete movie embeddings with tmdbId as index
movie_embeddings = pd.read_parquet("data/movie_embeddings_nn.parquet")

user_scaler = joblib.load('models/scaler_users.pkl')
movie_scaler = joblib.load('models/scaler_movies.pkl')
y_scaler = joblib.load('models/scaler_y.pkl')

# predict
# movie_embeddings - (87585, 32)
# user_embeddings - (1000, 32)
user_vecs = user_embeddings.loc[ratings_sample_2["userId"]].to_numpy()
movie_vecs = movie_embeddings.loc[ratings_sample_2["tmdbId"]].to_numpy()

# normalise before dot product
user_vecs = tf.keras.ops.normalize(user_vecs, axis=1)
movie_vecs = tf.keras.ops.normalize(movie_vecs, axis=1)

# vectorised dot product to get predictions for all user-movie pairs in the test set
predictions_scaled = np.sum(user_vecs * movie_vecs, axis=1)
# convert predictions back to rating scale
predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

#build candidate_preds dataframe for xgboost input
candidate_preds = pd.DataFrame({
    "userId": ratings_sample_2["userId"],
    "tmdbId": ratings_sample_2["tmdbId"],
    "rating": ratings_sample_2["rating"],
    # user_avg rating
    # user_std rating
    # user_count rating
    # movie_Release_Year
    # genre_affinity
    # decade_affinity
    "predicted_rating": predictions
})
candidate_preds = candidate_preds.merge(user_feature_frame[["user_avg_rating", "user_std_rating", "user_count_rating"]], on="userId", how="left")
candidate_preds = candidate_preds.merge(movie_feature_frame[["movie_Release_Year"]], on="tmdbId", how="left")

# calc genre affinity for each user-movie pair
genre_columns = pd.read_csv("data/genre_columns.csv").squeeze().tolist()
user_genre_cols = [f"user_{genre}" for genre in genre_columns]
movie_genre_cols = [f"movie_{genre}" for genre in genre_columns]
user_genre_values = (
    user_feature_frame
    .loc[candidate_preds["userId"], user_genre_cols]
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
    .loc[candidate_preds["userId"], user_decade_cols]
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
candidate_preds_ranked = pd.DataFrame()

for user in candidate_preds["userId"].unique():
    user_data = candidate_preds[candidate_preds["userId"] == user]
    user_ranked = initial_rank(user_data)
    candidate_preds_ranked = pd.concat([candidate_preds_ranked, user_ranked], ignore_index=True)

# loading xgboost features 
xgbmodel = xgb.XGBRanker()
xgbmodel.load_model("models/xgboostranker_model.json")

xgboost_input = candidate_preds_ranked.copy()
feature_cols = xgbmodel.get_booster().feature_names
xgboost_input = xgboost_input[feature_cols]

candidate_preds_ranked["xgb_predicted_rating"] = xgbmodel.predict(xgboost_input)

candidate_preds_ranked["xgb_rank"] = (
    candidate_preds_ranked
    .groupby("userId")["xgb_predicted_rating"]
    .rank(method="first", ascending=False)
)

candidate_preds_ranked["true_rank"] = (
    candidate_preds_ranked
    .groupby("userId")["rating"]
    .rank(method="first", ascending=False)
)

# keep top 15 from true rank
true_ranked = (
    candidate_preds_ranked[
        candidate_preds_ranked["true_rank"] <= 15
    ]
    .sort_values(
        ["userId", "true_rank"],
        ascending=[True, True]
    )
    .reset_index(drop=True)
)

# keep top 15 recs per user from xgboost ranked predictions
xgb_preds_ranked = (
    candidate_preds_ranked[
        candidate_preds_ranked["xgb_rank"] <= 15
    ]
    .sort_values(
        ["userId", "xgb_rank"],
        ascending=[True, True]
    )
    .reset_index(drop=True)
)

# top 15 recs per user from tt nn ranked predictions
tt_nn_preds_ranked = (
    candidate_preds_ranked[
        candidate_preds_ranked["user_rank"] <= 15
    ]
    .sort_values(
        ["userId", "user_rank"],
        ascending=[True, True]
    )
    .reset_index(drop=True)
)

# user_rank is tt-nn rank, xgb_rank is xgboost rank, true_rank is actual rating rank
# print(true_ranked[["userId", "tmdbId", "true_rank", "user_rank", "xgb_rank", "rating"]].head())
# print(tt_nn_preds_ranked[["userId", "tmdbId", "true_rank", "user_rank", "xgb_rank", "rating"]].head())
# print(xgb_preds_ranked[["userId", "tmdbId", "true_rank", "user_rank", "xgb_rank", "rating"]].head(10))

def ndcg(model_rank):
    # NDCG eval
    #Calc ideal relevance based on rank
    def z_to_relevance(z):
        if z > 1:
            return 3  #high link
        elif z > 0.5:
            return 2  #medium link
        elif z > 0:
            return 1  #above avergae
        else:
            return 0 #below average

    user_ndcgs = []

    for user_id, user_data in true_ranked.groupby("userId"):

        #ideal relevance
        user_data["relevance"] = ((user_data["rating"] - user_data["user_avg_rating"]) / user_data["user_std_rating"]).apply(z_to_relevance)
        y_ideal = user_data["relevance"].to_numpy().reshape(1, -1)

        # predicted ranking score
        y_score = -user_data[model_rank].to_numpy().reshape(1, -1)

        ndcg = ndcg_score(
            y_ideal,
            y_score,
            k=10
        )

        user_ndcgs.append(ndcg)
    mean_ndcg = np.mean(user_ndcgs)

    print(f"Mean NDCG@10 for {model_rank}: {mean_ndcg:.4f}")

ndcg("true_rank")
ndcg("user_rank")
ndcg("xgb_rank")

# hybrid weighting of ratings from models - xgb rank has highest ndcg score so keep it fully weighted for final rank
print(true_ranked[["userId", "tmdbId", "rating", "predicted_rating", "xgb_predicted_rating"]].head(10))

for i in range(1, 10):
    alpha = i / 10
    true_ranked["hybrid_predicted_rating"] = (
        alpha * true_ranked["predicted_rating"] + (1 - alpha) * true_ranked["xgb_predicted_rating"]
    )
    # rank by hybrid predicted rating
    true_ranked["hybrid_rank"] = (
        true_ranked
        .groupby("userId")["hybrid_predicted_rating"]
        .rank(method="first", ascending=False)
    )
    print(f"Hybrid weighting alpha: {alpha:.1f}")
    ndcg("hybrid_rank")