#for xgboost, fixed retrieval pipeline

import pandas as pd
import joblib
import numpy as np

# rank and group by user
# per user function 
def initial_rank(user_preds):
    """
    Rank the predictions for a single user and compute z-scores.
    Args:
        user_preds: DataFrame containing predictions for a single user
        input columns ['userId', 'movieId', 'rating', 'user_avg_rating', 'user_std_rating', 'user_count_rating', 'movie_Release_Year', 'genre_affinity', 'decade_affinity', 'predicted_rating']
    Returns:
        pd.DataFrame: DataFrame with additional columns for user rank and z-score
        Filters for top 300 predictions per user
    """
    predicted_rating = user_preds["predicted_rating"]
    user_preds_ranked = user_preds.nlargest(
        300, "predicted_rating", keep="first"
    ).reset_index(drop=True)
    user_preds_ranked["user_rank"] = (
        user_preds_ranked["predicted_rating"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    user_preds_ranked["z_score"] = (
        user_preds_ranked["predicted_rating"] - predicted_rating.mean()
    ) / predicted_rating.std()
    return user_preds_ranked

def main():
    #RUN FROM ROOT DIR

    from tensorflow.keras.models import load_model

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
        ["userId", "movieId", "rating","user_avg_rating", "user_std_rating", "user_count_rating", "movie_Release_Year", "genre_affinity", "decade_affinity", "embedding_cosine_similarity"]
    ].copy()

    candidate_preds["predicted_rating"] = predictions

    candidate_preds_ranked = pd.DataFrame()

    for user in candidate_preds["userId"].unique():
        user_data = candidate_preds[candidate_preds["userId"] == user]
        user_ranked = initial_rank(user_data)
        candidate_preds_ranked = pd.concat([candidate_preds_ranked, user_ranked], ignore_index=True)

    print("input columns", user_data.columns.tolist())

    print(candidate_preds_ranked)
    print(candidate_preds_ranked["user_rank"].describe())
    print(candidate_preds_ranked.columns.tolist())

    # export for xgboost
    candidate_preds_ranked.to_parquet(
        "data/train_candidate_preds_ranked.parquet",
        index=False
    )


if __name__ == "__main__":
    main()
