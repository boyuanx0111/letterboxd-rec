# for xgboost, fixed retrieval pipeline
# run before training xgboost model

import joblib
import numpy as np
import pandas as pd

try:  # Supports both `python -m src.recommender.pipelinept1` and script execution.
    from .cf import cf_predict_new_user
except ImportError:  # pragma: no cover - exercised by direct script invocation.
    from cf import cf_predict_new_user


# rank and group by user
# per user function
def initial_rank(user_preds):
    """
    Rank the predictions for a single user and compute z-scores.
    Args:
        user_preds: DataFrame containing predictions for a single user
        Input columns include user and movie IDs, ratings, user statistics,
        release year, and predicted rating.
    Returns:
        pd.DataFrame: DataFrame with additional columns for user rank and z-score
        Filters for top 300 predictions per user
    """
    if "predicted_rating" not in user_preds.columns:
        raise KeyError("user_preds must contain a 'predicted_rating' column")

    predicted_rating = user_preds["predicted_rating"]
    user_preds_ranked = user_preds.nlargest(500, "predicted_rating", keep="first").reset_index(
        drop=True
    )
    user_preds_ranked["user_rank"] = (
        user_preds_ranked["predicted_rating"].rank(method="first", ascending=False).astype(int)
    )
    rating_std = predicted_rating.std()
    if pd.isna(rating_std) or rating_std == 0:
        user_preds_ranked["z_score"] = 0.0
    else:
        user_preds_ranked["z_score"] = (
            user_preds_ranked["predicted_rating"] - predicted_rating.mean()
        ) / rating_std
    return user_preds_ranked


def main():
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[2]

    from tensorflow.keras.models import load_model

    model = load_model(root_dir / "models/two_tower.keras")

    user_scaler = joblib.load(root_dir / "models/scaler_users.pkl")
    movie_scaler = joblib.load(root_dir / "models/scaler_movies.pkl")
    y_scaler = joblib.load(root_dir / "models/scaler_y.pkl")

    # from 03, combined and enriched table
    training_data = pd.read_csv(root_dir / "data/test_data.csv")

    # Add tmdbId for collaborative filtering
    links = pd.read_csv(root_dir / "data/ml-32m/links.csv")

    training_data = training_data.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    user_cols = np.load(root_dir / "data/user_cols.npy", allow_pickle=True)
    movie_cols = np.load(root_dir / "data/movie_cols.npy", allow_pickle=True)

    # get test features
    X_user_training = training_data[user_cols]
    X_movie_training = training_data[movie_cols]

    # scale
    X_user_training_scaled = user_scaler.transform(X_user_training)
    X_movie_training_scaled = movie_scaler.transform(X_movie_training)

    # predict all test interactions
    predictions = model.predict([X_user_training_scaled, X_movie_training_scaled])

    # convert predictions back to rating scale
    predictions = y_scaler.inverse_transform(predictions).flatten()

    # loading xgboost features
    candidate_preds = training_data[
        [
            "userId",
            "movieId",
            "tmdbId",
            "rating",
            "user_avg_rating",
            "user_std_rating",
            "user_count_rating",
            "movie_Release_Year",
            "genre_affinity",
            "decade_affinity",
            "Original_Language_affinity",
            "Origin_country_affinity",
            "Director_affinity",
            "Top_cast_affinity",
            "Keyword_affinity",
            "Overview_affinity",
            "Production_company_affinity",
            "vote_count_affinity",
            "vote_average_affinity",
            "revenue_affinity",
            "svd_affinity",
        ]
    ].copy()

    candidate_preds["predicted_rating"] = predictions

    # Load collaborative filtering model
    cf_model = joblib.load(root_dir / "models/collaborative_filtering.pkl")

    candidate_results = []

    print("input columns:", candidate_preds.columns.tolist())
    print("input shape:", candidate_preds.shape)
    print("input sample:\n", candidate_preds.head())

    for _user_id, user_data in candidate_preds.groupby("userId"):
        # Two-tower retrieves top 300
        user_ranked = initial_rank(user_data)

        # Simulate uploaded Letterboxd ratings
        user_history: pd.DataFrame = user_data.loc[:, ["tmdbId", "rating"]].dropna()

        # Predict CF ratings for top 300
        user_ranked["cf_pred"] = cf_predict_new_user(
            cf_model, user_history, user_ranked["tmdbId"].to_numpy()
        )

        candidate_results.append(user_ranked)

    if candidate_results:
        candidate_preds_ranked = pd.concat(candidate_results, ignore_index=True)
    else:
        candidate_preds_ranked = candidate_preds.assign(
            user_rank=pd.Series(dtype="int64"),
            z_score=pd.Series(dtype="float64"),
            cf_pred=pd.Series(dtype="float64"),
        )

    print("ranked shape:", candidate_preds_ranked.shape)
    print("user_rank summary:\n", candidate_preds_ranked["user_rank"].describe())
    print("ranked columns:", candidate_preds_ranked.columns.tolist())

    # export for xgboost
    candidate_preds_ranked.to_parquet(
        root_dir / "data/test_candidate_preds_ranked.parquet", index=False
    )

    print("Exported ranked predictions to test_candidate_preds_ranked.parquet")


if __name__ == "__main__":
    main()
