# hybrid rec pipeline 
# tt-nn filters top 300 per user, then xgboost ranks and filters top 10
# input parsed user ratings (tmdbId, rating) to build user features and embeddings, then score all movies and rank with xgboost

from pathlib import Path

def hybrid_recommend(user_ratings):

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"

    """
    Recommender pipeline that combines a neural network for initial ranking and an XGBoost model for final ranking.
    Input: 
    - user_ratings:  containing user ratings with columns 'tmdbId' and 'rating'.
    Output:
    - DataFrame containing the top 10 recommended movies for the user, with columns 'tmdbId', 'movie_title', and 'xgb_rank'.
    """

    import pandas as pd
    import joblib
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from .pipelinept1 import initial_rank
    from .build_user_features import build_user_features
    from xgboost import XGBRegressor
    import xgboost as xgb
    from sklearn.metrics import ndcg_score
    from .cf import cf_predict_new_user

    # preprocess test data to tt-nn input

    # import movie feature frame
    movie_feature_frame = pd.read_parquet(DATA_DIR / "movie_feature_frame.parquet")

    # complete movie embeddings with tmdbId as index
    norm_movie_embeddings = pd.read_parquet(DATA_DIR / "norm_movie_embeddings_nn.parquet")

    # keep uploaded ratings that match the catalogue
    catalogue_ids = movie_feature_frame.index.intersection(norm_movie_embeddings.index)
    user_ratings = user_ratings[user_ratings["tmdbId"].isin(catalogue_ids)].copy()

    if user_ratings.empty:
        raise ValueError("No uploaded ratings match the movie catalogue.")

    # users feature frame from USER INPUT
    user_feature_frame = build_user_features(user_ratings, movie_feature_frame)

    user_scaler = joblib.load(MODEL_DIR / 'scaler_users.pkl')
    y_scaler = joblib.load(MODEL_DIR / 'scaler_y.pkl')

    # Match the exact feature names and order used when the scaler was fitted.
    user_feature_cols = getattr(user_scaler, "feature_names_in_", None)
    if user_feature_cols is None:
        user_feature_cols = np.load(DATA_DIR / "user_cols.npy", allow_pickle=True)
    user_feature_cols = list(user_feature_cols)
    missing_user_features = [
        col for col in user_feature_cols if col not in user_feature_frame.columns
    ]
    if missing_user_features:
        raise ValueError(
            "Built user features are missing scaler columns: "
            f"{missing_user_features}"
        )
    user_feature_frame = user_feature_frame.loc[:, user_feature_cols]

    # create user embeddings from user feature frame using userNN model
    user_feature_scaled = user_scaler.transform(user_feature_frame)
    userNN = load_model(MODEL_DIR / 'userNN.keras')
    user_embeddings = userNN.predict(user_feature_scaled)
    user_embeddings_df = pd.DataFrame(user_embeddings, index=user_feature_frame.index)

    # predict
    # movie_embeddings - (87585, 32)
    # user_embeddings - 
    user_vecs = user_embeddings_df.to_numpy()
    # create unwatched movie list of tmdbIds for user to extract corresponding movie embeddings
    movie_vecs = norm_movie_embeddings[~norm_movie_embeddings.index.isin(user_ratings["tmdbId"])]
    candidate_movie_ids = norm_movie_embeddings.index[~norm_movie_embeddings.index.isin(user_ratings["tmdbId"])]

    # normalise user before dot product (movie already normalised)
    user_vecs = tf.keras.ops.normalize(user_vecs, axis=1)

    # vectorised dot product to get predictions for all user-movie pairs in the test set
    predictions_scaled = movie_vecs.to_numpy() @ user_vecs[0].numpy()
    # convert predictions back to rating scale
    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Both towers are unit-normalized: the dot product is cosine similarity.
    embedding_cosine_similarity = np.clip(movie_vecs @ user_vecs[0], -1.0, 1.0)
    predictions_scaled = embedding_cosine_similarity

    #build candidate_preds dataframe for xgboost input
    candidate_preds = pd.DataFrame({
        "tmdbId": candidate_movie_ids.to_numpy(),
        # user_avg rating
        # user_std rating
        # user_count rating
        # movie_Release_Year
        # genre_affinity
        # decade_affinity
        "embedding_cosine_similarity": embedding_cosine_similarity,
        "predicted_rating": predictions
    })

    stat_cols = ["user_avg_rating", "user_std_rating", "user_count_rating"]
    for col in stat_cols:
        candidate_preds[col] = user_feature_frame[col].iloc[0]

    candidate_preds = candidate_preds.reset_index(drop=True)
    
    candidate_preds = candidate_preds.merge(movie_feature_frame[["movie_Release_Year"]], left_on="tmdbId", right_index=True, how="left")

    # calc genre affinity for each user-movie pair
    genre_columns = pd.read_csv(DATA_DIR / "genre_columns.csv").squeeze().tolist()
    user_genre_cols = [f"user_{genre}" for genre in genre_columns]
    movie_genre_cols = [f"movie_{genre}" for genre in genre_columns]
    user_genre_values = (
        user_feature_frame
        .loc[:, user_genre_cols]
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
        .loc[:, user_decade_cols]
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

    # calc language and country affinity features used to train XGBoost.
    user_language_cols = [f"user_tmdb_{i}_avg" for i in range(1, 12)]
    movie_language_cols = [f"movie_tmdb_{i}" for i in range(1, 12)]
    user_language_values = user_feature_frame[user_language_cols].fillna(0).to_numpy()
    movie_language_values = (
        movie_feature_frame
        .loc[candidate_preds["tmdbId"], movie_language_cols]
        .fillna(0)
        .to_numpy()
    )
    language_counts = movie_language_values.sum(axis=1)
    candidate_preds["Original_Language_affinity"] = np.divide(
        (user_language_values * movie_language_values).sum(axis=1),
        language_counts,
        out=np.full(len(candidate_preds), np.nan),
        where=language_counts != 0,
    )

    user_country_cols = [f"user_tmdb_{i}" for i in range(12, 27)]
    movie_country_cols = [f"movie_tmdb_{i}" for i in range(12, 27)]
    user_country_values = user_feature_frame[user_country_cols].fillna(0).to_numpy()
    movie_country_values = (
        movie_feature_frame
        .loc[candidate_preds["tmdbId"], movie_country_cols]
        .fillna(0)
        .to_numpy()
    )
    country_counts = movie_country_values.sum(axis=1)
    candidate_preds["Origin_country_affinity"] = np.divide(
        (user_country_values * movie_country_values).sum(axis=1),
        country_counts,
        out=np.full(len(candidate_preds), np.nan),
        where=country_counts != 0,
    )

    # test
    # Calculate SVD content affinity for each user-movie pair
    # User: user_svd_0 to user_svd_149
    # Movie: movie_tmdb_411 to movie_tmdb_560

    user_svd_cols = [f"user_user_svd_{i}" for i in range(150)]
    movie_svd_cols = [f"movie_tmdb_{i}" for i in range(411, 561)]

    # User SVD embedding (1, 150)
    user_svd_values = (
        user_feature_frame[user_svd_cols]
        .fillna(0)
        .to_numpy(dtype=float)[0]
    )

    # Candidate movie SVD embeddings (n_movies, 150)
    movie_svd_values = (
        movie_feature_frame
        .loc[candidate_preds["tmdbId"], movie_svd_cols]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # Cosine similarity
    dot_products = movie_svd_values @ user_svd_values

    user_norm = np.linalg.norm(user_svd_values)
    movie_norms = np.linalg.norm(movie_svd_values, axis=1)

    denominators = movie_norms * user_norm

    candidate_preds["svd_affinity"] = np.divide(
        dot_products,
        denominators,
        out=np.zeros(len(candidate_preds)),
        where=denominators > 0
    )
    # test

    # build candidate_preds_ranked dataframe for xgboost input
    candidate_preds_ranked = initial_rank(candidate_preds)
 
    # Movie popularity from TMDB metadata enhance
    candidate_preds_ranked["popularity"] = (
        movie_feature_frame
        .reindex(candidate_preds_ranked["tmdbId"])["movie_tmdb_564"]
        .to_numpy()
    )

    # User's average popularity preference
    user_popularity = user_feature_frame["user_popularity"].iloc[0]

    # Personalised popularity affinity
    candidate_preds_ranked["pop_affinity"] = -np.abs(
        np.log1p(candidate_preds_ranked["popularity"].clip(lower=0))
        - np.log1p(max(user_popularity, 0))
    )

    # Load collaborative filtering model
    cf_model = joblib.load(
        MODEL_DIR / "collaborative_filtering.pkl"
    )

    # Predict ratings for top 300 candidates
    candidate_preds_ranked["cf_pred"] = cf_predict_new_user(
        cf_model,
        user_ratings,
        candidate_preds_ranked["tmdbId"].to_numpy()
    )

    # loading xgboost features 
    xgbmodel = xgb.XGBRanker()
    xgbmodel.load_model(MODEL_DIR / "xgboostranker_model.json")

    xgboost_input = candidate_preds_ranked.copy()
    feature_cols = xgbmodel.get_booster().feature_names
    xgboost_input = xgboost_input[feature_cols]

    # XGBRanker predictions
    candidate_preds_ranked["xgb_score"] = xgbmodel.predict(xgboost_input)

        # Normalise each signal using percentile ranks
    candidate_preds_ranked["xgb_norm"] = (
        candidate_preds_ranked["xgb_score"].rank(pct=True)
    )

    candidate_preds_ranked["cf_norm"] = (
        candidate_preds_ranked["cf_pred"].rank(pct=True)
    )

    candidate_preds_ranked["pop_affinity_norm"] = (
        candidate_preds_ranked["pop_affinity"].rank(pct=True)
    )

    # Weighted hybrid score
    candidate_preds_ranked["hybrid_score"] = (
        0.50 * candidate_preds_ranked["xgb_norm"]
        + 0.40 * candidate_preds_ranked["cf_norm"]
        +0.10 * candidate_preds_ranked["pop_affinity_norm"]
    )

    # Final ranking
    candidate_preds_ranked["hybrid_rank"] = (
        candidate_preds_ranked["hybrid_score"]
        .rank(method="first", ascending=False)
    )
    # Select top 10 using hybrid score
    hybrid_preds_ranked = (
        candidate_preds_ranked
        .sort_values("hybrid_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Keep xgb_rank column name for frontend compatibility
    hybrid_preds_ranked["xgb_rank"] = np.arange(1, len(hybrid_preds_ranked) + 1)

    # Merge movie titles
    hybrid_preds_ranked = hybrid_preds_ranked.merge(
        movie_feature_frame[["movie_title"]],
        left_on="tmdbId",
        right_index=True,
        how="left"
    )

    # Required frontend output
    result = hybrid_preds_ranked[
        ["tmdbId", "movie_title", "xgb_rank"]
    ].copy()

    result.attrs["catalogue_rows"] = len(user_ratings)

    return result