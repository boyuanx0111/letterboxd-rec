"""Two-tower retrieval followed by XGBoost reranking."""

import threading
from functools import lru_cache
from pathlib import Path


class RecommenderInputError(ValueError):
    """Raised when supplied ratings cannot be used for recommendation."""


_INFERENCE_LOCK = threading.Lock()
_ARTIFACT_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_artifacts(root: Path):
    """Load immutable data and model artifacts once for each project root."""

    import joblib
    import pandas as pd
    import xgboost as xgb
    from tensorflow.keras.models import load_model

    data_dir = root / "data"
    model_dir = root / "models"

    movie_features = pd.read_parquet(data_dir / "movie_feature_frame.parquet")
    movie_embeddings = pd.read_parquet(data_dir / "movie_embeddings_nn.parquet")
    movie_features = movie_features[movie_features.index.notna()]
    movie_embeddings = movie_embeddings[movie_embeddings.index.notna()]
    catalogue_ids = movie_features.index.intersection(movie_embeddings.index)
    movie_features = movie_features.loc[catalogue_ids]
    movie_embeddings = movie_embeddings.loc[catalogue_ids]

    genre_columns = (
        pd.read_csv(data_dir / "genre_columns.csv")
        .squeeze("columns")
        .dropna()
        .astype(str)
        .tolist()
    )
    user_scaler = joblib.load(model_dir / "scaler_users.pkl")
    y_scaler = joblib.load(model_dir / "scaler_y.pkl")
    user_model = load_model(model_dir / "userNN.keras", compile=False)
    ranker = xgb.XGBRanker()
    ranker.load_model(str(model_dir / "xgboostranker_model.json"))
    ranker_features = ranker.get_booster().feature_names

    return (
        movie_features,
        movie_embeddings,
        genre_columns,
        user_scaler,
        y_scaler,
        user_model,
        ranker,
        ranker_features,
    )


def _get_artifacts(root: Path):
    with _ARTIFACT_LOAD_LOCK:
        return _load_artifacts(root)


def hybrid_recommend(user_ratings, root_dir=None):
    """Return the ten highest-ranked unwatched movies for one user.

    Args:
        user_ratings: DataFrame containing numeric ``tmdbId`` and ``rating``
            columns.
        root_dir: Project directory containing ``data`` and ``models``.
            Defaults to the repository containing this module.
    """

    import numpy as np
    import pandas as pd

    from .build_user_features import build_user_features
    from .pipelinept1 import initial_rank

    root = (
        Path(root_dir).expanduser().resolve()
        if root_dir is not None
        else Path(__file__).resolve().parents[2]
    )
    missing_columns = {"tmdbId", "rating"}.difference(user_ratings.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RecommenderInputError(
            f"Ratings are missing required columns: {missing}"
        )

    ratings = user_ratings[["tmdbId", "rating"]].copy()
    ratings["tmdbId"] = pd.to_numeric(ratings["tmdbId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["tmdbId", "rating"])
    ratings = ratings[
        np.isfinite(ratings["tmdbId"])
        & np.isfinite(ratings["rating"])
        & ratings["tmdbId"].gt(0)
        & ratings["rating"].gt(0)
        & ratings["rating"].le(5)
    ]
    ratings = ratings.drop_duplicates(subset="tmdbId", keep="last")

    if ratings.empty:
        raise RecommenderInputError("The uploaded file contains no valid ratings.")

    (
        movie_feature_frame,
        movie_embeddings,
        genre_columns,
        user_scaler,
        y_scaler,
        user_model,
        ranker,
        ranker_features,
    ) = _get_artifacts(root)

    catalogue_ids = movie_feature_frame.index
    ratings = ratings[ratings["tmdbId"].isin(catalogue_ids)]

    if ratings.empty:
        raise RecommenderInputError(
            "None of the uploaded ratings match movies in the recommendation catalogue."
        )
    catalogue_rating_count = len(ratings)

    user_feature_frame = build_user_features(ratings, movie_feature_frame)
    expected_user_columns = list(user_scaler.feature_names_in_)
    user_feature_frame = user_feature_frame.reindex(
        columns=expected_user_columns,
        fill_value=0,
    )
    user_feature_scaled = user_scaler.transform(user_feature_frame)

    with _INFERENCE_LOCK:
        user_vectors = user_model.predict(user_feature_scaled, verbose=0)

    watched_ids = set(ratings["tmdbId"])
    candidate_ids = movie_embeddings.index[~movie_embeddings.index.isin(watched_ids)]
    if candidate_ids.empty:
        raise RecommenderInputError(
            "There are no unwatched movies left to recommend."
        )

    movie_vectors = movie_embeddings.loc[candidate_ids].to_numpy(dtype="float32")
    user_vectors = np.asarray(user_vectors, dtype="float32")
    user_norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    movie_norms = np.linalg.norm(movie_vectors, axis=1, keepdims=True)
    user_vectors = np.divide(
        user_vectors,
        user_norms,
        out=np.zeros_like(user_vectors),
        where=user_norms != 0,
    )
    movie_vectors = np.divide(
        movie_vectors,
        movie_norms,
        out=np.zeros_like(movie_vectors),
        where=movie_norms != 0,
    )

    # Both towers are unit-normalized: the dot product is cosine similarity.
    embedding_cosine_similarity = np.clip(movie_vectors @ user_vectors[0], -1.0, 1.0)
    predictions_scaled = embedding_cosine_similarity
    predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    candidate_predictions = pd.DataFrame(
        {
            "tmdbId": candidate_ids.to_numpy(),
            "predicted_rating": predictions,
            "embedding_cosine_similarity": embedding_cosine_similarity,
        }
    )

    user_stats = user_feature_frame.iloc[0]
    for column in ("user_avg_rating", "user_std_rating", "user_count_rating"):
        candidate_predictions[column] = user_stats[column]

    candidate_predictions = candidate_predictions.join(
        movie_feature_frame[["movie_Release_Year"]],
        on="tmdbId",
    )

    user_genre_columns = [f"user_{genre}" for genre in genre_columns]
    movie_genre_columns = [f"movie_{genre}" for genre in genre_columns]
    user_genre_values = user_feature_frame.loc[
        :, user_genre_columns
    ].iloc[0].to_numpy(dtype="float64")
    movie_genre_values = movie_feature_frame.loc[
        candidate_ids, movie_genre_columns
    ].fillna(0).to_numpy(dtype="float64")
    genre_counts = movie_genre_values.sum(axis=1)
    candidate_predictions["genre_affinity"] = np.divide(
        (movie_genre_values * user_genre_values).sum(axis=1),
        genre_counts,
        out=np.full(len(candidate_predictions), np.nan, dtype="float64"),
        where=genre_counts != 0,
    )

    movie_decade_columns = [
        column
        for column in movie_feature_frame.columns
        if column.startswith("movie_Decade_")
    ]
    user_decade_columns = [
        column.replace("movie_", "user_", 1)
        for column in movie_decade_columns
    ]
    user_decade_values = user_feature_frame.loc[
        :, user_decade_columns
    ].iloc[0].to_numpy(dtype="float64")
    movie_decade_values = movie_feature_frame.loc[
        candidate_ids, movie_decade_columns
    ].fillna(0).to_numpy(dtype="float64")
    decade_counts = movie_decade_values.sum(axis=1)
    candidate_predictions["decade_affinity"] = np.divide(
        (movie_decade_values * user_decade_values).sum(axis=1),
        decade_counts,
        out=np.full(len(candidate_predictions), np.nan, dtype="float64"),
        where=decade_counts != 0,
    )

    ranked_candidates = initial_rank(candidate_predictions)
    with _INFERENCE_LOCK:
        ranked_candidates["xgb_predicted_rating"] = ranker.predict(
            ranked_candidates[ranker_features]
        )
    ranked_candidates["xgb_rank"] = (
        ranked_candidates["xgb_predicted_rating"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    recommendations = (
        ranked_candidates[ranked_candidates["xgb_rank"] <= 10]
        .sort_values("xgb_rank")
        .join(movie_feature_frame[["movie_title"]], on="tmdbId")
        .reset_index(drop=True)
    )
    recommendations["tmdbId"] = recommendations["tmdbId"].astype("int64")
    result = recommendations[["tmdbId", "movie_title", "xgb_rank"]].copy()
    result.attrs["catalogue_rows"] = catalogue_rating_count
    return result
