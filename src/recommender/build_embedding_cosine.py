"""Add learned two-tower cosine features to the saved train/test interaction tables.

Run from the letterboxd-rec repository root after notebooks 03, 04, and
Two-Tower-NN-Model have produced the CSVs, scalers, and standalone towers.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FEATURE = "embedding_cosine_similarity"


def _scaler_columns(scaler, fallback_path):
    """Use the exact ordered feature names captured when the scaler was fit."""
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return np.load(fallback_path, allow_pickle=True).tolist()


def _scale_frame(frame, columns, scaler, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} data is missing scaler columns: {missing[:10]}")

    # Column order must match the scaler's fit order. Replace infinities with
    # NaN before scaling, then use zero for any still-missing scaled values.
    values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    scaled = scaler.transform(values)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def main():
    train_path = DATA_DIR / "training_data.csv"
    test_path = DATA_DIR / "test_data.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    user_scaler = joblib.load(MODEL_DIR / "scaler_users.pkl")
    movie_scaler = joblib.load(MODEL_DIR / "scaler_movies.pkl")
    user_model = load_model(MODEL_DIR / "userNN.keras", compile=False)
    movie_model = load_model(MODEL_DIR / "movieNN.keras", compile=False)

    user_cols = _scaler_columns(user_scaler, DATA_DIR / "user_cols.npy")
    movie_cols = _scaler_columns(movie_scaler, DATA_DIR / "movie_cols.npy")
    all_rows = pd.concat([train, test], ignore_index=True)

    user_rows = (
        all_rows[["userId", *user_cols]]
        .drop_duplicates("userId", keep="first")
        .set_index("userId")
    )
    movie_rows = (
        all_rows[["movieId", *movie_cols]]
        .drop_duplicates("movieId", keep="first")
        .set_index("movieId")
    )

    user_inputs = _scale_frame(user_rows, user_cols, user_scaler, "User")
    movie_inputs = _scale_frame(movie_rows, movie_cols, movie_scaler, "Movie")
    user_vectors = np.asarray(user_model.predict(user_inputs, verbose=0), dtype="float32")
    movie_vectors = np.asarray(movie_model.predict(movie_inputs, verbose=0), dtype="float32")

    print("User embedding shape:", user_vectors.shape)
    print("Movie embedding shape:", movie_vectors.shape)
    if user_vectors.ndim != 2 or movie_vectors.ndim != 2:
        raise ValueError("Tower outputs must be two-dimensional matrices")
    if user_vectors.shape[1] != movie_vectors.shape[1]:
        raise ValueError("User and movie tower output dimensions must match")

    def attach(frame):
        user_pos = user_rows.index.get_indexer(frame["userId"])
        movie_pos = movie_rows.index.get_indexer(frame["movieId"])
        if (user_pos < 0).any() or (movie_pos < 0).any():
            raise ValueError("A pair row could not be mapped to its user/movie embedding")

        pair_users = user_vectors[user_pos]
        pair_movies = movie_vectors[movie_pos]
        dots = np.einsum("ij,ij->i", pair_users, pair_movies)
        norms = np.linalg.norm(pair_users, axis=1) * np.linalg.norm(pair_movies, axis=1)
        frame[FEATURE] = np.divide(
            dots, norms, out=np.zeros_like(dots), where=norms > 0
        )
        if len(frame[FEATURE]) != len(frame) or not np.isfinite(frame[FEATURE]).all():
            raise ValueError("Cosine scores are misaligned or contain NaN/inf values")
        if not frame[FEATURE].between(-1.00001, 1.00001).all():
            raise ValueError("Cosine scores fell outside [-1, 1]")

    attach(train)
    attach(test)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("Train cosine summary:\n", train[FEATURE].describe())
    print("Test cosine summary:\n", test[FEATURE].describe())
    print("Examples:\n", train[["userId", "movieId", "rating", FEATURE]].head())
    print("Rows: train =", len(train), ", test =", len(test))


if __name__ == "__main__":
    main()
