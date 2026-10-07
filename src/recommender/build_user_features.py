import pandas as pd


def build_user_features(ratings, movie_features):
    """Build the single user feature row expected by the saved user tower."""

    missing_columns = {"tmdbId", "rating"}.difference(ratings.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Ratings are missing required columns: {missing}")

    ratings = ratings.copy()
    ratings["tmdbId"] = pd.to_numeric(ratings["tmdbId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["tmdbId", "rating"])

    users_encoded = pd.DataFrame(index=[0])

    # genre avg ratings
    genre_columns = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "IMAX",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    for genre in genre_columns:
        genre_movie_ids = movie_features.index[
            movie_features[f"movie_{genre}"] == 1
        ]
        users_encoded[genre] = ratings.loc[
            ratings["tmdbId"].isin(genre_movie_ids),
            "rating",
        ].mean()

    # decade avg ratings
    decade_columns = [
        col for col in movie_features.columns if col.startswith("movie_Decade_")
    ]
    for decade in decade_columns:
        decade_movie_ids = movie_features.index[movie_features[decade] == 1]
        user_decade = decade.removeprefix("movie_")
        users_encoded[user_decade] = ratings.loc[
            ratings["tmdbId"].isin(decade_movie_ids),
            "rating",
        ].mean()

    users_encoded["avg_rating"] = ratings["rating"].mean()
    users_encoded["std_rating"] = ratings["rating"].std()
    users_encoded["count_rating"] = ratings["rating"].count()

    users_encoded = users_encoded.fillna(0)

    user_feature_frame = users_encoded.rename(
        columns={col: f"user_{col}" for col in users_encoded.columns}
    )

    return user_feature_frame
