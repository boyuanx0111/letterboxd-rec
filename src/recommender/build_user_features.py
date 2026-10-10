import pandas as pd

# lifted from user_embeddings notebook 

def build_user_features(ratings, movie_feature_frame):
    """
    ratings (per user ratings) - DataFrame containing tmdbId, rating, timestamp (optional)
    movie_embeddings - DataFrame containing tmdbId and ALL movie embeddings
    """

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
    # run first time or have saved under data when importing 
    # export genre columns
    # pd.Series(genre_columns, name="genre").to_csv(
    # "../data/genre_columns.csv",
    # index=False
    #)

    for genre in genre_columns:
        genre_movie_ids = movie_feature_frame.index[movie_feature_frame[f"movie_{genre}"] == 1]
        users_encoded[genre] = ratings[ratings.index.isin(genre_movie_ids)]["rating"].mean()

    # decade avg ratings
    decade_columns = [col for col in movie_feature_frame.columns if col.startswith("movie_Decade_")]
    for decade in decade_columns:
        decade_movie_ids = movie_feature_frame.index[movie_feature_frame[decade] == 1]
        user_decade = decade.removeprefix("movie_")
        users_encoded[user_decade] = ratings[ratings.index.isin(decade_movie_ids)]["rating"].mean()

    # Language average + frequency
    # tmdb_1 to tmdb_11

    language_cols = [f"tmdb_{i}" for i in range(1, 12)]

    total_rated_movies = ratings.index.nunique()

    for language_col in language_cols:

        movie_col = f"movie_{language_col}"

        language_movie_ids = movie_feature_frame.index[
            movie_feature_frame[movie_col] == 1
        ]

        language_ratings = ratings[
            ratings.index.isin(language_movie_ids)
        ]

        # Average rating for this language
        users_encoded[f"{language_col}_avg"] = (
            language_ratings["rating"].mean()
        )

        # Proportion of rated movies in this language
        users_encoded[f"{language_col}_freq"] = (
            language_ratings.index.nunique() / total_rated_movies
            if total_rated_movies > 0 else 0
        )

    # Country average ratings
    # tmdb_12 to tmdb_26

    country_cols = [f"tmdb_{i}" for i in range(12, 27)]

    for country_col in country_cols:

        movie_col = f"movie_{country_col}"

        country_movie_ids = movie_feature_frame.index[
            movie_feature_frame[movie_col] == 1
        ]

        users_encoded[country_col] = ratings[
            ratings.index.isin(country_movie_ids)
        ]["rating"].mean()

    # Numeric TMDB features
    # tmdb_561 = vote_count
    # tmdb_562 = vote_average
    # tmdb_563 = revenue

    numeric_tmdb_features = {
        "vote_count": "movie_tmdb_561",
        "vote_average": "movie_tmdb_562",
        "revenue": "movie_tmdb_563"
    }

    # Only include movies that the user has rated
    rated_movie_features = movie_feature_frame.reindex(ratings.index)

    for feature_name, movie_col in numeric_tmdb_features.items():

        users_encoded[feature_name] = (
            rated_movie_features[movie_col].mean()
        )

    users_encoded["avg_rating"] = ratings["rating"].mean()
    users_encoded["std_rating"] = ratings["rating"].std()
    users_encoded["count_rating"] = ratings["rating"].count()

    users_encoded = users_encoded.fillna(0)

    user_feature_frame = users_encoded.rename(
        columns={col: f"user_{col}" for col in users_encoded.columns}
    )

    return user_feature_frame
