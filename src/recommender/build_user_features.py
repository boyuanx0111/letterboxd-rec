import pandas as pd

# lifted from user_embeddings notebook 

def build_user_features(ratings, movie_embeddings):
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
    # export genre columns
    pd.Series(genre_columns, name="genre").to_csv(
    "../data/genre_columns.csv",
    index=False
    )

    for genre in genre_columns:
        genre_movie_ids = movie_embeddings.index[movie_embeddings[f"movie_{genre}"] == 1]
        users_encoded[genre] = ratings[ratings.index.isin(genre_movie_ids)]["rating"].mean()

    # decade avg ratings
    decade_columns = [col for col in movie_embeddings.columns if col.startswith("movie_Decade_")]
    for decade in decade_columns:
        decade_movie_ids = movie_embeddings.index[movie_embeddings[decade] == 1]
        user_decade = decade.removeprefix("movie_")
        users_encoded[user_decade] = ratings[ratings.index.isin(decade_movie_ids)]["rating"].mean()

    users_encoded["avg_rating"] = ratings["rating"].mean()
    users_encoded["std_rating"] = ratings["rating"].std()
    users_encoded["count_rating"] = ratings["rating"].count()

    users_encoded = users_encoded.fillna(0)

    user_feature_frame = users_encoded.rename(
        columns={col: f"user_{col}" for col in users_encoded.columns}
    )

    return user_feature_frame