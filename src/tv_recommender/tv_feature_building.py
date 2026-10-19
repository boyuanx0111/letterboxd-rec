import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_tv_data(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("file_path must point to a .csv file")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    pd_tv_data = pd.read_csv(file_path)
    if pd_tv_data.empty:
        raise ValueError(f"The CSV file at {file_path} is empty.")
    return pd_tv_data

def weight_tv_genres(genre_string):
    genre_mapping = {
        "action & adventure": {"Action": 1.0, "Adventure": 1.0},
    "animation": {"Animation": 1.0},
    "comedy": {"Comedy": 1.0},
    "crime": {"Crime": 1.0},
    "documentary": {"Documentary": 1.0},
    "drama": {"Drama": 1.0},
    "family": {"Children": 0.7},
    "kids": {"Children": 1.0},
    "mystery": {"Mystery": 1.0, "Thriller": 0.5},
    "news": {},
    "reality": {},
    "sci-fi & fantasy": {"Sci-Fi": 1.0, "Fantasy": 1.0},
    "film noir": {"Film-Noir": 1.0},
    "horror": {"Horror": 1.0, "Thriller": 0.5},
    "musical": {"Musical": 1.0},
    "soap": {"Drama": 0.8, "Romance": 0.6},
    "war": {"War": 1.0},
    "thriller": {"Thriller": 1.0, "Mystery": 0.5},
    "talk": {}, # Maybe add some weights here later 
    "western": {"Western": 1.0}
    }
    genres = genre_string.split('|')
    weighted_genres = {}
    for genre in genres:
        new_genres = genre_mapping.get(genre.lower(), {})
        for new_genre, weight in new_genres.items():
            if new_genre in weighted_genres:
                weighted_genres[new_genre] = max(weighted_genres[new_genre], weight)
            else:
                weighted_genres[new_genre] = weight
    return weighted_genres

def encode_genres(pd_tv_data):
    tv_genres = pd_tv_data['genres'].fillna("").apply(weight_tv_genres).tolist()
    genres_encoded = pd.DataFrame(tv_genres, index=pd_tv_data.index).fillna(0)
    return genres_encoded

def encode_release_decade(pd_tv_data):
    release_years = pd_tv_data['first_air_date'].fillna("0000-00-00").str.split("-").str[0].astype(int)
    release_decades = ((release_years // 10) * 10).astype(str).tolist()
    current_decade = (pd.Timestamp.now().year // 10) * 10
    release_decades = [release_decade if int(release_decade) <= current_decade else None for release_decade in release_decades]
    pd_release_decade = pd.DataFrame({'release_decade': release_decades}, index=pd_tv_data.index)
    encoded_release_decades = pd_release_decade['release_decade'].str.get_dummies().astype(int)
    return encoded_release_decades

def encode_vote_count(pd_tv_data):
    """Log-transform vote counts to reduce the effect of extreme values."""
    vote_counts = pd.to_numeric(pd_tv_data['vote_count'], errors='coerce').fillna(0)
    return pd.DataFrame(np.log1p(vote_counts.clip(lower=0)).rename('vote_count'), index=pd_tv_data.index)

def encode_vote_average(pd_tv_data):
    """Scale 0-10 TMDB ratings to 0-1 with less distance at the high end."""
    vote_averages = pd.to_numeric(pd_tv_data['vote_average'], errors='coerce').fillna(0)
    scaled = (vote_averages.clip(lower=0, upper=10) / 10) ** 0.5
    return pd.DataFrame(scaled.rename('vote_average'), index=pd_tv_data.index)

def encode_creators(pd_tv_data):
    creators_series = pd_tv_data['creators'].fillna("").str.get_dummies(sep='|').astype(int)
    return pd.DataFrame(creators_series, index=pd_tv_data.index)

if __name__ == "__main__":
    tv_data_path = "letterboxd-rec\\data\\tv_recommender\\tv_series_data.csv"
    tv_data = load_tv_data(tv_data_path)
    genre_encoded = encode_genres(tv_data)
    release_decade_encoded = encode_release_decade(tv_data)
    vote_count_encoded = encode_vote_count(tv_data)
    vote_average_encoded = encode_vote_average(tv_data)
    #creators_encoded = encode_creators(tv_data)
    print(vote_average_encoded.head())
    print(vote_count_encoded.head())
    print(release_decade_encoded.head())
    print(genre_encoded.head())


