import pandas as pd
import os
from pathlib import Path

def load_tv_data(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("file_path must point to a .csv file")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

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

if __name__ == "__main__":
    tv_data_path = "letterboxd-rec\\data\\tv_recommender\\tv_series_data.csv"
    tv_data = load_tv_data(tv_data_path)
    genre_encoded = encode_genres(tv_data)
    print(genre_encoded.head())
    print(genre_encoded.columns)