import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import requests
from dotenv import load_dotenv

def convert_tv_json_to_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tv_json = [json.loads(line) for line in file if line.strip()]
    tv_df = pd.DataFrame(tv_json)
    tv_df.to_csv("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.csv", index=False)

def fetch_tmdb_tv(tv_id):
    url = f"https://api.themoviedb.org/3/tv/{tv_id}"
    load_dotenv("letterboxd-rec\\src\\.env")
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        raise RuntimeError("TMDB_API_KEY is missing; set it in letterboxd-rec/src/.env")

    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json"
    }

    response = requests.get(
        url,
        headers=request_headers,
        timeout=20
    )
    response.raise_for_status()
    data = response.json()

    return {
        "tmdb_id": data.get("id"),
        "name": data.get("name"),
        "overview": data.get("overview"),
        "first_air_date": data.get("first_air_date"),
        "creators": "|".join(creator["name"] for creator in data.get("created_by", [])),
        "genres": "|".join(genre["name"] for genre in data.get("genres", [])),
        "vote_count": data.get("vote_count", 0),
        "vote_average": data.get("vote_average", 0.0),
    }

def fetch_several(tv_id_path, max_workers=5):
    """Fetch TV details for IDs in a CSV or JSON Lines file into a DataFrame."""
    if tv_id_path.endswith(".csv"):
        ids_df = pd.read_csv(tv_id_path)
        if "id" not in ids_df.columns:
            raise ValueError("CSV must contain an 'id' column")
        tv_ids = ids_df["id"].dropna().tolist()
    elif tv_id_path.endswith(".json"):
        with open(tv_id_path, "r", encoding="utf-8") as file:
            tv_ids = [json.loads(line)["id"] for line in file if line.strip()]
    else:
        raise ValueError("tv_id_path must point to a .csv or JSON Lines .json file")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_tmdb_tv, tv_id): tv_id for tv_id in tv_ids}
        for future in as_completed(futures):
            tv_id = futures[future]
            try:
                results.append(future.result())
            except requests.RequestException as error:
                print(f"Failed to fetch TV ID {tv_id}: {error}")

    return pd.DataFrame(results)

def create_tv_data_csv(tv_id_path):
    if tv_id_path.endswith('.csv'):
        tv_data = fetch_several(tv_id_path)
    else:
        raise ValueError("Convert tv_id_path to CSV first")
    tv_data.to_csv("letterboxd-rec\\data\\tv_recommender\\tv_series_data.csv", index=False)


def join_tv_show_data(tv_id_path, tv_data_path):
    if tv_id_path.endswith('.csv'):
        tv_ids = pd.read_csv(tv_id_path)
    else:
        raise ValueError("tv_id_path must be a CSV file")
    if tv_data_path.endswith('.csv'):
        tv_data = pd.read_csv(tv_data_path)
    else:
        raise ValueError("tv_data_path must be a CSV file")

    tv_mdata = pd.merge(tv_ids, tv_data, left_on='id', right_on='tmdb_id', how='left')
    tv_mdata.to_csv("letterboxd-rec\\data\\tv_recommender\\tv_mdata.csv", index=False)



if __name__ == "__main__":
    #tv_data = load_tv_data("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.json") 
    #print(tv_data[:5])
    convert_tv_json_to_csv("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.json")
    create_tv_data_csv("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.csv")
