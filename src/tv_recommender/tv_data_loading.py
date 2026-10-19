import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import pandas as pd
import os
import requests
from dotenv import load_dotenv

def convert_tv_json_to_csv(file_path):
    if os.path.isfile("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.csv"):
        print("CSV file already exists. Skipping conversion.")
        return
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
    first_air_date = data.get("first_air_date")
    if first_air_date and date.fromisoformat(first_air_date) > date.today():
        return None

    return {
        "tmdb_id": data.get("id"),
        "name": data.get("name"),
        "overview": data.get("overview").replace("\n", " "),
        "first_air_date": data.get("first_air_date"),
        "creators": "|".join(creator["name"] for creator in data.get("created_by", [])),
        "genres": "|".join(genre["name"] for genre in data.get("genres", [])),
        "vote_count": data.get("vote_count", 0),
        "vote_average": data.get("vote_average", 0.0),
    }

def fetch_several(tv_id_path, max_workers=5, batch_size=100, num_batches=None, verbose=True):
    """Fetch TV details for IDs in a CSV or JSON Lines file into a DataFrame."""
    if tv_id_path.endswith(".csv"):
        ids_df = pd.read_csv(tv_id_path)
        if "id" not in ids_df.columns:
            raise ValueError("CSV must contain an 'id' column")
        tv_ids = ids_df["id"].dropna().tolist()
    else:
        raise ValueError("tv_id_path must point to a .csv file")

    # Keep the first occurrence of each ID, preserving input order.
    tv_ids = list(dict.fromkeys(tv_ids)) # Instead of using list(), this is used to remove duplicates
    results = []
    batches_processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(tv_ids), batch_size):
            if num_batches is None or num_batches > 0:
                batch = tv_ids[start:start + batch_size]
                futures = {executor.submit(fetch_tmdb_tv, tv_id): tv_id for tv_id in batch}
                for future in as_completed(futures):
                    tv_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except requests.RequestException as error:
                        print(f"Failed to fetch TV ID {tv_id}: {error}")
                if num_batches is not None:
                    num_batches -= 1
                batches_processed += 1
                if verbose:
                    print(f"Processed batch {batches_processed}, total results: {len(results)}")
    pd_results = pd.DataFrame(results)
    pd_results = pd_results.sort_values(by="tmdb_id")
    return pd_results

def create_tv_data_csv(tv_id_path, max_workers=5, batch_size=100, num_batches=None):
    if os.path.isfile("letterboxd-rec\\data\\tv_recommender\\tv_series_data.csv"):
        print("CSV file already exists. Skipping data fetching.")
        return
    if tv_id_path.endswith('.csv'):
        tv_data = fetch_several(tv_id_path, max_workers, batch_size, num_batches)
        print(tv_data.head())
        print(tv_data['overview'].head())
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
    create_tv_data_csv("letterboxd-rec\\data\\tv_recommender\\tv_series_ids.csv", 12, 100)
