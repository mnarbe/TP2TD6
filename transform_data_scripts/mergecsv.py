import os
import json
import pandas as pd

def load_spotify_api_data(sp_api_dir="C:/Users/Agustin Mendez/Downloads/spotify_api_data"):
    """
    Load and flatten all Spotify API JSON files into a clean DataFrame.
    Handles both tracks and episodes with a unified schema.
    """
    rows = []

    for fname in os.listdir(sp_api_dir):
        fpath = os.path.join(sp_api_dir, fname)
        if not fname.endswith(".json"):
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {fpath}: {e}")
            continue

        if "uri" not in data:
            print(f"Skipping {fname}: no 'uri' field found")
            continue

        row = {
            "uri": data.get("uri"),
            "name": data.get("name"),
            "duration_ms": data.get("duration_ms"),
            "explicit": data.get("explicit"),
            "release_date": data.get("release_date"),
            # Track-specific
            "album_name": data.get("album", {}).get("name") if "album" in data else None,
            "album_release_date": data.get("album", {}).get("release_date") if "album" in data else None,
            "artist_name": data["artists"][0]["name"] if "artists" in data and data["artists"] else None,
            "popularity": data.get("popularity"),
            "track_number": data.get("track_number"),
            "genre1": data.get("genre1"),
            "genre2": data.get("genre2"),
            "genre3": data.get("genre3"),
            # Episode-specific
            "show_name": data.get("show", {}).get("name") if "show" in data else None,
            "show_publisher": data.get("show", {}).get("publisher") if "show" in data else None,
            "show_total_episodes": data.get("show", {}).get("total_episodes") if "show" in data else None,
        }

        rows.append(row)

    df_api = pd.DataFrame(rows)
    return df_api

def merge_train_with_api(train_path="train_data.txt", sp_api_dir="C:/Users/Agustin Mendez/Downloads/spotify_api_data", output_path="merged_data_2.csv"):
    """
    Merge train_data.txt with flattened Spotify API metadata.
    Tracks merge on 'spotify_track_uri', episodes on 'spotify_episode_uri'.
    """
    # Load train data
    df_train = pd.read_csv(train_path, sep="\t")  # adjust sep if needed
    print(f"Loaded train data: {df_train.shape[0]} rows")

    # Load Spotify API JSONs
    df_api = load_spotify_api_data(sp_api_dir)
    print(f"Loaded Spotify API data: {df_api.shape[0]} rows")

    # Columns to keep in the merge
    cols_to_keep = [
        "uri", "name", "duration_ms", "explicit", "release_date",
        "album_name", "album_release_date", "artist_name", "popularity",
        "track_number", "show_name", "show_publisher", "show_total_episodes",
        "genre1", "genre2", "genre3"
    ]

    # Merge track metadata
    merged = df_train.merge(
        df_api[cols_to_keep], left_on="spotify_track_uri", right_on="uri", how="left"
    )

    # Merge episode metadata
    merged = merged.merge(
        df_api[cols_to_keep], left_on="spotify_episode_uri", right_on="uri", how="left", suffixes=("", "_episode")
    )

    # Combine track vs episode columns
    for col in ["name", "duration_ms", "explicit", "release_date", "album_name",
                "album_release_date", "artist_name", "popularity", "track_number",
                "genre1", "genre2", "genre3",
                "show_name", "show_publisher", "show_total_episodes"]:
        col_episode = col + "_episode"
        if col_episode in merged.columns:
            merged[col] = merged[col].combine_first(merged[col_episode])
            merged.drop(columns=[col_episode], inplace=True)

    # Drop duplicate 'uri' columns
    merged.drop(columns=[c for c in merged.columns if c.startswith("uri")], inplace=True)

    print(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Save result
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

    return merged

def merge_test_with_api(train_path="test_data.txt", sp_api_dir="C:/Users/Agustin Mendez/Downloads/spotify_api_data", output_path="merged_test_data_2.csv"):
    """
    Merge test_data.txt with flattened Spotify API metadata.
    Tracks merge on 'spotify_track_uri', episodes on 'spotify_episode_uri'.
    """
    # Load train data
    df_train = pd.read_csv(train_path, sep="\t")  # adjust sep if needed
    print(f"Loaded train data: {df_train.shape[0]} rows")

    # Load Spotify API JSONs
    df_api = load_spotify_api_data(sp_api_dir)
    print(f"Loaded Spotify API data: {df_api.shape[0]} rows")

    # Columns to keep in the merge
    cols_to_keep = [
        "uri", "name", "duration_ms", "explicit", "release_date",
        "album_name", "album_release_date", "artist_name", "popularity",
        "track_number", "show_name", "show_publisher", "show_total_episodes",
        "genre1", "genre2", "genre3"
    ]

    # Merge track metadata
    merged = df_train.merge(
        df_api[cols_to_keep], left_on="spotify_track_uri", right_on="uri", how="left"
    )

    # Merge episode metadata
    merged = merged.merge(
        df_api[cols_to_keep], left_on="spotify_episode_uri", right_on="uri", how="left", suffixes=("", "_episode")
    )

    # Combine track vs episode columns
    for col in ["name", "duration_ms", "explicit", "release_date", "album_name",
                "album_release_date", "artist_name", "popularity", "track_number",
                "genre1", "genre2", "genre3",
                "show_name", "show_publisher", "show_total_episodes"]:
        col_episode = col + "_episode"
        if col_episode in merged.columns:
            merged[col] = merged[col].combine_first(merged[col_episode])
            merged.drop(columns=[col_episode], inplace=True)

    # Drop duplicate 'uri' columns
    merged.drop(columns=[c for c in merged.columns if c.startswith("uri")], inplace=True)

    print(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Save result
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

    return merged

if __name__ == "__main__":
    merge_train_with_api()
    print("-----------------MERGED TRAIN DATA--------------------")
    merge_test_with_api()
    print("-----------------MERGED TEST DATA--------------------")
