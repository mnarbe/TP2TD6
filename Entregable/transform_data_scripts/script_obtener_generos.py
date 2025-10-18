import os
import json
import requests
import time
import base64

# ==========================
# CONFIG
# ==========================
FOLDER_PATH = "C:/Users/Agustin Mendez/Downloads/spotify_api_data"
CLIENT_ID = "4d711678daf344d7b77500c2a6e55b28"
CLIENT_SECRET = "f7ff64b73a39454e85f3d4fa99b5ea20"
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.2
ARTIST_GENRES_FILE = "artist_genres.json"
ARTIST_IDS_FILE = "artist_ids.json"

# ==========================
# TOKEN CLIENT CREDENTIALS
# ==========================
TOKEN = None
TOKEN_EXPIRES_AT = 0

def get_new_token():
    global TOKEN, TOKEN_EXPIRES_AT
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    r = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    if r.status_code == 200:
        token_info = r.json()
        TOKEN = token_info["access_token"]
        TOKEN_EXPIRES_AT = time.time() + token_info["expires_in"]
        print(f"Nuevo token obtenido (expira en {token_info['expires_in']/60:.1f} min)")
    else:
        raise Exception(f"Error obteniendo token: {r.status_code} {r.text}")

def get_headers():
    if TOKEN is None or time.time() >= TOKEN_EXPIRES_AT - 30:
        get_new_token()
    return {"Authorization": f"Bearer {TOKEN}"}

# ==========================
# UTILIDADES
# ==========================
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==========================
# API CALLS
# ==========================
def get_artists_genres_batch(artist_ids):
    all_genres = {}
    for i in range(0, len(artist_ids), BATCH_SIZE):
        batch = artist_ids[i:i+BATCH_SIZE]
        url = f"https://api.spotify.com/v1/artists?ids={','.join(batch)}"

        while True:
            r = requests.get(url, headers=get_headers())
            if r.status_code == 200:
                data = r.json()
                for artist in data["artists"]:
                    all_genres[artist["id"]] = artist.get("genres", [])
                break
            elif r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 5))
                print(f"Rate limit alcanzado. Esperando {retry_after}s...")
                time.sleep(retry_after)
            else:
                print(f"Error {r.status_code} batch {i}-{i+BATCH_SIZE}")
                for artist_id in batch:
                    all_genres[artist_id] = []
                break

        time.sleep(RATE_LIMIT_DELAY)
    return all_genres

# ==========================
# ENRIQUECER TRACKS
# ==========================
def enrich_tracks_with_genres(folder_path, artist_genres_map):
    files = [f for f in os.listdir(folder_path) if f.endswith(".json") and "episode" not in f]
    print(f"Enriqueciendo {len(files)} tracks...")

    for file in files:
        filepath = os.path.join(folder_path, file)
        with open(filepath, "r", encoding="utf-8") as f:
            track = json.load(f)

        if not track.get("artists"):
            continue

        artist_id = track["artists"][0]["id"]
        genres = artist_genres_map.get(artist_id, [])

        track["genre1"] = genres[0] if len(genres) > 0 else ""
        track["genre2"] = genres[1] if len(genres) > 1 else ""
        track["genre3"] = genres[2] if len(genres) > 2 else ""

        track["enriched"] = True

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(track, f, indent=2, ensure_ascii=False)

# ==========================
# MAIN
# ==========================
def main():
    global TOKEN

    artist_genres_map = load_json(ARTIST_GENRES_FILE)
    artist_ids_cache = load_json(ARTIST_IDS_FILE)
    artist_ids_cache = set(artist_ids_cache) if isinstance(artist_ids_cache, list) else set()

    # --------------------------
    # Obtener todos los artistas
    # --------------------------
    if not artist_ids_cache:
        print("Escaneando archivos para obtener artist_ids...")
        artist_ids = set()
        files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json") and "episode" not in f]
        for file in files:
            filepath = os.path.join(FOLDER_PATH, file)
            with open(filepath, "r", encoding="utf-8") as f:
                track = json.load(f)
            if track.get("artists"):
                artist_ids.add(track["artists"][0]["id"])
        save_json(ARTIST_IDS_FILE, list(artist_ids))
        print(f"Guardados {len(artist_ids)} artist_ids en {ARTIST_IDS_FILE}")
    else:
        artist_ids = artist_ids_cache
        print(f"Cargados {len(artist_ids)} artist_ids desde cache.")

    # --------------------------
    # Filtrar los que faltan
    # --------------------------
    # artist_ids_to_fetch = [a for a in artist_ids if a not in artist_genres_map]
    # print(f"{len(artist_ids_to_fetch)} artistas sin géneros conocidos.")

    # # --------------------------
    # # Obtener géneros
    # # --------------------------
    # if artist_ids_to_fetch:
    #     new_artist_genres = get_artists_genres_batch(artist_ids_to_fetch)
    #     artist_genres_map.update(new_artist_genres)
    #     save_json(ARTIST_GENRES_FILE, artist_genres_map)
    #     print(f"Guardado {ARTIST_GENRES_FILE} con {len(artist_genres_map)} artistas.")

    # --------------------------
    # Enriquecer tracks (opcional)
    # --------------------------
    # Podés comentar esta línea si querés hacerlo después
    enrich_tracks_with_genres(FOLDER_PATH, artist_genres_map)

    print("Proceso completado.")

if __name__ == "__main__":
    main()
