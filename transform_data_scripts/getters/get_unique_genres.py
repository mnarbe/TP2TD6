import json

# Ruta al archivo JSON
INPUT_FILE = "artist_genres.json"
OUTPUT_FILE = "unique_genres.txt"

# Cargar el JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    artist_genres = json.load(f)

# Crear un set con todos los géneros únicos
unique_genres = set()

for genres in artist_genres.values():
    if isinstance(genres, list):
        unique_genres.update(genres)

# Guardar el listado ordenado
sorted_genres = sorted(unique_genres)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for genre in sorted_genres:
        f.write(genre + "\n")

print(f"{len(sorted_genres)} géneros únicos guardados en {OUTPUT_FILE}")
