import pandas as pd
import unicodedata

# ==============================
# CONFIGURACIÓN DE ENTRADAS
# ==============================

# Dataset principal con los géneros de cada registro
df = pd.read_csv("../merged_test_data_genresonly_c.csv", sep=",", low_memory=False)

# ==============================
# LISTAS DE REFERENCIA
# ==============================

popular_genres = ["children", "kids", "disney", "cartoon", "nursery", "family"]
rare_genres = ["comedy", "stand-up", "humor", "satire"]

kids_genres = ["children", "lullaby", "villancicos"]
comedy_genres = ["comedy", "spoken word", "humor"]

# english_genres = ["british", "american", "uk pop", "english indie"]
spanish_genres = ["argentine", "mexican", "spanish", "latino", "cumbia", "salsa", "bachata", "urbano", "folklore", "trap latino", "merengue", "reggaeton", "cuarteto", "tango"]
japanese_genres = ["j-", "japanese", "anime", "kayokyoku", "shibuya-kei"]

local_genres = ["argentine", "rock en español"]
latin_genres = ["latin", "urbano latino", "reggaeton", "salsa", "bachata", "merengue", "cumbia", "trap latino", "tropical", "urbano"]
low_energy_genres = ["ambient", "acoustic", "folk", "jazz", "classical", "lo-fi", "chill", "easy listening", "downtempo", "lounge"]
high_energy_genres = ["rock", "metal", "punk", "edm", "techno", "trance", "drum and bass", "hip hop", "trap", "house", "dance", "phonk"]
heavy_genres = ["metal", "hard"]
party_genres = ["dance", "house", "techno", "edm", "disco", "funk", "pop", "reggaeton", "salsa", "merengue", "tropical", "party", "cumbia"]
romantic_genres = ["r&b", "soul", "bolero", "romantic", "love", "bossa nova", "balada", "bachata"]
relaxing_genres = ["chill", "ambient", "lo-fi", "smooth jazz", "easy listening", "lounge", "soft", "downtempo", "space"]
instrumental_genres = ["classical", "instrumental", "soundtrack", "jazz", "orchestral", "symphonic", "piano", "ambient"]
# ==============================
# PREPROCESAMIENTO
# ==============================

# Asegurar que las columnas existan y sean texto (si faltan, las creamos vacías)
for col in ["genre1", "genre2", "genre3"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str).str.lower()

# Crear lista combinada por fila
df["all_genres"] = df[["genre1", "genre2", "genre3"]].values.tolist()

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def normalize_text(text):
    """Convierte texto a minúsculas y sin acentos."""
    if not isinstance(text, str):  # Maneja NaN, None, etc.
        return ""
    text = text.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def has_any(genres, genreset):
    """
    Devuelve True si alguno de los géneros en 'genres'
    contiene (como substring) alguno de los géneros del set.
    Ejemplo: si 'rock' ∈ genreset y 'rock nacional' está en genres → True.
    """
    if not isinstance(genres, list):
        return False

    # Filtrar valores vacíos o nulos
    norm_genres = [normalize_text(g) for g in genres if isinstance(g, str) and g.strip() != ""]
    norm_targets = [normalize_text(t) for t in genreset if isinstance(t, str) and t.strip() != ""]

    for g in norm_genres:
        for target in norm_targets:
            if target and target in g:  # búsqueda por substring
                return True
    return False

# ==============================
# CATEGORÍAS
# ==============================

df["has_popular_artist_genre"] = df["all_genres"].apply(lambda g: has_any(g, popular_genres))
df["has_rare_artist_genre"] = df["all_genres"].apply(lambda g: has_any(g, rare_genres))

df["is_kids_genre"] = df["all_genres"].apply(lambda g: has_any(g, kids_genres))
df["is_comedy_genre"] = df["all_genres"].apply(lambda g: has_any(g, comedy_genres))

# df["is_english"] = df["all_genres"].apply(lambda g: has_any(g, english_genres))
df["is_spanish"] = df["all_genres"].apply(lambda g: has_any(g, spanish_genres))
# df["is_english"] = False
df["has_japanese_genres"] = df["all_genres"].apply(lambda g: has_any(g, japanese_genres))

df["has_local_genre"] = df["all_genres"].apply(lambda g: has_any(g, local_genres))
df["has_latin_genre"] = df["all_genres"].apply(lambda g: has_any(g, latin_genres))
df["has_low_energy_genre"] = df["all_genres"].apply(lambda g: has_any(g, low_energy_genres))
df["has_high_energy_genre"] = df["all_genres"].apply(lambda g: has_any(g, high_energy_genres))
df["has_heavy_genre"] = df["all_genres"].apply(lambda g: has_any(g, heavy_genres))
df["has_party_genre"] = df["all_genres"].apply(lambda g: has_any(g, party_genres))
df["has_romantic_genre"] = df["all_genres"].apply(lambda g: has_any(g, romantic_genres))
df["has_relaxing_genre"] = df["all_genres"].apply(lambda g: has_any(g, relaxing_genres))
df["has_instrumental_genre"] = df["all_genres"].apply(lambda g: has_any(g, instrumental_genres))

# ==============================
# SALIDA FINAL
# ==============================

df.drop(columns=["all_genres"], inplace=True)
df.to_csv("../merged_test_data_new.csv", index=False)

print("✅ Nuevas columnas agregadas y archivo guardado como 'merged_test_data_new.csv'")