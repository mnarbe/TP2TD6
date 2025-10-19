import pandas as pd
from collections import Counter

# ==========================
# CONFIG
# ==========================
CSV_PATH_TRAIN = "merged_data.csv"
OUTPUT_PATH_TRAIN = "genre_counts_train.csv"
CSV_PATH_TEST = "merged_test_data.csv"
OUTPUT_PATH_TEST = "genre_counts_test.csv"

# ==========================
# PROCESO
# ==========================
def process(csv_path, output_path):
    # Cargar dataset
    df = pd.read_csv(csv_path)

    # Tomar las columnas relevantes
    genre_cols = ["genre1", "genre2", "genre3"]
    genres = []

    # Combinar y limpiar
    for col in genre_cols:
        if col in df.columns:
            genres.extend(df[col].dropna().astype(str).str.lower().str.strip())

    # Contar ocurrencias
    counter = Counter(genres)

    # Convertir a DataFrame ordenado
    genre_counts = pd.DataFrame(counter.items(), columns=["genre", "count"]).sort_values("count", ascending=False)

    # Guardar a CSV
    genre_counts.to_csv(output_path, index=False, encoding="utf-8")

    # Mostrar resumen
    print(f"Se encontraron {len(genre_counts)} géneros únicos.")
    print("Top 20 géneros más frecuentes:\n")
    print(genre_counts.head(20).to_string(index=False))

process(CSV_PATH_TEST, OUTPUT_PATH_TEST)
process(CSV_PATH_TRAIN, OUTPUT_PATH_TRAIN)