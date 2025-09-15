import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

COMPETITION_PATH = ""

train_file = os.path.join(COMPETITION_PATH, "merged_data.csv")
test_file = os.path.join(COMPETITION_PATH, "merged_test_data.csv")

train_df = pd.read_csv(train_file, sep=",", low_memory=False)
test_df = pd.read_csv(test_file, sep=",", low_memory=False)

# =========================================================
# Preparación: crear variable skip en train
# =========================================================
if "reason_end" in train_df.columns:
    train_df["skip"] = (train_df["reason_end"] == "fwdbtn").astype(int)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def plot_histograms(df, dataset_name):
    """Distribuciones de numéricas clave"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if "duration_ms" in df.columns:
        sns.histplot(df["duration_ms"]/1000, bins=50, ax=axes[0])
        axes[0].set_title(f"{dataset_name} - Duración (segundos)")
    if "popularity" in df.columns:
        sns.histplot(df["popularity"], bins=50, ax=axes[1])
        axes[1].set_title(f"{dataset_name} - Popularidad")
    plt.tight_layout()
    plt.show()

def plot_categorical_vs_target(df, dataset_name, col, top_n=10):
    """Categoría vs skip (solo en train)"""
    counts = (
        df.groupby(col)["skip"]
        .mean()
        .reset_index()
        .sort_values("skip", ascending=False)
    )
    if counts.shape[0] > top_n:
        counts = counts.head(top_n)
    plt.figure(figsize=(8,5))
    sns.barplot(x=col, y="skip", data=counts)
    plt.title(f"{dataset_name} - Proporción de skips por {col}")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_temporal_patterns(df, dataset_name, with_skip=False):
    if "timestamp" not in df.columns:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    if with_skip:
        # proporción de skips
        hourly = df.groupby("hour")["skip"].mean()
        plt.figure(figsize=(8,5))
        sns.lineplot(x=hourly.index, y=hourly.values)
        plt.title(f"{dataset_name} - Proporción de skips por hora")
        plt.show()
        pivot = df.groupby(["dayofweek","hour"])["skip"].mean().unstack()
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot, cmap="YlGnBu")
        plt.title(f"{dataset_name} - Heatmap skips (día vs hora)")
        plt.show()
    else:
        # cantidad de reproducciones
        plt.figure(figsize=(8,5))
        sns.countplot(x="hour", data=df)
        plt.title(f"{dataset_name} - Reproducciones por hora")
        plt.show()
        plt.figure(figsize=(8,5))
        sns.countplot(x="dayofweek", data=df)
        plt.title(f"{dataset_name} - Reproducciones por día (0=Lunes)")
        plt.show()

def plot_top_artists(df, dataset_name, with_skip=False, top_n=10):
    if "artist_name" not in df.columns: return
    if with_skip:
        stats = (
            df.groupby("artist_name")
            .agg(reproductions=("skip","count"), skip_rate=("skip","mean"))
            .reset_index()
        )
        top = stats.sort_values("reproductions", ascending=False).head(top_n)
        plt.figure(figsize=(10,6))
        sns.barplot(x="artist_name", y="skip_rate", data=top)
        plt.title(f"{dataset_name} - Tasa de skips en top {top_n} artistas")
        plt.xticks(rotation=45, ha="right")
        plt.show()
    else:
        counts = df["artist_name"].value_counts().head(top_n)
        plt.figure(figsize=(10,6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f"{dataset_name} - Top {top_n} artistas más reproducidos")
        plt.xticks(rotation=45, ha="right")
        plt.show()

def plot_correlation(df, dataset_name, with_skip=False):
    cols = ["duration_ms","popularity","track_number","show_total_episodes"]
    if with_skip:
        cols.append("skip")
    # Solo usar columnas que existen en el DataFrame
    cols_existentes = [c for c in cols if c in df.columns]
    if not cols_existentes:
        print(f"No hay columnas numéricas para correlación en {dataset_name}")
        return
    num_df = df[cols_existentes].dropna(axis=1, how="all")
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title(f"{dataset_name} - Matriz de correlación")
        plt.show()

def compare_train_test(train_df, test_df, col):
    """Comparación numérica (KDE)"""
    if col not in train_df.columns or col not in test_df.columns: return
    plt.figure(figsize=(8,5))
    sns.kdeplot(train_df[col].dropna(), label="Train", fill=True, alpha=0.4)
    sns.kdeplot(test_df[col].dropna(), label="Test", fill=True, alpha=0.4)
    plt.title(f"Comparación Train vs Test - {col}")
    plt.legend()
    plt.show()

def compare_categorical(train_df, test_df, col, top_n=10):
    if col in train_df.columns and col in test_df.columns:
        train_counts = train_df[col].value_counts(normalize=True).head(top_n)
        test_counts = test_df[col].value_counts(normalize=True).head(top_n)
        compare_df = pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0)
        compare_df.plot(kind="bar", figsize=(10,6))
        plt.title(f"Comparación Train vs Test - {col}")
        plt.ylabel("Proporción")
        plt.xticks(rotation=45, ha="right")
        plt.show()

# =========================================================
# 1) Gráficos comparativos Train vs Test
# =========================================================
for col in ["duration_ms","popularity","track_number","show_total_episodes"]:
    compare_train_test(train_df, test_df, col)
for col in ["platform","conn_country","shuffle","explicit","incognito_mode"]:
    compare_categorical(train_df, test_df, col)

# =========================================================
# 2) Gráficos en Train con skip
# =========================================================
print("\n=== Train con skip ===\n")
plot_histograms(train_df, "Train")
for col in ["platform","conn_country","shuffle","explicit","incognito_mode"]:
    if col in train_df.columns:
        plot_categorical_vs_target(train_df, "Train", col)
plot_temporal_patterns(train_df, "Train", with_skip=True)
plot_top_artists(train_df, "Train", with_skip=True)
plot_correlation(train_df, "Train", with_skip=True)

# =========================================================
# 3) Gráficos independientes Train y Test sin skip
# =========================================================
for df, name in [(train_df,"Train"),(test_df,"Test")]:
    print(f"\n=== {name} sin skip ===\n")
    plot_histograms(df, name)
    plot_temporal_patterns(df, name, with_skip=False)
    plot_top_artists(df, name, with_skip=False)
    plot_correlation(df, name, with_skip=False)
    for col in ["platform","conn_country","shuffle","explicit","incognito_mode"]:
        plot_categorical_counts = df[col].value_counts().head(10)
        plt.figure(figsize=(8,5))
        sns.barplot(x=plot_categorical_counts.index, y=plot_categorical_counts.values)
        plt.title(f"{name} - Frecuencia de {col} (Top 10)")
        plt.xticks(rotation=45, ha="right")
        plt.show()


# Feature engineering: A partir del análisis podés crear variables predictoras:
# * is_short_track (ej: duración < 1 min).
# * is_popular_track (umbral de popularity).
# * hour_of_day, day_of_week.
# * is_shuffle, is_incognito, is_offline.
# * country_top (agrupando países menos frecuentes en "other").
# * artist_popularity o tasa histórica de skip de un artista/canción.

# Encoding de variables categóricas:
# * platform, conn_country, explicit, shuffle, incognito_mode → one-hot encoding.
# * artist_name, track_name, album_name probablemente demasiado granulares → mejor transformarlas en métricas agregadas (ej: probabilidad histórica de salto por artista).