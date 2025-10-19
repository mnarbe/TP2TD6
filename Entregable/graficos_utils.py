import pandas as pd
import matplotlib.pyplot as plt
from database_utils import *

plt.style.use('seaborn-v0_8')


# ==========================
# PREPROCESAMIENTO
# ==========================
def load_dataset(file_path: str, sep: str = ',') -> pd.DataFrame:
    """
    Carga y preprocesa el dataset de Spotify.
    Convierte timestamps, filtra outliers y agrega columnas útiles.
    """
    df = pd.read_csv(file_path, sep=sep)
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df.dropna(subset=['ts'], inplace=True)

    # Agregar hora para análisis temporales
    df['hour'] = df['ts'].dt.hour

    return df


# ==========================
# ANÁLISIS DESCRIPTIVOS
# ==========================
def analyze_basic_info(df: pd.DataFrame) -> None:
    """
    Muestra información general sobre el dataset.
    """
    print("=== Información general ===")
    print(f"Período: {df['ts'].min()} — {df['ts'].max()}")
    print(f"Total de filas: {len(df)}")
    print(f"Usuarios únicos: {df['obs_id'].nunique() if 'obs_id' in df.columns else 'N/A'}")
    if 'master_metadata_album_artist_name' in df.columns:
        print(f"Artistas únicos: {df['master_metadata_album_artist_name'].nunique()}")
    if 'master_metadata_album_album_name' in df.columns:
        print(f"Álbumes únicos: {df['master_metadata_album_album_name'].nunique()}")

# ==========================
# ANÁLISIS TEMPORAL
# ==========================
def plot_activity_by_hour(df: pd.DataFrame) -> None:
    """
    Grafica la distribución de reproducciones por hora del día.
    """
    hourly = df['hour'].value_counts().sort_index()
    hourly.plot(kind='bar', figsize=(10, 5))
    plt.title("Actividad por hora del día")
    plt.xlabel("Hora")
    plt.ylabel("Cantidad de reproducciones")
    plt.tight_layout()
    plt.show()


def plot_daily_trend(df: pd.DataFrame) -> None:
    """
    Grafica la evolución temporal de reproducciones por día.
    """
    daily = df.resample('D', on='ts').size()
    daily.plot(figsize=(12, 5))
    plt.title("Evolución de reproducciones diarias")
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad de reproducciones")
    plt.tight_layout()
    plt.show()


# ==========================
# ANÁLISIS DE SKIPS (si existe la columna)
# ==========================
def plot_target_by_hour(df: pd.DataFrame) -> None:
    """
    Grafica la proporción de skips (target) por hora.
    """
    if 'target' not in df.columns:
        return

    hourly_skip_rate = df.groupby('hour')['target'].mean()
    hourly_skip_rate.plot(kind='line', marker='o', figsize=(10, 5))
    plt.title("Proporción de skips por hora")
    plt.xlabel("Hora")
    plt.ylabel("Skip rate")
    plt.tight_layout()
    plt.show()


def plot_target_vs_duration(df: pd.DataFrame, bins: int = 30) -> None:
    """
    Muestra relación entre duración de canciones y skips.
    """
    if 'target' not in df.columns or 'track_duration_ms' not in df.columns:
        return

    grouped = df.groupby(pd.cut(df['track_duration_ms'], bins))['target'].mean()

    grouped.plot(kind='bar', figsize=(12, 5))
    plt.title("Skip rate según duración de canción")
    plt.xlabel("Duración (min)")
    plt.ylabel("Skip rate")
    plt.tight_layout()
    plt.show()


# ==========================
# MAIN
# ==========================
def plotGraficos():
    # Cargar una sola vez
    file_path = "./merged_data.csv"
    df = load_dataset(file_path, sep=',')
    df = processTargetAndTestMask(df)

    # Análisis descriptivos
    # analyze_basic_info(df)

    # Análisis temporal
    # plot_activity_by_hour(df)
    plot_daily_trend(df)

    # # Análisis de skips (si aplica)
    # plot_target_by_hour(df)
    # plot_target_vs_duration(df)


# if __name__ == "__main__":
#     main()
