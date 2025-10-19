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



def plot_top_artists_with_skips_and_plays(df: pd.DataFrame, top_n: int = 8) -> None:
    """
    Genera un gráfico de barras que muestra los top N artistas más escuchados
    con barras agrupadas mostrando reproducciones totales y cantidad de skips.
    """
    if 'target' not in df.columns or 'master_metadata_album_artist_name' not in df.columns:
        print("Error: Se requieren las columnas 'target' y 'master_metadata_album_artist_name'")
        return

    # Filtrar datos válidos (excluir NaN en artistas)
    df_clean = df.dropna(subset=['master_metadata_album_artist_name'])
    
    if len(df_clean) == 0:
        print("Error: No hay datos válidos de artistas")
        return

    # Calcular métricas por artista
    artist_stats = df_clean.groupby('master_metadata_album_artist_name').agg({
        'target': ['sum', 'count']  # cantidad de skips y total de reproducciones
    }).round(0)
    
    # Aplanar columnas multi-nivel
    artist_stats.columns = ['total_skips', 'total_plays']
    artist_stats = artist_stats.reset_index()
    
    # Calcular reproducciones completadas (no skips)
    artist_stats['completed_plays'] = artist_stats['total_plays'] - artist_stats['total_skips']
    
    # Filtrar artistas con al menos 10 reproducciones para tener datos confiables
    artist_stats = artist_stats[artist_stats['total_plays'] >= 10]
    
    if len(artist_stats) == 0:
        print("Error: No hay artistas con suficientes reproducciones")
        return
    
    # Obtener los top N artistas por reproducciones
    top_artists = artist_stats.nlargest(top_n, 'total_plays')
    
    # Crear figura con un solo subplot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Configurar posiciones de las barras
    x = range(len(top_artists))
    width = 0.35
    
    # Crear barras agrupadas
    bars1 = ax.bar([i - width/2 for i in x], top_artists['completed_plays'], 
                   width, label='Reproducciones Completadas', color='lightgreen', 
                   alpha=0.8, edgecolor='darkgreen', linewidth=1)
    
    bars2 = ax.bar([i + width/2 for i in x], top_artists['total_skips'], 
                   width, label='Skips', color='lightcoral', 
                   alpha=0.8, edgecolor='darkred', linewidth=1)
    
    # Configurar el gráfico
    ax.set_xlabel('Artistas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad de Reproducciones', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_artists['master_metadata_album_artist_name'], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Valores en barras de reproducciones completadas
        if top_artists.iloc[i]['completed_plays'] > 0:
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(top_artists['total_plays'])*0.01, 
                    f'{int(top_artists.iloc[i]["completed_plays"]):,}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Valores en barras de skips
        if top_artists.iloc[i]['total_skips'] > 0:
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(top_artists['total_plays'])*0.01, 
                    f'{int(top_artists.iloc[i]["total_skips"]):,}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # === ANÁLISIS ESTADÍSTICO ===
    print("\n" + "="*80)
    print(f"ANÁLISIS DE LOS TOP {top_n} ARTISTAS: REPRODUCCIONES VS SKIPS")
    print("="*80)
    
    # Estadísticas generales
    print(f"\nESTADISTICAS GENERALES:")
    print(f"   Skip rate promedio general: {df['target'].mean():.4f}")
    print(f"   Total de reproducciones analizadas: {len(df_clean):,}")
    print(f"   Artistas unicos: {df_clean['master_metadata_album_artist_name'].nunique():,}")
    
    # Tabla detallada de los top artistas
    print(f"\nTOP {top_n} ARTISTAS DETALLADOS:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Artista':<25} {'Total':<8} {'Completadas':<12} {'Skips':<8} {'Skip%':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_artists.iterrows(), 1):
        skip_rate = row['total_skips'] / row['total_plays']
        skip_indicator = "[BAJO]" if skip_rate < 0.15 else "[MEDIO]" if skip_rate < 0.25 else "[ALTO]"
        print(f"{i:<4} {row['master_metadata_album_artist_name'][:24]:<25} "
              f"{row['total_plays']:,} {row['completed_plays']:,} {row['total_skips']:,} "
              f"{skip_rate:.1%} {skip_indicator}")
    
    # Análisis de categorías de skip rate
    print(f"\nANALISIS POR CATEGORIAS DE SKIP RATE:")
    top_artists['skip_rate'] = top_artists['total_skips'] / top_artists['total_plays']
    
    low_skip = top_artists[top_artists['skip_rate'] < 0.15]
    medium_skip = top_artists[(top_artists['skip_rate'] >= 0.15) & (top_artists['skip_rate'] < 0.25)]
    high_skip = top_artists[top_artists['skip_rate'] >= 0.25]
    
    print(f"   Skip rate bajo (< 15%): {len(low_skip)} artistas")
    if len(low_skip) > 0:
        print(f"      - {', '.join(low_skip['master_metadata_album_artist_name'].tolist())}")
    
    print(f"   Skip rate medio (15%-25%): {len(medium_skip)} artistas")
    if len(medium_skip) > 0:
        print(f"      - {', '.join(medium_skip['master_metadata_album_artist_name'].tolist())}")
    
    print(f"   Skip rate alto (>= 25%): {len(high_skip)} artistas")
    if len(high_skip) > 0:
        print(f"      - {', '.join(high_skip['master_metadata_album_artist_name'].tolist())}")
    
    # Estadísticas de skips
    print(f"\nESTADISTICAS DE SKIPS:")
    print(f"   Total de skips en top {top_n}: {top_artists['total_skips'].sum():,}")
    print(f"   Total de reproducciones completadas: {top_artists['completed_plays'].sum():,}")
    print(f"   Skip rate promedio en top {top_n}: {top_artists['skip_rate'].mean():.1%}")
    
    # Artista con más skips
    max_skips_idx = top_artists['total_skips'].idxmax()
    max_skips_artist = top_artists.loc[max_skips_idx]
    print(f"   Artista con mas skips: {max_skips_artist['master_metadata_album_artist_name']} "
          f"({max_skips_artist['total_skips']:,} skips)")
    
    # Artista con mejor retención
    min_skip_rate_idx = top_artists['skip_rate'].idxmin()
    min_skip_rate_artist = top_artists.loc[min_skip_rate_idx]
    print(f"   Artista con mejor retencion: {min_skip_rate_artist['master_metadata_album_artist_name']} "
          f"({min_skip_rate_artist['skip_rate']:.1%} skip rate)")


def plot_heatmap_skip_rate_by_hour_weekday(df: pd.DataFrame) -> None:
    """
    Genera solo el heatmap de skip rate por hora del día y día de la semana.
    Escala ajustada para el rango real de skip rates (0-0.45).
    """
    if 'target' not in df.columns or 'hour' not in df.columns:
        print("Error: Se requieren las columnas 'target' y 'hour'")
        return

    # Agregar día de la semana si no existe
    if 'weekday' not in df.columns:
        df['weekday'] = df['ts'].dt.dayofweek  # 0=Lunes, 6=Domingo
    
    # Crear figura solo para el heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Crear pivot table para heatmap
    heatmap_data = df.groupby(['weekday', 'hour'])['target'].mean().unstack(fill_value=0)
    
    # Mapear números a nombres de días
    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    heatmap_data.index = [day_names[i] for i in heatmap_data.index]
    
    # Crear heatmap con escala ajustada (0-0.45)
    im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.45)
    
    # Configurar ejes
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(7))
    ax.set_yticklabels(day_names)
    ax.set_xlabel('Hora del día', fontsize=12, fontweight='bold')
    ax.set_ylabel('Día de la semana', fontsize=12, fontweight='bold')
    
    # Agregar líneas de cuadrícula
    ax.set_xticks(range(24), minor=True)
    ax.set_yticks(range(7), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Agregar colorbar con escala personalizada
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Skip Rate', fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
    cbar.set_ticklabels(['0.0', '0.1', '0.2', '0.3', '0.4'])
    
    # Agregar valores en cada celda si el heatmap no es muy denso
    for i in range(7):
        for j in range(0, 24, 2):  # Solo cada 2 horas para no saturar
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value):
                text_color = 'white' if value > 0.2 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas del heatmap
    print("\n" + "="*60)
    print("ANÁLISIS DEL HEATMAP: SKIP RATE POR HORA Y DÍA")
    print("="*60)
    
    # Estadísticas generales del heatmap
    print(f"\nESTADISTICAS DEL HEATMAP:")
    print(f"   Skip rate minimo: {heatmap_data.min().min():.4f}")
    print(f"   Skip rate maximo: {heatmap_data.max().max():.4f}")
    print(f"   Skip rate promedio: {heatmap_data.mean().mean():.4f}")
    
    # Encontrar las celdas con mayor skip rate
    print(f"\nTOP 5 COMBINACIONES CON MAYOR SKIP RATE:")
    flat_data = heatmap_data.stack().reset_index()
    flat_data.columns = ['dia', 'hora', 'skip_rate']
    top_combinations = flat_data.nlargest(5, 'skip_rate')
    
    for _, row in top_combinations.iterrows():
        print(f"   {row['dia']} a las {row['hora']:2d}:00 - Skip rate: {row['skip_rate']:.4f}")
    
    # Análisis por día de la semana
    print(f"\nSKIP RATE PROMEDIO POR DIA:")
    daily_avg = heatmap_data.mean(axis=1).sort_values(ascending=False)
    for dia, rate in daily_avg.items():
        print(f"   {dia}: {rate:.4f}")
    
    # Análisis por hora del día
    print(f"\nSKIP RATE PROMEDIO POR HORA:")
    hourly_avg = heatmap_data.mean(axis=0).sort_values(ascending=False)
    for hora, rate in hourly_avg.head(10).items():
        print(f"   Hora {hora:2d}:00 - Skip rate: {rate:.4f}")



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
    
    # Gráfico de heatmap de skip rate por hora y día de la semana
    plot_heatmap_skip_rate_by_hour_weekday(df)
    
    # Gráfico de top artistas con reproducciones y skips
    plot_top_artists_with_skips_and_plays(df)


if __name__ == "__main__":
    plotGraficos()
