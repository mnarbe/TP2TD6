import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

COMPETITION_PATH = "D:/Usuario/Descargas/tp para clonar/TP2TD6/"

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    print(train_df.head())
    # Load test data
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined


df = load_competition_datasets(COMPETITION_PATH, sample_frac=0.2, random_state=1234)

print(df.info())      # Muestra tipos de datos y cantidad de valores no nulos
print(df.describe())  # Estadísticas básicas para columnas numéricas
print(df.head())      # Primeras filas del DataFrame
print(df.describe(include='all'))  # Estadísticas para todas las columnas