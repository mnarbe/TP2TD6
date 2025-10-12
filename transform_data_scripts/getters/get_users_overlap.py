import pandas as pd

# === 1. Leer archivos ===
train = pd.read_csv("merged_data.csv")
test = pd.read_csv("merged_test_data.csv")

# === 2. Extraer conjuntos de usernames ===
train_users = set(train["username"].unique())
test_users = set(test["username"].unique())

# === 3. Clasificar ===
common_users = train_users & test_users
train_only_users = train_users - test_users
test_only_users = test_users - train_users

# === 4. Crear DataFrame resumen ===
summary = pd.DataFrame({
    "username": list(train_users | test_users)
})
summary["in_train"] = summary["username"].isin(train_users)
summary["in_test"] = summary["username"].isin(test_users)

summary["category"] = summary.apply(
    lambda row: (
        "both" if row["in_train"] and row["in_test"]
        else "train_only" if row["in_train"]
        else "test_only"
    ),
    axis=1
)

# === 5. Guardar resultados ===
summary.to_csv("user_overlap_summary.csv", index=False)

# === 6. (Opcional) Mostrar conteo resumen ===
print("Resumen de usuarios:")
print(summary["category"].value_counts())
