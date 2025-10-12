import csv

input_file = "../merged_data_genresonly.csv"
output_file = "../merged_data_genresonly_c.csv"

# Cuántas columnas debería tener tu CSV
# Contalas del header (la primera fila)
with open(input_file, "r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n")
    expected_cols = len(header.split(","))
    print(f"El archivo debería tener {expected_cols} columnas.")

# Procesar el archivo línea por línea
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        # Detectar filas con el substring problemático
        if any("spotify:episode" in str(cell) for cell in row):
            # Si tiene menos columnas que las esperadas, agregamos comas vacías
            missing = expected_cols - len(row)
            if missing > 0:
                row.extend([""] * missing)
                print(f"Línea {i+1}: agregadas {missing} columnas vacías.")
        
        # Escribir siempre la fila (modificada o no)
        writer.writerow(row)

print(f"✅ Archivo corregido guardado en '{output_file}'")
