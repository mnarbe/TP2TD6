



# 

# has_popular_artist_genre: alguno de los 3 géneros es de los top N más populares
# has_rare_artist_genre: alguno de los 3 géneros es de los top N MENOS populares

# is_kids_genre: alguno de los 3 géneros es para niños
# is_comedy_genre: alguno de los 3 géneros es de comedia

# is_english: a partir de los géneros identificar si es ingles
# is_spanish: a partir de los géneros identificar si es español
# is_other_language: a partir de los géneros identificar si es otro idioma que no sea ingles o español
# has_japanese_genres: alguno de los 3 géneros es japonés o anime

# has_latin_genre: alguno de los 3 géneros es latino. reggaeton, salsa, bachata, cumbia, merengue, latino trap, urbano latino, etc.
# has_low_energy_genre: alguno de los 3 géneros es lento/tranquilo. incluye jazz, classical, ambient, lo-fi, acoustic, folk, etc.
# has_high_energy_genre: alguno de los 3 géneros es rápido. alguno de los géneros es rock, metal, edm, trap, hip hop, etc.
# has_heavy_genre: alguno de los 3 géneros es hard. alguno de los géneros es rock, metal, etc.
# has_party_genre: alguno de los 3 géneros es bailable. si hay dance, reggaeton, funk, house, techno, pop, disco, etc.
# has_romantic_genre: alguno de los 3 géneros es romántico. bolero, bachata, r&b, soul, love songs, etc.
# has_relaxing_genre: alguno de los 3 géneros es relajante. chillwave, ambient, lo-fi, lounge, easy listening, smooth jazz, etc.
# has_instrumental_genre: alguno de los tres generos es instrumental. classical, jazz, soundtrack, instrumental, ambient, etc.

# Calcular métricas por user y agregar columnas sobre esas variables

# Ej:
# user_likes_this_genre: proporción de canciones escuchadas por el usuario con alguno de los géneros actuales.
# user_rarely_listens_this_genre: si el género actual está fuera de su top N géneros más reproducidos.
# genre_novelty: qué tan distinto es el género actual respecto al historial (diversidad o exploración).
# genre_repetition: si repite un género que ya escuchó recientemente.
# proportion_danceable_listened
# proportion_romantic_listened
# proportion_spanish_listened
# proportion_modern_listened
# user_avg_genre_entropy
# user_main_language (por moda)
# user_top_genre_cluster (cluster dominante)
# user_prefers_high_energy (porcentaje de temas de alta energía escuchados)
# user_skips_rate_for_chill_genres (si podés cruzar con el target anterior)