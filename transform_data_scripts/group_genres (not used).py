import re
import json
from collections import defaultdict

def map_genre(genre: str) -> str:
    g = genre.lower().strip()

    # Argentina / Nacional
    if re.search(r'argentine|folklore argentino|cuarteto|chamamé|murga|rock en español|tango', g):
        return 'Argentina / Nacional'
    
    # Cumbia y cuarteto
    if re.search(r'cuarteto|cumbia', g):
        return 'Cumbia y cuarteto'

    # Pop
    if re.search(r'\bpop|bubblegum|synthpop|hyperpop|dance pop', g):
        return 'Pop'
    
    # H music
    if re.search(r'metal', g):
        return 'High Energy Music'

    # Rock
    if re.search(r'rock|metal|punk|grunge|garage|emo|stoner|shoegaze|ska', g):
        return 'Rock'

    # Hip Hop / Rap
    if re.search(r'hip hop|rap|drill|trap|phonk|crunk|boom bap', g):
        return 'Hip Hop / Rap'

    # R&B / Soul / Funk
    if re.search(r'r&b|soul|funk|motown|neo soul|groove|quiet storm', g):
        return 'R&B / Soul / Funk'

    # Jazz
    if re.search(r'jazz|bebop|bossa|swing|big band|cool jazz', g):
        return 'Jazz'

    # Electronic (no necesariamente bailable)
    if re.search(r'ambient|downtempo|idm|electronic|synthwave|chillwave|lo-fi|minimal|drone', g):
        return 'Electronic'

    # Dance (más club / house / trance / edm)
    if re.search(r'edm|electro|house|techno|trance|club|disco|progressive|bass|dubstep|drum and bass', g):
        return 'Dance'

    # Afro / African
    if re.search(r'afro|afrobeats|afropop|amapiano|afroswing|afrosoul|afropiano', g):
        return 'Afro / African'

    # Latin subdividido
    if re.search(r'latin pop|urbano|reggaeton|trap latino|latin r&b|pop urbano', g):
        return 'Latin Pop / Urban'
    if re.search(r'mexican|corridos|ranchera|mariachi|banda|norteño|sierreño|regional|duranguense|grupera|tropical|bachata|merengue|salsa|cumbia|vallenato|forró|sertanejo|pagode', g):
        return 'Regional Mexicano / Tropical'
    if re.search(r'folklore|tango|bossa nova|mpb|nueva trova|trova', g):
        return 'Latin Tradicional / Folklore'

    # Country / Folk / Americana
    if re.search(r'country|americana|folk|bluegrass|roots|singer-songwriter|outlaw', g):
        return 'Country / Folk / Americana'

    # Classical / Instrumental
    if re.search(r'classical|piano|orchestral|baroque|opera|chamber|symphonic|requiem|choral|medieval', g):
        return 'Classical / Instrumental'

    # Japanese / Anime
    if re.search(r'anime|j-pop|j-rock|j-rap|vocaloid|japanese|kayokyoku|vgm|shibuya-kei', g):
        return 'Japanese / Anime'

    # World / Ethnic / Traditional
    if re.search(r'arabic|balkan|indian|bollywood|celtic|turkish|mandopop|k-pop|thai|african|world|flamenco', g):
        return 'World / Ethnic / Traditional'

    # Reggae / Caribbean
    if re.search(r'reggae|dancehall|soca|calypso|dub|ragga|riddim|jamaican', g):
        return 'Reggae / Caribbean'

    # Gospel / Christian / Worship
    if re.search(r'gospel|christian|worship|ccm|pentecostal', g):
        return 'Gospel / Christian'

    # Children / Kids
    if re.search(r'children|kids|lullaby|infantil', g):
        return 'Children / Kids'

    # Comedy / Parody
    if re.search(r'comedy|parody|humor', g):
        return 'Comedy'

    # Soundtrack / Film / Stage
    if re.search(r'soundtrack|musical|score|broadway|film', g):
        return 'Soundtrack / Film'

    # Experimental / Avant-Garde
    if re.search(r'experimental|avant|noise|glitch|drone|minimal|idm', g):
        return 'Experimental / Avant-Garde'

    return 'Other'


# ====== EJECUCIÓN ======
with open("artist_genres.json", "r", encoding="utf-8") as f:
    artist_genres = json.load(f)

artist_groups = {}
others = set()
group_counts = defaultdict(int)

for artist_id, genres in artist_genres.items():
    mapped = set()
    for g in genres:
        category = map_genre(g)
        mapped.add(category)
        group_counts[category] += 1
        if category == "Other":
            others.add(g)
    artist_groups[artist_id] = list(mapped)

# Guardar resultados
with open("artist_genre_groups.json", "w", encoding="utf-8") as f:
    json.dump(artist_groups, f, indent=2, ensure_ascii=False)

# Mostrar resumen
print("✅ Agrupamiento completado.\n")
print("Distribución por categoría:")
for cat, count in sorted(group_counts.items(), key=lambda x: -x[1]):
    print(f" - {cat}: {count} géneros")

print("\n⚠️ Géneros no agrupados (Other):")
for g in sorted(others):
    print("  ", g)
