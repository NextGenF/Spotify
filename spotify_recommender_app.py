import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Cargar datos
df = pd.read_csv("spotify_scaled.csv")
df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
                'liveness', 'speechiness', 'loudness', 'tempo']

# Entrenar modelo
knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_model.fit(df[feature_cols])

# Agrupación de géneros
grupos_generos = {
    "reggaeton_latino": [
        "reggaeton", "latin", "latino", "salsa", "samba", "sertanejo", "mpb", "pagode", "romance", "spanish"
    ],
    "pop_comercial": [
        "pop", "dance", "dancehall", "electropop", "synth-pop", "power-pop", "pop-film", "show-tunes", "party"
    ],
    "rock_metal": [
        "rock", "alt-rock", "alternative", "grunge", "garage", "metal", "hard-rock", "metalcore", "heavy-metal",
        "death-metal", "black-metal", "rock-n-roll", "punk", "punk-rock", "psych-rock", "rockabilly"
    ],
    "hiphop_rap_urban": [
        "hip-hop", "rap", "trap", "r-n-b", "soul", "funk", "gospel", "reggae", "groove", "club", "guitar"
    ],
    "electronic_dance": [
        "electronic", "edm", "house", "techno", "deep-house", "disco", "electro", "progressive-house", "trance",
        "dubstep", "drum-and-bass", "detroit-techno", "minimal-techno", "breakbeat", "hardstyle", "idm", "industrial"
    ],
    "acoustic_folk_country": [
        "acoustic", "folk", "singer-songwriter", "country", "bluegrass", "honky-tonk", "americana", "guitar"
    ],
    "classical_jazz": [
        "classical", "jazz", "opera", "piano"
    ],
    "anime_jpop_kpop": [
        "anime", "j-pop", "j-dance", "j-idol", "j-rock", "k-pop", "cantopop", "mandopop"
    ],
    "world_international": [
        "brazil", "turkish", "indian", "malay", "iranian", "french", "german", "swedish", "world-music"
    ],
    "experimental_ambient": [
        "ambient", "chill", "trip-hop", "new-age", "idm"
    ],
    "niche_misc": [
        "study", "sleep", "children", "kids", "comedy", "disney", "happy", "sad", "blues", "ska"
    ]
}


def obtener_grupo_genero(genero_base):
    for grupo, lista in grupos_generos.items():
        if genero_base.lower() in lista:
            return lista
    return [genero_base]

# -------------------------
# 🎛️ SIDEBAR - FILTROS
# -------------------------
st.sidebar.header("🎛️ Filtros de Canciones")

with st.sidebar.expander("🎶 Géneros musicales"):
    generos = sorted(df['track_genre'].unique().tolist())
    genero_seleccionado = st.selectbox("Selecciona un género:", ["Seleccionar todos"] + generos)

with st.sidebar.expander("🧪 Filtrar por características"):
    filtros_rango = {}
    for col in ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence']:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        filtros_rango[col] = st.slider(
            f"{col.capitalize()}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

# Aplicar filtros al dataset base (NO afecta filtro de artista/canción)
df_filtrado = df.copy()
if genero_seleccionado != "Seleccionar todos":
    df_filtrado = df_filtrado[df_filtrado['track_genre'] == genero_seleccionado]

for col, (min_val, max_val) in filtros_rango.items():
    df_filtrado = df_filtrado[(df_filtrado[col] >= min_val) & (df_filtrado[col] <= max_val)]

df_filtrado = df_filtrado.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])
df_filtrado['combo'] = df_filtrado['track_name'] + " - " + df_filtrado['artists']

# -------------------------
# 🎧 INTERFAZ PRINCIPAL
# -------------------------
st.title("🎧 Recomendador de Canciones - Spotify")

if 'cancion_seleccionada' not in st.session_state:
    st.session_state['cancion_seleccionada'] = ""
if 'artista_filtro_ui' not in st.session_state:
    st.session_state['artista_filtro_ui'] = ""

# 🎤 Filtro de artista (solo afecta la lista de canciones)
artistas_unicos = sorted(df_filtrado['artists'].unique().tolist())
artista_ui = st.selectbox("Filtrar canciones por artista:", [""] + artistas_unicos, index=0)
st.session_state['artista_filtro_ui'] = artista_ui

# 🎵 Lista de canciones (filtrada por artista, ordenada alfabéticamente)
if artista_ui:
    canciones = df_filtrado[df_filtrado['artists'] == artista_ui]['combo'].sort_values().tolist()
else:
    canciones = df_filtrado['combo'].sort_values().tolist()

canciones_opciones = [""] + canciones
seleccion = st.selectbox(
    "Selecciona una canción:",
    canciones_opciones,
    index=canciones_opciones.index(st.session_state['cancion_seleccionada']) if st.session_state['cancion_seleccionada'] in canciones_opciones else 0
)
st.session_state['cancion_seleccionada'] = seleccion

n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=50, value=5)

# ---------------------------------------
# 🔍 Función de recomendación
# ---------------------------------------
def recomendar_knn(df, track_name, artist, n=5):
    seleccion = df[(df['track_name'] == track_name) & (df['artists'] == artist)].iloc[0]
    genero_base = seleccion['track_genre']
    grupo_relacionado = obtener_grupo_genero(genero_base)
    candidatos = df[df['track_genre'].isin(grupo_relacionado)].reset_index(drop=True)

    idx_local = candidatos[
        (candidatos['track_name'] == track_name) & (candidatos['artists'] == artist)
    ].index[0]

    X = candidatos[feature_cols]
    similitudes = cosine_similarity([X.iloc[idx_local]], X)[0]
    indices_similares = np.argsort(similitudes)[::-1][1:n+1]

    resultados = candidatos.iloc[indices_similares][['track_name', 'artists', 'track_genre']].copy()
    resultados['similitud'] = similitudes[indices_similares]
    resultados = resultados.drop_duplicates(subset=['track_name', 'artists'])

    resultados.reset_index(drop=True, inplace=True)
    resultados.index = resultados.index + 1
    resultados['similitud'] = (resultados['similitud'] * 100).round(2).astype(str) + '%'

    return resultados

# ---------------------------------------
# ▶️ Mostrar resultados
# ---------------------------------------
if seleccion:
    nombre, artista = seleccion.split(" - ", 1)
    if df_filtrado[(df_filtrado['track_name'] == nombre) & (df_filtrado['artists'] == artista)].empty:
        st.warning("⚠️ La canción seleccionada no está disponible con los filtros aplicados.")
    else:
        recomendaciones = recomendar_knn(df, nombre, artista, n=n_recomendaciones)  # usar df completo, no df_filtrado
        st.write(f"### Recomendaciones para: **{nombre} - {artista}**")
        st.dataframe(recomendaciones)
else:
    st.info("🎵 Selecciona una canción para ver recomendaciones.")
