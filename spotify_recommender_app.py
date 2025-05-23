import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("spotify_scaled.csv")
df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
                'liveness', 'speechiness', 'loudness', 'tempo']

st.sidebar.header("🎛️ Filtros de Canciones")

# -------------------------
# 🎶 GÉNEROS MUSICALES
# -------------------------
with st.sidebar.expander("🎶 Géneros musicales"):
    generos = sorted(df['track_genre'].unique().tolist())
    generos_opciones = ["Seleccionar todos"] + generos
    genero_seleccionado = st.selectbox("Selecciona un género:", options=generos_opciones)

# Filtros por características numéricas
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

# Aplicar filtros al dataframe
df_filtrado = df.copy()
if genero_seleccionado != "Seleccionar todos":
    df_filtrado = df_filtrado[df_filtrado['track_genre'] == genero_seleccionado]

for col, (min_val, max_val) in filtros_rango.items():
    df_filtrado = df_filtrado[(df_filtrado[col] >= min_val) & (df_filtrado[col] <= max_val)]

df_filtrado = df_filtrado.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])
df_filtrado['combo'] = df_filtrado['track_name'] + " - " + df_filtrado['artists']

# Estado para mantener selección
if 'cancion_seleccionada' not in st.session_state:
    st.session_state['cancion_seleccionada'] = ""

# UI principal
st.title("🎧 Recomendador de Canciones - Spotify")

# Solo si hay canciones tras los filtros
if not df_filtrado.empty:
    canciones_opciones = [""] + df_filtrado['combo'].tolist()
    seleccion = st.selectbox(
        "Selecciona una canción:",
        canciones_opciones,
        index=canciones_opciones.index(st.session_state['cancion_seleccionada']) if st.session_state['cancion_seleccionada'] in canciones_opciones else 0
    )
    st.session_state['cancion_seleccionada'] = seleccion
else:
    st.warning("⚠️ No hay resultados para los filtros aplicados.")
    seleccion = ""

n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=50, value=5)

# Función recomendadora
def recomendar_knn(df, track_name, artist, n=5):
    seleccion = df[(df['track_name'] == track_name) & (df['artists'] == artist)].iloc[0]
    genero_ref = seleccion['track_genre']
    candidatos = df[df['track_genre'] == genero_ref].reset_index(drop=True)

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

# Mostrar recomendaciones si hay selección
if seleccion:
    nombre, artista = seleccion.split(" - ", 1)
    if df_filtrado[(df_filtrado['track_name'] == nombre) & (df_filtrado['artists'] == artista)].empty:
        st.warning("⚠️ La canción seleccionada no está disponible con los filtros aplicados.")
    else:
        recomendaciones = recomendar_knn(df_filtrado, nombre, artista, n=n_recomendaciones)
        st.write(f"### Recomendaciones para: **{nombre} - {artista}**")
        st.dataframe(recomendaciones)
else:
    st.info("🎵 Selecciona una canción para ver recomendaciones.")
