import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("spotify_scaled.csv")

# Preprocesamiento: mantener solo una fila por canción+artista (la más popular)
df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

# Columnas que se usan para la recomendación
feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
                'liveness', 'speechiness', 'loudness', 'tempo']

# -----------------------------
# 🧭 SIDEBAR - PANEL DE FILTROS
# -----------------------------
st.sidebar.header("🎛️ Filtros de Canciones")

# 🎶 Géneros musicales con multiselect dentro de expander
with st.sidebar.expander("🎶 Géneros musicales"):
    generos = df['track_genre'].unique().tolist()
    generos_seleccionados = st.multiselect(
        "Selecciona uno o varios géneros:",
        options=generos,
        default=generos
    )

# 🎚️ Filtros numéricos (acousticness, etc.)
with st.sidebar.expander("🧪 Filtrar por características"):
    filtros_rango = {}
    for col in ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence']:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        filtros_rango[col] = st.slider(
            f"{col.capitalize()}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

# Aplicar los filtros al dataframe
df_filtrado = df[df['track_genre'].isin(generos_seleccionados)].copy()
for col, (min_val, max_val) in filtros_rango.items():
    df_filtrado = df_filtrado[(df_filtrado[col] >= min_val) & (df_filtrado[col] <= max_val)]

# Eliminar duplicados tras filtrar (por si acaso)
df_filtrado = df_filtrado.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])
df_filtrado['combo'] = df_filtrado['track_name'] + " - " + df_filtrado['artists']

# -----------------------------
# 🎧 INTERFAZ PRINCIPAL
# -----------------------------
st.title("🎧 Recomendador de Canciones - Spotify")

# Desplegable de canciones con opción vacía
canciones_opciones = [""] + df_filtrado['combo'].tolist()
cancion_seleccionada = st.selectbox("Selecciona una canción:", canciones_opciones, index=0)

# Selector de número de recomendaciones
n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=50, value=5)

# ---------------------------------------
# 🔍 Función para recomendar canciones
# ---------------------------------------
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

# ---------------------------------------
# ▶️ Mostrar resultados si hay selección
# ---------------------------------------
if cancion_seleccionada:
    nombre, artista = cancion_seleccionada.split(" - ", 1)
    recomendaciones = recomendar_knn(df_filtrado, nombre, artista, n=n_recomendaciones)
    st.write(f"### Recomendaciones para: **{nombre} - {artista}**")
    st.dataframe(recomendaciones)
else:
    st.info("🎵 Selecciona una canción para ver recomendaciones.")
