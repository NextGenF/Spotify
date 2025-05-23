import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("spotify_scaled.csv")

# Columnas de características
feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
                'liveness', 'speechiness', 'loudness', 'tempo']

# Sidebar - Panel de filtros
st.sidebar.header("🎛️ Filtros de Canciones")

# Géneros
generos = df['track_genre'].unique().tolist()
generos_seleccionados = st.sidebar.multiselect(
    "Filtrar por género musical",
    options=generos,
    default=generos
)

# Rango dinámico por cada característica
filtros_rango = {}
for col in ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence']:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    filtros_rango[col] = st.sidebar.slider(
        f"{col.capitalize()}",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

# Aplicar filtros
df_filtrado = df[df['track_genre'].isin(generos_seleccionados)].copy()

for col, (min_val, max_val) in filtros_rango.items():
    df_filtrado = df_filtrado[(df_filtrado[col] >= min_val) & (df_filtrado[col] <= max_val)]

# Crear columna combinada para identificar canciones únicas
df_filtrado['combo'] = df_filtrado['track_name'] + " - " + df_filtrado['artists']
df_filtrado = df_filtrado.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

# Interfaz principal
st.title("🎧 Recomendador de Canciones - Spotify")

cancion_seleccionada = st.selectbox("Selecciona una canción:", df_filtrado['combo'].tolist())
n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=50, value=5)

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

if st.button("Recomendar"):
    nombre, artista = cancion_seleccionada.split(" - ", 1)
    recomendaciones = recomendar_knn(df_filtrado, nombre, artista, n=n_recomendaciones)
    st.write("### Recomendaciones:")
    st.dataframe(recomendaciones)
    
