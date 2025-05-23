import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Cargar y preparar datos
df_scaled = pd.read_csv("spotify_scaled.csv")
df_scaled = df_scaled.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

cols_excluir = ['track_name', 'artists', 'track_genre', 'popularity']
feature_cols = df_scaled.drop(columns=cols_excluir).columns

# Función de recomendación
def recomendar_knn(track_name, artist_name, n=5):
    seleccion = df_scaled[
        (df_scaled['track_name'] == track_name) & (df_scaled['artists'] == artist_name)
    ].iloc[0]

    genero_ref = seleccion['track_genre']
    candidatos = df_scaled[df_scaled['track_genre'] == genero_ref].reset_index(drop=True)

    idx_local = candidatos[
        (candidatos['track_name'] == track_name) & (candidatos['artists'] == artist_name)
    ].index[0]

    X = candidatos[feature_cols]
    similitudes = cosine_similarity([X.iloc[idx_local]], X)[0]
    indices_similares = np.argsort(similitudes)[::-1][1:n+1]

    resultados = candidatos.iloc[indices_similares][['track_name', 'artists', 'track_genre', 'popularity']].copy()
    resultados['similitud'] = similitudes[indices_similares]
    resultados = resultados.drop_duplicates(subset=['track_name', 'artists'])

    resultados.reset_index(drop=True, inplace=True)
    resultados.index = resultados.index + 1  # Numeración desde 1
    resultados['similitud'] = (resultados['similitud'] * 100).round(2).astype(str) + '%'

    return resultados

# Streamlit UI
st.title("🎧 Recomendador de Canciones - Spotify")

# Selector de canciones del top
top_n = st.slider("Selecciona el top de canciones más populares para elegir", min_value=10, max_value=30, value=15)
top_tracks_df = df_scaled.head(top_n).copy()
top_tracks_df["combo"] = top_tracks_df["track_name"] + " - " + top_tracks_df["artists"]

selected_combo = st.selectbox("Selecciona una canción:", top_tracks_df["combo"].tolist())
track_name_selected, artist_name_selected = selected_combo.split(" - ", 1)

n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=10, value=5)

if st.button("Recomendar"):
    recomendaciones = recomendar_knn(track_name_selected, artist_name_selected, n=n_recomendaciones)
    st.write("### Recomendaciones:")
    st.dataframe(recomendaciones)
