import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df_scaled = pd.read_csv("spotify_scaled.csv")

# Preprocesamiento: dejar solo un registro por canción y artista
df_scaled = df_scaled.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

# Columnas para el modelo
cols_excluir = ['track_name', 'artists', 'track_genre', 'popularity']
feature_cols = df_scaled.drop(columns=cols_excluir).columns

# Ajustar modelo KNN (aunque no se usa directamente, lo dejamos para futura expansión)
knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_model.fit(df_scaled[feature_cols])

# Función de recomendación
def recomendar_knn(nombre_cancion, n=5):
    coincidencias = df_scaled[df_scaled['track_name'].str.lower() == nombre_cancion.lower()]

    if len(coincidencias) == 0:
        return pd.DataFrame([{"Mensaje": f"No se encontró la canción '{nombre_cancion}'."}])

    seleccion = coincidencias.iloc[0]
    genero_ref = seleccion['track_genre']
    candidatos = df_scaled[df_scaled['track_genre'] == genero_ref].reset_index(drop=True)

    idx_local = candidatos[
        (candidatos['track_name'].str.lower() == seleccion['track_name'].lower()) &
        (candidatos['artists'] == seleccion['artists'])
    ].index[0]

    X = candidatos[feature_cols]
    similitudes = cosine_similarity([X.iloc[idx_local]], X)[0]
    indices_similares = np.argsort(similitudes)[::-1][1:n+1]

    resultados = candidatos.iloc[indices_similares][['track_name', 'artists', 'track_genre', 'popularity']].copy()
    resultados['similitud'] = similitudes[indices_similares]
    resultados = resultados.drop_duplicates(subset=['track_name', 'artists'])

    return resultados

# Interfaz de Streamlit
st.title("🎧 Recomendador de Canciones - Spotify")
canciones = df_scaled['track_name'].drop_duplicates().sort_values().tolist()
cancion_seleccionada = st.selectbox("Selecciona una canción:", canciones)

n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=10, value=5)

if st.button("Recomendar"):
    recomendaciones = recomendar_knn(cancion_seleccionada, n=n_recomendaciones)
    st.write("### Recomendaciones:")
    st.dataframe(recomendaciones)