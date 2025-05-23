import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("spotify_scaled_final.csv")
    df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])
    return df

df = cargar_datos()

# ----------------------
# Preprocesamiento K-Means
# ----------------------
feature_cols_num = ['danceability', 'energy', 'valence', 'acousticness',
                    'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo', 'popularity']
feature_col_cat = ['track_genre']

scaler = StandardScaler()
X_num = scaler.fit_transform(df[feature_cols_num])

ohe = OneHotEncoder(sparse_output=False)
X_cat = ohe.fit_transform(df[feature_col_cat])

df_num = pd.DataFrame(X_num, columns=feature_cols_num)
df_cat = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(feature_col_cat))

X_all = pd.concat([df_num, df_cat], axis=1)

kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_all)

# Combinar nombre-artista
df['combo'] = df['track_name'] + " - " + df['artists']

# ----------------------
# Interfaz Streamlit
# ----------------------
st.title("🎧 Recomendador de Canciones (K-Means)")

# Filtro de artista para acotar lista de canciones
artistas = sorted(df['artists'].unique())
artista_ui = st.selectbox("Filtrar canciones por artista (opcional):", [""] + artistas)

# Filtrar canciones
if artista_ui:
    canciones_opciones = sorted(df[df['artists'] == artista_ui]['combo'].tolist())
else:
    canciones_opciones = sorted(df['combo'].tolist())

# Selector de canción
cancion_seleccionada = st.selectbox("Selecciona una canción:", [""] + canciones_opciones)

# Slider recomendaciones
n_recomendaciones = st.slider("Número de recomendaciones", min_value=1, max_value=50, value=5)

# ----------------------
# Función recomendadora
# ----------------------
def recomendar_kmeans(df, track_name, artist, n=5):
    seleccion = df[(df['track_name'] == track_name) & (df['artists'] == artist)].iloc[0]
    cluster = seleccion['cluster']
    df_cluster = df[df['cluster'] == cluster].copy()
    df_cluster = df_cluster[(df_cluster['track_name'] != track_name) | (df_cluster['artists'] != artist)]

    features = df_num.columns.tolist() + list(ohe.get_feature_names_out(feature_col_cat))
    seleccion_vec = X_all[(df['track_name'] == track_name) & (df['artists'] == artist)].iloc[0].values.reshape(1, -1)
    feature_data = X_all[df_cluster.index]

    similitudes = cosine_similarity(seleccion_vec, feature_data)[0]
    df_cluster['similitud'] = similitudes
    resultados = df_cluster.sort_values(by='similitud', ascending=False).head(n)

    resultados = resultados[['track_name', 'artists', 'track_genre', 'similitud']].copy()
    resultados.reset_index(drop=True, inplace=True)
    resultados.index = resultados.index + 1
    resultados['similitud'] = (resultados['similitud'] * 100).round(2).astype(str) + '%'
    return resultados

# Mostrar resultados
if cancion_seleccionada:
    nombre, artista = cancion_seleccionada.split(" - ", 1)
    if df[(df['track_name'] == nombre) & (df['artists'] == artista)].empty:
        st.warning("⚠️ La canción seleccionada no está disponible.")
    else:
        recomendaciones = recomendar_kmeans(df, nombre, artista, n=n_recomendaciones)
        st.write(f"### Recomendaciones para: **{nombre} - {artista}**")
        st.dataframe(recomendaciones)
else:
    st.info("🎵 Selecciona una canción para ver recomendaciones.")
