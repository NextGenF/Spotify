import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def cargar_y_preprocesar():
    df = pd.read_csv("spotify_scaled.csv")
    df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])

    feature_cols_num = ['danceability', 'energy', 'valence', 'acousticness',
                        'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo', 'popularity']
    feature_col_cat = ['track_genre']

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[feature_cols_num])

    encoder = OneHotEncoder(sparse_output=False)
    X_cat = encoder.fit_transform(df[feature_col_cat])

    df_num = pd.DataFrame(X_num, columns=feature_cols_num)
    df_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(feature_col_cat))

    X_all = pd.concat([df_num, df_cat], axis=1, ignore_index=True)
    X_all.index = df.index

    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_all)
    df['combo'] = df['track_name'] + " - " + df['artists']

    return df, X_all, scaler, encoder

df, X_all, scaler, encoder = cargar_y_preprocesar()


# BARRA DE FILTROS

st.sidebar.header("🎛️ Filtros de Canciones")

with st.sidebar.expander("🎶 Género musical"):
    generos = sorted(df['track_genre'].dropna().unique().tolist())
    genero_seleccionado = st.selectbox("Selecciona un género:", ["Seleccionar todos"] + generos)

with st.sidebar.expander("🎚️ Filtrar por características musicales"):
    filtros_rango = {}
    for col in ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence']:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        filtros_rango[col] = st.slider(
            f"{col.capitalize()}", min_value=min_val, max_value=max_val, value=(min_val, max_val)
        )

if 'cancion_seleccionada' not in st.session_state:
    st.session_state['cancion_seleccionada'] = ""

# Aplicar filtros
df_filtrado = df.copy()
if genero_seleccionado != "Seleccionar todos":
    df_filtrado = df_filtrado[df_filtrado['track_genre'] == genero_seleccionado]

for col, (min_val, max_val) in filtros_rango.items():
    df_filtrado = df_filtrado[(df_filtrado[col] >= min_val) & (df_filtrado[col] <= max_val)]


# INTERFAZ 

st.title("🎧 Recomendador de Canciones Spotify")

canciones_opciones = sorted(df_filtrado['combo'].tolist())
seleccion = st.selectbox(
    "🎵 Selecciona una canción:",
    [""] + canciones_opciones,
    index=canciones_opciones.index(st.session_state['cancion_seleccionada']) + 1 if st.session_state['cancion_seleccionada'] in canciones_opciones else 0
)
st.session_state['cancion_seleccionada'] = seleccion

n_recomendaciones = st.slider("📊 Número de recomendaciones", min_value=1, max_value=50, value=5)


# Grupos de géneros

grupos_generos = {
    "reggaeton": ["latin", "latino", "reggaeton", "salsa", "samba", "sertanejo", "mpb", "pagode"],
    "pop": ["pop", "dance", "dancehall", "electropop", "synth-pop", "indie pop", "power-pop", "pop-film", "show-tunes", "j-pop", "k-pop"],
    "rock": ["rock", "alt-rock", "alternative", "hard-rock", "psych-rock", "punk", "punk-rock", "pop-rock", "grunge", "garage", "metal", "metalcore", "heavy-metal", "death-metal", "black-metal", "classic rock", "rock-n-roll"],
    "hiphop_urban": ["hip-hop", "rap", "trap", "r-n-b", "funk", "soul", "gospel", "reggae"],
    "electronic": ["electronic", "edm", "house", "techno", "progressive-house", "deep-house", "electro", "detroit-techno", "disco", "trance", "dubstep", "drum-and-bass", "minimal-techno"],
    "acoustic_folk": ["acoustic", "folk", "singer-songwriter", "country", "bluegrass", "honky-tonk", "americana"],
    "classical_jazz": ["classical", "jazz", "opera", "piano"],
    "world": ["world-music", "brazil", "turkish", "mandopop", "cantopop", "indian", "malay", "forro"],
    "soundtrack_misc": ["study", "sleep", "sad", "children", "comedy", "christmas", "anime", "disney"]
}

def obtener_grupo_genero(genero):
    for grupo, lista in grupos_generos.items():
        if genero in lista:
            return lista
    return [genero]


# Recomendación

def recomendar_kmeans(df, track_name, artist, n=5):
    seleccion = df[(df['track_name'] == track_name) & (df['artists'] == artist)].iloc[0]
    cluster = seleccion['cluster']
    grupo_genero = obtener_grupo_genero(seleccion['track_genre'])

    df_cluster = df[(df['cluster'] == cluster) & (df['track_genre'].isin(grupo_genero))].copy()
    df_cluster = df_cluster[(df_cluster['track_name'] != track_name) | (df_cluster['artists'] != artist)]

    seleccion_idx = seleccion.name
    seleccion_vec = X_all.loc[seleccion_idx].values.reshape(1, -1)
    feature_data = X_all.loc[df_cluster.index]

    similitudes = cosine_similarity(seleccion_vec, feature_data)[0]
    df_cluster['similitud'] = similitudes

    resultados = df_cluster.sort_values(by='similitud', ascending=False).head(n)
    resultados = resultados[['track_name', 'artists', 'track_genre', 'similitud']].copy()

    resultados.reset_index(drop=True, inplace=True)
    resultados.index = resultados.index + 1
    resultados['similitud'] = (resultados['similitud'] * 100).round(2).astype(str) + '%'

    return resultados


# Mostrar resultados

if seleccion:
    nombre, artista = seleccion.split(" - ", 1)
    if df_filtrado[(df_filtrado['track_name'] == nombre) & (df_filtrado['artists'] == artista)].empty:
        st.warning("⚠️ La canción seleccionada no está disponible con los filtros aplicados.")
    else:
        recomendaciones = recomendar_kmeans(df, nombre, artista, n=n_recomendaciones)
        st.write(f"### Recomendaciones para: **{nombre} - {artista}**")
        st.dataframe(recomendaciones)
else:
    st.info("🎵 Selecciona una canción para ver recomendaciones.")
