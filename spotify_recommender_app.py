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

def obtener_grupo_genero(genero_base):
    for grupo, lista in grupos_generos.items():
        if genero_base.lower() in lista:
            return lista
    return [genero_base]

# -------------------------
# 🎛️ SIDEBAR - FILTROS
# -------------------------
st.sidebar.header("🎛️ Filtros de Cancion
