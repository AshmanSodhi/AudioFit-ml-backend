from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# --------------------------------------------------
# Load model files
# --------------------------------------------------

scaler_hin = joblib.load("scaler_hin.pkl")
knn_hin = joblib.load("knn_model_hin.pkl")
df_hin = pd.read_pickle("hin_song_catalog.pkl")

scaler_eng = joblib.load("scaler_eng.pkl")
knn_eng = joblib.load("knn_model_eng.pkl")
df_eng = pd.read_pickle("eng_song_catalog.pkl")

# --------------------------------------------------
# Features used by the model
# --------------------------------------------------

FEATURE_COLUMNS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence"
]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Music Recommendation API",
    description="Content-based music recommendation system",
    version="1.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class SongFeatures(BaseModel):

    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    tempo: float
    valence: float

    n_recommendations: int = 5


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Music Recommendation API is running"
    }


# --------------------------------------------------
# Recommendation endpoint - Eng Songs
# --------------------------------------------------

@app.post("/recommend_eng")
def recommend_eng(song: SongFeatures):

    if song.n_recommendations < 1:
        raise HTTPException(
            status_code=400,
            detail="n_recommendations must be at least 1"
        )

    if song.n_recommendations > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 recommendations allowed"
        )

    # Create input dataframe
    new_song = pd.DataFrame([{
        "acousticness": song.acousticness,
        "danceability": song.danceability,
        "energy": song.energy,
        "instrumentalness": song.instrumentalness,
        "liveness": song.liveness,
        "loudness": song.loudness,
        "speechiness": song.speechiness,
        "tempo": song.tempo,
        "valence": song.valence
    }])

    # Scale using the scaler trained on the dataset
    new_song_scaled = scaler_eng.transform(
        new_song[FEATURE_COLUMNS]
    )

    # Find nearest songs
    distances, indices = knn_eng.kneighbors(
        new_song_scaled,
        n_neighbors=song.n_recommendations
    )

    # Get recommended songs
    recommendations = df_eng.iloc[
        indices[0]
    ].copy()

    # Convert cosine distance to similarity
    recommendations["similarity"] = (
        1 - distances[0]
    )

    # Columns returned to mobile app
    result_columns = [
        "track_id",
        "track_name",
        "artist_name",
        "album_name",
        "year",
        "language",
        "popularity",
        "similarity"
    ]

    # Only return columns that actually exist
    result_columns = [
        col for col in result_columns
        if col in recommendations.columns
    ]

    results = recommendations[
        result_columns
    ].to_dict(orient="records")

    return {
        "recommendations": results
    }


# --------------------------------------------------
# Recommendation endpoint - Hin Songs
# --------------------------------------------------


@app.post("/recommend_hin")
def recommend_hin(song: SongFeatures):

    if song.n_recommendations < 1:
        raise HTTPException(
            status_code=400,
            detail="n_recommendations must be at least 1"
        )

    if song.n_recommendations > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 recommendations allowed"
        )

    # Create input dataframe
    new_song = pd.DataFrame([{
        "acousticness": song.acousticness,
        "danceability": song.danceability,
        "energy": song.energy,
        "instrumentalness": song.instrumentalness,
        "liveness": song.liveness,
        "loudness": song.loudness,
        "speechiness": song.speechiness,
        "tempo": song.tempo,
        "valence": song.valence
    }])

    # Scale using the scaler trained on the dataset
    new_song_scaled = scaler_hin.transform(
        new_song[FEATURE_COLUMNS]
    )

    # Find nearest songs
    distances, indices = knn_hin.kneighbors(
        new_song_scaled,
        n_neighbors=song.n_recommendations
    )

    # Get recommended songs
    recommendations = df_hin.iloc[
        indices[0]
    ].copy()

    # Convert cosine distance to similarity
    recommendations["similarity"] = (
        1 - distances[0]
    )

    # Columns returned to mobile app
    result_columns = [
        "track_id",
        "track_name",
        "artist_name",
        "album_name",
        "year",
        "language",
        "popularity",
        "similarity"
    ]

    # Only return columns that actually exist
    result_columns = [
        col for col in result_columns
        if col in recommendations.columns
    ]

    results = recommendations[
        result_columns
    ].to_dict(orient="records")

    return {
        "recommendations": results
    }