from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re
import json
import requests


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

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

AI_SYSTEM_PROMPT = (
    "You are AudioFit AI, a workout music recommender. "
    "Given a user prompt about running/workout/mood, return ONLY a JSON array of songs. "
    "Each element: {\"title\": \"song name\", \"artist\": \"artist name\", \"reason\": \"<=12 words why it fits the prompt\"}. "
    "Rules: The songs suggested, their total duration should be atleast 5 mins more than the requested. If no time given, then take the default time to be 20Mins or suggest max 10 songs. Match language if user said English/Hindi/Mix. "
    "Prefer high-energy for runs, chill for warmup/cooldown. "
    "Return valid JSON array and nothing else — no markdown, no explanation."
)

def _extract_json_array(text: str):
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _call_mistral(user_prompt: str, language: str = "mix", count: int = 10):
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set on server (Render > Environment)")
    lang_hint = {"english": "English only", "hindi": "Hindi/Bollywood only", "mix": "mix of English and Hindi"}.get(language.lower(), "mix of English and Hindi")
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": f'User prompt: "{user_prompt}"\nLanguage preference: {lang_hint}\nReturn exactly {count} songs as JSON array.'},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=12)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Mistral {r.status_code}: {r.text[:400]}")
    content = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    arr = _extract_json_array(content)
    if not arr:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {content[:400]}")
    out = []
    for item in arr[:count]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        artist = str(item.get("artist", "")).strip()
        reason = str(item.get("reason", "")).strip()[:80]
        if not title or not artist:
            continue
        out.append({"title": title, "artist": artist, "reason": reason or "Fits your prompt"})
    return out

class AIRequest(BaseModel):
    prompt: str
    language: str = "mix"
    count: int = 10
    q: str | None = None  # alias for prompt

@app.post("/ai-recommend")
def ai_recommend(body: AIRequest):
    prompt = (body.prompt or body.q or "").strip()
    language = (body.language or "mix").strip()
    count = max(1, min(int(body.count or 10), 12))
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > 500:
        raise HTTPException(status_code=400, detail="prompt too long (max 500 chars)")
    songs = _call_mistral(prompt, language, count)
    return {"songs": songs, "model": MISTRAL_MODEL}