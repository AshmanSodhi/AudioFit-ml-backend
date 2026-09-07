from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import re
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Load model files
# --------------------------------------------------

scaler_hin = joblib.load("scaler_hin.pkl")
knn_hin = joblib.load("knn_model_hin.pkl")
df_hin = pd.read_pickle("hin_song_catalog.pkl")

scaler_eng = joblib.load("scaler_eng.pkl")
knn_eng = joblib.load("knn_model_eng.pkl")
df_eng = pd.read_pickle("eng_song_catalog.pkl")

# ============================================================
# LOAD V2 MODEL CONFIGURATION
# ============================================================

v2_model = joblib.load("v2_model.pkl")

v2_scaler = v2_model["scaler"]

V2_USER_WEIGHT = v2_model["user_weight"]
V2_CONTENT_WEIGHT = v2_model["content_weight"]

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

AI_SYSTEM_PROMPT = (
    "You are AudioFit AI, a workout music recommender. "
    "Given a user prompt about running/workout/mood, return ONLY a JSON array of songs. "
    "Each element: {\"title\": \"song name\", \"artist\": \"artist name\", \"reason\": \"<=12 words why it fits the prompt\"}. "
    "Rules: The songs suggested, their total duration should be at least 5 mins more than the requested. If no time given, default to 20 mins. Match language if user said English/Hindi/Mix. "
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

def _call_gemini(user_prompt: str, language: str = "mix", count: int = 10):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server (Render > Environment)")
    lang_hint = {"english": "English only", "hindi": "Hindi/Bollywood only", "mix": "mix of English and Hindi"}.get(language.lower(), "mix of English and Hindi")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": AI_SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": f'User prompt: "{user_prompt}"\nLanguage preference: {lang_hint}\nReturn exactly {count} songs as JSON array.'}]},
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 3000,
        },
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=12)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Gemini {r.status_code}: {r.text[:400]}")
    try:
        data = r.json()
        candidates = data.get("candidates", [])
        parts = (candidates[0].get("content", {}).get("parts", []) if candidates else [])
        content = "".join([str(p.get("text", "")) for p in parts if isinstance(p, dict)]).strip()
    except Exception:
        content = ""
    if not content:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {r.text[:400]}")
    arr = _extract_json_array(content)
    if not arr:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {content[:400]}")
    out = []
    for item in arr:
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
    count = max(3, int(body.count or 10))
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > 500:
        raise HTTPException(status_code=400, detail="prompt too long (max 500 chars)")
    songs = _call_gemini(prompt, language, count)
    return {"songs": songs, "model": GEMINI_MODEL}



# ============================================================
# V2 FAVORITE SONG
# ============================================================

class FavoriteSong(BaseModel):

    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    tempo: float
    valence: float


# ============================================================
# V2 PROFILE CREATION REQUEST
# ============================================================

class CreateProfileRequest(BaseModel):

    favorite_songs: list[FavoriteSong]


# ============================================================
# V2 RECOMMENDATION REQUEST
# ============================================================

class V2Request(BaseModel):

    user_profile: list[float]

    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    tempo: float
    valence: float

    n_recommendations: int = 10


# ============================================================
# HELPER
# ============================================================

def create_song_dataframe(song):

    return pd.DataFrame([{
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

# ============================================================
# V2 - CREATE USER PROFILE
# ============================================================

@app.post("/create_profile")
def create_profile(request: CreateProfileRequest):

    # --------------------------------------------------------
    # Minimum 10 favorites
    # --------------------------------------------------------

    if len(request.favorite_songs) < 10:

        raise HTTPException(
            status_code=400,
            detail="At least 10 favorite songs are required."
        )


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    songs = pd.DataFrame([
        song.model_dump()
        for song in request.favorite_songs
    ])


    # --------------------------------------------------------
    # Correct feature order
    # --------------------------------------------------------

    songs = songs[
        FEATURE_COLUMNS
    ]


    # --------------------------------------------------------
    # Scale using V2 scaler
    # --------------------------------------------------------

    songs_scaled = v2_scaler.transform(
        songs[FEATURE_COLUMNS]
    )


    # --------------------------------------------------------
    # Create profile
    #
    # All selected songs are positive preferences.
    # --------------------------------------------------------

    user_profile = np.mean(
        songs_scaled,
        axis=0
    )


    # --------------------------------------------------------
    # Return profile to mobile app
    # --------------------------------------------------------

    return {

        "status": "success",

        "profile": user_profile.tolist(),

        "profile_dimensions": len(
            user_profile
        ),

        "songs_used": len(songs)
    }


# ============================================================
# V2 - RECOMMEND
# ============================================================

@app.post("/recommend_v2")
def recommend_v2(request: V2Request):

    # --------------------------------------------------------
    # Validate recommendation count
    # --------------------------------------------------------

    if request.n_recommendations < 1:

        raise HTTPException(
            status_code=400,
            detail="n_recommendations must be at least 1"
        )

    if request.n_recommendations > 100:

        raise HTTPException(
            status_code=400,
            detail="Maximum 100 recommendations allowed"
        )


    # --------------------------------------------------------
    # Validate user profile
    # --------------------------------------------------------

    if len(request.user_profile) != len(
        FEATURE_COLUMNS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"user_profile must contain "
                f"{len(FEATURE_COLUMNS)} values."
            )
        )


    # --------------------------------------------------------
    # Create current-song DataFrame
    # --------------------------------------------------------

    current_song = create_song_dataframe(
        request
    )


    # ========================================================
    # STEP 1
    # V1 → 50 ENGLISH
    # ========================================================

    current_eng_scaled = (
        scaler_eng.transform(
            current_song[
                FEATURE_COLUMNS
            ]
        )
    )

    eng_n = min(
        51,
        len(df_eng)
    )

    eng_distances, eng_indices = (
        knn_eng.kneighbors(
            current_eng_scaled,
            n_neighbors=eng_n
        )
    )

    eng_candidates = df_eng.iloc[
        eng_indices[0]
    ].copy()

    eng_candidates = (
        eng_candidates
        .drop_duplicates(
            subset=["track_id"]
        )
        .head(50)
    )


    # ========================================================
    # STEP 2
    # V1 → 50 HINDI
    # ========================================================

    current_hin_scaled = (
        scaler_hin.transform(
            current_song[
                FEATURE_COLUMNS
            ]
        )
    )

    hin_n = min(
        51,
        len(df_hin)
    )

    hin_distances, hin_indices = (
        knn_hin.kneighbors(
            current_hin_scaled,
            n_neighbors=hin_n
        )
    )

    hin_candidates = df_hin.iloc[
        hin_indices[0]
    ].copy()

    hin_candidates = (
        hin_candidates
        .drop_duplicates(
            subset=["track_id"]
        )
        .head(50)
    )


    # ========================================================
    # STEP 3
    # COMBINE V1 CANDIDATES
    # ========================================================

    candidates = pd.concat(
        [
            eng_candidates,
            hin_candidates
        ],
        ignore_index=True
    )

    candidates = (
        candidates
        .drop_duplicates(
            subset=["track_id"]
        )
        .reset_index(drop=True)
    )


    if len(candidates) == 0:

        raise HTTPException(
            status_code=500,
            detail="V1 generated no candidates."
        )


    # ========================================================
    # STEP 4
    # SCALE CANDIDATES USING V2 SCALER
    # ========================================================

    candidate_features = (
        candidates[
            FEATURE_COLUMNS
        ]
    )

    candidate_scaled = (
        v2_scaler.transform(
            candidate_features
        )
    )


    # ========================================================
    # STEP 5
    # SCALE CURRENT SONG
    # ========================================================

    current_scaled = (
        v2_scaler.transform(
            current_song[
                FEATURE_COLUMNS
            ]
        )
    )


    # ========================================================
    # STEP 6
    # CONTENT SIMILARITY
    # ========================================================

    content_similarity = (
        cosine_similarity(
            current_scaled,
            candidate_scaled
        )[0]
    )


    # ========================================================
    # STEP 7
    # USER SIMILARITY
    # ========================================================

    user_profile = np.asarray(
        request.user_profile,
        dtype=float
    ).reshape(1, -1)


    user_similarity = (
        cosine_similarity(
            user_profile,
            candidate_scaled
        )[0]
    )


    # ========================================================
    # STEP 8
    # V2 FINAL SCORE
    # ========================================================

    v2_scores = (
        V2_USER_WEIGHT *
        user_similarity
        +
        V2_CONTENT_WEIGHT *
        content_similarity
    )


    # ========================================================
    # STEP 9
    # RANK
    # ========================================================

    ranked_indices = np.argsort(
        v2_scores
    )[::-1]

    ranked_indices = ranked_indices[
        :request.n_recommendations
    ]


    recommendations = candidates.iloc[
        ranked_indices
    ].copy()


    # ========================================================
    # STEP 10
    # ADD SCORES
    # ========================================================

    recommendations[
        "v2_score"
    ] = v2_scores[
        ranked_indices
    ]

    recommendations[
        "content_similarity"
    ] = content_similarity[
        ranked_indices
    ]

    recommendations[
        "user_similarity"
    ] = user_similarity[
        ranked_indices
    ]


    # ========================================================
    # STEP 11
    # RESPONSE
    # ========================================================

    result_columns = [
        "track_id",
        "track_name",
        "artist_name",
        "album_name",
        "year",
        "language",
        "popularity",
        "v2_score",
        "content_similarity",
        "user_similarity"
    ]

    result_columns = [
        col
        for col in result_columns
        if col in recommendations.columns
    ]


    results = recommendations[
        result_columns
    ].to_dict(
        orient="records"
    )


    return {

        "model": "V2",

        "candidate_count": len(
            candidates
        ),

        "recommendation_count": len(
            results
        ),

        "recommendations": results
    }