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
from dotenv import load_dotenv

load_dotenv()

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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

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
    r = requests.post(url, headers=headers, json=payload, timeout=120)
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

def _call_mistral(user_prompt: str, language: str = "mix", count: int = 10):
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set on server (Render > Environment)")
    lang_hint = {"english": "English only", "hindi": "Hindi/Bollywood only", "mix": "mix of English and Hindi"}.get(language.lower(), "mix of English and Hindi")
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": f'User prompt: "{user_prompt}"\nLanguage preference: {lang_hint}\nReturn exactly {count} songs as JSON array.'},
        ],
        "temperature": 0.7,
        "max_tokens": 3000,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_API_KEY}"}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Mistral {r.status_code}: {r.text[:400]}")
    try:
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
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

class AIRequest(BaseModel):
    prompt: str
    language: str = "mix"
    count: int = 10
    q: str | None = None  # alias for prompt
    # --- LLM Taste Card / hybrid (all optional, backward compatible) ---
    provider: str = "gemini"  # "gemini" | "mistral"
    taste_card: dict | str | None = None
    user_profile: list[float] | None = None
    current_song: dict | None = None  # 9 audio features (kept as dict to avoid forward-ref)

# ============================================================
# LLM TASTE CARD + HYBRID HELPERS
# (additive only — existing endpoints untouched)
# ============================================================

TASTE_CARD_SYSTEM_PROMPT = (
    "You are AudioFit AI, a music taste profiler. "
    "Given aggregate audio-feature stats of a user's favorite songs plus optional free-text vibe, "
    "return ONLY a JSON object (no markdown, no explanation) with keys: "
    '{"taste_summary": "<=25 words overall taste", '
    '"energy_bias": "low|medium|high", '
    '"tempo_bias": "slow|moderate|fast", '
    '"languages": "e.g. mix 60% English / 40% Hindi", '
    '"likes": ["3-5 short traits"], '
    '"avoids": ["2-4 short traits"], '
    '"workout_use": "e.g. run + gym + cooldown", '
    '"archetype_hint": "one of high_energy_workout/edm_fan/pop_listener/rock_fan/chill_listener/acoustic_listener/hiphop_listener/low_energy_listener"}.'
)

HYBRID_RERANK_SYSTEM_PROMPT = (
    "You are AudioFit AI, a workout music recommender. "
    "You are given the user's taste card, their workout/mood request, and a list of candidate songs from the catalog. "
    "Pick ONLY songs from the candidate list (by exact track_id), ranked best-first for the request + taste. "
    "Return ONLY a JSON array, each element: {\"track_id\": \"...\", \"reason\": \"<=12 words why it fits\"}. "
    "No markdown, no explanation, no songs outside the candidate list."
)


def _chat_gemini_text(system_prompt: str, user_text: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server (Render > Environment)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3000},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Gemini {r.status_code}: {r.text[:400]}")
    try:
        data = r.json()
        candidates = data.get("candidates", [])
        parts = (candidates[0].get("content", {}).get("parts", []) if candidates else [])
        return "".join([str(p.get("text", "")) for p in parts if isinstance(p, dict)]).strip()
    except Exception:
        return ""


def _chat_mistral_text(system_prompt: str, user_text: str) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set on server (Render > Environment)")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": 3000,
        },
        timeout=120,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Mistral {r.status_code}: {r.text[:400]}")
    try:
        return r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def _chat_llm(provider: str, system_prompt: str, user_text: str) -> str:
    if (provider or "gemini").lower() == "mistral":
        return _chat_mistral_text(system_prompt, user_text)
    return _chat_gemini_text(system_prompt, user_text)


def _extract_json_object(text: str):
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _summarize_favorites(songs_df: pd.DataFrame) -> dict:
    means = songs_df[FEATURE_COLUMNS].mean().to_dict()
    energy = float(means.get("energy", 0.5))
    tempo = float(means.get("tempo", 100))
    # tempo is raw BPM (~60-200), energy is 0-1
    return {
        "means": {k: round(float(v), 3) for k, v in means.items()},
        "energy_bias": "high" if energy > 0.65 else ("low" if energy < 0.4 else "medium"),
        "tempo_bias": "fast" if tempo > 120 else ("slow" if tempo < 90 else "moderate"),
        "count": len(songs_df),
    }


def _build_hybrid_candidates(current_song_df: pd.DataFrame, user_profile: list[float] | None, top_n: int = 40) -> pd.DataFrame:
    # V1: 50 eng + 50 hin nearest to current song (same as /recommend_v2 steps 1-3)
    eng_n = min(51, len(df_eng))
    _, eng_idx = knn_eng.kneighbors(scaler_eng.transform(current_song_df[FEATURE_COLUMNS]), n_neighbors=eng_n)
    eng_c = df_eng.iloc[eng_idx[0]].copy().drop_duplicates(subset=["track_id"]).head(50)
    hin_n = min(51, len(df_hin))
    _, hin_idx = knn_hin.kneighbors(scaler_hin.transform(current_song_df[FEATURE_COLUMNS]), n_neighbors=hin_n)
    hin_c = df_hin.iloc[hin_idx[0]].copy().drop_duplicates(subset=["track_id"]).head(50)
    cands = pd.concat([eng_c, hin_c], ignore_index=True).drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    if len(cands) == 0:
        raise HTTPException(status_code=500, detail="V1 generated no candidates.")
    cand_scaled = v2_scaler.transform(cands[FEATURE_COLUMNS])
    cur_scaled = v2_scaler.transform(current_song_df[FEATURE_COLUMNS])
    content_sim = cosine_similarity(cur_scaled, cand_scaled)[0]
    if user_profile is not None:
        up = np.asarray(user_profile, dtype=float).reshape(1, -1)
        user_sim = cosine_similarity(up, cand_scaled)[0]
        scores = V2_USER_WEIGHT * user_sim + V2_CONTENT_WEIGHT * content_sim
        cands["user_similarity"] = user_sim
    else:
        scores = content_sim
    cands["content_similarity"] = content_sim
    cands["v2_score"] = scores
    return cands.sort_values("v2_score", ascending=False).head(max(10, min(top_n, len(cands)))).reset_index(drop=True)


def _slim_candidates_for_llm(cands: pd.DataFrame) -> list:
    slim = []
    for _, r in cands.iterrows():
        slim.append({
            "track_id": str(r.get("track_id", "")),
            "title": str(r.get("track_name", ""))[:60],
            "artist": str(r.get("artist_name", ""))[:60],
            "language": str(r.get("language", "")),
            "energy": round(float(r.get("energy", 0)), 2),
            "tempo": round(float(r.get("tempo", 0)), 2),
            "valence": round(float(r.get("valence", 0)), 2),
            "popularity": int(r.get("popularity", 0)) if pd.notna(r.get("popularity", 0)) else 0,
        })
    return slim

@app.post("/ai-recommend")
def ai_recommend(body: AIRequest):
    prompt = (body.prompt or body.q or "").strip()
    language = (body.language or "mix").strip()
    count = max(3, int(body.count or 10))
    provider = (body.provider or "gemini").lower()
    if provider not in ("gemini", "mistral"):
        raise HTTPException(status_code=400, detail="provider must be 'gemini' or 'mistral'")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > 500:
        raise HTTPException(status_code=400, detail="prompt too long (max 500 chars)")

    # --- Hybrid path: taste card and/or V2 profile + current song ---
    if body.taste_card is not None or body.user_profile is not None or body.current_song is not None:
        if body.current_song is None:
            raise HTTPException(status_code=400, detail="current_song is required for personalized (hybrid) recommendations")
        if body.user_profile is not None and len(body.user_profile) != len(FEATURE_COLUMNS):
            raise HTTPException(status_code=400, detail=f"user_profile must contain {len(FEATURE_COLUMNS)} values.")
        try:
            cur_df = pd.DataFrame([{k: float(body.current_song[k]) for k in FEATURE_COLUMNS}])
        except Exception:
            raise HTTPException(status_code=400, detail="current_song must contain all 9 audio features as numbers.")
        cands = _build_hybrid_candidates(cur_df, body.user_profile, top_n=40)
        slim = _slim_candidates_for_llm(cands)
        taste_str = body.taste_card if isinstance(body.taste_card, str) else json.dumps(body.taste_card or {})
        llm_text = (
            f"Taste card: {taste_str[:2000]}\n"
            f'User request: "{prompt}"\nLanguage preference: {language}\n'
            f"Return exactly {count} songs.\nCandidates (JSON): {json.dumps(slim)}"
        )
        try:
            content = _chat_llm(provider, HYBRID_RERANK_SYSTEM_PROMPT, llm_text)
            arr = _extract_json_array(content)
            if not arr:
                raise ValueError("no json array")
            by_id = {str(r.get("track_id")): r for _, r in cands.iterrows()}
            out, seen = [], set()
            for item in arr:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("track_id", "")).strip()
                if not tid or tid in seen or tid not in by_id:
                    continue  # strict grounding: ignore hallucinated IDs
                seen.add(tid)
                r = by_id[tid]
                out.append({
                    "track_id": tid,
                    "track_name": str(r.get("track_name", "")),
                    "artist_name": str(r.get("artist_name", "")),
                    "album_name": str(r.get("album_name", "")),
                    "year": int(r.get("year")) if pd.notna(r.get("year")) else None,
                    "language": str(r.get("language", "")),
                    "popularity": int(r.get("popularity", 0)) if pd.notna(r.get("popularity", 0)) else 0,
                    "reason": str(item.get("reason", ""))[:80] or "Matches your taste",
                    "v2_score": float(r.get("v2_score", 0)),
                })
                if len(out) >= count:
                    break
            if out:
                model_name = MISTRAL_MODEL if provider == "mistral" else GEMINI_MODEL
                return {"songs": out, "model": model_name, "mode": "hybrid", "candidate_count": len(cands)}
        except HTTPException:
            raise
        except Exception:
            pass
        # Fallback: pure V2 ranking with generic reasons
        fb = cands.head(count)
        return {
            "songs": [{
                "track_id": str(r.get("track_id", "")),
                "track_name": str(r.get("track_name", "")),
                "artist_name": str(r.get("artist_name", "")),
                "album_name": str(r.get("album_name", "")),
                "year": int(r.get("year")) if pd.notna(r.get("year")) else None,
                "language": str(r.get("language", "")),
                "popularity": int(r.get("popularity", 0)) if pd.notna(r.get("popularity", 0)) else 0,
                "reason": "Matches your taste profile",
                "v2_score": float(r.get("v2_score", 0)),
            } for _, r in fb.iterrows()],
            "model": "V2-fallback",
            "mode": "hybrid-fallback",
            "candidate_count": len(cands),
        }

    # --- Legacy open-world path (unchanged behavior) ---
    if provider == "mistral":
        songs = _call_mistral(prompt, language, count)
        return {"songs": songs, "model": MISTRAL_MODEL, "mode": "open-world"}
    songs = _call_gemini(prompt, language, count)
    return {"songs": songs, "model": GEMINI_MODEL, "mode": "open-world"}


# ============================================================
# LLM TASTE CARD CREATION
# Client stores the returned card, sends it back with /ai-recommend
# ============================================================

class TasteSong(FavoriteSong):
    track_name: str | None = None
    artist_name: str | None = None
    language: str | None = None


class CreateTasteCardRequest(BaseModel):
    favorite_songs: list[TasteSong]
    vibe_text: str | None = None
    language: str = "mix"
    provider: str = "gemini"  # "gemini" | "mistral"


@app.post("/create_taste_card")
def create_taste_card(request: CreateTasteCardRequest):
    if len(request.favorite_songs) < 3:
        raise HTTPException(status_code=400, detail="At least 3 favorite songs are required.")
    provider = (request.provider or "gemini").lower()
    if provider not in ("gemini", "mistral"):
        raise HTTPException(status_code=400, detail="provider must be 'gemini' or 'mistral'")
    vibe = (request.vibe_text or "").strip()[:500]

    songs_df = pd.DataFrame([s.model_dump() for s in request.favorite_songs])
    songs_df = songs_df[FEATURE_COLUMNS]
    stats = _summarize_favorites(songs_df)

    samples = []
    for s in request.favorite_songs[:10]:
        samples.append({
            "title": (s.track_name or "")[:60],
            "artist": (s.artist_name or "")[:60],
            "language": s.language or "",
        })
    lang_counts = pd.Series([(s.language or "").strip() or "unknown" for s in request.favorite_songs]).value_counts().to_dict()

    user_text = (
        f"Feature means (0-1 raw): {json.dumps(stats['means'])}\n"
        f"Energy bias: {stats['energy_bias']}, Tempo bias: {stats['tempo_bias']}, "
        f"Songs analyzed: {stats['count']}, Language counts: {json.dumps(lang_counts)}\n"
        f"Sample favorites: {json.dumps(samples)}\n"
        f"User vibe text: \"{vibe or 'none'}\"\n"
        f"Preferred language: {request.language}\n"
        "Return ONLY the JSON taste-card object."
    )
    content = _chat_llm(provider, TASTE_CARD_SYSTEM_PROMPT, user_text)
    card = _extract_json_object(content)
    if not card:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {content[:400]}")
    model_name = MISTRAL_MODEL if provider == "mistral" else GEMINI_MODEL
    return {"status": "success", "taste_card": card, "stats": stats, "model": model_name, "songs_used": len(songs_df)}



# ============================================================
# V2 PROFILE CREATION REQUEST (FavoriteSong defined above with AIRequest)
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