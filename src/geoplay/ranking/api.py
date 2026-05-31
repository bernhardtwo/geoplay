"""FastAPI serving app for the ranking model.

Run locally from the repo root:

    uv run uvicorn geoplay.ranking.api:app --port 8000

The serving directory and the model source are read from the environment so
the same app runs unchanged in a container:

    GEOPLAY_SERVING_DIR   directory holding serving_features.parquet and
                          serving_meta.json (default: the local data path)
    GEOPLAY_MODEL_URI     model location. When set (e.g. a local exported
                          model path), the service loads it directly and
                          never touches the MLflow registry. When unset, the
                          service resolves the latest registry version.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from geoplay.ranking.serving import RankingService

DEFAULT_SERVING_DIR = "data/processed/ranking/serving"

SERVING_DIR = Path(os.environ.get("GEOPLAY_SERVING_DIR", DEFAULT_SERVING_DIR))
# An unset or empty variable means "use the registry".
MODEL_URI = os.environ.get("GEOPLAY_MODEL_URI") or None


class RankRequest(BaseModel):
    """A ranking request for one player in one (day_of_week, period) window."""

    player_id: str
    day_of_week: int = Field(ge=0, le=6, description="0 = Monday, 6 = Sunday")
    period: Literal["morning", "afternoon", "evening", "night"]
    top_k: int = Field(default=10, ge=1, le=100)


class RankedHex(BaseModel):
    """A single ranked hex and its model score."""

    hex_id: str
    score: float


class RankResponse(BaseModel):
    """The ranked result for a request."""

    player_id: str
    n_candidates: int
    ranked: list[RankedHex]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the serving artifacts once at startup."""
    app.state.service = RankingService(serving_dir=SERVING_DIR, model_uri=MODEL_URI)
    yield


app = FastAPI(title="geoplay ranking", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness check that also reports the size of the served universe."""
    service: RankingService = app.state.service
    return {"status": "ok", "known_players": service.known_players()}


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    """Rank the hexes a player already knows for the given time window."""
    service: RankingService = app.state.service
    try:
        result = service.rank(
            player_id=request.player_id,
            day_of_week=request.day_of_week,
            period=request.period,
            top_k=request.top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown player: {request.player_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RankResponse(
        player_id=request.player_id,
        n_candidates=result.n_candidates,
        ranked=[RankedHex(hex_id=h, score=s) for h, s in result.ranked],
    )
