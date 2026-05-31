"""FastAPI serving app for the geoplay ranker.

Run from the repository root so the relative artifact paths and the SQLite
tracking URI resolve correctly:

    uv run uvicorn geoplay.ranking.api:app --port 8000

Interactive docs are then at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from geoplay.ranking.serving import RankingService

SERVING_DIR = Path("data/processed/ranking/serving")


class RankRequest(BaseModel):
    """A ranking request for one player and time window."""

    player_id: str
    day_of_week: int = Field(ge=0, le=6, description="0=Monday ... 6=Sunday")
    period: Literal["night", "morning", "afternoon", "evening"]
    top_k: int = Field(default=10, ge=1, le=100)


class RankedHex(BaseModel):
    """One ranked hex with its score."""

    hex_id: str
    score: float


class RankResponse(BaseModel):
    """The ranked list for a request, plus the size of the player's universe."""

    player_id: str
    day_of_week: int
    period: str
    n_candidates: int
    results: list[RankedHex]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ranking service once at startup."""
    app.state.service = RankingService(serving_dir=SERVING_DIR)
    yield


app = FastAPI(title="geoplay ranker", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest, request: Request) -> RankResponse:
    """Rank a player's known hexes for a (day_of_week, period) window."""
    service: RankingService = request.app.state.service
    try:
        result = service.rank(
            player_id=req.player_id,
            day_of_week=req.day_of_week,
            period=req.period,
            top_k=req.top_k,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"player '{req.player_id}' not found in the known universe",
        ) from exc

    return RankResponse(
        player_id=req.player_id,
        day_of_week=req.day_of_week,
        period=req.period,
        n_candidates=result.n_candidates,
        results=[RankedHex(hex_id=h, score=s) for h, s in result.ranked],
    )
