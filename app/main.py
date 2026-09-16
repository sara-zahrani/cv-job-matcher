"""
HTTP API: the front door.

Everything underneath is plain functions. This layer turns an uploaded
file into a call to `match` and the result into JSON. Open the docs at
http://localhost:8000/docs once it is running.

    uv run uvicorn app.main:app --reload
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import APIStatusError, APITimeoutError
from pydantic import BaseModel

from app.config import settings
from app.rag.embeddings import Embedder
from app.rag.llm import LLMClient
from app.rag.matcher import MIN_SCORE, load_cv, match
from app.rag.vectorstore import VectorStore

# Opened once at startup, shared by every request (see lifespan below).
resources: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: fail here, before any request, if the key is missing or the
    # store can't open. Not on the first upload.
    settings.require_api_key()
    resources["store"] = VectorStore(Embedder())
    resources["llm"] = LLMClient()
    yield
    # Shutdown: flush the store to disk.
    resources["store"].close()


app = FastAPI(
    title="CV → Job Matcher",
    description="Upload a CV; get the best-fitting job postings from the indexed pool, with an explanation of each.",
    lifespan=lifespan,
)


# --- response shapes ------------------------------------------------------
# Pydantic models double as documentation: they appear in /docs and are
# validated on the way out.

class JobMatch(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    score: float


class MatchResponse(BaseModel):
    strong_match: bool
    min_score: float
    matches: list[JobMatch]
    assessment: str | None
    steps: list[str]


# --- endpoints ------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The one-page UI. Plain HTML that calls /match; no build step."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_jobs": resources["store"].count()}


@app.post("/match", response_model=MatchResponse)
async def match_cv(
    cv: UploadFile = File(..., description="CV as PDF or plain text"),
    name: str | None = Form(None, description="Candidate's name, to strip it before embedding"),
    top_k: int = Form(5, ge=1, le=20),
) -> MatchResponse:
    suffix = Path(cv.filename or "").suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(status_code=415, detail="Upload a .pdf or .txt file.")

    # The upload lives in memory; load_cv wants a path. A temp file bridges them.
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await cv.read())
        tmp.flush()
        try:
            cv_text = load_cv(Path(tmp.name), name)
        except ValueError as exc:  # e.g. a scanned PDF with no text layer
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = match(cv_text, resources["store"], resources["llm"], top_k=top_k)
    except APIStatusError as exc:
        # The model provider refused (rate limit, bad key, model down).
        # Surface its message; a generic 500 hides the one thing the user needs.
        detail = exc.body.get("message") if isinstance(exc.body, dict) else str(exc)
        raise HTTPException(status_code=503, detail=f"Model provider error: {detail}") from exc
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="The model did not answer in time. Try again.") from exc

    return MatchResponse(
        strong_match=result["strong_match"],
        min_score=MIN_SCORE,
        matches=[JobMatch(**m) for m in result["matches"]],
        assessment=result["assessment"],
        steps=result["steps"],
    )
