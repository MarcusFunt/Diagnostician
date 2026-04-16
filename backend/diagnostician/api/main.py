from __future__ import annotations

from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from diagnostician.core.config import get_settings
from diagnostician.core.schemas import (
    CaseReview,
    DiagnosisSubmission,
    PlayerTurnRequest,
    RunCreateRequest,
    RunSnapshot,
    TruthCase,
    TurnResponse,
)
from diagnostician.db.session import get_db_session
from diagnostician.llm.ollama_client import OllamaClient
from diagnostician.services.store import GameStore, SqlAlchemyGameStore
from diagnostician.services.workflows import DiagnosticWorkflow


app = FastAPI(title="Diagnostician API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store(session: Session = Depends(get_db_session)) -> GameStore:
    return SqlAlchemyGameStore(session)


def get_workflow(store: GameStore = Depends(get_store)) -> DiagnosticWorkflow:
    return DiagnosticWorkflow(store=store)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    ollama = OllamaClient(settings).health()
    return {
        "ok": True,
        "ollama": ollama,
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
    }


@app.get("/cases/approved", response_model=list[TruthCase])
def list_approved_cases(store: GameStore = Depends(get_store)) -> list[TruthCase]:
    return store.list_approved_cases()


@app.post("/runs", response_model=TurnResponse)
def create_run(
    request: RunCreateRequest,
    workflow: DiagnosticWorkflow = Depends(get_workflow),
) -> TurnResponse:
    try:
        return workflow.create_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}", response_model=RunSnapshot)
def get_run(
    run_id: UUID,
    workflow: DiagnosticWorkflow = Depends(get_workflow),
) -> RunSnapshot:
    try:
        return workflow.get_snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs/{run_id}/turns", response_model=TurnResponse)
def submit_turn(
    run_id: UUID,
    request: PlayerTurnRequest,
    workflow: DiagnosticWorkflow = Depends(get_workflow),
) -> TurnResponse:
    try:
        return workflow.handle_turn(run_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs/{run_id}/diagnosis", response_model=CaseReview)
def submit_diagnosis(
    run_id: UUID,
    request: DiagnosisSubmission,
    workflow: DiagnosticWorkflow = Depends(get_workflow),
) -> CaseReview:
    try:
        return workflow.submit_diagnosis(run_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/review", response_model=CaseReview)
def get_review(
    run_id: UUID,
    workflow: DiagnosticWorkflow = Depends(get_workflow),
) -> CaseReview:
    try:
        return workflow.get_review(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
