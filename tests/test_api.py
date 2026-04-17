from fastapi.testclient import TestClient

from diagnostician.api.main import app, get_store, get_workflow
from diagnostician.services.store import InMemoryGameStore
from diagnostician.services.workflows import DiagnosticWorkflow

from tests.helpers import FakeLLMClient, demo_store, populated_store


def test_api_lists_safe_case_summaries():
    store = demo_store()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    app.dependency_overrides[get_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.get("/cases")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_estimate"] == 8
        assert len(payload["items"]) == 8
        assert payload["next_cursor"] is None
        assert "final_diagnosis" not in payload["items"][0]
        assert all("pulmonary embolism" not in tag.casefold() for item in payload["items"] for tag in item["tags"])
    finally:
        app.dependency_overrides.clear()


def test_api_run_turn_diagnosis_review_flow():
    store = populated_store()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    try:
        client = TestClient(app)

        created = client.post("/runs", json={})
        assert created.status_code == 200
        created_payload = created.json()
        run_id = created_payload["run_state"]["id"]
        assert "case_story" in created_payload["run_state"]
        assert "run_summary" in created_payload["run_state"]
        assert "story_fact_ids" in created_payload["run_state"]

        turn = client.post(
            f"/runs/{run_id}/turns",
            json={"action_type": "order_lab", "target": "D-dimer", "player_text": "Order d-dimer"},
        )
        assert turn.status_code == 200
        assert turn.json()["validation"]["status"] == "pass"

        diagnosis = client.post(
            f"/runs/{run_id}/diagnosis",
            json={"diagnosis": "Pulmonary embolism", "rationale": "CTA and risk factors"},
        )
        assert diagnosis.status_code == 200
        assert diagnosis.json()["player_score"]["correct"] is True

        review = client.get(f"/runs/{run_id}/review")
        assert review.status_code == 200
        assert review.json()["diagnosis"] == "Pulmonary embolism"
    finally:
        app.dependency_overrides.clear()


def test_api_case_listing_supports_pagination_and_search():
    store = demo_store()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    app.dependency_overrides[get_store] = lambda: store
    try:
        client = TestClient(app)
        first = client.get("/cases?limit=3")
        assert first.status_code == 200
        payload = first.json()
        assert len(payload["items"]) == 3
        assert payload["total_estimate"] == 8
        assert payload["next_cursor"] == "3"

        second = client.get(f"/cases?limit=3&cursor={payload['next_cursor']}")
        assert second.status_code == 200
        assert len(second.json()["items"]) == 3

        searched = client.get("/cases?q=chest")
        assert searched.status_code == 200
        assert searched.json()["total_estimate"] >= 1
        assert any(
            "chest" in " ".join([item["title"], item["chief_complaint"], *item["tags"]]).casefold()
            for item in searched.json()["items"]
        )
    finally:
        app.dependency_overrides.clear()


def test_api_invalid_run_and_no_approved_cases_errors():
    store = InMemoryGameStore()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    try:
        client = TestClient(app)

        missing_cases = client.post("/runs", json={})
        assert missing_cases.status_code == 409

        missing_run = client.get("/runs/00000000-0000-0000-0000-000000000000")
        assert missing_run.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_api_completed_run_turn_returns_complete_status():
    store = populated_store()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    try:
        client = TestClient(app)

        created = client.post("/runs", json={})
        run_id = created.json()["run_state"]["id"]
        client.post(
            f"/runs/{run_id}/diagnosis",
            json={"diagnosis": "Pulmonary embolism", "rationale": "CTA filling defects"},
        )
        turn = client.post(
            f"/runs/{run_id}/turns",
            json={"action_type": "order_lab", "target": "D-dimer", "player_text": "Order d-dimer"},
        )

        assert turn.status_code == 200
        assert turn.json()["run_state"]["status"] == "complete"
        assert turn.json()["display_blocks"][0]["title"] == "Run Complete"
    finally:
        app.dependency_overrides.clear()
