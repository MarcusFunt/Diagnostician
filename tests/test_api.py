from fastapi.testclient import TestClient

from diagnostician.api.main import app, get_workflow
from diagnostician.services.workflows import DiagnosticWorkflow

from tests.helpers import FakeLLMClient, populated_store


def test_api_run_turn_diagnosis_review_flow():
    store = populated_store()

    def override_workflow():
        return DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    app.dependency_overrides[get_workflow] = override_workflow
    try:
        client = TestClient(app)

        created = client.post("/runs", json={})
        assert created.status_code == 200
        run_id = created.json()["run_state"]["id"]

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
