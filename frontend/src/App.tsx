import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRun, submitDiagnosis, submitTurn } from "./api";
import type {
  ActionType,
  CaseFact,
  CaseReview,
  DisplayBlock,
  PlayerTurnRequest,
  RunState,
  TurnResponse,
} from "./types";

type ActionOption = {
  value: ActionType;
  label: string;
  targetHint: string;
  requestHint: string;
};

const DEFAULT_ACTION: ActionOption = {
  value: "ask_patient_question",
  label: "Ask patient",
  targetHint: "Symptom, history, or concern",
  requestHint: "What would you like to ask?",
};

const ACTIONS: ActionOption[] = [
  DEFAULT_ACTION,
  {
    value: "request_exam_detail",
    label: "Exam detail",
    targetHint: "Vitals, cardiopulmonary, abdomen...",
    requestHint: "Which exam findings should be clarified?",
  },
  {
    value: "order_lab",
    label: "Order lab",
    targetHint: "CBC, D-dimer, troponin...",
    requestHint: "Add clinical context for the lab request.",
  },
  {
    value: "order_imaging",
    label: "Order imaging",
    targetHint: "Chest x-ray, CT, ultrasound...",
    requestHint: "Which imaging study should be ordered?",
  },
  {
    value: "request_pathology_detail",
    label: "Pathology detail",
    targetHint: "Biopsy site or report section",
    requestHint: "Which pathology detail should be reviewed?",
  },
  {
    value: "submit_differential",
    label: "Submit differential",
    targetHint: "Optional focus",
    requestHint: "List the working differential.",
  },
  {
    value: "request_hint",
    label: "Request hint",
    targetHint: "Optional topic",
    requestHint: "Where are you stuck?",
  },
];

const EMPTY_MESSAGE = "Start the backend, then begin a case.";

export default function App() {
  const [runState, setRunState] = useState<RunState | null>(null);
  const [blocks, setBlocks] = useState<DisplayBlock[]>([]);
  const [evidence, setEvidence] = useState<CaseFact[]>([]);
  const [actionType, setActionType] = useState<ActionType>("ask_patient_question");
  const [target, setTarget] = useState("");
  const [playerText, setPlayerText] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [rationale, setRationale] = useState("");
  const [review, setReview] = useState<CaseReview | null>(null);
  const [status, setStatus] = useState("Connecting...");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const didStartRef = useRef(false);

  const selectedAction = useMemo(
    () => ACTIONS.find((action) => action.value === actionType) ?? DEFAULT_ACTION,
    [actionType],
  );

  const runId = runState?.id ?? null;
  const isComplete = runState?.status === "complete" || review !== null;

  const applyTurnResponse = useCallback((response: TurnResponse, replaceBlocks = false) => {
    setRunState(response.run_state);
    setEvidence(response.visible_evidence.facts);
    setBlocks((current) =>
      replaceBlocks ? response.display_blocks : [...current, ...response.display_blocks],
    );
    setStatus(readValidationStatus(response));
  }, []);

  const startCase = useCallback(async () => {
    setIsBusy(true);
    setError(null);
    setStatus("Starting case...");
    setReview(null);
    setDiagnosis("");
    setRationale("");
    setTarget("");
    setPlayerText("");

    try {
      const response = await createRun();
      applyTurnResponse(response, true);
    } catch (caught) {
      setRunState(null);
      setBlocks([]);
      setEvidence([]);
      setStatus("Backend unavailable");
      setError(caught instanceof Error ? caught.message : "Unable to start a case.");
    } finally {
      setIsBusy(false);
    }
  }, [applyTurnResponse]);

  useEffect(() => {
    if (didStartRef.current) {
      return;
    }

    didStartRef.current = true;
    void startCase();
  }, [startCase]);

  async function handleTurnSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (runId === null) {
      setError("No active run is available.");
      return;
    }

    const payload: PlayerTurnRequest = {
      action_type: actionType,
      player_text: playerText.trim(),
      target: target.trim() || null,
      client_timestamp: new Date().toISOString(),
    };

    setIsBusy(true);
    setError(null);
    setStatus("Sending action...");

    try {
      const response = await submitTurn(runId, payload);
      applyTurnResponse(response);
      setTarget("");
      setPlayerText("");
    } catch (caught) {
      setStatus("Action failed");
      setError(caught instanceof Error ? caught.message : "Unable to submit the action.");
    } finally {
      setIsBusy(false);
    }
  }

  async function handleDiagnosisSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (runId === null) {
      setError("No active run is available.");
      return;
    }

    setIsBusy(true);
    setError(null);
    setStatus("Submitting diagnosis...");

    try {
      const nextReview = await submitDiagnosis(runId, {
        diagnosis: diagnosis.trim(),
        rationale: rationale.trim(),
        client_timestamp: new Date().toISOString(),
      });
      setReview(nextReview);
      setEvidence(nextReview.key_findings);
      setBlocks(nextReview.turn_timeline);
      setRunState((current) =>
        current === null
          ? current
          : {
              ...current,
              status: "complete",
              stage: "review",
              score: nextReview.player_score.final_score,
            },
      );
      setStatus(nextReview.player_score.correct ? "Diagnosis accepted" : "Review ready");
    } catch (caught) {
      setStatus("Diagnosis failed");
      setError(caught instanceof Error ? caught.message : "Unable to submit the diagnosis.");
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <img src="/diagnostic_monitor.svg" alt="" className="brand-icon" />
          <div>
            <p className="eyebrow">Diagnostic workstation</p>
            <h1>Diagnostician</h1>
          </div>
        </div>
        <div className="run-summary" aria-live="polite">
          <span className={`status-dot ${isBusy ? "status-dot-busy" : ""}`} />
          <span>{status}</span>
          {runState ? <span>Turn {runState.turn_count}</span> : null}
          {runState?.score !== null && runState?.score !== undefined ? (
            <span>{runState.score} / 100</span>
          ) : null}
        </div>
      </header>

      {error ? (
        <section className="notice notice-error" role="alert">
          <strong>{error}</strong>
          <button type="button" onClick={startCase} disabled={isBusy}>
            Retry
          </button>
        </section>
      ) : null}

      {review ? <ReviewBanner review={review} /> : null}

      <section className="workspace" aria-label="Diagnostic case workspace">
        <section className="case-stream" aria-labelledby="case-stream-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Case stream</p>
              <h2 id="case-stream-title">Source-grounded updates</h2>
            </div>
            <button type="button" onClick={startCase} disabled={isBusy}>
              New case
            </button>
          </div>

          <div className="block-list">
            {blocks.length > 0 ? (
              blocks.map((block) => <DisplayBlockCard key={block.id} block={block} />)
            ) : (
              <p className="empty-state">{EMPTY_MESSAGE}</p>
            )}
          </div>
        </section>

        <aside className="side-panel" aria-label="Actions and evidence">
          <section className="evidence-panel" aria-labelledby="evidence-title">
            <div className="section-heading compact">
              <div>
                <p className="eyebrow">Visible evidence</p>
                <h2 id="evidence-title">{evidence.length} findings</h2>
              </div>
            </div>
            <div className="evidence-list">
              {evidence.length > 0 ? (
                evidence.map((fact) => <EvidenceItem key={fact.id} fact={fact} />)
              ) : (
                <p className="empty-state">No evidence is visible yet.</p>
              )}
            </div>
          </section>

          <form className="action-panel" onSubmit={handleTurnSubmit}>
            <div className="section-heading compact">
              <div>
                <p className="eyebrow">Next action</p>
                <h2>Investigate</h2>
              </div>
            </div>

            <label>
              Action
              <select
                value={actionType}
                onChange={(event) => setActionType(event.target.value as ActionType)}
                disabled={isBusy || isComplete}
              >
                {ACTIONS.map((action) => (
                  <option value={action.value} key={action.value}>
                    {action.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Target
              <input
                value={target}
                onChange={(event) => setTarget(event.target.value)}
                placeholder={selectedAction.targetHint}
                disabled={isBusy || isComplete}
              />
            </label>

            <label>
              Request
              <textarea
                value={playerText}
                onChange={(event) => setPlayerText(event.target.value)}
                placeholder={selectedAction.requestHint}
                rows={4}
                disabled={isBusy || isComplete}
              />
            </label>

            <button type="submit" disabled={isBusy || isComplete || runId === null}>
              Submit action
            </button>
          </form>

          <form className="diagnosis-panel" onSubmit={handleDiagnosisSubmit}>
            <div className="section-heading compact">
              <div>
                <p className="eyebrow">Final answer</p>
                <h2>Diagnosis</h2>
              </div>
            </div>

            <label>
              Diagnosis
              <input
                value={diagnosis}
                onChange={(event) => setDiagnosis(event.target.value)}
                placeholder="Final diagnosis"
                disabled={isBusy || isComplete}
                required
              />
            </label>

            <label>
              Rationale
              <textarea
                value={rationale}
                onChange={(event) => setRationale(event.target.value)}
                placeholder="Source-grounded rationale"
                rows={3}
                disabled={isBusy || isComplete}
              />
            </label>

            <button type="submit" disabled={isBusy || isComplete || runId === null || !diagnosis.trim()}>
              Submit final diagnosis
            </button>
          </form>
        </aside>
      </section>
    </main>
  );
}

function DisplayBlockCard({ block }: { block: DisplayBlock }) {
  return (
    <article className={`display-block severity-${block.severity}`}>
      <div className="block-header">
        <h3>{block.title}</h3>
        <span>{formatToken(block.type)}</span>
      </div>
      <p>{block.body}</p>
    </article>
  );
}

function EvidenceItem({ fact }: { fact: CaseFact }) {
  return (
    <article className="evidence-item">
      <div>
        <h3>{fact.label}</h3>
        <span>{formatToken(fact.category)}</span>
      </div>
      <p>{fact.value}</p>
    </article>
  );
}

function ReviewBanner({ review }: { review: CaseReview }) {
  const score = review.player_score;

  return (
    <section className={`notice ${score.correct ? "notice-success" : "notice-warning"}`}>
      <div>
        <p className="eyebrow">Case review</p>
        <h2>{review.diagnosis}</h2>
        <p>
          Score {score.final_score} / 100. {score.rationale.join(" ")}
        </p>
      </div>
    </section>
  );
}

function readValidationStatus(response: TurnResponse): string {
  if (response.validation.fallback_used) {
    return "Ready, fallback used";
  }

  if (response.newly_revealed_facts.length > 0) {
    return `${response.newly_revealed_facts.length} new findings`;
  }

  return "Ready";
}

function formatToken(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
