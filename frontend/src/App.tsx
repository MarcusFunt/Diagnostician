import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { createRun, listCases, submitDiagnosis, submitTurn } from "./api";
import type {
  ActionType,
  CaseFact,
  CaseReview,
  CaseSummary,
  DisplayBlock,
  PlayerTurnRequest,
  RunState,
  TurnResponse,
  UUID,
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
  targetHint: "Symptom, history, risk factor...",
  requestHint: "What would you like to ask?",
};

const ACTIONS: ActionOption[] = [
  DEFAULT_ACTION,
  {
    value: "request_exam_detail",
    label: "Exam detail",
    targetHint: "Vitals, abdomen, neurologic exam...",
    requestHint: "Which exam findings should be clarified?",
  },
  {
    value: "order_lab",
    label: "Order lab",
    targetHint: "CBC, D-dimer, hCG, CSF profile...",
    requestHint: "Add clinical context for the lab request.",
  },
  {
    value: "order_imaging",
    label: "Order imaging",
    targetHint: "Chest x-ray, CTA, ultrasound...",
    requestHint: "Which imaging study should be ordered?",
  },
  {
    value: "request_pathology_detail",
    label: "Procedure detail",
    targetHint: "Endoscopy, pathology, procedure...",
    requestHint: "Which report or procedure detail should be reviewed?",
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

const SEEN_CASES_KEY = "diagnostician.seenCaseIds";

export default function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<UUID | "">("");
  const [specialtyFilter, setSpecialtyFilter] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("");
  const [avoidReplay, setAvoidReplay] = useState(true);
  const [seenCaseIds, setSeenCaseIds] = useState<UUID[]>(() => readSeenCases());

  const [runState, setRunState] = useState<RunState | null>(null);
  const [blocks, setBlocks] = useState<DisplayBlock[]>([]);
  const [evidence, setEvidence] = useState<CaseFact[]>([]);
  const [actionType, setActionType] = useState<ActionType>("ask_patient_question");
  const [target, setTarget] = useState("");
  const [playerText, setPlayerText] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [rationale, setRationale] = useState("");
  const [review, setReview] = useState<CaseReview | null>(null);
  const [status, setStatus] = useState("Choose a case.");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  const selectedAction = useMemo(
    () => ACTIONS.find((action) => action.value === actionType) ?? DEFAULT_ACTION,
    [actionType],
  );

  const specialties = useMemo(() => uniqueValues(cases.map((item) => item.specialty)), [cases]);
  const difficulties = useMemo(() => uniqueValues(cases.map((item) => item.difficulty)), [cases]);

  const filteredCases = useMemo(
    () =>
      cases.filter(
        (item) =>
          (!specialtyFilter || item.specialty === specialtyFilter) &&
          (!difficultyFilter || item.difficulty === difficultyFilter),
      ),
    [cases, difficultyFilter, specialtyFilter],
  );

  const selectedCase = useMemo(
    () => cases.find((item) => item.id === selectedCaseId) ?? null,
    [cases, selectedCaseId],
  );
  const activeCase = useMemo(
    () => cases.find((item) => item.id === runState?.case_id) ?? null,
    [cases, runState?.case_id],
  );

  const runId = runState?.id ?? null;
  const isComplete = runState?.status === "complete" || review !== null;
  const evidenceGroups = useMemo(() => groupEvidence(evidence), [evidence]);

  const loadCaseLibrary = useCallback(async () => {
    setStatus("Loading cases...");
    setError(null);
    try {
      const items = await listCases();
      setCases(items);
      setStatus(items.length > 0 ? `${items.length} demo cases ready.` : "No demo cases available.");
    } catch (caught) {
      setStatus("Backend unavailable");
      setError(caught instanceof Error ? caught.message : "Unable to load cases.");
    }
  }, []);

  useEffect(() => {
    void loadCaseLibrary();
  }, [loadCaseLibrary]);

  const applyTurnResponse = useCallback((response: TurnResponse, replaceBlocks = false) => {
    setRunState(response.run_state);
    setEvidence(response.visible_evidence.facts);
    setBlocks((current) =>
      replaceBlocks ? response.display_blocks : [...current, ...response.display_blocks],
    );
    setStatus(readValidationStatus(response));
  }, []);

  const rememberSeenCase = useCallback((caseId: UUID) => {
    setSeenCaseIds((current) => {
      const next = current.includes(caseId) ? current : [...current, caseId];
      window.localStorage.setItem(SEEN_CASES_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const startCase = useCallback(
    async (caseId: UUID | "" = selectedCaseId) => {
      setIsBusy(true);
      setError(null);
      setStatus("Starting case...");
      setReview(null);
      setDiagnosis("");
      setRationale("");
      setTarget("");
      setPlayerText("");
      setActionType("ask_patient_question");

      try {
        const response = await createRun({
          case_id: caseId || null,
          specialty: caseId ? null : specialtyFilter || null,
          difficulty: caseId ? null : difficultyFilter || null,
          exclude_case_ids: caseId || !avoidReplay ? [] : seenCaseIds,
          randomize: !caseId,
        });
        applyTurnResponse(response, true);
        rememberSeenCase(response.run_state.case_id);
      } catch (caught) {
        setRunState(null);
        setBlocks([]);
        setEvidence([]);
        setStatus("Case start failed");
        setError(caught instanceof Error ? caught.message : "Unable to start a case.");
      } finally {
        setIsBusy(false);
      }
    },
    [
      applyTurnResponse,
      avoidReplay,
      difficultyFilter,
      rememberSeenCase,
      seenCaseIds,
      selectedCaseId,
      specialtyFilter,
    ],
  );

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

  function returnToLibrary() {
    setRunState(null);
    setBlocks([]);
    setEvidence([]);
    setReview(null);
    setDiagnosis("");
    setRationale("");
    setTarget("");
    setPlayerText("");
    setStatus(cases.length > 0 ? `${cases.length} demo cases ready.` : "Choose a case.");
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
          {activeCase ? <span>{activeCase.title}</span> : null}
          {runState ? <span>Turn {runState.turn_count}</span> : null}
          {runState?.score !== null && runState?.score !== undefined ? (
            <span>{runState.score} / 100</span>
          ) : null}
        </div>
      </header>

      {error ? (
        <section className="notice notice-error" role="alert">
          <strong>{error}</strong>
          <button
            type="button"
            onClick={() => void (runState ? startCase(runState.case_id) : loadCaseLibrary())}
            disabled={isBusy}
          >
            Retry
          </button>
        </section>
      ) : null}

      {runState === null ? (
        <StartScreen
          cases={filteredCases}
          allCasesCount={cases.length}
          selectedCaseId={selectedCaseId}
          selectedCase={selectedCase}
          specialties={specialties}
          difficulties={difficulties}
          specialtyFilter={specialtyFilter}
          difficultyFilter={difficultyFilter}
          avoidReplay={avoidReplay}
          seenCaseIds={seenCaseIds}
          isBusy={isBusy}
          onSelectCase={setSelectedCaseId}
          onSpecialtyChange={setSpecialtyFilter}
          onDifficultyChange={setDifficultyFilter}
          onAvoidReplayChange={setAvoidReplay}
          onStartRandom={() => void startCase("")}
          onStartSelected={() => void startCase(selectedCaseId)}
        />
      ) : (
        <>
          {review ? <ReviewBanner review={review} /> : null}

          <section className="workspace" aria-label="Diagnostic case workspace">
            <section className="case-stream" aria-labelledby="case-stream-title">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Case stream</p>
                  <h2 id="case-stream-title">Source-grounded updates</h2>
                </div>
                <div className="button-row">
                  <button type="button" onClick={returnToLibrary} disabled={isBusy}>
                    Case library
                  </button>
                  <button type="button" onClick={() => void startCase("")} disabled={isBusy}>
                    Random case
                  </button>
                </div>
              </div>

              <div className="block-list">
                {blocks.map((block) => (
                  <DisplayBlockCard key={block.id} block={block} />
                ))}
              </div>
            </section>

            <aside className="side-panel" aria-label="Actions and evidence">
              <EvidencePanel groups={evidenceGroups} />

              <section className="differential-panel" aria-labelledby="differential-title">
                <div className="section-heading compact">
                  <div>
                    <p className="eyebrow">Working list</p>
                    <h2 id="differential-title">Differential</h2>
                  </div>
                </div>
                {runState.submitted_differentials.length > 0 ? (
                  <ul className="token-list">
                    {runState.submitted_differentials.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">Submit a differential when the pattern starts to form.</p>
                )}
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
                    placeholder="Tie your answer to revealed findings."
                    rows={3}
                    disabled={isBusy || isComplete}
                  />
                </label>

                <button type="submit" disabled={isBusy || isComplete || runId === null || !diagnosis.trim()}>
                  Submit final diagnosis
                </button>
              </form>

              {review ? <ReviewPanel review={review} /> : null}
            </aside>
          </section>
        </>
      )}
    </main>
  );
}

function StartScreen({
  cases,
  allCasesCount,
  selectedCaseId,
  selectedCase,
  specialties,
  difficulties,
  specialtyFilter,
  difficultyFilter,
  avoidReplay,
  seenCaseIds,
  isBusy,
  onSelectCase,
  onSpecialtyChange,
  onDifficultyChange,
  onAvoidReplayChange,
  onStartRandom,
  onStartSelected,
}: {
  cases: CaseSummary[];
  allCasesCount: number;
  selectedCaseId: UUID | "";
  selectedCase: CaseSummary | null;
  specialties: string[];
  difficulties: string[];
  specialtyFilter: string;
  difficultyFilter: string;
  avoidReplay: boolean;
  seenCaseIds: UUID[];
  isBusy: boolean;
  onSelectCase: (id: UUID | "") => void;
  onSpecialtyChange: (value: string) => void;
  onDifficultyChange: (value: string) => void;
  onAvoidReplayChange: (value: boolean) => void;
  onStartRandom: () => void;
  onStartSelected: () => void;
}) {
  return (
    <section className="start-grid" aria-label="Case library">
      <section className="case-picker">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Case library</p>
            <h2>{cases.length} available cases</h2>
          </div>
          <button type="button" onClick={onStartRandom} disabled={isBusy || allCasesCount === 0}>
            Start random case
          </button>
        </div>

        <div className="filters">
          <label>
            Specialty
            <select value={specialtyFilter} onChange={(event) => onSpecialtyChange(event.target.value)}>
              <option value="">Any specialty</option>
              {specialties.map((specialty) => (
                <option value={specialty} key={specialty}>
                  {specialty}
                </option>
              ))}
            </select>
          </label>
          <label>
            Difficulty
            <select value={difficultyFilter} onChange={(event) => onDifficultyChange(event.target.value)}>
              <option value="">Any difficulty</option>
              {difficulties.map((difficulty) => (
                <option value={difficulty} key={difficulty}>
                  {formatToken(difficulty)}
                </option>
              ))}
            </select>
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              checked={avoidReplay}
              onChange={(event) => onAvoidReplayChange(event.target.checked)}
            />
            Avoid played cases
          </label>
        </div>

        <div className="case-card-grid">
          {cases.length > 0 ? (
            cases.map((item) => (
              <button
                type="button"
                className={`case-card ${item.id === selectedCaseId ? "case-card-selected" : ""}`}
                onClick={() => onSelectCase(item.id)}
                key={item.id}
              >
                <span className="case-card-header">
                  <strong>{item.title}</strong>
                  {seenCaseIds.includes(item.id) ? <span>Played</span> : null}
                </span>
                <span>{item.chief_complaint}</span>
                <span className="case-meta">
                  {item.specialty ?? "general"} / {formatToken(item.difficulty)}
                </span>
              </button>
            ))
          ) : (
            <p className="empty-state">No cases match the selected filters.</p>
          )}
        </div>
      </section>

      <aside className="selected-case-panel" aria-label="Selected case">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Selected case</p>
            <h2>{selectedCase ? selectedCase.title : "Randomized"}</h2>
          </div>
        </div>
        {selectedCase ? (
          <>
            <p>{selectedCase.chief_complaint}</p>
            <div className="token-list inline">
              <span>{selectedCase.specialty ?? "general"}</span>
              <span>{formatToken(selectedCase.difficulty)}</span>
              {selectedCase.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <button type="button" onClick={onStartSelected} disabled={isBusy}>
              Start selected case
            </button>
          </>
        ) : (
          <>
            <p>Use filters for a random case, or choose a specific case from the library.</p>
            <button type="button" onClick={onStartRandom} disabled={isBusy || allCasesCount === 0}>
              Start random case
            </button>
          </>
        )}
      </aside>
    </section>
  );
}

function EvidencePanel({ groups }: { groups: Array<[string, CaseFact[]]> }) {
  return (
    <section className="evidence-panel" aria-labelledby="evidence-title">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Visible evidence</p>
          <h2 id="evidence-title">{groups.reduce((count, [, items]) => count + items.length, 0)} findings</h2>
        </div>
      </div>
      <div className="evidence-list">
        {groups.length > 0 ? (
          groups.map(([category, items]) => (
            <section className="evidence-group" key={category}>
              <h3>{formatToken(category)}</h3>
              {items.map((fact) => (
                <EvidenceItem key={fact.id} fact={fact} />
              ))}
            </section>
          ))
        ) : (
          <p className="empty-state">No evidence is visible yet.</p>
        )}
      </div>
    </section>
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

function ReviewPanel({ review }: { review: CaseReview }) {
  return (
    <section className="review-panel" aria-labelledby="review-title">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Review</p>
          <h2 id="review-title">Teaching points</h2>
        </div>
      </div>
      <ul className="review-list">
        {review.teaching_points.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>
      <h3>Key findings</h3>
      <div className="mini-finding-list">
        {review.key_findings.map((fact) => (
          <span key={fact.id}>{fact.label}</span>
        ))}
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

function groupEvidence(facts: CaseFact[]): Array<[string, CaseFact[]]> {
  const groups = new Map<string, CaseFact[]>();
  for (const fact of facts) {
    groups.set(fact.category, [...(groups.get(fact.category) ?? []), fact]);
  }
  return Array.from(groups.entries());
}

function uniqueValues(values: Array<string | null>): string[] {
  return Array.from(new Set(values.filter((value): value is string => Boolean(value)))).sort();
}

function readSeenCases(): UUID[] {
  try {
    const raw = window.localStorage.getItem(SEEN_CASES_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item): item is UUID => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function formatToken(value: string): string {
  return value
    .split("_")
    .join(" ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
