export type UUID = string;

export interface CaseSummary {
  id: UUID;
  title: string;
  chief_complaint: string;
  difficulty: string;
  specialty: string | null;
  tags: string[];
  curation_notes: string[];
  created_at: string;
}

export interface CaseListResponse {
  items: CaseSummary[];
  next_cursor: string | null;
  total_estimate: number;
}

export type ActionType =
  | "ask_patient_question"
  | "request_exam_detail"
  | "order_lab"
  | "order_ecg"
  | "order_imaging"
  | "request_pathology_detail"
  | "give_treatment"
  | "request_consult"
  | "observe_patient"
  | "submit_differential"
  | "request_hint";

export type DisplayBlockType =
  | "narrative"
  | "patient_dialogue"
  | "attending_comment"
  | "lab_result"
  | "ecg_report"
  | "imaging_report"
  | "pathology_report"
  | "treatment_update"
  | "consult_note"
  | "observation_update"
  | "warning"
  | "hint"
  | "system_status";

export type Severity = "info" | "warning" | "error" | "success";

export type RunStatus = "active" | "complete" | "abandoned";

export type ValidationStatus = "pass" | "fail" | "fallback_used";

export interface CaseFact {
  id: UUID;
  case_id: UUID;
  category: string;
  label: string;
  value: string;
  provenance_ids: UUID[];
  spoiler: boolean;
  initially_visible: boolean;
  reveal_actions: ActionType[];
  tags: string[];
  search_text: string;
}

export interface Provenance {
  id: UUID;
  source_document_id: UUID;
  kind: string;
  locator: string | null;
  quote: string | null;
  note: string | null;
}

export interface VisibleEvidence {
  facts: CaseFact[];
}

export interface DisplayBlock {
  id: UUID;
  type: DisplayBlockType;
  title: string;
  body: string;
  fact_ids: UUID[];
  provenance_ids: UUID[];
  severity: Severity;
}

export interface RunState {
  id: UUID;
  case_id: UUID;
  status: RunStatus;
  stage: string;
  visible_fact_ids: UUID[];
  ordered_tests: string[];
  interventions: string[];
  consults: string[];
  observations: string[];
  requested_clues: string[];
  submitted_differentials: string[];
  hint_count: number;
  turn_count: number;
  score: number | null;
  case_story: string;
  run_summary: string;
  story_fact_ids: UUID[];
  created_at: string;
  updated_at: string;
}

export interface ValidationReport {
  id: UUID;
  status: ValidationStatus;
  hard_errors: string[];
  warnings: string[];
  allowed_fact_ids: UUID[];
  revealed_fact_ids: UUID[];
  soft_audit: SoftAuditScore;
  fallback_used: boolean;
  created_at: string;
}

export interface TurnResponse {
  run_state: RunState;
  display_blocks: DisplayBlock[];
  newly_revealed_facts: CaseFact[];
  visible_evidence: VisibleEvidence;
  validation: ValidationReport;
}

export interface PlayerTurnRequest {
  action_type: ActionType;
  player_text: string;
  target: string | null;
  selected_evidence_ids?: UUID[];
  client_timestamp?: string;
}

export interface RunCreateRequest {
  case_id?: UUID | null;
  specialty?: string | null;
  difficulty?: string | null;
  exclude_case_ids?: UUID[];
  randomize?: boolean;
}

export interface DiagnosisSubmission {
  diagnosis: string;
  rationale: string;
  client_timestamp?: string;
}

export interface ScoreSummary {
  correct: boolean;
  final_score: number;
  diagnosis_points: number;
  differential_points: number;
  efficiency_penalty: number;
  testing_penalty: number;
  hint_penalty: number;
  dangerous_miss_penalty: number;
  rationale_points: number;
  missed_key_findings_penalty: number;
  rationale: string[];
}

export interface SoftAuditScore {
  spoiler_risk: number;
  contradiction_risk: number;
  plausibility: number;
  tone_fit: number;
  notes: string[];
}

export interface CaseReview {
  run_id: UUID;
  case_id: UUID;
  diagnosis: string;
  player_diagnosis: string;
  player_rationale: string;
  player_score: ScoreSummary;
  key_findings: CaseFact[];
  revealed_key_findings: CaseFact[];
  missed_key_findings: CaseFact[];
  reasoning_path: ReasoningStep[];
  rationale_feedback: string[];
  teaching_points: string[];
  provenance: Provenance[];
  turn_timeline: DisplayBlock[];
}

export interface RunSnapshot {
  run_state: RunState;
  visible_evidence: VisibleEvidence;
  display_blocks: DisplayBlock[];
}

export interface ReasoningStep {
  turn_index: number;
  action_type: string;
  target: string | null;
  player_text: string;
  response_titles: string[];
  revealed_fact_labels: string[];
}

export interface HealthStatus {
  ok: boolean;
  ollama: {
    ok: boolean;
    models?: Array<{ name?: string }>;
    error?: string;
  };
  generation_model: string;
  case_generator_model: string;
  medical_check_model: string;
  medical_check_enabled: boolean;
  embedding_model: string;
}
