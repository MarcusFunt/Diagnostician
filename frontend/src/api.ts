import type {
  CaseListResponse,
  CaseReview,
  DiagnosisSubmission,
  HealthStatus,
  PlayerTurnRequest,
  RunCreateRequest,
  RunSnapshot,
  TurnResponse,
  UUID,
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

export interface CaseListParams {
  specialty?: string;
  difficulty?: string;
  q?: string;
  limit?: number;
  cursor?: string | null;
}

export async function listCases(params: CaseListParams = {}): Promise<CaseListResponse> {
  const query = new URLSearchParams();
  if (params.specialty) {
    query.set("specialty", params.specialty);
  }
  if (params.difficulty) {
    query.set("difficulty", params.difficulty);
  }
  if (params.q) {
    query.set("q", params.q);
  }
  if (params.limit) {
    query.set("limit", String(params.limit));
  }
  if (params.cursor) {
    query.set("cursor", params.cursor);
  }
  const suffix = query.toString() ? `?${query}` : "";
  return request<CaseListResponse>(`/cases${suffix}`);
}

export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

export async function createRun(payload: RunCreateRequest = {}): Promise<TurnResponse> {
  return request<TurnResponse>("/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getRun(runId: UUID): Promise<RunSnapshot> {
  return request<RunSnapshot>(`/runs/${runId}`);
}

export async function abandonRun(runId: UUID): Promise<RunSnapshot> {
  return request<RunSnapshot>(`/runs/${runId}/abandon`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function submitTurn(runId: UUID, payload: PlayerTurnRequest): Promise<TurnResponse> {
  return request<TurnResponse>(`/runs/${runId}/turns`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function submitDiagnosis(
  runId: UUID,
  payload: DiagnosisSubmission,
): Promise<CaseReview> {
  return request<CaseReview>(`/runs/${runId}/diagnosis`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getReview(runId: UUID): Promise<CaseReview> {
  return request<CaseReview>(`/runs/${runId}/review`);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init.headers,
    },
  });

  const text = await response.text();
  const payload: unknown = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new Error(readErrorMessage(payload, response.status));
  }

  return payload as T;
}

function readErrorMessage(payload: unknown, status: number): string {
  if (isRecord(payload)) {
    const detail = payload.detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((item) => (isRecord(item) && typeof item.msg === "string" ? item.msg : "Invalid request"))
        .join("; ");
    }
  }

  return `Backend request failed with status ${status}.`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
