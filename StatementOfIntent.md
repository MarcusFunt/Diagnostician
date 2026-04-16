Engineering specification: Godot-first AI medical diagnostics game
Product vision

Build a single-player PC diagnostic reasoning game in Godot where the player works through realistic medical cases step by step. The player should feel like they are doing actual clinical reasoning rather than chatting with a hallucinating chatbot.

The game must therefore be built around a strict separation between:

source-grounded medical truth
game presentation
run/session state
AI-generated narrative and interaction

The system should use real de-identified medical cases as the backbone, transformed into a structured internal schema. AI is then used in a constrained and audited way to turn those source-backed facts into a playable case experience.

This is not a pure LLM chat product. It is a backend-authoritative stateful game system with LLM-assisted generation at controlled points.

The final product should support:

starting a new case run,
gradually revealing history, exam findings, labs, imaging, pathology, and other data,
allowing the player to ask questions, order tests, form differentials, and submit diagnoses,
checking that every turn is consistent with the underlying case,
preventing accidental spoilers,
scoring the player’s reasoning and efficiency,
and presenting the full case review after the run ends.

The final user experience should feel like a polished diagnostic workstation or teaching simulation, not a raw chat window.

Core architectural philosophy
1. Source truth is the center of the system

The underlying case data must come from real source material and be normalized into a structured format. The AI must never be treated as the medical source of truth.

The AI may:

rephrase,
narrate,
roleplay,
pace information release,
generate safe connective dialogue,
and present available facts in a more natural way.

The AI may not:

invent critical diagnostic facts,
alter the actual diagnosis,
contradict the source case,
or reveal locked information before the system allows it.

This architecture is chosen because it gives a much more accurate and controllable experience than freeform case generation. The point is to produce something that is medically grounded, auditable, and replayable.

2. The backend is authoritative

Gameplay-critical logic must live in the backend, not in the Godot client and not inside prompts alone.

The backend is responsible for:

selecting and loading the case,
maintaining run state,
determining which facts are visible,
deciding what the player is allowed to see next,
running the LLM workflow,
validating generated output,
and persisting the session.

The Godot client is responsible for:

rendering the experience,
collecting user input,
presenting structured result blocks,
and acting as a clean game frontend.

This is chosen because it keeps the game logic centralized, testable, and secure from drift.

3. The system is a workflow, not a free-roaming agent

The gameplay engine should be implemented as an explicit multi-step workflow. The model should not be allowed to “decide how the whole system works” on its own.

That means:

use deterministic state transitions,
explicit nodes for generation and validation,
fixed guardrails,
and structured outputs.

This is exactly why LangGraph is appropriate: it fits stateful, multi-step orchestration with checkpoints and controlled data flow. LangChain may be used as glue, wrappers, and structured generation helpers, but it should not define the architecture by itself.

Technology choices and reasoning
Godot for the client

The game is planned around Godot from the start, not as an afterthought.

Godot is chosen because:

it is a real game engine rather than a generic desktop UI framework,
it handles scenes, structured UI flows, input, transitions, persistence, and packaging cleanly,
it is well suited to a 2D interface-heavy game,
and it leaves room for future expansion into more game-like features without forcing a rewrite.

The project should target Godot 4.x and begin with GDScript unless there is a compelling implementation reason to use C#.

The game should be scene-based and UI-driven. The visual design should feel like a professional diagnostic terminal or workstation, with clearly separated panels for story, evidence, tests, notes, and player actions.

Python for the backend

Use Python as the main backend language because it is the most practical choice for:

LLM orchestration,
data processing,
retrieval pipelines,
schema validation,
case normalization,
and AI workflow integration.

Python is also a natural fit with FastAPI, Pydantic, LangGraph, database tooling, and offline ingestion pipelines.

Ollama for local LLM serving

Use Ollama as the LLM and embeddings runtime.

This is chosen because:

it supports local inference,
it works well for experimentation and deployment without depending on hosted APIs,
it supports embeddings as well as chat generation,
and it is a practical fit for a self-contained local-first stack.

Ollama should be used both for:

case/narrative generation steps,
and embeddings-based retrieval where appropriate.
LangGraph for orchestration

Use LangGraph as the AI workflow engine.

This is chosen because the product needs:

a stateful run model,
multi-step controlled execution,
durable checkpoints,
explicit workflow stages,
and audit/validation gates.

The game flow is not a general-purpose agent problem. It is a structured orchestration problem:

fetch case,
generate presentation,
validate,
reveal,
update state,
generate next turn,
validate again,
and continue.

LangGraph is a good fit because it supports this graph-style stateful execution directly.

LangChain as supporting glue

Use LangChain where it adds convenience, but do not let it become the core architecture.

Appropriate uses:

model wrappers,
prompt templates,
structured-output helpers,
tool abstractions if needed.

Inappropriate use:

making the whole game depend on one giant agent loop.

The project should be designed so that the hard game logic remains understandable even if the LLM layer is replaced later.

FastAPI for the API layer

Use FastAPI for the backend API.

This is chosen because:

it works cleanly with Pydantic,
it is fast enough for this use case,
it encourages typed request/response contracts,
and it is a clean interface between Godot and the backend.

The client-backend boundary should be JSON over HTTP.

Pydantic for schemas and validation

Use Pydantic for all core data contracts.

This is one of the most important implementation decisions in the project.

All major system objects should be validated:

source cases,
truth-layer facts,
revealable facts,
turn inputs,
turn outputs,
validation reports,
and run state.

This is chosen because prompt-only correctness is not reliable enough. The system needs typed, structured contracts around every critical data boundary.

Postgres as the main data store

Use Postgres as the primary persistent database.

This is chosen because:

it is mature and reliable,
it handles structured persistent data very well,
it fits both gameplay persistence and content storage,
and it works well with migrations and embeddings extensions.

Postgres should store:

normalized cases,
run state,
turn history,
validation logs,
source metadata,
and possibly embedding vectors via pgvector.
pgvector for retrieval

Use pgvector inside Postgres for initial retrieval.

This is chosen because:

it keeps the system simpler than introducing a separate vector database immediately,
it is good enough for the expected initial scale,
and it allows semantic retrieval without fragmenting infrastructure too early.

The project can later migrate or expand to a dedicated vector database if scale or performance genuinely requires it, but that is not the default assumption.

Alembic for schema migrations

Use Alembic for DB migrations.

This is necessary because the schema will evolve as:

new case types are added,
reveal policies become more sophisticated,
validation becomes richer,
and analytics/logging expands.
Docker Compose for local orchestration

Use Docker Compose to run the backend stack.

This is chosen because the system naturally consists of multiple moving parts:

backend API,
Postgres,
optional worker service,
optional local model service setup,
and possibly ingestion utilities.

The system should be easy to bring up locally and easy to reason about operationally. Docker Compose is the right level of complexity for that.

GitHub for versioning

Use GitHub for versioning, collaboration, issue tracking, and repository structure.

That choice is straightforward.

Product behavior and experience
What a player should experience

When a new run starts, the player should be presented with a realistic opening case presentation. This might be an ED arrival note, clinic presentation, pathology consult, or ward handoff depending on the case type.

The player should then be able to:

ask questions,
order tests,
request physical exam details,
review available findings,
build a differential,
ask for hints,
and eventually submit a final diagnosis.

Each response should feel coherent and context-aware. The patient should answer like a patient. The attending should comment like an attending. A pathologist or radiologist should sound like a specialist when the game reveals those modalities.

But that realism must sit on top of hard structure. The illusion of flexibility should be supported by a rigorous hidden truth model.

UI philosophy

The UI should avoid looking like a generic chat app.

The main gameplay screen should have distinct panels for:

case narrative / current events,
available evidence and revealed facts,
labs and imaging,
pathology and histology,
differential tracking,
action input,
run summary / notes.

Player interaction should support both:

free-text input, for expressiveness,
and structured actions, for clarity and game control.

Structured actions should include at least:

ask patient question,
request exam detail,
order lab,
order imaging,
request pathology detail,
submit differential,
submit final diagnosis,
request hint.

This is chosen because free text alone is too ambiguous and too hard to validate, while buttons alone make the game feel rigid.

Case model and data separation

The system must represent cases in layers.

Truth layer

This is the structured source-grounded representation of the real case. It must be immutable during the run.

It should include things like:

demographics,
chief complaint,
timeline,
symptoms,
past medical history,
medications,
social history,
vitals,
physical exam findings,
labs,
imaging,
pathology,
procedures,
microbiology,
final diagnosis,
differential tags,
teaching points,
spoiler-critical facts,
provenance and citation metadata.

This is the layer the system trusts.

Presentation layer

This is the playable representation of the case. It is derived from the truth layer and changes during the run.

It includes:

opening story text,
revealed narrative,
visible labs,
visible exam findings,
dialogue already shown,
already unlocked reports,
and any visualized game blocks.

This is what the player sees.

Run state

This is the session-specific progression model.

It includes:

stage of the case,
ordered tests,
requested clues,
visible fact IDs,
player dialogue history,
turn summaries,
hint count,
score state,
guesses and hypotheses,
and session status.

This is what evolves turn by turn.

Policy layer

This contains the reveal rules and gameplay constraints.

Examples:

the diagnosis may not be named before final unlock,
certain hallmark clues must stay hidden until the player requests the relevant test,
certain patient statements should only become available if the player asks the right type of question,
and specific results can only be unlocked once conditions are satisfied.

This is what prevents the AI from leaking the case.

This layered design is the single most important structural rule in the project.

AI workflow behavior
Start-of-run flow

When a run is created, the system should:

select a case according to game rules and metadata filters,
load the source-grounded structured truth case,
initialize run state,
choose the initial visible facts,
generate an opening presentation,
validate it for consistency,
validate it for spoiler leakage,
persist the result,
and return a structured payload to the Godot client.

The opening should feel natural, but it should only be built from facts that are legitimately visible at that stage.

Turn flow

On each player turn, the system should:

receive the turn request from the Godot client,
classify the action type,
determine which facts are currently visible and which new facts are allowed,
retrieve only the relevant structured case slice for this turn,
generate the next response,
validate it against hard rules,
run soft audits for consistency and spoilers,
update run state,
persist the turn and any revealed facts,
and return a structured turn response.

The model should never simply be handed the entire story and told to continue. It should receive a scoped, explicit view of:

the player request,
the current visible facts,
the allowed newly revealable facts,
the run summary,
and the output schema.

That keeps the system tighter and reduces drift.

Validation and guardrails
Hard validation

Hard validation must be implemented in code and treated as authoritative.

This includes rules such as:

output must match the response schema,
newly revealed facts must be in the allowed set,
diagnosis fields must stay hidden until allowed,
output must not contradict known facts,
test results must correspond to valid case facts or explicitly approved synthetic-safe filler,
and every critical reveal must link back to a source fact or approved derived fact.

If hard validation fails, the system should not expose the generated output directly. It should either:

regenerate with tighter constraints,
or fall back to a safe deterministic response.
Soft validation

Soft validation can be model-assisted and should score things like:

whether a response is too spoilery,
whether the wording over-implies the diagnosis,
whether the narrative is medically plausible,
whether the dialogue sounds consistent with the patient profile,
and whether specialist commentary is appropriately calibrated.

Soft validation informs quality and safety, but it does not replace hard rules.

Provenance awareness

Every medically meaningful fact should carry provenance metadata or at least a clear classification, such as:

source-grounded exact,
source-grounded paraphrase,
approved synthetic-safe connective detail,
or forbidden synthetic diagnostic fact.

The gameplay system should be able to audit where a clue came from.

Data ingestion and retrieval strategy
Source-first content pipeline

The content library should be built from real source cases that are de-identified and suitable for structured educational use.

The ingestion pipeline should:

fetch the source case,
parse and normalize it,
extract structured fields,
mark uncertainty where extraction confidence is low,
attach provenance metadata,
generate embeddings,
store the normalized case in the database,
and make it available to retrieval.

The extraction layer should support partial human review later, even if the first implementation is automated.

Retrieval strategy

At run start, the system should retrieve a case using:

specialty,
modality,
difficulty,
tags,
novelty and repetition avoidance,
and semantic relevance if needed.

Retrieval should be based on normalized case objects rather than raw PDFs or raw prose. The purpose of retrieval is not just to fetch text; it is to fetch a structured truth case that the rest of the engine can trust.

Godot client structure
Scene-first design

The Godot project should be scene-based.

Expected major scenes include:

main menu,
case selection or new-run setup,
the main case-play scene,
evidence and review panels,
settings,
and end-of-case summary.

The main case scene should be the primary play surface and contain the major UI panels.

Client responsibilities

The Godot client should:

request a new run from the backend,
submit player turn requests,
render the backend’s structured response blocks,
maintain local UI state,
display current evidence cleanly,
manage scene transitions,
and preserve local settings.

The client should not try to infer gameplay-critical truth on its own.

Structured rendering model

The backend should return structured display blocks rather than one giant text field.

Examples of response block types:

narrative,
patient dialogue,
attending comment,
lab result,
imaging report,
pathology report,
warning,
hint,
system status.

The Godot client should render these block types differently. That gives a much better experience than dumping all content into a chat transcript.

API philosophy

The client-backend contract should be stable, typed, and narrow.

The API should support:

creating runs,
retrieving run state,
submitting turn actions,
requesting hints,
submitting diagnoses,
and retrieving review/summary data.

Responses should include:

display blocks,
newly revealed facts,
updated run state fragments,
and validation-aware system messages where appropriate.

The API design should assume that the backend remains the source of truth for game state.

Scoring and educational feedback

The final product should score runs on:

final diagnosis correctness,
differential quality,
time or turn efficiency,
unnecessary testing,
use of hints,
and whether dangerous or important possibilities were considered.

After the case ends, the player should receive:

the correct diagnosis,
a case summary,
an explanation of the key findings,
their score and efficiency,
and educational takeaways.

The review system should also be source-grounded and consistent with the actual truth case.

Difficulty model

Difficulty should be implemented as changes to rules and presentation, not just wording.

Difficulty can affect:

how much is revealed in the opening,
how explicit symptoms are,
how much irrelevant noise is present,
how strict the hint system is,
how much prompting the attending gives,
and how hard the player is judged for unnecessary tests.

This allows the same truth case to produce different gameplay experiences without changing the underlying medicine.

Persistence and observability
Backend persistence

The backend should persist:

normalized cases,
run state,
turn history,
validation logs,
scoring summaries,
and debugging data.

This is needed both for the actual product and for development/debugging.

Client persistence

The Godot client may persist:

settings,
presentation preferences,
local cached summaries,
and non-authoritative UI state.
Debug and inspection support

The system should be built so developers can inspect:

which facts were visible at each turn,
which facts were eligible to be revealed,
what the model generated,
what validation accepted or rejected,
and how the run state changed.

This is extremely important because AI-heavy systems become painful to debug if they are not observable.

Engineering priorities

The final product should optimize for the following, in this order:

medical consistency
state correctness
spoiler control
clear architecture
playable user experience
content scalability
visual polish

The temptation will be to make the game feel clever and fluid by giving the LLM too much freedom. Resist that. The system should instead feel clever because the architecture is good enough to support believable flexibility without losing control.

Implementation rules

The coding agent should follow these rules throughout the build:

Keep source truth separate from presentation and run state.
Never store gameplay-critical truth only in prompt text.
Treat the backend as authoritative.
Require structured schemas for all important inputs and outputs.
Make AI generation nodes small and specific rather than relying on giant prompts.
Pass scoped structured state into the model, not the entire raw story unless necessary.
Enforce hard validation before anything reaches the player.
Support safe fallback behavior for invalid model outputs.
Preserve provenance for medically meaningful clues.
Make the system debuggable at every turn.
Final framing

This product should be implemented as a Godot-first, source-grounded, backend-authoritative diagnostic reasoning game. It uses real medical case data as the immutable truth layer, a Python backend for orchestration and persistence, Ollama for local inference and embeddings, LangGraph for stateful workflow execution, and strict schema validation to control what the model may say and when it may say it.

The point is not to build a chatbot that pretends to know medicine. The point is to build a game engine that can present medicine in a realistic, interactive, and controlled way.