extends Control

const API_BASE_URL := "http://127.0.0.1:8000"

var run_id := ""
var active_request := ""
var pending_payload := {}
var blocks_box: VBoxContainer
var evidence_list: ItemList
var action_select: OptionButton
var player_text: TextEdit
var target_text: LineEdit
var diagnosis_text: LineEdit
var rationale_text: TextEdit
var status_label: Label
var score_label: Label
var http: HTTPRequest

var actions := [
	["ask_patient_question", "Ask patient"],
	["request_exam_detail", "Exam detail"],
	["order_lab", "Order lab"],
	["order_imaging", "Order imaging"],
	["request_pathology_detail", "Pathology detail"],
	["submit_differential", "Submit differential"],
	["request_hint", "Request hint"]
]

func _ready() -> void:
	_build_ui()
	_start_case()

func _build_ui() -> void:
	var root := VBoxContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.add_theme_constant_override("separation", 8)
	add_child(root)

	var header := HBoxContainer.new()
	header.custom_minimum_size = Vector2(0, 72)
	root.add_child(header)

	var icon := TextureRect.new()
	icon.texture = load("res://assets/diagnostic_monitor.svg")
	icon.custom_minimum_size = Vector2(52, 52)
	icon.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	header.add_child(icon)

	var title := Label.new()
	title.text = "Diagnostician Workstation"
	title.add_theme_font_size_override("font_size", 26)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(title)

	status_label = Label.new()
	status_label.text = "Connecting..."
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	status_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(status_label)

	var body := HSplitContainer.new()
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(body)

	var left := VBoxContainer.new()
	left.custom_minimum_size = Vector2(460, 0)
	left.add_theme_constant_override("separation", 8)
	body.add_child(left)

	var narrative_label := Label.new()
	narrative_label.text = "Case stream"
	narrative_label.add_theme_font_size_override("font_size", 18)
	left.add_child(narrative_label)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(scroll)

	blocks_box = VBoxContainer.new()
	blocks_box.add_theme_constant_override("separation", 8)
	scroll.add_child(blocks_box)

	var right := VBoxContainer.new()
	right.custom_minimum_size = Vector2(520, 0)
	right.add_theme_constant_override("separation", 8)
	body.add_child(right)

	var evidence_label := Label.new()
	evidence_label.text = "Visible evidence"
	evidence_label.add_theme_font_size_override("font_size", 18)
	right.add_child(evidence_label)

	evidence_list = ItemList.new()
	evidence_list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right.add_child(evidence_list)

	var controls := VBoxContainer.new()
	controls.add_theme_constant_override("separation", 6)
	right.add_child(controls)

	action_select = OptionButton.new()
	for action in actions:
		action_select.add_item(action[1])
	controls.add_child(action_select)

	target_text = LineEdit.new()
	target_text.placeholder_text = "Target test, modality, or topic"
	controls.add_child(target_text)

	player_text = TextEdit.new()
	player_text.placeholder_text = "Clinical question, differential, or request"
	player_text.custom_minimum_size = Vector2(0, 84)
	controls.add_child(player_text)

	var send_button := Button.new()
	send_button.text = "Submit action"
	send_button.pressed.connect(_submit_turn)
	controls.add_child(send_button)

	diagnosis_text = LineEdit.new()
	diagnosis_text.placeholder_text = "Final diagnosis"
	controls.add_child(diagnosis_text)

	rationale_text = TextEdit.new()
	rationale_text.placeholder_text = "Rationale"
	rationale_text.custom_minimum_size = Vector2(0, 70)
	controls.add_child(rationale_text)

	var diagnosis_button := Button.new()
	diagnosis_button.text = "Submit final diagnosis"
	diagnosis_button.pressed.connect(_submit_diagnosis)
	controls.add_child(diagnosis_button)

	score_label = Label.new()
	score_label.text = ""
	controls.add_child(score_label)

	http = HTTPRequest.new()
	http.request_completed.connect(_on_request_completed)
	add_child(http)

func _start_case() -> void:
	_request_json("start", HTTPClient.METHOD_POST, "/runs", {})

func _submit_turn() -> void:
	if run_id == "":
		_set_status("No active run.")
		return
	var index := action_select.get_selected_id()
	var payload := {
		"action_type": actions[index][0],
		"player_text": player_text.text,
		"target": target_text.text
	}
	_request_json("turn", HTTPClient.METHOD_POST, "/runs/%s/turns" % run_id, payload)

func _submit_diagnosis() -> void:
	if run_id == "":
		_set_status("No active run.")
		return
	var payload := {
		"diagnosis": diagnosis_text.text,
		"rationale": rationale_text.text
	}
	_request_json("diagnosis", HTTPClient.METHOD_POST, "/runs/%s/diagnosis" % run_id, payload)

func _request_json(kind: String, method: int, path: String, payload: Dictionary) -> void:
	active_request = kind
	pending_payload = payload
	_set_status("Sending...")
	var headers := ["Content-Type: application/json"]
	var body := JSON.stringify(payload)
	var err := http.request(API_BASE_URL + path, headers, method, body)
	if err != OK:
		_set_status("Request failed before sending.")

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS:
		_set_status("Backend request failed.")
		return
	var text := body.get_string_from_utf8()
	var parsed = JSON.parse_string(text)
	if response_code < 200 or response_code >= 300:
		_set_status("Backend error %s: %s" % [response_code, text])
		return
	if typeof(parsed) != TYPE_DICTIONARY:
		_set_status("Unexpected backend response.")
		return

	if active_request == "diagnosis":
		_render_review(parsed)
	else:
		_render_turn_response(parsed)
	_set_status("Ready")

func _render_turn_response(payload: Dictionary) -> void:
	if payload.has("run_state"):
		run_id = str(payload["run_state"].get("id", run_id))
	_render_blocks(payload.get("display_blocks", []))
	_render_evidence(payload.get("visible_evidence", {}).get("facts", []))
	player_text.text = ""
	target_text.text = ""

func _render_review(payload: Dictionary) -> void:
	var score := payload.get("player_score", {})
	score_label.text = "Score: %s / 100" % score.get("final_score", "?")
	_add_block("Final review", "Correct diagnosis: %s\n%s" % [payload.get("diagnosis", ""), _score_lines(score)], "success")
	_render_evidence(payload.get("key_findings", []))

func _render_blocks(blocks: Array) -> void:
	for block in blocks:
		_add_block(block.get("title", "Update"), block.get("body", ""), block.get("type", "system_status"))

func _add_block(title: String, body: String, kind: String) -> void:
	var panel := PanelContainer.new()
	panel.custom_minimum_size = Vector2(0, 96)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 4)
	panel.add_child(box)

	var title_label := Label.new()
	title_label.text = "%s  [%s]" % [title, kind]
	title_label.add_theme_font_size_override("font_size", 16)
	box.add_child(title_label)

	var body_label := Label.new()
	body_label.text = body
	body_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(body_label)
	blocks_box.add_child(panel)

func _render_evidence(facts: Array) -> void:
	evidence_list.clear()
	for fact in facts:
		var line := "%s: %s" % [fact.get("label", ""), fact.get("value", "")]
		evidence_list.add_item(line)

func _score_lines(score: Dictionary) -> String:
	var lines := []
	for item in score.get("rationale", []):
		lines.append(str(item))
	return "\n".join(lines)

func _set_status(message: String) -> void:
	status_label.text = message
