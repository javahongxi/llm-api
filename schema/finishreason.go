package schema

// finishReasonMap normalizes provider-specific finish/stop reasons to OpenAI's
// finish_reason vocabulary ("stop", "length", "tool_calls", "content_filter").
// It mirrors litellm's _FINISH_REASON_MAP in litellm_core_utils/core_helpers.py.
var finishReasonMap = map[string]string{
	// Anthropic
	"stop_sequence":                 "stop",
	"end_turn":                      "stop",
	"max_tokens":                    "length",
	"tool_use":                      "tool_calls",
	"refusal":                       "content_filter",
	"compaction":                    "length",
	"pause_turn":                    "length",
	"model_context_window_exceeded": "length",
	// Gemini / Vertex AI
	"STOP":                      "stop",
	"MAX_TOKENS":                "length",
	"CONTINUATION":              "length",
	"SAFETY":                    "content_filter",
	"RECITATION":                "content_filter",
	"FINISH_REASON_UNSPECIFIED": "stop",
	"MALFORMED_FUNCTION_CALL":   "stop",
	"LANGUAGE":                  "content_filter",
	"OTHER":                     "content_filter",
	"BLOCKLIST":                 "content_filter",
	"PROHIBITED_CONTENT":        "content_filter",
	"SPII":                      "content_filter",
	"IMAGE_SAFETY":              "content_filter",
	"IMAGE_PROHIBITED_CONTENT":  "content_filter",
	"TOO_MANY_TOOL_CALLS":       "stop",
	"MALFORMED_RESPONSE":        "stop",
	// OpenAI / common
	"stop":           "stop",
	"length":         "length",
	"tool_calls":     "tool_calls",
	"function_call":  "function_call",
	"content_filter": "content_filter",
}

// MapFinishReason normalizes a provider finish reason. Unknown values are
// returned unchanged (litellm falls back to the original value).
func MapFinishReason(reason string) string {
	if mapped, ok := finishReasonMap[reason]; ok {
		return mapped
	}
	return reason
}
