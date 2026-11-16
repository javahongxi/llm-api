package schema

import "testing"

func TestMapHTTPErrorStatusCodes(t *testing.T) {
	cases := []struct {
		status int
		kind   ErrorKind
	}{
		{400, KindBadRequestError},
		{401, KindAuthenticationError},
		{403, KindPermissionDeniedError},
		{404, KindNotFoundError},
		{408, KindTimeoutError},
		{422, KindUnprocessableEntityError},
		{429, KindRateLimitError},
		{500, KindInternalServerError},
		{502, KindBadGatewayError},
		{503, KindServiceUnavailableError},
		{504, KindTimeoutError},
	}
	for _, tc := range cases {
		err := MapHTTPError(tc.status, []byte(`{"error":{"message":"boom"}}`), "openai", "gpt-4o")
		e, ok := AsError(err)
		if !ok {
			t.Fatalf("status %d: expected *Error, got %T", tc.status, err)
		}
		if e.Kind != tc.kind {
			t.Errorf("status %d: expected kind %s, got %s", tc.status, tc.kind, e.Kind)
		}
		if e.StatusCode != tc.status {
			t.Errorf("status %d: expected StatusCode %d, got %d", tc.status, tc.status, e.StatusCode)
		}
		if e.Message != "boom" {
			t.Errorf("status %d: expected message boom, got %q", tc.status, e.Message)
		}
	}
}

func TestMapHTTPErrorContextWindow(t *testing.T) {
	body := []byte(`{"error":{"message":"This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."}}`)
	err := MapHTTPError(400, body, "openai", "gpt-4o")
	if !IsContextWindowExceeded(err) {
		t.Fatalf("expected ContextWindowExceededError, got %v", err)
	}
}

func TestMapHTTPErrorContentPolicy(t *testing.T) {
	body := []byte(`{"error":{"message":"The response was filtered due to the prompt triggering content management policy."}}`)
	err := MapHTTPError(400, body, "openai", "gpt-4o")
	if !IsContentPolicyViolation(err) {
		t.Fatalf("expected ContentPolicyViolationError, got %v", err)
	}
}

func TestIsRetryable(t *testing.T) {
	retryable := []int{429, 500, 502, 503, 408}
	for _, status := range retryable {
		if !IsRetryable(MapHTTPError(status, nil, "openai", "m")) {
			t.Errorf("status %d should be retryable", status)
		}
	}
	if IsRetryable(MapHTTPError(401, nil, "openai", "m")) {
		t.Error("401 should not be retryable")
	}
}

func TestExtractErrorMessageShapes(t *testing.T) {
	shapes := []string{
		`{"error":{"message":"nested"}}`,
		`{"message":"top-level"}`,
		`{"error":"string-error"}`,
	}
	want := []string{"nested", "top-level", "string-error"}
	for i, body := range shapes {
		if got := extractErrorMessage([]byte(body)); got != want[i] {
			t.Errorf("shape %d: expected %q, got %q", i, want[i], got)
		}
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"end_turn":    "stop",
		"max_tokens":  "length",
		"tool_use":    "tool_calls",
		"STOP":        "stop",
		"MAX_TOKENS":  "length",
		"SAFETY":      "content_filter",
		"stop":        "stop",
		"unknown_xyz": "unknown_xyz",
	}
	for in, want := range cases {
		if got := MapFinishReason(in); got != want {
			t.Errorf("MapFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}
