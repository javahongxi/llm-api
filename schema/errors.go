package schema

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ErrorKind names a unified error class. The values mirror litellm's exception
// class names (litellm/exceptions.py) so behaviour maps one-to-one.
type ErrorKind string

const (
	KindAPIError                 ErrorKind = "APIError"
	KindAPIConnectionError       ErrorKind = "APIConnectionError"
	KindAuthenticationError      ErrorKind = "AuthenticationError"
	KindPermissionDeniedError    ErrorKind = "PermissionDeniedError"
	KindBadRequestError          ErrorKind = "BadRequestError"
	KindNotFoundError            ErrorKind = "NotFoundError"
	KindUnprocessableEntityError ErrorKind = "UnprocessableEntityError"
	KindTimeoutError             ErrorKind = "Timeout"
	KindRateLimitError           ErrorKind = "RateLimitError"
	KindInternalServerError      ErrorKind = "InternalServerError"
	KindBadGatewayError          ErrorKind = "BadGatewayError"
	KindServiceUnavailableError  ErrorKind = "ServiceUnavailableError"
	KindContextWindowExceeded    ErrorKind = "ContextWindowExceededError"
	KindContentPolicyViolation   ErrorKind = "ContentPolicyViolationError"
)

// Error is the unified error returned by every provider. It carries the same
// fields litellm attaches to its exceptions: status_code, message, llm_provider
// and model. Use errors.As to recover it and the Is* helpers to classify it.
type Error struct {
	Kind        ErrorKind
	StatusCode  int
	Message     string
	LLMProvider string
	Model       string
	// Body is the raw provider error payload, when available.
	Body string
}

func (e *Error) Error() string {
	provider := e.LLMProvider
	if provider == "" {
		provider = "llm-api"
	}
	return fmt.Sprintf("llm-api.%s: %s (provider=%s, model=%s, status=%d)",
		e.Kind, e.Message, provider, e.Model, e.StatusCode)
}

// newError builds a unified error.
func newError(kind ErrorKind, status int, message, provider, model, body string) *Error {
	return &Error{Kind: kind, StatusCode: status, Message: message, LLMProvider: provider, Model: model, Body: body}
}

// AsError recovers the unified *Error from an error chain, if present.
func AsError(err error) (*Error, bool) {
	var e *Error
	ok := errors.As(err, &e)
	return e, ok
}

func kindIs(err error, kind ErrorKind) bool {
	e, ok := AsError(err)
	return ok && e.Kind == kind
}

// IsAuthentication reports whether err is an authentication (401) error.
func IsAuthentication(err error) bool { return kindIs(err, KindAuthenticationError) }

// IsPermissionDenied reports whether err is a permission (403) error.
func IsPermissionDenied(err error) bool { return kindIs(err, KindPermissionDeniedError) }

// IsBadRequest reports whether err is a bad-request (400) error.
func IsBadRequest(err error) bool { return kindIs(err, KindBadRequestError) }

// IsNotFound reports whether err is a not-found (404) error.
func IsNotFound(err error) bool { return kindIs(err, KindNotFoundError) }

// IsRateLimit reports whether err is a rate-limit (429) error.
func IsRateLimit(err error) bool { return kindIs(err, KindRateLimitError) }

// IsTimeout reports whether err is a timeout error.
func IsTimeout(err error) bool { return kindIs(err, KindTimeoutError) }

// IsContextWindowExceeded reports whether err is a context-window-exceeded error.
func IsContextWindowExceeded(err error) bool { return kindIs(err, KindContextWindowExceeded) }

// IsContentPolicyViolation reports whether err is a content-policy violation.
func IsContentPolicyViolation(err error) bool { return kindIs(err, KindContentPolicyViolation) }

// IsServerError reports whether err is a 5xx server-side error.
func IsServerError(err error) bool {
	e, ok := AsError(err)
	return ok && e.StatusCode >= 500 && e.StatusCode < 600
}

// IsRetryable reports whether err is generally worth retrying: rate limits,
// timeouts, connection errors and 5xx responses (mirrors litellm's retry logic).
func IsRetryable(err error) bool {
	e, ok := AsError(err)
	if !ok {
		return false
	}
	switch e.Kind {
	case KindRateLimitError, KindTimeoutError, KindAPIConnectionError,
		KindInternalServerError, KindBadGatewayError, KindServiceUnavailableError:
		return true
	}
	return e.StatusCode >= 500 && e.StatusCode < 600
}

// MapHTTPError maps an HTTP status code and provider error body to a unified
// error, mirroring litellm's exception_mapping_utils. It inspects the message
// text for the special context-window / content-policy cases.
func MapHTTPError(status int, body []byte, provider, model string) error {
	msg := extractErrorMessage(body)
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	if msg == "" {
		msg = fmt.Sprintf("provider returned HTTP %d", status)
	}
	raw := string(body)
	lower := strings.ToLower(msg)

	// Special bad-request subtypes detected from the message text.
	if status == 400 || status == 413 {
		if isContextWindowMessage(lower) {
			return newError(KindContextWindowExceeded, 400, msg, provider, model, raw)
		}
		if isContentPolicyMessage(lower) {
			return newError(KindContentPolicyViolation, 400, msg, provider, model, raw)
		}
	}

	var kind ErrorKind
	switch status {
	case 400, 413:
		kind = KindBadRequestError
	case 401:
		kind = KindAuthenticationError
	case 403:
		kind = KindPermissionDeniedError
	case 404:
		kind = KindNotFoundError
	case 408:
		kind = KindTimeoutError
	case 422:
		kind = KindUnprocessableEntityError
	case 429:
		kind = KindRateLimitError
	case 500, 529:
		kind = KindInternalServerError
	case 502:
		kind = KindBadGatewayError
	case 503:
		kind = KindServiceUnavailableError
	case 504:
		kind = KindTimeoutError
	default:
		kind = KindAPIError
	}
	return newError(kind, status, msg, provider, model, raw)
}

// NewConnectionError builds an APIConnectionError (network-level failure).
func NewConnectionError(message, provider, model string) error {
	return newError(KindAPIConnectionError, 0, message, provider, model, "")
}

// NewTimeoutError builds a Timeout error.
func NewTimeoutError(message, provider, model string) error {
	return newError(KindTimeoutError, 408, message, provider, model, "")
}

// NewBadRequestError builds a BadRequestError (e.g. unsupported params).
func NewBadRequestError(message, provider, model string) error {
	return newError(KindBadRequestError, 400, message, provider, model, "")
}

func isContextWindowMessage(lower string) bool {
	patterns := []string{
		"context window", "context length", "maximum context", "max tokens",
		"too many tokens", "token limit", "reduce the length", "context_window",
		"maximum allowed length", "input length", "prompt is too long",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func isContentPolicyMessage(lower string) bool {
	patterns := []string{
		"content_policy", "content policy", "content management policy",
		"flagged for safety", "safety reasons", "responsible ai", "content filter",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractErrorMessage pulls a human-readable message out of the common provider
// error shapes: OpenAI/Anthropic {"error":{"message":...}}, {"message":...},
// Gemini {"error":{"message":...,"status":...}}.
func extractErrorMessage(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var generic map[string]any
	if err := json.Unmarshal(body, &generic); err != nil {
		return ""
	}
	if errObj, ok := generic["error"].(map[string]any); ok {
		if m, ok := errObj["message"].(string); ok && m != "" {
			return m
		}
	}
	if m, ok := generic["message"].(string); ok && m != "" {
		return m
	}
	if m, ok := generic["error"].(string); ok && m != "" {
		return m
	}
	return ""
}
