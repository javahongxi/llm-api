package schema

import (
	"net/http"
	"time"
)

// CallConfig holds every parameter for a single completion/embedding call. It is
// the Go analogue of litellm's completion(**kwargs): the OpenAI-shaped sampling
// and tool parameters plus call-level configuration (credentials, base URL,
// HTTP client, timeout). Providers read it to build their native request.
//
// Pointer fields distinguish "unset" from a zero value, so providers only send
// parameters the caller actually supplied (litellm's non_default_params idea).
type CallConfig struct {
	// --- call-level configuration ---

	// APIKey overrides the provider credential. When empty, the root package
	// resolves it from the provider's environment variables.
	APIKey string
	// APIBase overrides the provider endpoint base URL.
	APIBase string
	// Provider explicitly selects the backend (litellm custom_llm_provider).
	Provider string
	// HTTPClient is the client used for the request. Defaults to a shared client.
	HTTPClient *http.Client
	// Timeout bounds the whole call. Zero means no explicit timeout.
	Timeout time.Duration
	// ExtraHeaders are merged into the outgoing request headers.
	ExtraHeaders map[string]string
	// ExtraQuery are merged into the outgoing URL query parameters.
	ExtraQuery map[string]string

	// --- OpenAI-shaped chat parameters ---

	Temperature         *float64
	TopP                *float64
	TopK                *int
	MaxTokens           *int
	MaxCompletionTokens *int
	N                   *int
	Stop                []string
	Stream              bool
	Tools               []Tool
	ToolChoice          any
	ParallelToolCalls   *bool
	ResponseFormat      any
	ReasoningEffort     string
	PresencePenalty     *float64
	FrequencyPenalty    *float64
	Seed                *int
	Logprobs            *bool
	TopLogprobs         *int
	User                string
	// ExtraBody holds provider-specific passthrough fields merged into the
	// request body (litellm's extra_body / provider-specific kwargs).
	ExtraBody map[string]any

	// --- embedding parameters ---

	Dimensions     *int
	EncodingFormat string
}

// Clone returns a shallow copy of the config so providers can normalize fields
// (e.g. resolve a default max_tokens) without mutating the caller's config.
func (c *CallConfig) Clone() *CallConfig {
	if c == nil {
		return &CallConfig{}
	}
	cp := *c
	return &cp
}

// MaxOutputTokens returns the effective output token limit, preferring
// max_completion_tokens over max_tokens (OpenAI's newer parameter).
func (c *CallConfig) MaxOutputTokens() int {
	if c.MaxCompletionTokens != nil {
		return *c.MaxCompletionTokens
	}
	if c.MaxTokens != nil {
		return *c.MaxTokens
	}
	return 0
}
