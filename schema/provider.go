package schema

// Provider transforms between the unified OpenAI-shaped API and one specific
// backend. It is the Go analogue of litellm's per-provider Config classes
// (transform_request / transform_response / get_complete_url /
// validate_environment / get_error_class).
//
// Implementations are pure transformers: they build request bodies, URLs and
// headers, and parse response payloads. The root llmapi package owns the HTTP
// execution and credential resolution, so providers never perform I/O. Before
// any method is called, the root package fills c.APIBase and c.APIKey (from
// DefaultAPIBase / APIKeyEnvVars when the caller did not supply them).
type Provider interface {
	// Name is the provider identifier used in model prefixes ("anthropic/...").
	Name() string
	// DefaultAPIBase is the canonical endpoint base used when none is supplied.
	DefaultAPIBase() string
	// APIKeyEnvVars lists environment variables checked for the credential, in
	// priority order (e.g. ["OPENAI_API_KEY"]).
	APIKeyEnvVars() []string

	// --- chat completion ---

	// ChatRequest builds the provider-native request body.
	ChatRequest(model string, messages []Message, c *CallConfig) (any, error)
	// ChatEndpoint returns the full request URL for chat (stream-aware).
	ChatEndpoint(c *CallConfig, model string, stream bool) string
	// ChatHeaders returns auth + content-type headers for chat.
	ChatHeaders(c *CallConfig) map[string]string
	// ParseChat converts a non-streaming response body to the unified shape.
	ParseChat(model string, body []byte) (*ModelResponse, error)
	// NewStreamDecoder creates a stateful decoder for a streaming response. The
	// decoder consumes SSE events and emits unified chunks (litellm's
	// per-provider ModelResponseIterator). State lets providers map Anthropic
	// content-block indices to tool-call indices, accumulate fragments, etc.
	NewStreamDecoder(model string) StreamDecoder

	// --- embeddings ---

	// SupportsEmbedding reports whether the provider implements embeddings.
	SupportsEmbedding() bool
	// EmbeddingRequest builds the provider-native embedding request body.
	EmbeddingRequest(model string, input []string, c *CallConfig) (any, error)
	// EmbeddingEndpoint returns the full embedding request URL.
	EmbeddingEndpoint(c *CallConfig, model string) string
	// ParseEmbedding converts an embedding response body to the unified shape.
	ParseEmbedding(model string, body []byte) (*EmbeddingResponse, error)

	// --- errors ---

	// MapError maps an HTTP error status + body to a unified *Error.
	MapError(status int, body []byte, model string) error
}

// StreamDecoder consumes SSE events from a streaming chat response and produces
// unified chunks. Decode may return zero chunks (bookkeeping events), one chunk
// (OpenAI/Gemini), or several. Done reports end-of-stream so the reader can stop
// even if the connection stays open.
type StreamDecoder interface {
	// Decode processes one SSE event. eventType is the SSE "event:" field (empty
	// for OpenAI-style streams); data is the joined "data:" payload.
	Decode(eventType string, data []byte) ([]*ModelResponseStream, error)
	// Done reports whether the stream has signalled completion.
	Done() bool
}
