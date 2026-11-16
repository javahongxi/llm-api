// Package schema defines the unified, OpenAI-shaped data types shared across
// every provider in llm-api. It mirrors litellm's types/utils.py (ModelResponse,
// Message, Delta, Usage, ...) so that callers see one consistent response shape
// regardless of the backend (OpenAI, Anthropic, Gemini, DeepSeek, DashScope).
//
// This package has no dependency on any provider or on the root llmapi package,
// which keeps the import graph acyclic: providers import schema, and the root
// package imports both.
package schema

// Message is a single chat message. It is used both for requests (role + content
// supplied by the caller) and for responses (the assistant message returned by a
// provider). Content is a plain string in the core text/chat scope; multimodal
// content parts can be added later without breaking this shape.
type Message struct {
	// Role is one of "system", "user", "assistant", "tool".
	Role string `json:"role"`
	// Content is the textual content of the message.
	Content string `json:"content,omitempty"`
	// Name optionally identifies the author of the message (OpenAI name field).
	Name string `json:"name,omitempty"`
	// ToolCalls is set on assistant messages that invoke tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a "tool" role message back to the assistant tool call it
	// answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// FunctionCall is the legacy OpenAI function-calling field.
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
	// ReasoningContent carries provider thinking/reasoning text (e.g. DeepSeek
	// reasoner, Anthropic extended thinking) normalized to a single field.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// SystemMessage builds a system message.
func SystemMessage(content string) Message { return Message{Role: "system", Content: content} }

// UserMessage builds a user message.
func UserMessage(content string) Message { return Message{Role: "user", Content: content} }

// AssistantMessage builds an assistant message.
func AssistantMessage(content string) Message { return Message{Role: "assistant", Content: content} }

// ToolMessage builds a tool-result message answering the given tool call id.
func ToolMessage(toolCallID, content string) Message {
	return Message{Role: "tool", ToolCallID: toolCallID, Content: content}
}

// FunctionCall is a function name plus its JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// ToolCall is a single tool invocation. Index is only meaningful inside streaming
// deltas, where it identifies which tool call a fragment belongs to.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"` // "function"
	Function FunctionCall `json:"function"`
}

// Tool is a request-side tool definition (OpenAI chat completion tool param).
type Tool struct {
	Type     string             `json:"type"` // "function"
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition describes a callable function exposed to the model.
type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object describing the function arguments.
	Parameters any   `json:"parameters,omitempty"`
	Strict     *bool `json:"strict,omitempty"`
}

// NewFunctionTool is a convenience constructor for a function tool.
func NewFunctionTool(name, description string, parameters any) Tool {
	return Tool{Type: "function", Function: FunctionDefinition{
		Name: name, Description: description, Parameters: parameters,
	}}
}

// Choice is a single non-streaming completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
	Logprobs     any     `json:"logprobs,omitempty"`
}

// ModelResponse is the unified non-streaming chat completion response. Its JSON
// shape matches OpenAI's /chat/completions response, like litellm.ModelResponse.
type ModelResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"` // "chat.completion"
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	// Cost is the computed USD cost of the call (0 when the model price is
	// unknown), populated by the SDK like litellm's response_cost.
	Cost float64 `json:"cost,omitempty"`
}

// Delta is the incremental content of a streaming choice.
type Delta struct {
	Role             string        `json:"role,omitempty"`
	Content          string        `json:"content,omitempty"`
	ToolCalls        []ToolCall    `json:"tool_calls,omitempty"`
	FunctionCall     *FunctionCall `json:"function_call,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
}

// StreamChoice is a single streaming completion choice.
type StreamChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
	Logprobs     any    `json:"logprobs,omitempty"`
}

// ModelResponseStream is one unified streaming chunk. Its JSON shape matches
// OpenAI's chat.completion.chunk, like litellm.ModelResponseStream.
type ModelResponseStream struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"` // "chat.completion.chunk"
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
}

// PromptTokensDetails breaks down prompt token usage.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

// CompletionTokensDetails breaks down completion token usage.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
	TextTokens      int `json:"text_tokens,omitempty"`
}

// Usage is the unified token usage object, matching OpenAI's completion usage.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// EmbeddingData is a single embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"` // "embedding"
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingUsage is the token usage for an embedding request.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingResponse is the unified embedding response, matching OpenAI's
// /embeddings response shape.
type EmbeddingResponse struct {
	Object string          `json:"object"` // "list"
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  *EmbeddingUsage `json:"usage,omitempty"`
	// Cost is the computed USD cost of the call (0 when unknown).
	Cost float64 `json:"cost,omitempty"`
}

// ModelInfo describes a single model in a model listing.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the unified /models listing response.
type ModelList struct {
	Object string      `json:"object"` // "list"
	Data   []ModelInfo `json:"data"`
}
