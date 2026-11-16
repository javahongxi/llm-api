package llmapi

import (
	"context"

	"github.com/javahongxi/llm-api/schema"
)

// Unified type aliases so callers can use llmapi.Message, llmapi.ModelResponse,
// etc. without importing the schema package directly.
type (
	Message             = schema.Message
	FunctionCall        = schema.FunctionCall
	ToolCall            = schema.ToolCall
	Tool                = schema.Tool
	FunctionDefinition  = schema.FunctionDefinition
	Choice              = schema.Choice
	ModelResponse       = schema.ModelResponse
	Delta               = schema.Delta
	StreamChoice        = schema.StreamChoice
	ModelResponseStream = schema.ModelResponseStream
	Usage               = schema.Usage
	EmbeddingData       = schema.EmbeddingData
	EmbeddingResponse   = schema.EmbeddingResponse
	ModelInfo           = schema.ModelInfo
	ModelList           = schema.ModelList
	Error               = schema.Error
	CallConfig          = schema.CallConfig
	Provider            = schema.Provider
)

// Message constructors (re-exported from schema for convenience).
func SystemMessage(content string) Message    { return schema.SystemMessage(content) }
func UserMessage(content string) Message      { return schema.UserMessage(content) }
func AssistantMessage(content string) Message { return schema.AssistantMessage(content) }
func ToolMessage(toolCallID, content string) Message {
	return schema.ToolMessage(toolCallID, content)
}

// NewFunctionTool builds a function tool definition.
func NewFunctionTool(name, description string, parameters any) Tool {
	return schema.NewFunctionTool(name, description, parameters)
}

// Completion performs a non-streaming chat completion against any supported
// provider, returning the unified OpenAI-shaped response. It is the Go analogue
// of litellm.completion(model, messages, ...).
func Completion(ctx context.Context, model string, messages []Message, opts ...Option) (*ModelResponse, error) {
	c := buildConfig(opts)
	p, resolvedModel, err := resolveProvider(model, c.Provider)
	if err != nil {
		return nil, err
	}
	resolveDefaults(p, c)

	body, err := p.ChatRequest(resolvedModel, messages, c)
	if err != nil {
		return nil, err
	}
	endpoint := p.ChatEndpoint(c, resolvedModel, false)
	respBody, err := doRequest(ctx, c, p, resolvedModel, endpoint, p.ChatHeaders(c), body)
	if err != nil {
		return nil, err
	}
	return p.ParseChat(resolvedModel, respBody)
}

// CompletionStream performs a streaming chat completion, returning a Stream of
// unified chunks. It is the Go analogue of litellm.completion(..., stream=True).
func CompletionStream(ctx context.Context, model string, messages []Message, opts ...Option) (*Stream, error) {
	c := buildConfig(opts)
	c.Stream = true
	p, resolvedModel, err := resolveProvider(model, c.Provider)
	if err != nil {
		return nil, err
	}
	resolveDefaults(p, c)

	body, err := p.ChatRequest(resolvedModel, messages, c)
	if err != nil {
		return nil, err
	}
	endpoint := p.ChatEndpoint(c, resolvedModel, true)
	resp, err := doStreamRequest(ctx, c, p, resolvedModel, endpoint, p.ChatHeaders(c), body)
	if err != nil {
		return nil, err
	}
	return newStream(resp, p.NewStreamDecoder(resolvedModel)), nil
}

// Embedding creates embeddings for the input (a string or []string), returning
// the unified OpenAI-shaped response. It is the Go analogue of
// litellm.embedding(model, input, ...).
func Embedding(ctx context.Context, model string, input any, opts ...Option) (*EmbeddingResponse, error) {
	inputs, err := normalizeEmbeddingInput(input)
	if err != nil {
		return nil, err
	}
	c := buildConfig(opts)
	p, resolvedModel, err := resolveProvider(model, c.Provider)
	if err != nil {
		return nil, err
	}
	if !p.SupportsEmbedding() {
		return nil, schema.NewBadRequestError("provider "+p.Name()+" does not support embeddings", p.Name(), resolvedModel)
	}
	resolveDefaults(p, c)

	body, err := p.EmbeddingRequest(resolvedModel, inputs, c)
	if err != nil {
		return nil, err
	}
	endpoint := p.EmbeddingEndpoint(c, resolvedModel)
	respBody, err := doRequest(ctx, c, p, resolvedModel, endpoint, p.ChatHeaders(c), body)
	if err != nil {
		return nil, err
	}
	return p.ParseEmbedding(resolvedModel, respBody)
}

func normalizeEmbeddingInput(input any) ([]string, error) {
	switch v := input.(type) {
	case string:
		return []string{v}, nil
	case []string:
		return v, nil
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, schema.NewBadRequestError("embedding input elements must be strings", "", "")
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, schema.NewBadRequestError("embedding input must be a string or []string", "", "")
	}
}
