// Package openaicompat implements the shared transformation for OpenAI and every
// OpenAI-compatible backend (DeepSeek, DashScope compatible-mode, ...). It is the
// Go analogue of litellm's OpenAIGPTConfig, which DeepSeekChatConfig and
// DashScopeChatConfig subclass. The request body is already OpenAI-shaped, so the
// transformation is mostly parameter selection plus response normalization.
package openaicompat

import (
	"encoding/json"
	"strings"

	"github.com/javahongxi/llm-api/schema"
)

// Provider is a configurable OpenAI-compatible backend. Construct one per
// provider by setting Name/Base/EnvKeys (see the openai, deepseek, dashscope
// packages).
type Provider struct {
	ProviderName string
	Base         string
	EnvKeys      []string
	// Embeddings enables the /embeddings endpoint for this backend.
	Embeddings bool
}

func (p *Provider) Name() string            { return p.ProviderName }
func (p *Provider) DefaultAPIBase() string  { return p.Base }
func (p *Provider) APIKeyEnvVars() []string { return p.EnvKeys }
func (p *Provider) SupportsEmbedding() bool { return p.Embeddings }

// ChatRequest builds the OpenAI-shaped request body, including only the
// parameters the caller set (litellm's non_default_params behaviour).
func (p *Provider) ChatRequest(model string, messages []schema.Message, c *schema.CallConfig) (any, error) {
	body := map[string]any{
		"model":    model,
		"messages": messages,
	}
	if c.Temperature != nil {
		body["temperature"] = *c.Temperature
	}
	if c.TopP != nil {
		body["top_p"] = *c.TopP
	}
	if c.MaxCompletionTokens != nil {
		body["max_completion_tokens"] = *c.MaxCompletionTokens
	} else if c.MaxTokens != nil {
		body["max_tokens"] = *c.MaxTokens
	}
	if c.N != nil {
		body["n"] = *c.N
	}
	if len(c.Stop) > 0 {
		body["stop"] = c.Stop
	}
	if c.PresencePenalty != nil {
		body["presence_penalty"] = *c.PresencePenalty
	}
	if c.FrequencyPenalty != nil {
		body["frequency_penalty"] = *c.FrequencyPenalty
	}
	if c.Seed != nil {
		body["seed"] = *c.Seed
	}
	if c.Logprobs != nil {
		body["logprobs"] = *c.Logprobs
	}
	if c.TopLogprobs != nil {
		body["top_logprobs"] = *c.TopLogprobs
	}
	if c.User != "" {
		body["user"] = c.User
	}
	if len(c.Tools) > 0 {
		body["tools"] = c.Tools
	}
	if c.ToolChoice != nil {
		body["tool_choice"] = c.ToolChoice
	}
	if c.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *c.ParallelToolCalls
	}
	if c.ResponseFormat != nil {
		body["response_format"] = c.ResponseFormat
	}
	if c.ReasoningEffort != "" {
		body["reasoning_effort"] = c.ReasoningEffort
	}
	if c.Stream {
		body["stream"] = true
	}
	for k, v := range c.ExtraBody {
		body[k] = v
	}
	return body, nil
}

// ChatEndpoint joins the resolved base URL with the chat/completions route.
func (p *Provider) ChatEndpoint(c *schema.CallConfig, model string, stream bool) string {
	return joinEndpoint(c.APIBase, "chat/completions")
}

// ChatHeaders returns bearer auth + JSON content type.
func (p *Provider) ChatHeaders(c *schema.CallConfig) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	if c.APIKey != "" {
		h["Authorization"] = "Bearer " + c.APIKey
	}
	return h
}

// ParseChat unmarshals the OpenAI-shaped response and normalizes finish reasons.
func (p *Provider) ParseChat(model string, body []byte) (*schema.ModelResponse, error) {
	var resp schema.ModelResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, schema.NewBadRequestError("unable to parse chat response: "+err.Error(), p.Name(), model)
	}
	if resp.Object == "" {
		resp.Object = "chat.completion"
	}
	if resp.Model == "" {
		resp.Model = model
	}
	for i := range resp.Choices {
		resp.Choices[i].FinishReason = schema.MapFinishReason(resp.Choices[i].FinishReason)
	}
	return &resp, nil
}

// wireChunk captures the streaming chunk, including the "reasoning" alias used
// by some OpenAI-compatible backends (Groq / vLLM) that litellm maps to
// reasoning_content.
type wireChunk struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           int64  `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Role             string               `json:"role"`
			Content          string               `json:"content"`
			Reasoning        string               `json:"reasoning"`
			ReasoningContent string               `json:"reasoning_content"`
			ToolCalls        []schema.ToolCall    `json:"tool_calls"`
			FunctionCall     *schema.FunctionCall `json:"function_call"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *schema.Usage `json:"usage"`
}

// NewStreamDecoder returns a decoder for OpenAI-style SSE streams.
func (p *Provider) NewStreamDecoder(model string) schema.StreamDecoder {
	return &streamDecoder{p: p, model: model}
}

type streamDecoder struct {
	p     *Provider
	model string
	done  bool
}

func (d *streamDecoder) Decode(eventType string, data []byte) ([]*schema.ModelResponseStream, error) {
	if strings.TrimSpace(string(data)) == "[DONE]" {
		d.done = true
		return nil, nil
	}
	chunk, err := d.p.parseChunk(d.model, data)
	if err != nil {
		return nil, err
	}
	return []*schema.ModelResponseStream{chunk}, nil
}

func (d *streamDecoder) Done() bool { return d.done }

// parseChunk converts one OpenAI-style SSE payload to a unified chunk.
func (p *Provider) parseChunk(model string, data []byte) (*schema.ModelResponseStream, error) {
	// Some backends embed an error payload inside an HTTP 200 stream.
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(data, &probe); err == nil {
		if errRaw, ok := probe["error"]; ok && len(errRaw) > 0 && string(errRaw) != "null" {
			return nil, schema.MapHTTPError(500, errRaw, p.Name(), model)
		}
	}

	var w wireChunk
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, schema.NewBadRequestError("unable to parse stream chunk: "+err.Error(), p.Name(), model)
	}
	chunk := &schema.ModelResponseStream{
		ID:                w.ID,
		Object:            "chat.completion.chunk",
		Created:           w.Created,
		Model:             w.Model,
		SystemFingerprint: w.SystemFingerprint,
		Usage:             w.Usage,
	}
	if chunk.Model == "" {
		chunk.Model = model
	}
	for _, wc := range w.Choices {
		reasoning := wc.Delta.ReasoningContent
		if reasoning == "" {
			reasoning = wc.Delta.Reasoning
		}
		chunk.Choices = append(chunk.Choices, schema.StreamChoice{
			Index:        wc.Index,
			FinishReason: schema.MapFinishReason(wc.FinishReason),
			Delta: schema.Delta{
				Role:             wc.Delta.Role,
				Content:          wc.Delta.Content,
				ReasoningContent: reasoning,
				ToolCalls:        wc.Delta.ToolCalls,
				FunctionCall:     wc.Delta.FunctionCall,
			},
		})
	}
	return chunk, nil
}

// EmbeddingRequest builds the OpenAI-shaped /embeddings body.
func (p *Provider) EmbeddingRequest(model string, input []string, c *schema.CallConfig) (any, error) {
	body := map[string]any{
		"model": model,
		"input": input,
	}
	if c.Dimensions != nil {
		body["dimensions"] = *c.Dimensions
	}
	if c.EncodingFormat != "" {
		body["encoding_format"] = c.EncodingFormat
	}
	if c.User != "" {
		body["user"] = c.User
	}
	for k, v := range c.ExtraBody {
		body[k] = v
	}
	return body, nil
}

// EmbeddingEndpoint joins the resolved base URL with the embeddings route.
func (p *Provider) EmbeddingEndpoint(c *schema.CallConfig, model string) string {
	return joinEndpoint(c.APIBase, "embeddings")
}

// ParseEmbedding unmarshals the OpenAI-shaped embedding response.
func (p *Provider) ParseEmbedding(model string, body []byte) (*schema.EmbeddingResponse, error) {
	var resp schema.EmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, schema.NewBadRequestError("unable to parse embedding response: "+err.Error(), p.Name(), model)
	}
	if resp.Object == "" {
		resp.Object = "list"
	}
	if resp.Model == "" {
		resp.Model = model
	}
	for i := range resp.Data {
		if resp.Data[i].Object == "" {
			resp.Data[i].Object = "embedding"
		}
	}
	return &resp, nil
}

// MapError maps an HTTP error to the unified error type.
func (p *Provider) MapError(status int, body []byte, model string) error {
	return schema.MapHTTPError(status, body, p.Name(), model)
}

// joinEndpoint appends route to base unless base already contains it.
func joinEndpoint(base, route string) string {
	base = strings.TrimRight(base, "/")
	if strings.Contains(base, route) {
		return base
	}
	return base + "/" + route
}
