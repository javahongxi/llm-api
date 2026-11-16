// Package anthropic implements the Anthropic Messages API backend
// (api.anthropic.com/v1/messages). It mirrors litellm's AnthropicConfig:
// system messages are hoisted to the top-level "system" field, OpenAI tools map
// to Anthropic input_schema tools, tool results become user-role tool_result
// blocks, and stop_reason maps to OpenAI finish_reason.
package anthropic

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

const defaultMaxTokens = 4096

// Provider is the Anthropic backend.
type Provider struct{}

// New returns the Anthropic provider.
func New() *Provider { return &Provider{} }

func (p *Provider) Name() string            { return "anthropic" }
func (p *Provider) DefaultAPIBase() string  { return "https://api.anthropic.com" }
func (p *Provider) APIKeyEnvVars() []string { return []string{"ANTHROPIC_API_KEY"} }
func (p *Provider) SupportsEmbedding() bool { return false }

// ChatEndpoint returns the /v1/messages URL.
func (p *Provider) ChatEndpoint(c *schema.CallConfig, model string, stream bool) string {
	base := strings.TrimRight(c.APIBase, "/")
	if strings.HasSuffix(base, "/v1/messages") {
		return base
	}
	if strings.HasSuffix(base, "/v1") {
		return base + "/messages"
	}
	return base + "/v1/messages"
}

// ChatHeaders returns Anthropic auth headers.
func (p *Provider) ChatHeaders(c *schema.CallConfig) map[string]string {
	h := map[string]string{
		"Content-Type":      "application/json",
		"anthropic-version": "2023-06-01",
	}
	if c.APIKey != "" {
		h["x-api-key"] = c.APIKey
	}
	return h
}

// ChatRequest builds the Anthropic Messages request body.
func (p *Provider) ChatRequest(model string, messages []schema.Message, c *schema.CallConfig) (any, error) {
	system, msgs := convertMessages(messages)

	maxTokens := c.MaxOutputTokens()
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	body := map[string]any{
		"model":      model,
		"messages":   msgs,
		"max_tokens": maxTokens,
	}
	if system != "" {
		body["system"] = system
	}
	if c.Temperature != nil {
		body["temperature"] = *c.Temperature
	}
	if c.TopP != nil {
		body["top_p"] = *c.TopP
	}
	if c.TopK != nil {
		body["top_k"] = *c.TopK
	}
	if len(c.Stop) > 0 {
		body["stop_sequences"] = c.Stop
	}
	if c.Stream {
		body["stream"] = true
	}

	// tool_choice "none" disables tools entirely (Anthropic has no "none" mode).
	disableTools := c.ToolChoice == "none"
	if len(c.Tools) > 0 && !disableTools {
		body["tools"] = convertTools(c.Tools)
		if tc := convertToolChoice(c.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}

	for k, v := range c.ExtraBody {
		body[k] = v
	}
	return body, nil
}

// convertMessages hoists system messages and converts the rest to Anthropic
// content blocks, merging consecutive same-role messages (Anthropic requires
// alternating roles and tool results live in user-role messages).
func convertMessages(messages []schema.Message) (string, []map[string]any) {
	var systemParts []string
	out := make([]map[string]any, 0, len(messages))

	for _, m := range messages {
		switch m.Role {
		case "system":
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}
		case "tool":
			block := map[string]any{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     m.Content,
			}
			appendBlocks(&out, "user", []any{block})
		case "assistant":
			var blocks []any
			if m.Content != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": m.Content})
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": parseArgs(tc.Function.Arguments),
				})
			}
			if len(blocks) > 0 {
				appendBlocks(&out, "assistant", blocks)
			}
		default: // user
			if m.Content != "" {
				appendBlocks(&out, "user", []any{map[string]any{"type": "text", "text": m.Content}})
			}
		}
	}
	return strings.Join(systemParts, "\n\n"), out
}

func appendBlocks(out *[]map[string]any, role string, blocks []any) {
	if n := len(*out); n > 0 {
		last := (*out)[n-1]
		if last["role"] == role {
			last["content"] = append(last["content"].([]any), blocks...)
			return
		}
	}
	*out = append(*out, map[string]any{"role": role, "content": blocks})
}

func parseArgs(arguments string) any {
	if strings.TrimSpace(arguments) == "" {
		return map[string]any{}
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(arguments), &obj); err != nil {
		return map[string]any{}
	}
	return obj
}

func convertTools(tools []schema.Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		tool := map[string]any{"name": t.Function.Name}
		if t.Function.Description != "" {
			tool["description"] = t.Function.Description
		}
		if t.Function.Parameters != nil {
			tool["input_schema"] = t.Function.Parameters
		} else {
			tool["input_schema"] = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, tool)
	}
	return out
}

func convertToolChoice(tc any) any {
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto":
			return map[string]any{"type": "auto"}
		case "required", "any":
			return map[string]any{"type": "any"}
		}
	case map[string]any:
		if v["type"] == "function" {
			if fn, ok := v["function"].(map[string]any); ok {
				return map[string]any{"type": "tool", "name": fn["name"]}
			}
		}
		if _, ok := v["type"]; ok {
			return v // already Anthropic-shaped
		}
	}
	return nil
}

// anthropicResponse is the wire shape of a non-streaming Messages response.
type anthropicResponse struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ID       string `json:"id"`
		Name     string `json:"name"`
		Input    any    `json:"input"`
		Thinking string `json:"thinking"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

// ParseChat converts an Anthropic Messages response to the unified shape.
func (p *Provider) ParseChat(model string, body []byte) (*schema.ModelResponse, error) {
	var r anthropicResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, schema.NewBadRequestError("unable to parse anthropic response: "+err.Error(), p.Name(), model)
	}

	var text, reasoning strings.Builder
	var toolCalls []schema.ToolCall
	for _, block := range r.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "thinking":
			reasoning.WriteString(block.Thinking)
		case "tool_use":
			args, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, schema.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: schema.FunctionCall{
					Name:      block.Name,
					Arguments: string(args),
				},
			})
		}
	}

	finishReason := schema.MapFinishReason(r.StopReason)
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	msg := schema.Message{Role: "assistant", Content: text.String(), ToolCalls: toolCalls}
	if reasoning.Len() > 0 {
		msg.ReasoningContent = reasoning.String()
	}

	resp := &schema.ModelResponse{
		ID:      r.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   r.Model,
		Choices: []schema.Choice{{Index: 0, Message: msg, FinishReason: finishReason}},
		Usage: &schema.Usage{
			PromptTokens:     r.Usage.InputTokens,
			CompletionTokens: r.Usage.OutputTokens,
			TotalTokens:      r.Usage.InputTokens + r.Usage.OutputTokens,
		},
	}
	if resp.Model == "" {
		resp.Model = model
	}
	if r.Usage.CacheReadInputTokens > 0 {
		resp.Usage.PromptTokensDetails = &schema.PromptTokensDetails{CachedTokens: r.Usage.CacheReadInputTokens}
	}
	return resp, nil
}

// EmbeddingRequest is unsupported by Anthropic.
func (p *Provider) EmbeddingRequest(model string, input []string, c *schema.CallConfig) (any, error) {
	return nil, schema.NewBadRequestError("anthropic does not support embeddings", p.Name(), model)
}

// EmbeddingEndpoint is unsupported by Anthropic.
func (p *Provider) EmbeddingEndpoint(c *schema.CallConfig, model string) string { return "" }

// ParseEmbedding is unsupported by Anthropic.
func (p *Provider) ParseEmbedding(model string, body []byte) (*schema.EmbeddingResponse, error) {
	return nil, schema.NewBadRequestError("anthropic does not support embeddings", p.Name(), model)
}

// MapError maps an HTTP error to the unified error type.
func (p *Provider) MapError(status int, body []byte, model string) error {
	return schema.MapHTTPError(status, body, p.Name(), model)
}
