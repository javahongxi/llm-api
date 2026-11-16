// Package gemini implements the Google AI Studio Gemini backend
// (generativelanguage.googleapis.com). It mirrors litellm's
// GoogleAIStudioGeminiConfig: OpenAI messages map to Gemini contents/parts,
// system messages become systemInstruction, tools become functionDeclarations,
// and finishReason maps to OpenAI finish_reason.
package gemini

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// Provider is the Gemini (Google AI Studio) backend.
type Provider struct{}

// New returns the Gemini provider.
func New() *Provider { return &Provider{} }

func (p *Provider) Name() string            { return "gemini" }
func (p *Provider) DefaultAPIBase() string  { return "https://generativelanguage.googleapis.com" }
func (p *Provider) APIKeyEnvVars() []string { return []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"} }
func (p *Provider) SupportsEmbedding() bool { return true }

// ChatEndpoint builds the generateContent / streamGenerateContent URL.
func (p *Provider) ChatEndpoint(c *schema.CallConfig, model string, stream bool) string {
	method := "generateContent"
	if stream {
		method = "streamGenerateContent"
	}
	url := p.modelBase(c, model) + ":" + method
	if stream {
		url += "?alt=sse"
	}
	return url
}

// modelBase returns "{base}/v1beta/models/{model}".
func (p *Provider) modelBase(c *schema.CallConfig, model string) string {
	base := strings.TrimRight(c.APIBase, "/")
	if !strings.HasSuffix(base, "/v1beta") && !strings.HasSuffix(base, "/v1") {
		base += "/v1beta"
	}
	model = strings.TrimPrefix(model, "models/")
	return base + "/models/" + model
}

// ChatHeaders returns Gemini auth headers.
func (p *Provider) ChatHeaders(c *schema.CallConfig) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	if c.APIKey != "" {
		h["x-goog-api-key"] = c.APIKey
	}
	return h
}

// ChatRequest builds the Gemini generateContent request body.
func (p *Provider) ChatRequest(model string, messages []schema.Message, c *schema.CallConfig) (any, error) {
	system, contents := convertMessages(messages)
	body := map[string]any{"contents": contents}
	if system != "" {
		body["systemInstruction"] = map[string]any{
			"parts": []any{map[string]any{"text": system}},
		}
	}

	genConfig := map[string]any{}
	if c.Temperature != nil {
		genConfig["temperature"] = *c.Temperature
	}
	if maxTok := c.MaxOutputTokens(); maxTok > 0 {
		genConfig["maxOutputTokens"] = maxTok
	}
	if c.TopP != nil {
		genConfig["topP"] = *c.TopP
	}
	if c.TopK != nil {
		genConfig["topK"] = *c.TopK
	}
	if len(c.Stop) > 0 {
		genConfig["stopSequences"] = c.Stop
	}
	if c.N != nil {
		genConfig["candidateCount"] = *c.N
	}
	if c.ResponseFormat != nil {
		if rf, ok := c.ResponseFormat.(map[string]any); ok && rf["type"] == "json_object" {
			genConfig["responseMimeType"] = "application/json"
		}
	}
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	if len(c.Tools) > 0 {
		body["tools"] = []any{map[string]any{"functionDeclarations": convertTools(c.Tools)}}
	}
	if tc := convertToolChoice(c.ToolChoice); tc != nil {
		body["toolConfig"] = map[string]any{"functionCallingConfig": tc}
	}

	for k, v := range c.ExtraBody {
		body[k] = v
	}
	return body, nil
}

// convertMessages hoists system messages and converts the rest to Gemini
// contents/parts, merging consecutive same-role contents. Tool results become
// functionResponse parts (role user); the function name is recovered from the
// preceding assistant tool_calls via tool_call_id.
func convertMessages(messages []schema.Message) (string, []map[string]any) {
	var systemParts []string
	contents := make([]map[string]any, 0, len(messages))

	toolNames := map[string]string{}
	for _, m := range messages {
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				toolNames[tc.ID] = tc.Function.Name
			}
		}
	}

	for _, m := range messages {
		switch m.Role {
		case "system":
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}
		case "tool":
			name := toolNames[m.ToolCallID]
			if name == "" {
				name = m.ToolCallID
			}
			part := map[string]any{"functionResponse": map[string]any{
				"name":     name,
				"response": parseResponse(m.Content),
			}}
			appendParts(&contents, "user", []any{part})
		case "assistant":
			var parts []any
			if m.Content != "" {
				parts = append(parts, map[string]any{"text": m.Content})
			}
			for _, tc := range m.ToolCalls {
				parts = append(parts, map[string]any{"functionCall": map[string]any{
					"name": tc.Function.Name,
					"args": parseArgs(tc.Function.Arguments),
				}})
			}
			if len(parts) > 0 {
				appendParts(&contents, "model", parts)
			}
		default: // user
			if m.Content != "" {
				appendParts(&contents, "user", []any{map[string]any{"text": m.Content}})
			}
		}
	}
	return strings.Join(systemParts, "\n\n"), contents
}

func appendParts(out *[]map[string]any, role string, parts []any) {
	if n := len(*out); n > 0 {
		last := (*out)[n-1]
		if last["role"] == role {
			last["parts"] = append(last["parts"].([]any), parts...)
			return
		}
	}
	*out = append(*out, map[string]any{"role": role, "parts": parts})
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

func parseResponse(content string) any {
	var obj map[string]any
	if err := json.Unmarshal([]byte(content), &obj); err == nil {
		return obj
	}
	return map[string]any{"content": content}
}

func convertTools(tools []schema.Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		fn := map[string]any{"name": t.Function.Name}
		if t.Function.Description != "" {
			fn["description"] = t.Function.Description
		}
		if t.Function.Parameters != nil {
			fn["parameters"] = t.Function.Parameters
		}
		out = append(out, fn)
	}
	return out
}

func convertToolChoice(tc any) map[string]any {
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto":
			return map[string]any{"mode": "AUTO"}
		case "required", "any":
			return map[string]any{"mode": "ANY"}
		case "none":
			return map[string]any{"mode": "NONE"}
		}
	case map[string]any:
		if v["type"] == "function" {
			if fn, ok := v["function"].(map[string]any); ok {
				return map[string]any{"mode": "ANY", "allowedFunctionNames": []any{fn["name"]}}
			}
		}
	}
	return nil
}

// geminiResponse is the wire shape of a generateContent response (also used for
// each streaming chunk).
type geminiResponse struct {
	Candidates    []candidate `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type candidate struct {
	Content struct {
		Parts []part `json:"parts"`
		Role  string `json:"role"`
	} `json:"content"`
	FinishReason string `json:"finishReason"`
	Index        int    `json:"index"`
}

type part struct {
	Text         string        `json:"text"`
	Thought      bool          `json:"thought"`
	FunctionCall *functionCall `json:"functionCall"`
}

type functionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

// ParseChat converts a Gemini generateContent response to the unified shape.
func (p *Provider) ParseChat(model string, body []byte) (*schema.ModelResponse, error) {
	var r geminiResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, schema.NewBadRequestError("unable to parse gemini response: "+err.Error(), p.Name(), model)
	}

	resp := &schema.ModelResponse{
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Usage: &schema.Usage{
			PromptTokens:     r.UsageMetadata.PromptTokenCount,
			CompletionTokens: r.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      r.UsageMetadata.TotalTokenCount,
		},
	}
	for _, cand := range r.Candidates {
		msg, finish := candidateToMessage(cand)
		resp.Choices = append(resp.Choices, schema.Choice{
			Index:        cand.Index,
			Message:      msg,
			FinishReason: finish,
		})
	}
	if len(resp.Choices) == 0 {
		resp.Choices = []schema.Choice{{Index: 0, Message: schema.Message{Role: "assistant"}, FinishReason: "stop"}}
	}
	return resp, nil
}

// candidateToMessage extracts text, tool calls and reasoning from a candidate.
func candidateToMessage(cand candidate) (schema.Message, string) {
	var text, reasoning strings.Builder
	var toolCalls []schema.ToolCall
	for i, part := range cand.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			args, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, schema.ToolCall{
				ID:   part.FunctionCall.Name + "-" + strconv.Itoa(i),
				Type: "function",
				Function: schema.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				},
			})
		case part.Thought:
			reasoning.WriteString(part.Text)
		default:
			text.WriteString(part.Text)
		}
	}
	finish := schema.MapFinishReason(cand.FinishReason)
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	}
	msg := schema.Message{Role: "assistant", Content: text.String(), ToolCalls: toolCalls}
	if reasoning.Len() > 0 {
		msg.ReasoningContent = reasoning.String()
	}
	return msg, finish
}

// MapError maps an HTTP error to the unified error type.
func (p *Provider) MapError(status int, body []byte, model string) error {
	return schema.MapHTTPError(status, body, p.Name(), model)
}
