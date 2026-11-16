package openaicompat

import (
	"testing"

	"github.com/javahongxi/llm-api/schema"
)

func newTestProvider() *Provider {
	return &Provider{ProviderName: "openai", Base: "https://api.openai.com/v1", EnvKeys: []string{"OPENAI_API_KEY"}, Embeddings: true}
}

func TestChatRequestOnlySetParams(t *testing.T) {
	p := newTestProvider()
	temp := 0.7
	maxTok := 100
	c := &schema.CallConfig{Temperature: &temp, MaxTokens: &maxTok}
	body, err := p.ChatRequest("gpt-4o", []schema.Message{schema.UserMessage("hi")}, c)
	if err != nil {
		t.Fatal(err)
	}
	m := body.(map[string]any)
	if m["model"] != "gpt-4o" || m["temperature"] != 0.7 || m["max_tokens"] != 100 {
		t.Fatalf("params not mapped: %+v", m)
	}
	if _, ok := m["top_p"]; ok {
		t.Error("unset top_p should not be sent")
	}
	if _, ok := m["stream"]; ok {
		t.Error("stream should be absent when false")
	}
}

func TestChatRequestExtraBody(t *testing.T) {
	p := newTestProvider()
	c := &schema.CallConfig{ExtraBody: map[string]any{"custom_flag": true}}
	body, _ := p.ChatRequest("gpt-4o", []schema.Message{schema.UserMessage("hi")}, c)
	if body.(map[string]any)["custom_flag"] != true {
		t.Error("extra_body not merged")
	}
}

func TestChatEndpointJoin(t *testing.T) {
	p := newTestProvider()
	c := &schema.CallConfig{APIBase: "https://api.openai.com/v1"}
	if got := p.ChatEndpoint(c, "gpt-4o", false); got != "https://api.openai.com/v1/chat/completions" {
		t.Errorf("unexpected endpoint: %s", got)
	}
	// base already containing the route is not double-appended
	c2 := &schema.CallConfig{APIBase: "http://localhost:8080/v1/chat/completions"}
	if got := p.ChatEndpoint(c2, "gpt-4o", false); got != "http://localhost:8080/v1/chat/completions" {
		t.Errorf("endpoint should not double-append: %s", got)
	}
}

func TestParseChat(t *testing.T) {
	p := newTestProvider()
	body := []byte(`{
		"id":"chatcmpl-1","object":"chat.completion","created":123,"model":"gpt-4o",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
	}`)
	resp, err := p.ParseChat("gpt-4o", body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "hi there" || resp.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestStreamDecoderReasoningAlias(t *testing.T) {
	p := newTestProvider()
	d := p.NewStreamDecoder("gpt-4o")
	// "reasoning" alias should normalize to reasoning_content
	chunks, err := d.Decode("", []byte(`{"id":"1","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].Choices[0].Delta.ReasoningContent != "thinking..." {
		t.Errorf("reasoning alias not mapped: %+v", chunks[0].Choices[0].Delta)
	}
	// [DONE] terminates
	if _, err := d.Decode("", []byte("[DONE]")); err != nil {
		t.Fatal(err)
	}
	if !d.Done() {
		t.Error("decoder should be done after [DONE]")
	}
}

func TestStreamDecoderErrorPayload(t *testing.T) {
	p := newTestProvider()
	d := p.NewStreamDecoder("gpt-4o")
	_, err := d.Decode("", []byte(`{"error":{"message":"upstream failed","code":400}}`))
	if err == nil {
		t.Fatal("expected error from in-stream error payload")
	}
}

func TestEmbeddingRequestAndParse(t *testing.T) {
	p := newTestProvider()
	dim := 128
	c := &schema.CallConfig{Dimensions: &dim}
	body, _ := p.EmbeddingRequest("text-embedding-3-small", []string{"a"}, c)
	m := body.(map[string]any)
	if m["dimensions"] != 128 {
		t.Errorf("dimensions not mapped: %v", m)
	}
	resp, err := p.ParseEmbedding("text-embedding-3-small", []byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1,"total_tokens":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 1 || resp.Data[0].Embedding[1] != 0.2 {
		t.Fatalf("unexpected embedding response: %+v", resp)
	}
}
