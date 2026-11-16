package gemini

import (
	"testing"

	"github.com/javahongxi/llm-api/schema"
)

func TestChatRequestContentsAndSystem(t *testing.T) {
	p := New()
	maxTok := 100
	c := &schema.CallConfig{MaxTokens: &maxTok}
	messages := []schema.Message{
		schema.SystemMessage("be brief"),
		schema.UserMessage("hi"),
	}
	body, err := p.ChatRequest("gemini-1.5-pro", messages, c)
	if err != nil {
		t.Fatal(err)
	}
	m := body.(map[string]any)
	if _, ok := m["systemInstruction"]; !ok {
		t.Error("expected systemInstruction")
	}
	contents := m["contents"].([]map[string]any)
	if len(contents) != 1 || contents[0]["role"] != "user" {
		t.Fatalf("expected single user content, got %+v", contents)
	}
	gc := m["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"] != 100 {
		t.Errorf("maxOutputTokens not mapped: %v", gc)
	}
}

func TestChatEndpoint(t *testing.T) {
	p := New()
	c := &schema.CallConfig{APIBase: "https://generativelanguage.googleapis.com"}
	if got := p.ChatEndpoint(c, "gemini-1.5-pro", false); got != "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent" {
		t.Errorf("unexpected non-stream endpoint: %s", got)
	}
	if got := p.ChatEndpoint(c, "gemini-1.5-pro", true); got != "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse" {
		t.Errorf("unexpected stream endpoint: %s", got)
	}
}

func TestParseChatTextAndFunctionCall(t *testing.T) {
	p := New()
	body := []byte(`{
		"candidates":[{"content":{"parts":[{"text":"Hello"},{"functionCall":{"name":"get_weather","args":{"city":"BJ"}}}],"role":"model"},"finishReason":"STOP","index":0}],
		"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":7,"totalTokenCount":10}
	}`)
	resp, err := p.ParseChat("gemini-1.5-pro", body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("expected text Hello, got %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("functionCall should yield tool_calls finish, got %q", resp.Choices[0].FinishReason)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
	}
	if resp.Usage.TotalTokens != 10 {
		t.Errorf("usage not mapped: %+v", resp.Usage)
	}
}

func TestParseChatSafetyFinish(t *testing.T) {
	p := New()
	body := []byte(`{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"SAFETY","index":0}]}`)
	resp, _ := p.ParseChat("gemini-1.5-pro", body)
	if resp.Choices[0].FinishReason != "content_filter" {
		t.Errorf("SAFETY should map to content_filter, got %q", resp.Choices[0].FinishReason)
	}
}

func TestEmbeddingRequestAndParse(t *testing.T) {
	p := New()
	c := &schema.CallConfig{APIBase: "https://generativelanguage.googleapis.com"}
	body, err := p.EmbeddingRequest("text-embedding-004", []string{"a", "b"}, c)
	if err != nil {
		t.Fatal(err)
	}
	reqs := body.(map[string]any)["requests"].([]map[string]any)
	if len(reqs) != 2 || reqs[0]["model"] != "models/text-embedding-004" {
		t.Fatalf("unexpected embedding requests: %+v", reqs)
	}
	resp, err := p.ParseEmbedding("text-embedding-004", []byte(`{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 2 || resp.Data[1].Index != 1 || resp.Data[0].Embedding[0] != 0.1 {
		t.Fatalf("unexpected embedding response: %+v", resp)
	}
}

func TestStreamDecoder(t *testing.T) {
	p := New()
	d := p.NewStreamDecoder("gemini-1.5-pro")
	chunks1, err := d.Decode("", []byte(`{"candidates":[{"content":{"parts":[{"text":"Hel"}]},"index":0}]}`))
	if err != nil {
		t.Fatal(err)
	}
	chunks2, err := d.Decode("", []byte(`{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}`))
	if err != nil {
		t.Fatal(err)
	}
	if chunks1[0].Choices[0].Delta.Content != "Hel" || chunks2[0].Choices[0].Delta.Content != "lo" {
		t.Error("stream content not decoded")
	}
	if chunks2[0].Choices[0].FinishReason != "stop" {
		t.Errorf("expected stop finish, got %q", chunks2[0].Choices[0].FinishReason)
	}
	if !d.Done() {
		t.Error("decoder should be done after finishReason")
	}
}
