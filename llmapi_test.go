package llmapi

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/javahongxi/llm-api/schema"
)

func TestResolveProvider(t *testing.T) {
	cases := []struct {
		model, explicit string
		wantProvider    string
		wantModel       string
	}{
		{"gpt-4o", "", "openai", "gpt-4o"},
		{"claude-3-5-sonnet-latest", "", "anthropic", "claude-3-5-sonnet-latest"},
		{"gemini-1.5-pro", "", "gemini", "gemini-1.5-pro"},
		{"deepseek-chat", "", "deepseek", "deepseek-chat"},
		{"qwen-plus", "", "dashscope", "qwen-plus"},
		{"text-embedding-v3", "", "dashscope", "text-embedding-v3"},
		{"anthropic/claude-3-5-sonnet-latest", "", "anthropic", "claude-3-5-sonnet-latest"},
		{"gemini/gemini-1.5-pro", "", "gemini", "gemini-1.5-pro"},
		{"openai/gpt-4o", "", "openai", "gpt-4o"},
		{"claude-3-5-sonnet-latest", "openai", "openai", "claude-3-5-sonnet-latest"},
		{"openai/gpt-4o", "openai", "openai", "gpt-4o"},
	}
	for _, tc := range cases {
		p, model, err := resolveProvider(tc.model, tc.explicit)
		if err != nil {
			t.Fatalf("%s/%s: %v", tc.model, tc.explicit, err)
		}
		if p.Name() != tc.wantProvider || model != tc.wantModel {
			t.Errorf("resolveProvider(%q,%q) = (%s,%s), want (%s,%s)",
				tc.model, tc.explicit, p.Name(), model, tc.wantProvider, tc.wantModel)
		}
	}

	if _, _, err := resolveProvider("nope/model", ""); err == nil {
		t.Error("unknown prefix should error")
	}
	if _, _, err := resolveProvider("gpt-4o", "nope"); err == nil {
		t.Error("unknown explicit provider should error")
	}
}

func TestGetProviderRegistry(t *testing.T) {
	for _, name := range []string{"openai", "anthropic", "gemini", "deepseek", "dashscope"} {
		if _, ok := GetProvider(name); !ok {
			t.Errorf("provider %s not registered", name)
		}
	}
	if _, ok := GetProvider("missing"); ok {
		t.Error("missing provider should not be found")
	}
}

// openAIMock returns a test server speaking the OpenAI wire format and records
// the last request body/path/headers for assertions.
type capturedRequest struct {
	path   string
	body   map[string]any
	auth   string
	stream bool
}

func newOpenAIMock(t *testing.T, capture *capturedRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		if capture != nil {
			capture.path = r.URL.Path
			capture.body = body
			capture.auth = r.Header.Get("Authorization")
			if s, ok := body["stream"].(bool); ok {
				capture.stream = s
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/embeddings") {
			_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2,0.3],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":2,"total_tokens":2}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hello back"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`))
	}))
}

func TestCompletion(t *testing.T) {
	var cap capturedRequest
	srv := newOpenAIMock(t, &cap)
	defer srv.Close()

	resp, err := Completion(context.Background(), "gpt-4o",
		[]Message{UserMessage("hi")},
		WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"), WithTemperature(0.5))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "hello back" {
		t.Errorf("unexpected content: %q", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 5 {
		t.Errorf("usage not parsed: %+v", resp.Usage)
	}
	if cap.path != "/v1/chat/completions" {
		t.Errorf("unexpected path: %s", cap.path)
	}
	if cap.auth != "Bearer sk-test" {
		t.Errorf("unexpected auth header: %q", cap.auth)
	}
	if cap.body["temperature"] != 0.5 {
		t.Errorf("temperature not forwarded: %+v", cap.body)
	}
}

func TestCompletionErrorMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	}))
	defer srv.Close()

	_, err := Completion(context.Background(), "gpt-4o",
		[]Message{UserMessage("hi")}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("bad"))
	if err == nil {
		t.Fatal("expected error")
	}
	if !schema.IsAuthentication(err) {
		t.Errorf("expected authentication error, got %v", err)
	}
	e, _ := schema.AsError(err)
	if e.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", e.StatusCode)
	}
}

func TestCompletionStream(t *testing.T) {
	var cap capturedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		cap.body = body
		if s, ok := body["stream"].(bool); ok {
			cap.stream = s
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	stream, err := CompletionStream(context.Background(), "gpt-4o",
		[]Message{UserMessage("hi")}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var content string
	var finish string
	for stream.Next() {
		ch := stream.Current()
		if len(ch.Choices) > 0 {
			content += ch.Choices[0].Delta.Content
			if ch.Choices[0].FinishReason != "" {
				finish = ch.Choices[0].FinishReason
			}
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if content != "Hello" {
		t.Errorf("expected Hello, got %q", content)
	}
	if finish != "stop" {
		t.Errorf("expected stop finish, got %q", finish)
	}
	if !cap.stream {
		t.Error("stream flag should be set on request")
	}
}

func TestEmbedding(t *testing.T) {
	var cap capturedRequest
	srv := newOpenAIMock(t, &cap)
	defer srv.Close()

	resp, err := Embedding(context.Background(), "text-embedding-3-small",
		[]string{"a", "b"}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 1 || resp.Data[0].Embedding[2] != 0.3 {
		t.Fatalf("unexpected embedding: %+v", resp)
	}
	if cap.path != "/v1/embeddings" {
		t.Errorf("unexpected path: %s", cap.path)
	}
}

func TestEmbeddingUnsupportedProvider(t *testing.T) {
	_, err := Embedding(context.Background(), "deepseek-chat", "a", WithAPIKey("x"))
	if err == nil {
		t.Fatal("expected error for provider without embeddings")
	}
}

func TestNormalizeEmbeddingInput(t *testing.T) {
	if got, _ := normalizeEmbeddingInput("a"); len(got) != 1 || got[0] != "a" {
		t.Errorf("string input: %+v", got)
	}
	if got, _ := normalizeEmbeddingInput([]string{"a", "b"}); len(got) != 2 {
		t.Errorf("slice input: %+v", got)
	}
	if got, _ := normalizeEmbeddingInput([]any{"a", "b"}); len(got) != 2 {
		t.Errorf("any slice input: %+v", got)
	}
	if _, err := normalizeEmbeddingInput([]any{"a", 1}); err == nil {
		t.Error("non-string element should error")
	}
	if _, err := normalizeEmbeddingInput(42); err == nil {
		t.Error("unsupported type should error")
	}
}
