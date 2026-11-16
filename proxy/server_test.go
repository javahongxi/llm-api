package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newBackend returns an OpenAI-wire mock backend that the proxy routes to.
func newBackend(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		stream, _ := body["stream"].(bool)
		if strings.HasSuffix(r.URL.Path, "/embeddings") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1,"total_tokens":1}}`))
			return
		}
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			for _, c := range []string{
				`{"id":"1","choices":[{"index":0,"delta":{"content":"Hi"}}]}`,
				`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+c+"\n\n")
				if flusher != nil {
					flusher.Flush()
				}
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"proxy reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
}

func newTestServer(t *testing.T, backendURL string) *httptest.Server {
	t.Helper()
	cfg := &Config{ModelList: []ModelEntry{
		{ModelName: "my-gpt", Params: Params{Model: "openai/gpt-4o", APIKey: "sk-test", APIBase: backendURL + "/v1"}},
		{ModelName: "my-embed", Params: Params{Model: "openai/text-embedding-3-small", APIKey: "sk-test", APIBase: backendURL + "/v1"}},
	}}
	return httptest.NewServer(New(cfg).Handler())
}

func postJSON(t *testing.T, url, body string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, data
}

func TestProxyChat(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"my-gpt","messages":[{"role":"user","content":"hi"}],"temperature":0.3}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, data)
	}
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	choices := out["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "proxy reply" {
		t.Errorf("unexpected content: %v", msg["content"])
	}
}

func TestProxyChatStream(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"my-gpt","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, data)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected SSE content-type, got %q", ct)
	}
	text := string(data)
	if !strings.Contains(text, "data: [DONE]") {
		t.Errorf("missing [DONE]: %s", text)
	}
	if !strings.Contains(text, `"content":"Hi"`) {
		t.Errorf("missing streamed content: %s", text)
	}
}

func TestProxyEmbeddings(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/embeddings",
		`{"model":"my-embed","input":["a","b"]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, data)
	}
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	if len(out["data"].([]any)) != 1 {
		t.Errorf("unexpected embedding data: %s", data)
	}
}

func TestProxyModels(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	if out["object"] != "list" {
		t.Errorf("expected list object: %s", data)
	}
	list := out["data"].([]any)
	if len(list) != 2 {
		t.Errorf("expected 2 models, got %d", len(list))
	}
	// sorted: my-embed before my-gpt
	if list[0].(map[string]any)["id"] != "my-embed" {
		t.Errorf("models should be sorted: %s", data)
	}
}

func TestProxyHealth(t *testing.T) {
	cfg := &Config{}
	srv := httptest.NewServer(New(cfg).Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(data), `"status":"ok"`) {
		t.Errorf("unexpected health: %s", data)
	}
}

func TestProxyModelNotFound(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404, got %d: %s", resp.StatusCode, data)
	}
}

func TestProxyUpstreamError(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"my-gpt","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("expected 429 passthrough, got %d: %s", resp.StatusCode, data)
	}
	if !strings.Contains(string(data), "rate limited") {
		t.Errorf("error message not forwarded: %s", data)
	}
}

func TestResolveSecret(t *testing.T) {
	t.Setenv("LLM_API_TEST_KEY", "secret-value")
	if got := ResolveSecret("os.environ/LLM_API_TEST_KEY"); got != "secret-value" {
		t.Errorf("env not resolved: %q", got)
	}
	if got := ResolveSecret("plain-key"); got != "plain-key" {
		t.Errorf("plain value changed: %q", got)
	}
}

func TestConfigAddrDefault(t *testing.T) {
	cfg := &Config{}
	if cfg.Addr() != "0.0.0.0:0" {
		t.Errorf("unexpected addr: %s", cfg.Addr())
	}
	cfg.Server.Host = "127.0.0.1"
	cfg.Server.Port = 4000
	if cfg.Addr() != "127.0.0.1:4000" {
		t.Errorf("unexpected addr: %s", cfg.Addr())
	}
}
