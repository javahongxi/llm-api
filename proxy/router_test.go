package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const proxyOKBody = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func TestProxyRouterRetry(t *testing.T) {
	var count int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		n := atomic.AddInt32(&count, 1)
		w.Header().Set("Content-Type", "application/json")
		if n < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"slow down"}}`)
			return
		}
		_, _ = io.WriteString(w, proxyOKBody)
	}))
	defer backend.Close()

	cfg := &Config{
		ModelList: []ModelEntry{{ModelName: "gpt", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: backend.URL + "/v1"}}},
		Router:    RouterSettings{NumRetries: 2},
	}
	srv := httptest.NewServer(New(cfg).Handler())
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"gpt","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected success after retry, got %d: %s", resp.StatusCode, data)
	}
	if atomic.LoadInt32(&count) != 2 {
		t.Errorf("expected 2 backend attempts, got %d", count)
	}
}

func TestProxyRouterFallback(t *testing.T) {
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"message":"down"}}`)
	}))
	defer primary.Close()
	backup := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, proxyOKBody)
	}))
	defer backup.Close()

	cfg := &Config{
		ModelList: []ModelEntry{
			{ModelName: "primary", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: primary.URL + "/v1"}},
			{ModelName: "backup", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: backup.URL + "/v1"}},
		},
		Router: RouterSettings{Fallbacks: []map[string][]string{{"primary": {"backup"}}}},
	}
	srv := httptest.NewServer(New(cfg).Handler())
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"primary","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected fallback success, got %d: %s", resp.StatusCode, data)
	}
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	msg := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "ok" {
		t.Errorf("expected backup content, got %v", msg["content"])
	}
}

func TestProxyRouterLoadBalance(t *testing.T) {
	var a, b int32
	backendA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		atomic.AddInt32(&a, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, proxyOKBody)
	}))
	defer backendA.Close()
	backendB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		atomic.AddInt32(&b, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, proxyOKBody)
	}))
	defer backendB.Close()

	cfg := &Config{
		ModelList: []ModelEntry{
			{ModelName: "gpt", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: backendA.URL + "/v1"}},
			{ModelName: "gpt", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: backendB.URL + "/v1"}},
		},
		Router: RouterSettings{RoutingStrategy: "round-robin"},
	}
	srv := httptest.NewServer(New(cfg).Handler())
	defer srv.Close()

	for i := 0; i < 4; i++ {
		resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
			`{"model":"gpt","messages":[{"role":"user","content":"hi"}]}`)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", resp.StatusCode, data)
		}
	}
	if atomic.LoadInt32(&a) != 2 || atomic.LoadInt32(&b) != 2 {
		t.Errorf("round-robin should split 2/2, got %d/%d", a, b)
	}
	// model_list groups collapse to a single public model name.
	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if strings.Count(string(data), `"id":"gpt"`) != 1 {
		t.Errorf("expected one grouped model entry: %s", data)
	}
}
