package router

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llmapi "github.com/javahongxi/llm-api"
	"github.com/javahongxi/llm-api/schema"
)

const okChatBody = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// countingBackend returns a server that counts requests and replies via reply.
func countingBackend(t *testing.T, reply func(n int32) (int, string)) (*httptest.Server, *int32) {
	t.Helper()
	var count int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		n := atomic.AddInt32(&count, 1)
		status, body := reply(n)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, &count
}

func okBackend(t *testing.T) (*httptest.Server, *int32) {
	return countingBackend(t, func(int32) (int, string) { return http.StatusOK, okChatBody })
}

func fastSettings() Settings {
	return Settings{
		NumRetries:       3,
		RetryBackoffBase: time.Millisecond,
		MaxRetryBackoff:  5 * time.Millisecond,
		CooldownTime:     time.Hour,
	}
}

func dep(group, base string) Deployment {
	return Deployment{ModelName: group, Model: "openai/gpt-4o", APIKey: "x", APIBase: base + "/v1"}
}

func TestRoundRobinDistribution(t *testing.T) {
	a, ca := okBackend(t)
	b, cb := okBackend(t)
	r := New([]Deployment{dep("gpt", a.URL), dep("gpt", b.URL)},
		Settings{Strategy: RoundRobin})

	for i := 0; i < 6; i++ {
		if _, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")}); err != nil {
			t.Fatal(err)
		}
	}
	if atomic.LoadInt32(ca) != 3 || atomic.LoadInt32(cb) != 3 {
		t.Errorf("round-robin should split 3/3, got %d/%d", *ca, *cb)
	}
}

func TestLeastBusyPick(t *testing.T) {
	r := New([]Deployment{dep("gpt", "http://a"), dep("gpt", "http://b")},
		Settings{Strategy: LeastBusy})
	deps := r.groups["gpt"]
	r.begin(deps[0])
	r.begin(deps[0])
	r.begin(deps[1])
	if got := r.pick("gpt", deps); got != deps[1] {
		t.Errorf("least-busy should pick the less loaded deployment")
	}
}

func TestLatencyBasedPick(t *testing.T) {
	r := New([]Deployment{dep("gpt", "http://a"), dep("gpt", "http://b")},
		Settings{Strategy: LatencyBased})
	deps := r.groups["gpt"]
	deps[0].latencyEWMA, deps[0].hasLatency = 1.0, true
	deps[1].latencyEWMA, deps[1].hasLatency = 0.1, true
	if got := r.pick("gpt", deps); got != deps[1] {
		t.Errorf("latency-based should pick the faster deployment")
	}
}

func TestWeightedShuffle(t *testing.T) {
	heavy := dep("gpt", "http://heavy")
	heavy.Weight = 100
	light := dep("gpt", "http://light")
	light.Weight = 1
	r := New([]Deployment{heavy, light}, Settings{Strategy: SimpleShuffle})
	deps := r.groups["gpt"]

	heavyHits := 0
	for i := 0; i < 2000; i++ {
		if r.pick("gpt", deps) == deps[0] {
			heavyHits++
		}
	}
	if heavyHits < 1600 {
		t.Errorf("weighted shuffle should favour heavy deployment, got %d/2000", heavyHits)
	}
}

func TestRetryThenSuccess(t *testing.T) {
	srv, count := countingBackend(t, func(n int32) (int, string) {
		if n < 3 {
			return http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`
		}
		return http.StatusOK, okChatBody
	})
	r := New([]Deployment{dep("gpt", srv.URL)}, fastSettings())

	resp, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")})
	if err != nil {
		t.Fatalf("expected success after retries: %v", err)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Errorf("unexpected content: %q", resp.Choices[0].Message.Content)
	}
	if atomic.LoadInt32(count) != 3 {
		t.Errorf("expected 3 attempts, got %d", *count)
	}
}

func TestNoRetryOnBadRequest(t *testing.T) {
	srv, count := countingBackend(t, func(int32) (int, string) {
		return http.StatusBadRequest, `{"error":{"message":"bad input"}}`
	})
	r := New([]Deployment{dep("gpt", srv.URL)}, fastSettings())

	_, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")})
	if !schema.IsBadRequest(err) {
		t.Fatalf("expected bad request error, got %v", err)
	}
	if atomic.LoadInt32(count) != 1 {
		t.Errorf("bad request should not retry, got %d attempts", *count)
	}
}

func TestRetryPolicyOverride(t *testing.T) {
	srv, count := countingBackend(t, func(n int32) (int, string) {
		if n < 3 {
			return http.StatusBadRequest, `{"error":{"message":"bad"}}`
		}
		return http.StatusOK, okChatBody
	})
	s := fastSettings()
	two := 2
	s.RetryPolicy = &RetryPolicy{BadRequestErrorRetries: &two}
	r := New([]Deployment{dep("gpt", srv.URL)}, s)

	if _, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")}); err != nil {
		t.Fatalf("expected success via policy override: %v", err)
	}
	if atomic.LoadInt32(count) != 3 {
		t.Errorf("policy should allow 2 retries (3 attempts), got %d", *count)
	}
}

func TestFallback(t *testing.T) {
	primary, pc := countingBackend(t, func(int32) (int, string) {
		return http.StatusInternalServerError, `{"error":{"message":"down"}}`
	})
	backup, bc := okBackend(t)
	s := fastSettings()
	s.NumRetries = 1
	s.Fallbacks = map[string][]string{"primary": {"backup"}}
	r := New([]Deployment{dep("primary", primary.URL), dep("backup", backup.URL)}, s)

	resp, err := r.Completion(context.Background(), "primary", []llmapi.Message{llmapi.UserMessage("hi")})
	if err != nil {
		t.Fatalf("expected fallback success: %v", err)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Errorf("unexpected content: %q", resp.Choices[0].Message.Content)
	}
	if atomic.LoadInt32(pc) != 2 { // 1 attempt + 1 retry
		t.Errorf("primary should be tried twice, got %d", *pc)
	}
	if atomic.LoadInt32(bc) != 1 {
		t.Errorf("backup should be tried once, got %d", *bc)
	}
}

func TestCooldownSkipsFailingDeployment(t *testing.T) {
	bad, _ := countingBackend(t, func(int32) (int, string) {
		return http.StatusInternalServerError, `{"error":{"message":"down"}}`
	})
	good, gc := okBackend(t)
	s := fastSettings()
	s.NumRetries = 0
	s.AllowedFails = 1
	r := New([]Deployment{dep("gpt", bad.URL), dep("gpt", good.URL)}, s)
	r.settings.Strategy = RoundRobin
	deps := r.groups["gpt"]

	// First call hits the bad deployment (rr index 0) and fails.
	if _, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")}); err == nil {
		t.Fatal("expected first call to fail")
	}
	if !deps[0].cooldownUntil.After(time.Now()) {
		t.Fatal("bad deployment should be in cooldown")
	}
	// Second call must skip the cooled-down deployment and hit the good one.
	if _, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")}); err != nil {
		t.Fatalf("expected second call to succeed via healthy deployment: %v", err)
	}
	if atomic.LoadInt32(gc) != 1 {
		t.Errorf("good deployment should serve the second call, got %d", *gc)
	}
}

func TestUnknownModel(t *testing.T) {
	r := New(nil, Settings{})
	_, err := r.Completion(context.Background(), "missing", []llmapi.Message{llmapi.UserMessage("hi")})
	if !schema.IsBadRequest(err) {
		t.Fatalf("expected bad request for unknown model, got %v", err)
	}
}

func TestStreamRetry(t *testing.T) {
	var count int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		n := atomic.AddInt32(&count, 1)
		if n < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"unavailable"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: "+`{"id":"1","choices":[{"index":0,"delta":{"content":"hi"}}]}`+"\n\n")
		_, _ = io.WriteString(w, "data: "+`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	r := New([]Deployment{dep("gpt", srv.URL)}, fastSettings())

	stream, err := r.CompletionStream(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")})
	if err != nil {
		t.Fatalf("expected stream after retry: %v", err)
	}
	defer stream.Close()
	var content string
	for stream.Next() {
		if len(stream.Current().Choices) > 0 {
			content += stream.Current().Choices[0].Delta.Content
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if content != "hi" {
		t.Errorf("expected hi, got %q", content)
	}
	if atomic.LoadInt32(&count) != 2 {
		t.Errorf("expected 2 attempts, got %d", count)
	}
}

func TestEmbeddingRouting(t *testing.T) {
	srv, _ := countingBackend(t, func(int32) (int, string) {
		return http.StatusOK, `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1,"total_tokens":1}}`
	})
	r := New([]Deployment{{ModelName: "embed", Model: "openai/text-embedding-3-small", APIKey: "x", APIBase: srv.URL + "/v1"}}, Settings{})
	resp, err := r.Embedding(context.Background(), "embed", []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 1 || resp.Data[0].Embedding[1] != 0.2 {
		t.Fatalf("unexpected embedding: %+v", resp)
	}
}

func TestModelsAndHasModel(t *testing.T) {
	r := New([]Deployment{dep("zeta", "http://a"), dep("alpha", "http://b"), dep("alpha", "http://c")}, Settings{})
	models := r.Models()
	if len(models) != 2 || models[0] != "alpha" || models[1] != "zeta" {
		t.Errorf("expected sorted unique groups [alpha zeta], got %v", models)
	}
	if !r.HasModel("alpha") || r.HasModel("missing") {
		t.Error("HasModel mismatch")
	}
}
