package proxy

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/javahongxi/llm-api/cost"
)

func TestProxySpend(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	// newBackend chat usage: prompt 1, completion 2 -> gpt-4o cost 2.25e-5.
	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"my-gpt","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status %d: %s", resp.StatusCode, data)
	}
	var chat map[string]any
	_ = json.Unmarshal(data, &chat)
	if cost, _ := chat["cost"].(float64); math.Abs(cost-2.25e-5) > 1e-12 {
		t.Errorf("expected response cost 2.25e-5, got %v", cost)
	}

	// /spend should reflect the tracked call.
	sresp, err := http.Get(srv.URL + "/spend")
	if err != nil {
		t.Fatal(err)
	}
	sdata, _ := io.ReadAll(sresp.Body)
	sresp.Body.Close()
	var report struct {
		TotalSpend float64            `json:"total_spend"`
		ByModel    map[string]float64 `json:"spend_by_model"`
	}
	if err := json.Unmarshal(sdata, &report); err != nil {
		t.Fatal(err)
	}
	if math.Abs(report.TotalSpend-2.25e-5) > 1e-12 {
		t.Errorf("expected total spend 2.25e-5, got %v", report.TotalSpend)
	}
	if math.Abs(report.ByModel["my-gpt"]-2.25e-5) > 1e-12 {
		t.Errorf("expected spend by model my-gpt 2.25e-5, got %v", report.ByModel["my-gpt"])
	}
}

func TestProxyModelCostOverride(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	cfg := &Config{
		ModelList: []ModelEntry{{ModelName: "my-gpt", Params: Params{Model: "openai/custom-priced-model", APIKey: "x", APIBase: backend.URL + "/v1"}}},
		ModelCost: map[string]cost.ModelCost{"custom-priced-model": {InputCostPerToken: 1e-3, OutputCostPerToken: 1e-3}},
	}
	srv := httptest.NewServer(New(cfg).Handler())
	defer srv.Close()

	resp, data := postJSON(t, srv.URL+"/v1/chat/completions",
		`{"model":"my-gpt","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, data)
	}
	var chat map[string]any
	_ = json.Unmarshal(data, &chat)
	// overridden price: (1+2)*1e-3 = 3e-3
	if c, _ := chat["cost"].(float64); math.Abs(c-3e-3) > 1e-12 {
		t.Errorf("expected overridden cost 3e-3, got %v", c)
	}
}
