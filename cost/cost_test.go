package cost

import (
	"math"
	"testing"

	"github.com/javahongxi/llm-api/schema"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestCostKnownModel(t *testing.T) {
	// gpt-4o: 2.5e-6 input, 1e-5 output. 1000 in + 500 out = 0.0075.
	got := Cost("gpt-4o", 1000, 500)
	if !approx(got, 0.0075) {
		t.Errorf("expected 0.0075, got %v", got)
	}
}

func TestCostUnknownModel(t *testing.T) {
	if got := Cost("totally-unknown-model", 1000, 500); got != 0 {
		t.Errorf("unknown model should cost 0, got %v", got)
	}
}

func TestLookupProviderPrefix(t *testing.T) {
	if _, ok := Lookup("openai/gpt-4o"); !ok {
		t.Error("prefixed model should resolve to bare price")
	}
	if _, ok := Lookup("dashscope/qwen-plus"); !ok {
		t.Error("dashscope/qwen-plus should resolve")
	}
	if c, ok := Lookup("deepseek/deepseek-chat"); !ok || c.InputCostPerToken != 2.8e-7 {
		t.Errorf("deepseek price mismatch: %+v ok=%v", c, ok)
	}
}

func TestRegisterOverride(t *testing.T) {
	Register("test-custom-model", ModelCost{InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6})
	got := Cost("test-custom-model", 1000, 1000)
	if !approx(got, 0.003) {
		t.Errorf("expected 0.003 from custom price, got %v", got)
	}
	// custom wins over default for a known model name
	Register("test-override-model", ModelCost{InputCostPerToken: 1, OutputCostPerToken: 1})
	RegisterAll(map[string]ModelCost{"test-override-model": {InputCostPerToken: 5, OutputCostPerToken: 5}})
	if c, _ := Lookup("test-override-model"); c.InputCostPerToken != 5 {
		t.Errorf("RegisterAll should override, got %+v", c)
	}
}

func TestUsageCost(t *testing.T) {
	if got := UsageCost("gpt-4o", nil); got != 0 {
		t.Errorf("nil usage should cost 0, got %v", got)
	}
	u := &schema.Usage{PromptTokens: 2, CompletionTokens: 3}
	// 2*2.5e-6 + 3*1e-5 = 3.5e-5
	if got := UsageCost("gpt-4o", u); !approx(got, 3.5e-5) {
		t.Errorf("expected 3.5e-5, got %v", got)
	}
}

func TestSpendTracker(t *testing.T) {
	s := NewSpendTracker()
	s.Add("gpt-4o", 0.01)
	s.Add("gpt-4o", 0.02)
	s.AddForKey("key-1", "claude", 0.05)
	s.Add("ignored", -1) // negative ignored

	if !approx(s.Total(), 0.08) {
		t.Errorf("expected total 0.08, got %v", s.Total())
	}
	rep := s.Report()
	if !approx(rep.ByModel["gpt-4o"], 0.03) {
		t.Errorf("byModel gpt-4o expected 0.03, got %v", rep.ByModel["gpt-4o"])
	}
	if !approx(rep.ByKey["key-1"], 0.05) {
		t.Errorf("byKey key-1 expected 0.05, got %v", rep.ByKey["key-1"])
	}
	// report is a snapshot copy
	rep.ByModel["gpt-4o"] = 999
	if !approx(s.Report().ByModel["gpt-4o"], 0.03) {
		t.Error("report should be an independent copy")
	}
}
