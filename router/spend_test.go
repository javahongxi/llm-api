package router

import (
	"context"
	"math"
	"net/http"
	"testing"

	llmapi "github.com/javahongxi/llm-api"
)

func TestSpendTracking(t *testing.T) {
	srv, _ := okBackend(t)
	r := New([]Deployment{dep("gpt", srv.URL)}, Settings{})

	for i := 0; i < 2; i++ {
		resp, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")})
		if err != nil {
			t.Fatal(err)
		}
		// okChatBody usage: prompt 1, completion 1 -> 1*2.5e-6 + 1*1e-5 = 1.25e-5
		if math.Abs(resp.Cost-1.25e-5) > 1e-12 {
			t.Fatalf("expected per-call cost 1.25e-5, got %v", resp.Cost)
		}
	}
	report := r.Spend()
	if math.Abs(report.TotalSpend-2.5e-5) > 1e-12 {
		t.Errorf("expected total spend 2.5e-5, got %v", report.TotalSpend)
	}
	if math.Abs(report.ByModel["gpt"]-2.5e-5) > 1e-12 {
		t.Errorf("expected spend by model gpt 2.5e-5, got %v", report.ByModel["gpt"])
	}
}

func TestSpendNotRecordedOnFailure(t *testing.T) {
	srv, _ := countingBackend(t, func(int32) (int, string) {
		return http.StatusInternalServerError, `{"error":{"message":"down"}}`
	})
	r := New([]Deployment{dep("gpt", srv.URL)}, Settings{NumRetries: 0})
	if _, err := r.Completion(context.Background(), "gpt", []llmapi.Message{llmapi.UserMessage("hi")}); err == nil {
		t.Fatal("expected failure")
	}
	if r.Spend().TotalSpend != 0 {
		t.Errorf("failed calls should not accrue spend, got %v", r.Spend().TotalSpend)
	}
}
