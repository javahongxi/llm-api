package llmapi

import (
	"context"
	"math"
	"testing"
)

func TestCompletionCost(t *testing.T) {
	srv := newOpenAIMock(t, nil)
	defer srv.Close()

	resp, err := Completion(context.Background(), "gpt-4o",
		[]Message{UserMessage("hi")}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"))
	if err != nil {
		t.Fatal(err)
	}
	// mock usage: prompt 2, completion 3 -> 2*2.5e-6 + 3*1e-5 = 3.5e-5
	want := 3.5e-5
	if math.Abs(resp.Cost-want) > 1e-12 {
		t.Errorf("expected cost %v, got %v", want, resp.Cost)
	}
}

func TestEmbeddingCost(t *testing.T) {
	srv := newOpenAIMock(t, nil)
	defer srv.Close()

	resp, err := Embedding(context.Background(), "text-embedding-3-small",
		[]string{"a"}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"))
	if err != nil {
		t.Fatal(err)
	}
	// mock usage: prompt 2 -> 2*2e-8 = 4e-8
	want := 4e-8
	if math.Abs(resp.Cost-want) > 1e-15 {
		t.Errorf("expected cost %v, got %v", want, resp.Cost)
	}
}

func TestCompletionCostUnknownModel(t *testing.T) {
	srv := newOpenAIMock(t, nil)
	defer srv.Close()

	resp, err := Completion(context.Background(), "openai/some-unknown-model",
		[]Message{UserMessage("hi")}, WithAPIBase(srv.URL+"/v1"), WithAPIKey("sk-test"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Cost != 0 {
		t.Errorf("unknown model should have zero cost, got %v", resp.Cost)
	}
}
