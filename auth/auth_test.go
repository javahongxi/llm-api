package auth

import (
	"testing"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

func TestOpenModeAuthorize(t *testing.T) {
	s := NewStore("")
	if s.Enabled() {
		t.Fatal("empty store should be disabled (open mode)")
	}
	key, err := s.Authorize("", "gpt-4o")
	if err != nil || key != nil {
		t.Fatalf("open mode should be a no-op, got key=%v err=%v", key, err)
	}
}

func TestMasterBypass(t *testing.T) {
	s := NewStore("sk-master")
	if !s.Enabled() {
		t.Fatal("store with master key should be enabled")
	}
	if !s.IsMaster("sk-master") {
		t.Fatal("IsMaster should recognize the master key")
	}
	key, err := s.Authorize("sk-master", "anything")
	if err != nil || key != nil {
		t.Fatalf("master should bypass, got key=%v err=%v", key, err)
	}
}

func TestMissingAndInvalidKey(t *testing.T) {
	s := NewStore("sk-master")
	if _, err := s.Authorize("", "gpt-4o"); !schema.IsAuthentication(err) {
		t.Errorf("missing key should be 401, got %v", err)
	}
	if _, err := s.Authorize("sk-nope", "gpt-4o"); !schema.IsAuthentication(err) {
		t.Errorf("invalid key should be 401, got %v", err)
	}
}

func TestGenerateGetListDelete(t *testing.T) {
	s := NewStore("sk-master")
	k, err := s.Generate(GenerateRequest{Alias: "team-a", MaxBudget: 10, RPMLimit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if k.Token == "" || k.Alias != "team-a" || k.MaxBudget != 10 {
		t.Fatalf("unexpected key: %+v", k)
	}
	if got, ok := s.Get(k.Token); !ok || got.Token != k.Token {
		t.Fatal("Get should return the generated key")
	}
	if len(s.List()) != 1 {
		t.Fatalf("expected 1 key, got %d", len(s.List()))
	}
	if !s.Delete(k.Token) || s.Delete(k.Token) {
		t.Fatal("Delete should report existence once")
	}
	if len(s.List()) != 0 {
		t.Fatal("store should be empty after delete")
	}
}

func TestModelAllowlist(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{Models: []string{"gpt-4o"}})
	if _, err := s.Authorize(k.Token, "gpt-4o"); err != nil {
		t.Fatalf("allowed model should pass: %v", err)
	}
	_, err := s.Authorize(k.Token, "claude")
	if !schema.IsPermissionDenied(err) {
		t.Errorf("disallowed model should be 403, got %v", err)
	}
}

func TestBudgetExceeded(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{MaxBudget: 1})
	if _, err := s.Authorize(k.Token, "gpt-4o"); err != nil {
		t.Fatalf("under budget should pass: %v", err)
	}
	s.AddSpend(k.Token, 1.5)
	_, err := s.Authorize(k.Token, "gpt-4o")
	if !schema.IsBadRequest(err) {
		t.Errorf("over budget should be 400, got %v", err)
	}
}

func TestBudgetReset(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{MaxBudget: 1, BudgetDuration: time.Millisecond})
	s.AddSpend(k.Token, 2)
	time.Sleep(5 * time.Millisecond)
	if _, err := s.Authorize(k.Token, "gpt-4o"); err != nil {
		t.Fatalf("budget should reset after duration: %v", err)
	}
	if k.Spend != 0 {
		t.Errorf("spend should reset to 0, got %v", k.Spend)
	}
}

func TestRPMLimit(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{RPMLimit: 2})
	for i := 0; i < 2; i++ {
		if _, err := s.Authorize(k.Token, "gpt-4o"); err != nil {
			t.Fatalf("request %d under rpm should pass: %v", i, err)
		}
	}
	_, err := s.Authorize(k.Token, "gpt-4o")
	if !schema.IsRateLimit(err) {
		t.Errorf("over rpm should be 429, got %v", err)
	}
}

func TestTPMLimit(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{TPMLimit: 10})
	s.AddTokens(k.Token, 10)
	_, err := s.Authorize(k.Token, "gpt-4o")
	if !schema.IsRateLimit(err) {
		t.Errorf("over tpm should be 429, got %v", err)
	}
}

func TestExpiry(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{Duration: time.Millisecond})
	time.Sleep(5 * time.Millisecond)
	_, err := s.Authorize(k.Token, "gpt-4o")
	if !schema.IsAuthentication(err) {
		t.Errorf("expired key should be 401, got %v", err)
	}
}

func TestUpdate(t *testing.T) {
	s := NewStore("sk-master")
	k, _ := s.Generate(GenerateRequest{MaxBudget: 1, Alias: "old"})
	budget := 5.0
	alias := "new"
	updated, ok := s.Update(k.Token, UpdateRequest{MaxBudget: &budget, Alias: &alias})
	if !ok {
		t.Fatal("Update should find the key")
	}
	if updated.MaxBudget != 5 || updated.Alias != "new" {
		t.Fatalf("unexpected update: %+v", updated)
	}
	if _, ok := s.Update("sk-missing", UpdateRequest{}); ok {
		t.Fatal("Update of unknown key should report false")
	}
}
