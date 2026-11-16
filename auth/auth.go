// Package auth implements proxy virtual API keys with budgets and rate limits,
// the Go analogue of litellm's proxy auth/budgets. Clients authenticate with a
// proxy-issued key (Authorization: Bearer <key>); each key carries a max_budget,
// optional rpm/tpm limits, an allowed-model list and an expiry. Spend is tracked
// per key and enforced against the budget.
package auth

import (
	"crypto/rand"
	"encoding/hex"
	"strconv"
	"sync"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// Key is a virtual API key and its limits/spend.
type Key struct {
	Token          string        `json:"key"`
	Alias          string        `json:"key_alias,omitempty"`
	MaxBudget      float64       `json:"max_budget,omitempty"` // USD; 0 = unlimited
	Spend          float64       `json:"spend"`                // accumulated USD
	RPMLimit       int           `json:"rpm_limit,omitempty"`  // requests/min; 0 = unlimited
	TPMLimit       int           `json:"tpm_limit,omitempty"`  // tokens/min; 0 = unlimited
	Models         []string      `json:"models,omitempty"`     // allowed models; empty = all
	Expires        time.Time     `json:"expires,omitempty"`
	BudgetDuration time.Duration `json:"-"` // reset spend after this window
	Created        time.Time     `json:"created_at"`

	windowStart time.Time
	requests    []time.Time
	tokens      []tokenEvent
}

type tokenEvent struct {
	at     time.Time
	tokens int
}

// GenerateRequest describes a new key to mint.
type GenerateRequest struct {
	Alias          string
	MaxBudget      float64
	RPMLimit       int
	TPMLimit       int
	Models         []string
	Duration       time.Duration // key expires after this long (0 = never)
	BudgetDuration time.Duration // spend resets after this window (0 = never)
}

// Store holds virtual keys and the master key. It is safe for concurrent use.
type Store struct {
	mu        sync.Mutex
	keys      map[string]*Key
	masterKey string
}

// NewStore creates a store. A non-empty masterKey enables auth and grants admin
// access to key management; the master key itself bypasses budget/model checks.
func NewStore(masterKey string) *Store {
	return &Store{keys: map[string]*Key{}, masterKey: masterKey}
}

// Enabled reports whether auth is active (a master key is set or keys exist).
// When false the proxy runs in open mode and Authorize is a no-op.
func (s *Store) Enabled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.masterKey != "" || len(s.keys) > 0
}

// IsMaster reports whether token is the master key.
func (s *Store) IsMaster(token string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.masterKey != "" && token == s.masterKey
}

// Generate mints a new random key.
func (s *Store) Generate(req GenerateRequest) (*Key, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return nil, schema.NewConnectionError("unable to generate key: "+err.Error(), "proxy", "")
	}
	now := time.Now()
	k := &Key{
		Token:          "sk-" + hex.EncodeToString(buf),
		Alias:          req.Alias,
		MaxBudget:      req.MaxBudget,
		RPMLimit:       req.RPMLimit,
		TPMLimit:       req.TPMLimit,
		Models:         req.Models,
		BudgetDuration: req.BudgetDuration,
		Created:        now,
		windowStart:    now,
	}
	if req.Duration > 0 {
		k.Expires = now.Add(req.Duration)
	}
	s.mu.Lock()
	s.keys[k.Token] = k
	s.mu.Unlock()
	return k, nil
}

// Get returns a key by token.
func (s *Store) Get(token string) (*Key, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k, ok := s.keys[token]
	return k, ok
}

// List returns all keys (snapshots of the exported fields).
func (s *Store) List() []*Key {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Key, 0, len(s.keys))
	for _, k := range s.keys {
		out = append(out, k)
	}
	return out
}

// Delete removes a key, reporting whether it existed.
func (s *Store) Delete(token string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.keys[token]
	delete(s.keys, token)
	return ok
}

// UpdateRequest describes mutable key fields; nil fields are left unchanged.
type UpdateRequest struct {
	Alias          *string
	MaxBudget      *float64
	RPMLimit       *int
	TPMLimit       *int
	Models         *[]string
	Duration       *time.Duration // recompute expiry from now (0 = never expires)
	BudgetDuration *time.Duration
}

// Update mutates an existing key's limits/metadata, returning the updated key.
func (s *Store) Update(token string, req UpdateRequest) (*Key, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k, ok := s.keys[token]
	if !ok {
		return nil, false
	}
	if req.Alias != nil {
		k.Alias = *req.Alias
	}
	if req.MaxBudget != nil {
		k.MaxBudget = *req.MaxBudget
	}
	if req.RPMLimit != nil {
		k.RPMLimit = *req.RPMLimit
	}
	if req.TPMLimit != nil {
		k.TPMLimit = *req.TPMLimit
	}
	if req.Models != nil {
		k.Models = *req.Models
	}
	if req.Duration != nil {
		if *req.Duration > 0 {
			k.Expires = time.Now().Add(*req.Duration)
		} else {
			k.Expires = time.Time{}
		}
	}
	if req.BudgetDuration != nil {
		k.BudgetDuration = *req.BudgetDuration
	}
	return k, true
}

// AddSpend records spend against a key (no-op for unknown keys or non-positive
// amounts).
func (s *Store) AddSpend(token string, amount float64) {
	if amount <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if k, ok := s.keys[token]; ok {
		s.resetBudgetIfDue(k, time.Now())
		k.Spend += amount
	}
}

// AddTokens records token usage against a key for tpm limiting.
func (s *Store) AddTokens(token string, tokens int) {
	if tokens <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if k, ok := s.keys[token]; ok {
		k.tokens = append(k.tokens, tokenEvent{at: time.Now(), tokens: tokens})
	}
}

// Authorize validates a request token for the given model. In open mode it
// returns (nil, nil). The master key bypasses checks. Otherwise it enforces
// existence, expiry, model allowlist, budget and rpm/tpm limits, returning a
// unified error on failure. On success it returns the key (for spend attribution)
// and records the request against the rate limits.
func (s *Store) Authorize(token, model string) (*Key, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.masterKey == "" && len(s.keys) == 0 {
		return nil, nil // open mode
	}
	if token == "" {
		return nil, schema.NewAuthenticationError("missing API key (Authorization: Bearer <key>)", "proxy", model)
	}
	if s.masterKey != "" && token == s.masterKey {
		return nil, nil // master bypass
	}
	k, ok := s.keys[token]
	if !ok {
		return nil, schema.NewAuthenticationError("invalid API key", "proxy", model)
	}
	now := time.Now()
	if !k.Expires.IsZero() && now.After(k.Expires) {
		return nil, schema.NewAuthenticationError("API key has expired", "proxy", model)
	}
	if len(k.Models) > 0 && !contains(k.Models, model) {
		return nil, schema.NewPermissionDeniedError("key not allowed to access model: "+model, "proxy", model)
	}
	s.resetBudgetIfDue(k, now)
	if k.MaxBudget > 0 && k.Spend >= k.MaxBudget {
		return nil, schema.NewBadRequestError(
			"Budget has been exceeded! Current cost: "+formatUSD(k.Spend)+", Max budget: "+formatUSD(k.MaxBudget),
			"proxy", model)
	}
	// rpm: sliding 60s window.
	if k.RPMLimit > 0 {
		k.requests = pruneRequests(k.requests, now)
		if len(k.requests) >= k.RPMLimit {
			return nil, schema.NewRateLimitError("key rpm limit exceeded", "proxy", model)
		}
		k.requests = append(k.requests, now)
	}
	// tpm: sliding 60s window of token usage.
	if k.TPMLimit > 0 {
		k.tokens = pruneTokens(k.tokens, now)
		if sumTokens(k.tokens) >= k.TPMLimit {
			return nil, schema.NewRateLimitError("key tpm limit exceeded", "proxy", model)
		}
	}
	return k, nil
}

func (s *Store) resetBudgetIfDue(k *Key, now time.Time) {
	if k.BudgetDuration > 0 && now.Sub(k.windowStart) >= k.BudgetDuration {
		k.Spend = 0
		k.windowStart = now
	}
}

func pruneRequests(reqs []time.Time, now time.Time) []time.Time {
	cutoff := now.Add(-time.Minute)
	i := 0
	for _, t := range reqs {
		if t.After(cutoff) {
			reqs[i] = t
			i++
		}
	}
	return reqs[:i]
}

func pruneTokens(events []tokenEvent, now time.Time) []tokenEvent {
	cutoff := now.Add(-time.Minute)
	i := 0
	for _, e := range events {
		if e.at.After(cutoff) {
			events[i] = e
			i++
		}
	}
	return events[:i]
}

func sumTokens(events []tokenEvent) int {
	total := 0
	for _, e := range events {
		total += e.tokens
	}
	return total
}

func contains(list []string, v string) bool {
	for _, item := range list {
		if item == v {
			return true
		}
	}
	return false
}

func formatUSD(v float64) string {
	return "$" + strconv.FormatFloat(v, 'f', 6, 64)
}
