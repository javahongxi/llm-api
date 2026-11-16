package proxy

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/javahongxi/llm-api/auth"
)

// requireAdmin guards admin endpoints (key management, /spend). In open mode it
// is a no-op; when auth is enabled it requires the master key. It writes a 401
// and returns false on failure.
func (s *Server) requireAdmin(w http.ResponseWriter, r *http.Request) bool {
	if !s.keys.Enabled() {
		return true
	}
	if s.keys.IsMaster(bearerToken(r)) {
		return true
	}
	writeError(w, http.StatusUnauthorized, "admin endpoints require the master key")
	return false
}

// keyGenerateRequest is the /key/generate body (litellm GenerateKeyRequest).
// Durations accept Go duration strings ("24h", "30d" is not supported; use
// "720h") or a number of seconds.
type keyGenerateRequest struct {
	KeyAlias       string   `json:"key_alias"`
	MaxBudget      float64  `json:"max_budget"`
	RPMLimit       int      `json:"rpm_limit"`
	TPMLimit       int      `json:"tpm_limit"`
	Models         []string `json:"models"`
	Duration       string   `json:"duration"`
	BudgetDuration string   `json:"budget_duration"`
}

func (s *Server) handleKeyGenerate(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	var req keyGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	duration, err := parseDuration(req.Duration)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid duration: "+err.Error())
		return
	}
	budgetDuration, err := parseDuration(req.BudgetDuration)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid budget_duration: "+err.Error())
		return
	}
	key, err := s.keys.Generate(auth.GenerateRequest{
		Alias:          req.KeyAlias,
		MaxBudget:      req.MaxBudget,
		RPMLimit:       req.RPMLimit,
		TPMLimit:       req.TPMLimit,
		Models:         req.Models,
		Duration:       duration,
		BudgetDuration: budgetDuration,
	})
	if err != nil {
		writeLLMError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, key)
}

// keyUpdateRequest is the /key/update body; omitted fields are left unchanged.
type keyUpdateRequest struct {
	Key            string    `json:"key"`
	KeyAlias       *string   `json:"key_alias"`
	MaxBudget      *float64  `json:"max_budget"`
	RPMLimit       *int      `json:"rpm_limit"`
	TPMLimit       *int      `json:"tpm_limit"`
	Models         *[]string `json:"models"`
	Duration       *string   `json:"duration"`
	BudgetDuration *string   `json:"budget_duration"`
}

func (s *Server) handleKeyUpdate(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	var req keyUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Key == "" {
		writeError(w, http.StatusBadRequest, "key is required")
		return
	}
	update := auth.UpdateRequest{
		Alias:     req.KeyAlias,
		MaxBudget: req.MaxBudget,
		RPMLimit:  req.RPMLimit,
		TPMLimit:  req.TPMLimit,
		Models:    req.Models,
	}
	if req.Duration != nil {
		d, err := parseDuration(*req.Duration)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid duration: "+err.Error())
			return
		}
		update.Duration = &d
	}
	if req.BudgetDuration != nil {
		d, err := parseDuration(*req.BudgetDuration)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid budget_duration: "+err.Error())
			return
		}
		update.BudgetDuration = &d
	}
	key, ok := s.keys.Update(req.Key, update)
	if !ok {
		writeError(w, http.StatusNotFound, "key not found: "+req.Key)
		return
	}
	writeJSON(w, http.StatusOK, key)
}

func (s *Server) handleKeyDelete(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	var req struct {
		Keys []string `json:"keys"`
		Key  string   `json:"key"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	tokens := req.Keys
	if req.Key != "" {
		tokens = append(tokens, req.Key)
	}
	deleted := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if s.keys.Delete(token) {
			deleted = append(deleted, token)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted_keys": deleted})
}

func (s *Server) handleKeyInfo(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	token := r.URL.Query().Get("key")
	if token == "" {
		token = bearerToken(r)
	}
	key, ok := s.keys.Get(token)
	if !ok {
		writeError(w, http.StatusNotFound, "key not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"key": token, "info": key})
}

func (s *Server) handleKeyList(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	keys := s.keys.List()
	tokens := make([]string, 0, len(keys))
	for _, k := range keys {
		tokens = append(tokens, k.Token)
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": tokens, "total_count": len(tokens)})
}

// parseDuration accepts a Go duration string ("24h", "1h30m") or a bare number
// of seconds. Empty input yields 0.
func parseDuration(s string) (time.Duration, error) {
	if s == "" {
		return 0, nil
	}
	if d, err := time.ParseDuration(s); err == nil {
		return d, nil
	}
	secs, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return time.Duration(secs * float64(time.Second)), nil
}
