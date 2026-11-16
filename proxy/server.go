package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	llmapi "github.com/javahongxi/llm-api"
	"github.com/javahongxi/llm-api/auth"
	"github.com/javahongxi/llm-api/cost"
	"github.com/javahongxi/llm-api/router"
	"github.com/javahongxi/llm-api/schema"
)

// Server is the OpenAI-compatible proxy gateway. Requests are dispatched through
// a router.Router, so model groups get load balancing, retries and fallbacks.
// When a master key is configured, data endpoints require a virtual API key and
// spend is tracked and budget-enforced per key.
type Server struct {
	cfg     *Config
	router  *router.Router
	keys    *auth.Store
	handler http.Handler
}

// New builds a Server from the configuration.
func New(cfg *Config) *Server {
	if len(cfg.ModelCost) > 0 {
		cost.RegisterAll(cfg.ModelCost)
	}
	deployments := make([]router.Deployment, 0, len(cfg.ModelList))
	for _, entry := range cfg.ModelList {
		deployments = append(deployments, router.Deployment{
			ModelName:         entry.ModelName,
			Model:             entry.Params.Model,
			APIKey:            entry.Params.APIKey,
			APIBase:           entry.Params.APIBase,
			CustomLLMProvider: entry.Params.CustomLLMProvider,
			Weight:            entry.Params.Weight,
		})
	}
	s := &Server{
		cfg:    cfg,
		router: router.New(deployments, cfg.Router.toRouter()),
		keys:   auth.NewStore(ResolveSecret(cfg.GeneralSettings.MasterKey)),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleModels)
	mux.HandleFunc("GET /spend", s.handleSpend)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /key/generate", s.handleKeyGenerate)
	mux.HandleFunc("POST /key/update", s.handleKeyUpdate)
	mux.HandleFunc("POST /key/delete", s.handleKeyDelete)
	mux.HandleFunc("GET /key/info", s.handleKeyInfo)
	mux.HandleFunc("GET /key/list", s.handleKeyList)
	s.handler = mux
	return s
}

// Handler returns the root HTTP handler (useful for tests and embedding).
func (s *Server) Handler() http.Handler { return s.handler }

// ListenAndServe starts the gateway on the configured address.
func (s *Server) ListenAndServe() error {
	return http.ListenAndServe(s.cfg.Addr(), s.handler)
}

// chatRequest is the incoming OpenAI-shaped chat completion request.
type chatRequest struct {
	Model               string           `json:"model"`
	Messages            []schema.Message `json:"messages"`
	Stream              bool             `json:"stream"`
	Temperature         *float64         `json:"temperature"`
	TopP                *float64         `json:"top_p"`
	MaxTokens           *int             `json:"max_tokens"`
	MaxCompletionTokens *int             `json:"max_completion_tokens"`
	N                   *int             `json:"n"`
	Stop                any              `json:"stop"`
	Tools               []schema.Tool    `json:"tools"`
	ToolChoice          any              `json:"tool_choice"`
	PresencePenalty     *float64         `json:"presence_penalty"`
	FrequencyPenalty    *float64         `json:"frequency_penalty"`
	ResponseFormat      any              `json:"response_format"`
	ReasoningEffort     string           `json:"reasoning_effort"`
	User                string           `json:"user"`
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if !s.router.HasModel(req.Model) {
		writeError(w, http.StatusNotFound, "model not found in model_list: "+req.Model)
		return
	}
	key, err := s.keys.Authorize(bearerToken(r), req.Model)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	opts := requestOptions(&req)

	if req.Stream {
		s.streamChat(w, r, req.Model, req.Messages, opts, key)
		return
	}
	resp, err := s.router.Completion(r.Context(), req.Model, req.Messages, opts...)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	s.recordSpend(key, req.Model, resp.Cost, usageTotal(resp.Usage))
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, model string, messages []schema.Message, opts []llmapi.Option, key *auth.Key) {
	stream, err := s.router.CompletionStream(r.Context(), model, messages, opts...)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	defer stream.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	var lastUsage *schema.Usage
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			continue
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher != nil {
			flusher.Flush()
		}
	}
	if err := stream.Err(); err != nil {
		data, _ := json.Marshal(errorBody(err))
		fmt.Fprintf(w, "data: %s\n\n", data)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	if lastUsage != nil {
		s.recordSpend(key, model, cost.UsageCost(model, lastUsage), usageTotal(lastUsage))
	}
}

// recordSpend attributes a call's cost and token usage to the calling virtual
// key (when present) and to the router's per-key spend breakdown.
func (s *Server) recordSpend(key *auth.Key, model string, amount float64, tokens int) {
	if key == nil {
		return
	}
	s.keys.AddSpend(key.Token, amount)
	s.keys.AddTokens(key.Token, tokens)
	s.router.SpendTracker().AddForKey(key.Token, model, amount)
}

func usageTotal(u *schema.Usage) int {
	if u == nil {
		return 0
	}
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.PromptTokens + u.CompletionTokens
}

// bearerToken extracts the API key from Authorization: Bearer <key> or x-api-key.
func bearerToken(r *http.Request) string {
	if h := r.Header.Get("Authorization"); h != "" {
		if rest, ok := strings.CutPrefix(h, "Bearer "); ok {
			return strings.TrimSpace(rest)
		}
		return strings.TrimSpace(h)
	}
	return r.Header.Get("x-api-key")
}

// requestOptions translates the incoming request into SDK options. Backend
// credentials/routing come from the router's deployment, not the request.
func requestOptions(req *chatRequest) []llmapi.Option {
	var opts []llmapi.Option
	if req.Temperature != nil {
		opts = append(opts, llmapi.WithTemperature(*req.Temperature))
	}
	if req.TopP != nil {
		opts = append(opts, llmapi.WithTopP(*req.TopP))
	}
	if req.MaxTokens != nil {
		opts = append(opts, llmapi.WithMaxTokens(*req.MaxTokens))
	}
	if req.MaxCompletionTokens != nil {
		opts = append(opts, llmapi.WithMaxCompletionTokens(*req.MaxCompletionTokens))
	}
	if req.N != nil {
		opts = append(opts, llmapi.WithN(*req.N))
	}
	if req.PresencePenalty != nil {
		opts = append(opts, llmapi.WithPresencePenalty(*req.PresencePenalty))
	}
	if req.FrequencyPenalty != nil {
		opts = append(opts, llmapi.WithFrequencyPenalty(*req.FrequencyPenalty))
	}
	if req.ResponseFormat != nil {
		opts = append(opts, llmapi.WithResponseFormat(req.ResponseFormat))
	}
	if req.ReasoningEffort != "" {
		opts = append(opts, llmapi.WithReasoningEffort(req.ReasoningEffort))
	}
	if req.User != "" {
		opts = append(opts, llmapi.WithUser(req.User))
	}
	if len(req.Tools) > 0 {
		opts = append(opts, llmapi.WithTools(req.Tools...))
	}
	if req.ToolChoice != nil {
		opts = append(opts, llmapi.WithToolChoice(req.ToolChoice))
	}
	if stop := normalizeStop(req.Stop); len(stop) > 0 {
		opts = append(opts, llmapi.WithStop(stop...))
	}
	return opts
}

func normalizeStop(stop any) []string {
	switch v := stop.(type) {
	case string:
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return v
	}
	return nil
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model          string `json:"model"`
		Input          any    `json:"input"`
		Dimensions     *int   `json:"dimensions"`
		EncodingFormat string `json:"encoding_format"`
		User           string `json:"user"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if !s.router.HasModel(req.Model) {
		writeError(w, http.StatusNotFound, "model not found in model_list: "+req.Model)
		return
	}
	key, err := s.keys.Authorize(bearerToken(r), req.Model)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	var opts []llmapi.Option
	if req.Dimensions != nil {
		opts = append(opts, llmapi.WithDimensions(*req.Dimensions))
	}
	if req.EncodingFormat != "" {
		opts = append(opts, llmapi.WithEncodingFormat(req.EncodingFormat))
	}
	if req.User != "" {
		opts = append(opts, llmapi.WithUser(req.User))
	}
	resp, err := s.router.Embedding(r.Context(), req.Model, req.Input, opts...)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	tokens := 0
	if resp.Usage != nil {
		tokens = resp.Usage.TotalTokens
	}
	s.recordSpend(key, req.Model, resp.Cost, tokens)
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	list := schema.ModelList{Object: "list"}
	now := time.Now().Unix()
	for _, name := range s.router.Models() {
		list.Data = append(list.Data, schema.ModelInfo{
			ID: name, Object: "model", Created: now, OwnedBy: "llm-api",
		})
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

// handleSpend reports accumulated spend tracked by the router (litellm /spend).
// When auth is enabled it requires the master key.
func (s *Server) handleSpend(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	writeJSON(w, http.StatusOK, s.router.Spend())
}

// --- response helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
			"code":    status,
		},
	})
}

func writeLLMError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	if e, ok := schema.AsError(err); ok && e.StatusCode > 0 {
		status = e.StatusCode
	}
	writeJSON(w, status, errorBody(err))
}

func errorBody(err error) map[string]any {
	errType := "api_error"
	if e, ok := schema.AsError(err); ok {
		errType = string(e.Kind)
	}
	return map[string]any{
		"error": map[string]any{
			"message": err.Error(),
			"type":    errType,
		},
	}
}
