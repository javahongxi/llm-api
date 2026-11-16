package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	llmapi "github.com/javahongxi/llm-api"
	"github.com/javahongxi/llm-api/cost"
	"github.com/javahongxi/llm-api/router"
	"github.com/javahongxi/llm-api/schema"
)

// Server is the OpenAI-compatible proxy gateway. Requests are dispatched through
// a router.Router, so model groups get load balancing, retries and fallbacks.
type Server struct {
	cfg     *Config
	router  *router.Router
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
	s := &Server{cfg: cfg, router: router.New(deployments, cfg.Router.toRouter())}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleModels)
	mux.HandleFunc("GET /spend", s.handleSpend)
	mux.HandleFunc("GET /health", s.handleHealth)
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
	opts := requestOptions(&req)

	if req.Stream {
		s.streamChat(w, r, req.Model, req.Messages, opts)
		return
	}
	resp, err := s.router.Completion(r.Context(), req.Model, req.Messages, opts...)
	if err != nil {
		writeLLMError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, model string, messages []schema.Message, opts []llmapi.Option) {
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

	for stream.Next() {
		data, err := json.Marshal(stream.Current())
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
func (s *Server) handleSpend(w http.ResponseWriter, r *http.Request) {
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
