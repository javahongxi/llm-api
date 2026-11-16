// Package router is the Go analogue of litellm's Router: it load-balances calls
// across multiple deployments of a model group, retries transient failures with
// exponential backoff, cools down failing deployments, and falls back to other
// model groups when a group is exhausted. It is built on top of the llmapi SDK.
//
//	r := router.New([]router.Deployment{
//	    {ModelName: "gpt-4o", Model: "openai/gpt-4o", APIKey: "os.environ/OPENAI_API_KEY"},
//	    {ModelName: "gpt-4o", Model: "azure/gpt-4o", APIKey: "...", APIBase: "..."},
//	}, router.Settings{NumRetries: 3, Fallbacks: map[string][]string{"gpt-4o": {"claude"}}})
//	resp, err := r.Completion(ctx, "gpt-4o", messages)
package router

import (
	"context"
	"math/rand"
	"sort"
	"sync"
	"time"

	llmapi "github.com/javahongxi/llm-api"
	"github.com/javahongxi/llm-api/cost"
	"github.com/javahongxi/llm-api/schema"
)

// RoutingStrategy selects how a deployment is chosen within a model group. The
// values mirror litellm's routing_strategy names.
type RoutingStrategy string

const (
	// SimpleShuffle picks a deployment at random, honouring weights (default).
	SimpleShuffle RoutingStrategy = "simple-shuffle"
	// RoundRobin cycles through healthy deployments in order.
	RoundRobin RoutingStrategy = "round-robin"
	// LeastBusy picks the deployment with the fewest in-flight requests.
	LeastBusy RoutingStrategy = "least-busy"
	// LatencyBased picks the deployment with the lowest observed latency.
	LatencyBased RoutingStrategy = "latency-based-routing"
)

// Deployment is one backend behind a public model group name (litellm's
// model_list entry). ModelName is what callers request; Model is the SDK model
// string actually dispatched (e.g. "openai/gpt-4o", "anthropic/claude-...").
type Deployment struct {
	ModelName         string  `yaml:"model_name" json:"model_name"`
	Model             string  `yaml:"model" json:"model"`
	APIKey            string  `yaml:"api_key" json:"api_key"`
	APIBase           string  `yaml:"api_base" json:"api_base"`
	CustomLLMProvider string  `yaml:"custom_llm_provider" json:"custom_llm_provider"`
	Weight            float64 `yaml:"weight" json:"weight"`
	ID                string  `yaml:"id" json:"id"`
}

// RetryPolicy overrides the number of retries per error class (litellm's
// RetryPolicy). A nil field falls back to Settings.NumRetries for retryable
// errors, or to DefaultRetries when set.
type RetryPolicy struct {
	BadRequestErrorRetries             *int `yaml:"BadRequestErrorRetries" json:"BadRequestErrorRetries"`
	AuthenticationErrorRetries         *int `yaml:"AuthenticationErrorRetries" json:"AuthenticationErrorRetries"`
	TimeoutErrorRetries                *int `yaml:"TimeoutErrorRetries" json:"TimeoutErrorRetries"`
	RateLimitErrorRetries              *int `yaml:"RateLimitErrorRetries" json:"RateLimitErrorRetries"`
	ContentPolicyViolationErrorRetries *int `yaml:"ContentPolicyViolationErrorRetries" json:"ContentPolicyViolationErrorRetries"`
	InternalServerErrorRetries         *int `yaml:"InternalServerErrorRetries" json:"InternalServerErrorRetries"`
	ServiceUnavailableErrorRetries     *int `yaml:"ServiceUnavailableErrorRetries" json:"ServiceUnavailableErrorRetries"`
	NotFoundErrorRetries               *int `yaml:"NotFoundErrorRetries" json:"NotFoundErrorRetries"`
	DefaultRetries                     *int `yaml:"DefaultRetries" json:"DefaultRetries"`
}

// Settings configures routing, retries, cooldown and fallbacks.
type Settings struct {
	// Strategy is the load-balancing strategy (default SimpleShuffle).
	Strategy RoutingStrategy
	// NumRetries is the default retry count for retryable errors.
	NumRetries int
	// RetryPolicy optionally overrides retries per error class.
	RetryPolicy *RetryPolicy
	// Timeout bounds each individual attempt (0 = no per-attempt timeout).
	Timeout time.Duration
	// RetryBackoffBase is the base delay for exponential backoff (default 500ms).
	RetryBackoffBase time.Duration
	// MaxRetryBackoff caps the backoff delay (default 8s).
	MaxRetryBackoff time.Duration
	// AllowedFails is the number of failures before a deployment is cooled down
	// (default 1).
	AllowedFails int
	// CooldownTime is how long a failing deployment stays unavailable
	// (default 5s).
	CooldownTime time.Duration
	// Fallbacks maps a model group to ordered fallback groups tried when the
	// primary group exhausts its retries (litellm fallbacks).
	Fallbacks map[string][]string
}

func (s *Settings) applyDefaults() {
	if s.Strategy == "" {
		s.Strategy = SimpleShuffle
	}
	if s.AllowedFails <= 0 {
		s.AllowedFails = 1
	}
	if s.CooldownTime <= 0 {
		s.CooldownTime = 5 * time.Second
	}
	if s.RetryBackoffBase <= 0 {
		s.RetryBackoffBase = 500 * time.Millisecond
	}
	if s.MaxRetryBackoff <= 0 {
		s.MaxRetryBackoff = 8 * time.Second
	}
}

// Router dispatches calls across deployment groups with retries and fallbacks.
// It is safe for concurrent use.
type Router struct {
	groups   map[string][]*deployment
	settings Settings
	spend    *cost.SpendTracker

	mu  sync.Mutex
	rng *rand.Rand
	rr  map[string]uint64
}

// New builds a Router from deployments and settings, grouping deployments by
// ModelName.
func New(deployments []Deployment, settings Settings) *Router {
	settings.applyDefaults()
	r := &Router{
		groups:   map[string][]*deployment{},
		settings: settings,
		spend:    cost.NewSpendTracker(),
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
		rr:       map[string]uint64{},
	}
	for _, d := range deployments {
		r.groups[d.ModelName] = append(r.groups[d.ModelName], &deployment{Deployment: d})
	}
	return r
}

// Models returns the sorted public model group names.
func (r *Router) Models() []string {
	names := make([]string, 0, len(r.groups))
	for name := range r.groups {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasModel reports whether a model group is configured.
func (r *Router) HasModel(name string) bool {
	_, ok := r.groups[name]
	return ok
}

// Spend returns a snapshot of accumulated spend (USD) across successful calls.
func (r *Router) Spend() cost.SpendReport { return r.spend.Report() }

// SpendTracker exposes the underlying tracker for attributed spend recording.
func (r *Router) SpendTracker() *cost.SpendTracker { return r.spend }

// Completion routes a non-streaming chat completion through the model group,
// retrying and falling back as configured.
func (r *Router) Completion(ctx context.Context, model string, messages []llmapi.Message, opts ...llmapi.Option) (*llmapi.ModelResponse, error) {
	resp, err := execute(r, ctx, model, opts, func(ctx context.Context, d *deployment, o []llmapi.Option) (*llmapi.ModelResponse, error) {
		return llmapi.Completion(ctx, d.Model, messages, o...)
	})
	if err == nil {
		r.spend.Add(model, resp.Cost)
	}
	return resp, err
}

// CompletionStream routes a streaming chat completion. Retries apply while
// establishing the stream; once streaming begins, mid-stream errors surface via
// Stream.Err.
func (r *Router) CompletionStream(ctx context.Context, model string, messages []llmapi.Message, opts ...llmapi.Option) (*llmapi.Stream, error) {
	return execute(r, ctx, model, opts, func(ctx context.Context, d *deployment, o []llmapi.Option) (*llmapi.Stream, error) {
		return llmapi.CompletionStream(ctx, d.Model, messages, o...)
	})
}

// Embedding routes an embedding request through the model group.
func (r *Router) Embedding(ctx context.Context, model string, input any, opts ...llmapi.Option) (*llmapi.EmbeddingResponse, error) {
	resp, err := execute(r, ctx, model, opts, func(ctx context.Context, d *deployment, o []llmapi.Option) (*llmapi.EmbeddingResponse, error) {
		return llmapi.Embedding(ctx, d.Model, input, o...)
	})
	if err == nil {
		r.spend.Add(model, resp.Cost)
	}
	return resp, err
}

// execute runs the group -> fallback chain. For each group it retries up to the
// allowed count, picking a (healthy) deployment per attempt.
func execute[T any](r *Router, ctx context.Context, model string, opts []llmapi.Option, call func(context.Context, *deployment, []llmapi.Option) (T, error)) (T, error) {
	var zero T
	chain := append([]string{model}, r.settings.Fallbacks[model]...)

	var lastErr error
	tried := false
	for _, group := range chain {
		deps := r.groups[group]
		if len(deps) == 0 {
			continue
		}
		tried = true
		res, err := tryGroup(r, ctx, group, deps, opts, call)
		if err == nil {
			return res, nil
		}
		lastErr = err
		if ctx.Err() != nil {
			break
		}
	}
	if !tried {
		return zero, schema.NewBadRequestError("model not found in router model_list: "+model, "router", model)
	}
	return zero, lastErr
}

// tryGroup attempts a single group with retries and backoff.
func tryGroup[T any](r *Router, ctx context.Context, group string, deps []*deployment, opts []llmapi.Option, call func(context.Context, *deployment, []llmapi.Option) (T, error)) (T, error) {
	var zero T
	var lastErr error
	for attempt := 0; ; attempt++ {
		d := r.pick(group, deps)
		merged := append(deploymentOptions(d), opts...)

		attemptCtx := ctx
		var cancel context.CancelFunc
		if r.settings.Timeout > 0 {
			attemptCtx, cancel = context.WithTimeout(ctx, r.settings.Timeout)
		}
		r.begin(d)
		start := time.Now()
		res, err := call(attemptCtx, d, merged)
		r.end(d, time.Since(start), err)
		if cancel != nil {
			cancel()
		}
		if err == nil {
			return res, nil
		}
		lastErr = err

		allowed := r.retriesFor(err)
		if attempt >= allowed || ctx.Err() != nil {
			return zero, lastErr
		}
		if !r.backoff(ctx, attempt) {
			return zero, lastErr
		}
	}
}
