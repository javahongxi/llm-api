// Package cost computes per-call LLM spend from token usage and a model price
// table, and accumulates spend over time. It is the Go analogue of litellm's
// cost_calculator (completion_cost / cost_per_token) and spend tracking.
//
// Prices are USD per token. A built-in table covers common models; override or
// extend it with Register (or the proxy's model_cost config). Unknown models
// cost 0.
package cost

import (
	"strings"
	"sync"

	"github.com/javahongxi/llm-api/schema"
)

// ModelCost holds the USD price per token for a model.
type ModelCost struct {
	InputCostPerToken  float64 `yaml:"input_cost_per_token" json:"input_cost_per_token"`
	OutputCostPerToken float64 `yaml:"output_cost_per_token" json:"output_cost_per_token"`
}

// defaultPrices is a built-in price table (USD per token) for common models.
// Values are approximate public list prices and can be overridden via Register.
var defaultPrices = map[string]ModelCost{
	// OpenAI
	"gpt-6-astra":            {1e-5, 5e-5},
	"gpt-4o":                 {2.5e-6, 1e-5},
	"gpt-4o-2024-08-06":      {2.5e-6, 1e-5},
	"gpt-4o-2024-05-13":      {5e-6, 1.5e-5},
	"gpt-4o-mini":            {1.5e-7, 6e-7},
	"gpt-4o-mini-2024-07-18": {1.5e-7, 6e-7},
	"gpt-4-turbo":            {1e-5, 3e-5},
	"gpt-4":                  {3e-5, 6e-5},
	"gpt-3.5-turbo":          {5e-7, 1.5e-6},
	"o1":                     {1.5e-5, 6e-5},
	"o1-mini":                {3e-6, 1.2e-5},
	"o3-mini":                {1.1e-6, 4.4e-6},
	"text-embedding-3-small": {2e-8, 0},
	"text-embedding-3-large": {1.3e-7, 0},
	"text-embedding-ada-002": {1e-7, 0},

	// Anthropic
	"claude-opus-5-5":            {4e-6, 2e-5},
	"claude-fable-5-2":           {8e-6, 4e-5},
	"claude-3-5-sonnet-latest":   {3e-6, 1.5e-5},
	"claude-3-5-sonnet-20241022": {3e-6, 1.5e-5},
	"claude-3-5-sonnet-20240620": {3e-6, 1.5e-5},
	"claude-sonnet-4-5":          {3e-6, 1.5e-5},
	"claude-3-opus-20240229":     {1.5e-5, 7.5e-5},
	"claude-3-haiku-20240307":    {2.5e-7, 1.25e-6},

	// Gemini
	"gemini-4-carbon":  {2e-6, 1e-5}, // intro price until 2026-11-30; then {4e-6, 2e-5}
	"gemini-1.5-pro":   {1.25e-6, 5e-6},
	"gemini-1.5-flash": {7.5e-8, 3e-7},
	"gemini-2.0-flash": {1e-7, 4e-7},
	"gemini-1.0-pro":   {5e-7, 1.5e-6},

	// DeepSeek
	"deepseek-chat":     {2.8e-7, 4.2e-7},
	"deepseek-reasoner": {5.5e-7, 2.19e-6},
	"deepseek-v4-flash": {3e-7, 1.2e-6},

	// DashScope
	"qwen-plus":         {4e-7, 1.2e-6},
	"qwen-turbo":        {5e-8, 2e-7},
	"qwen-max":          {1.6e-6, 6.4e-6},
	"qwen3.7-plus":      {4e-7, 1.2e-6},
	"qwen4-max":         {1.5e-6, 4.5e-6}, // QwenCloud international list price
	"qwen3.8-flash":     {1.5e-7, 4.7e-7}, // QwenCloud international list price
	"text-embedding-v3": {7e-8, 0},
	"text-embedding-v4": {7e-8, 0},
}

var (
	mu     sync.RWMutex
	custom = map[string]ModelCost{}
)

// Register sets a custom price for a model, overriding the built-in table. It is
// safe for concurrent use. The model may be bare ("gpt-4o") or prefixed
// ("openai/gpt-4o").
func Register(model string, c ModelCost) {
	mu.Lock()
	custom[model] = c
	mu.Unlock()
}

// RegisterAll merges a price table (e.g. from proxy config) into the registry.
func RegisterAll(prices map[string]ModelCost) {
	mu.Lock()
	for model, c := range prices {
		custom[model] = c
	}
	mu.Unlock()
}

// Lookup returns the price for a model. Custom registrations win over the
// built-in table; a "provider/model" prefix is stripped if the full name is not
// found. The boolean reports whether a price is known.
func Lookup(model string) (ModelCost, bool) {
	mu.RLock()
	c, ok := custom[model]
	mu.RUnlock()
	if ok {
		return c, true
	}
	if c, ok := defaultPrices[model]; ok {
		return c, true
	}
	if bare := stripProvider(model); bare != model {
		mu.RLock()
		c, ok := custom[bare]
		mu.RUnlock()
		if ok {
			return c, true
		}
		if c, ok := defaultPrices[bare]; ok {
			return c, true
		}
	}
	return ModelCost{}, false
}

func stripProvider(model string) string {
	if _, rest, ok := strings.Cut(model, "/"); ok {
		return rest
	}
	return model
}

// Cost computes the USD cost for a call given prompt/completion token counts.
// Unknown models cost 0.
func Cost(model string, promptTokens, completionTokens int) float64 {
	c, ok := Lookup(model)
	if !ok {
		return 0
	}
	return float64(promptTokens)*c.InputCostPerToken + float64(completionTokens)*c.OutputCostPerToken
}

// UsageCost computes the cost from a chat completion Usage (nil-safe).
func UsageCost(model string, u *schema.Usage) float64 {
	if u == nil {
		return 0
	}
	return Cost(model, u.PromptTokens, u.CompletionTokens)
}
