// Package llmapi is a Go port of litellm: one unified, OpenAI-shaped API to call
// many LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, DashScope). Call
// Completion / CompletionStream / Embedding with a model name and messages; the
// package resolves the provider, translates the request, and normalizes the
// response and errors to a single shape.
//
//	resp, err := llmapi.Completion(ctx, "gpt-4o", []llmapi.Message{
//	    llmapi.UserMessage("Hello!"),
//	}, llmapi.WithAPIKey("sk-..."))
//
// Model names may carry a provider prefix ("anthropic/claude-3-5-sonnet-latest",
// "gemini/gemini-1.5-pro", "deepseek/deepseek-chat", "dashscope/qwen-plus"), or
// the provider is inferred from the model name, mirroring litellm's
// get_llm_provider.
package llmapi

import (
	"os"
	"strings"

	"github.com/javahongxi/llm-api/providers/anthropic"
	"github.com/javahongxi/llm-api/providers/dashscope"
	"github.com/javahongxi/llm-api/providers/deepseek"
	"github.com/javahongxi/llm-api/providers/gemini"
	"github.com/javahongxi/llm-api/providers/openai"
	"github.com/javahongxi/llm-api/schema"
)

// registry holds the known providers keyed by name.
var registry = map[string]schema.Provider{}

func init() {
	Register(openai.New())
	Register(anthropic.New())
	Register(gemini.New())
	Register(deepseek.New())
	Register(dashscope.New())
}

// Register adds a provider to the registry, enabling custom backends. It is safe
// to call at init time to extend llm-api with new providers.
func Register(p schema.Provider) { registry[p.Name()] = p }

// GetProvider looks up a registered provider by name.
func GetProvider(name string) (schema.Provider, bool) {
	p, ok := registry[name]
	return p, ok
}

// Providers returns the names of all registered providers.
func Providers() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	return names
}

// resolveProvider mirrors litellm's get_llm_provider: an explicit provider wins,
// then a "provider/model" prefix, then inference from the model name. It returns
// the provider and the model name with any provider prefix stripped.
func resolveProvider(model, explicit string) (schema.Provider, string, error) {
	if explicit != "" {
		p, ok := registry[explicit]
		if !ok {
			return nil, "", schema.NewBadRequestError("unknown provider: "+explicit, explicit, model)
		}
		return p, stripPrefix(model, explicit), nil
	}

	if prefix, rest, ok := strings.Cut(model, "/"); ok {
		if p, registered := registry[prefix]; registered {
			return p, rest, nil
		}
		return nil, "", schema.NewBadRequestError("unsupported provider prefix: "+prefix, prefix, model)
	}

	name := inferProvider(model)
	return registry[name], model, nil
}

func stripPrefix(model, provider string) string {
	if rest, ok := strings.CutPrefix(model, provider+"/"); ok {
		return rest
	}
	return model
}

// inferProvider guesses the provider from a bare model name (litellm's model
// pattern matching). Unknown models default to OpenAI, as litellm does.
func inferProvider(model string) string {
	m := strings.ToLower(model)
	switch {
	case strings.HasPrefix(m, "claude"):
		return "anthropic"
	case strings.HasPrefix(m, "gemini"):
		return "gemini"
	case strings.HasPrefix(m, "deepseek"):
		return "deepseek"
	case strings.HasPrefix(m, "qwen"), strings.HasPrefix(m, "qwq"), strings.HasPrefix(m, "qvq"),
		strings.HasPrefix(m, "text-embedding-v"):
		return "dashscope"
	default:
		return "openai"
	}
}

// resolveDefaults fills APIBase and APIKey from the provider defaults and
// environment when the caller did not supply them (litellm's get_api_base /
// get_api_key / validate_environment).
func resolveDefaults(p schema.Provider, c *schema.CallConfig) {
	if c.APIBase == "" {
		c.APIBase = p.DefaultAPIBase()
	}
	if c.APIKey == "" {
		for _, env := range p.APIKeyEnvVars() {
			if v := os.Getenv(env); v != "" {
				c.APIKey = v
				break
			}
		}
	}
}
