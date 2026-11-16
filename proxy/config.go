// Package proxy implements an OpenAI-compatible HTTP gateway on top of the
// llmapi SDK, mirroring litellm's Proxy Server. A model_list maps public model
// names to backend provider/model + credentials; incoming /v1/chat/completions,
// /v1/embeddings and /v1/models requests are routed through the unified SDK.
package proxy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/javahongxi/llm-api/cost"
	"github.com/javahongxi/llm-api/router"
	"gopkg.in/yaml.v3"
)

// Config is the proxy configuration (litellm proxy config.yaml shape).
type Config struct {
	ModelList []ModelEntry              `yaml:"model_list" json:"model_list"`
	Server    ServerSettings            `yaml:"server" json:"server"`
	Router    RouterSettings            `yaml:"router_settings" json:"router_settings"`
	ModelCost map[string]cost.ModelCost `yaml:"model_cost" json:"model_cost"`
}

// RouterSettings mirrors litellm's router_settings block.
type RouterSettings struct {
	NumRetries      int                   `yaml:"num_retries" json:"num_retries"`
	RoutingStrategy string                `yaml:"routing_strategy" json:"routing_strategy"`
	Timeout         float64               `yaml:"timeout" json:"timeout"` // seconds, per attempt
	AllowedFails    int                   `yaml:"allowed_fails" json:"allowed_fails"`
	CooldownTime    float64               `yaml:"cooldown_time" json:"cooldown_time"` // seconds
	RetryPolicy     *router.RetryPolicy   `yaml:"retry_policy" json:"retry_policy"`
	Fallbacks       []map[string][]string `yaml:"fallbacks" json:"fallbacks"`
}

// toRouter converts the proxy router_settings into router.Settings.
func (s RouterSettings) toRouter() router.Settings {
	settings := router.Settings{
		NumRetries:   s.NumRetries,
		Strategy:     router.RoutingStrategy(s.RoutingStrategy),
		AllowedFails: s.AllowedFails,
		RetryPolicy:  s.RetryPolicy,
	}
	if s.Timeout > 0 {
		settings.Timeout = time.Duration(s.Timeout * float64(time.Second))
	}
	if s.CooldownTime > 0 {
		settings.CooldownTime = time.Duration(s.CooldownTime * float64(time.Second))
	}
	if len(s.Fallbacks) > 0 {
		settings.Fallbacks = map[string][]string{}
		for _, rule := range s.Fallbacks {
			for model, targets := range rule {
				settings.Fallbacks[model] = append(settings.Fallbacks[model], targets...)
			}
		}
	}
	return settings
}

// ServerSettings controls the listen address.
type ServerSettings struct {
	Host string `yaml:"host" json:"host"`
	Port int    `yaml:"port" json:"port"`
}

// ModelEntry maps a public model_name to backend litellm_params.
type ModelEntry struct {
	ModelName string `yaml:"model_name" json:"model_name"`
	Params    Params `yaml:"litellm_params" json:"litellm_params"`
}

// Params are the backend routing parameters (litellm litellm_params).
type Params struct {
	// Model is the SDK model string, e.g. "openai/gpt-4o",
	// "anthropic/claude-3-5-sonnet-latest", "gemini/gemini-1.5-pro".
	Model string `yaml:"model" json:"model"`
	// APIKey is the credential; "os.environ/VAR" reads an environment variable.
	APIKey string `yaml:"api_key" json:"api_key"`
	// APIBase optionally overrides the provider endpoint.
	APIBase string `yaml:"api_base" json:"api_base"`
	// CustomLLMProvider optionally forces the backend provider.
	CustomLLMProvider string `yaml:"custom_llm_provider" json:"custom_llm_provider"`
	// Weight optionally biases simple-shuffle load balancing for this deployment.
	Weight float64 `yaml:"weight" json:"weight"`
}

// LoadConfig reads a JSON or YAML config file (YAML for .yaml/.yml).
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	cfg := &Config{}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse yaml config: %w", err)
		}
	default:
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse json config: %w", err)
		}
	}
	if cfg.Server.Port == 0 {
		cfg.Server.Port = 4000 // litellm proxy default port
	}
	return cfg, nil
}

// ResolveSecret resolves "os.environ/VAR" references to environment values;
// any other value is returned unchanged (litellm secret manager behaviour).
func ResolveSecret(value string) string {
	if rest, ok := strings.CutPrefix(value, "os.environ/"); ok {
		return os.Getenv(rest)
	}
	return value
}

// Addr returns the configured listen address.
func (c *Config) Addr() string {
	host := c.Server.Host
	if host == "" {
		host = "0.0.0.0"
	}
	return fmt.Sprintf("%s:%d", host, c.Server.Port)
}
