package llmapi

import (
	"net/http"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// Option configures a single call. Options are the Go analogue of litellm's
// completion(**kwargs): sampling parameters, tools, credentials and HTTP
// behaviour are all supplied as options.
type Option func(*schema.CallConfig)

func buildConfig(opts []Option) *schema.CallConfig {
	c := &schema.CallConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(c)
		}
	}
	return c
}

// WithAPIKey sets the provider credential (overrides environment).
func WithAPIKey(key string) Option { return func(c *schema.CallConfig) { c.APIKey = key } }

// WithAPIBase overrides the provider endpoint base URL.
func WithAPIBase(base string) Option { return func(c *schema.CallConfig) { c.APIBase = base } }

// WithProvider explicitly selects the backend (litellm custom_llm_provider).
func WithProvider(name string) Option { return func(c *schema.CallConfig) { c.Provider = name } }

// WithHTTPClient sets the HTTP client used for the call.
func WithHTTPClient(client *http.Client) Option {
	return func(c *schema.CallConfig) { c.HTTPClient = client }
}

// WithTimeout bounds the whole call.
func WithTimeout(d time.Duration) Option { return func(c *schema.CallConfig) { c.Timeout = d } }

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option { return func(c *schema.CallConfig) { c.Temperature = &t } }

// WithTopP sets nucleus sampling.
func WithTopP(p float64) Option { return func(c *schema.CallConfig) { c.TopP = &p } }

// WithTopK sets top-k sampling (Anthropic / Gemini).
func WithTopK(k int) Option { return func(c *schema.CallConfig) { c.TopK = &k } }

// WithMaxTokens sets the maximum output tokens.
func WithMaxTokens(n int) Option { return func(c *schema.CallConfig) { c.MaxTokens = &n } }

// WithMaxCompletionTokens sets max_completion_tokens (OpenAI's newer parameter).
func WithMaxCompletionTokens(n int) Option {
	return func(c *schema.CallConfig) { c.MaxCompletionTokens = &n }
}

// WithN sets the number of choices to generate.
func WithN(n int) Option { return func(c *schema.CallConfig) { c.N = &n } }

// WithStop sets stop sequences.
func WithStop(stop ...string) Option { return func(c *schema.CallConfig) { c.Stop = stop } }

// WithTools supplies function tools the model may call.
func WithTools(tools ...Tool) Option { return func(c *schema.CallConfig) { c.Tools = tools } }

// WithToolChoice controls tool selection ("auto", "none", "required", or a
// specific function object).
func WithToolChoice(choice any) Option { return func(c *schema.CallConfig) { c.ToolChoice = choice } }

// WithParallelToolCalls enables/disables parallel tool calls.
func WithParallelToolCalls(b bool) Option {
	return func(c *schema.CallConfig) { c.ParallelToolCalls = &b }
}

// WithResponseFormat sets the response format (e.g. {"type":"json_object"}).
func WithResponseFormat(format any) Option {
	return func(c *schema.CallConfig) { c.ResponseFormat = format }
}

// WithReasoningEffort sets reasoning effort for reasoning models.
func WithReasoningEffort(effort string) Option {
	return func(c *schema.CallConfig) { c.ReasoningEffort = effort }
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(p float64) Option {
	return func(c *schema.CallConfig) { c.PresencePenalty = &p }
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(p float64) Option {
	return func(c *schema.CallConfig) { c.FrequencyPenalty = &p }
}

// WithSeed sets a deterministic sampling seed.
func WithSeed(seed int) Option { return func(c *schema.CallConfig) { c.Seed = &seed } }

// WithLogprobs enables returning log probabilities.
func WithLogprobs(b bool) Option { return func(c *schema.CallConfig) { c.Logprobs = &b } }

// WithTopLogprobs sets the number of top log probabilities to return.
func WithTopLogprobs(n int) Option { return func(c *schema.CallConfig) { c.TopLogprobs = &n } }

// WithUser sets an end-user identifier.
func WithUser(user string) Option { return func(c *schema.CallConfig) { c.User = user } }

// WithExtraBody merges provider-specific passthrough fields into the request
// body (litellm extra_body).
func WithExtraBody(body map[string]any) Option {
	return func(c *schema.CallConfig) { c.ExtraBody = body }
}

// WithExtraHeaders merges additional headers into the request.
func WithExtraHeaders(headers map[string]string) Option {
	return func(c *schema.CallConfig) { c.ExtraHeaders = headers }
}

// WithExtraQuery merges additional URL query parameters into the request.
func WithExtraQuery(query map[string]string) Option {
	return func(c *schema.CallConfig) { c.ExtraQuery = query }
}

// WithDimensions sets the embedding output dimensionality.
func WithDimensions(n int) Option { return func(c *schema.CallConfig) { c.Dimensions = &n } }

// WithEncodingFormat sets the embedding encoding format (e.g. "float").
func WithEncodingFormat(format string) Option {
	return func(c *schema.CallConfig) { c.EncodingFormat = format }
}
