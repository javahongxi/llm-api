// Package deepseek provides the DeepSeek backend (api.deepseek.com). DeepSeek is
// OpenAI-compatible; its reasoner models emit reasoning_content, which the shared
// transformation normalizes. Matches litellm's DeepSeekChatConfig.
package deepseek

import "github.com/javahongxi/llm-api/providers/openaicompat"

// New returns the DeepSeek provider.
func New() *openaicompat.Provider {
	return &openaicompat.Provider{
		ProviderName: "deepseek",
		Base:         "https://api.deepseek.com/v1",
		EnvKeys:      []string{"DEEPSEEK_API_KEY"},
		Embeddings:   false,
	}
}
