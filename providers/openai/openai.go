// Package openai provides the OpenAI backend (api.openai.com). It reuses the
// OpenAI-compatible transformation, matching litellm's OpenAIGPTConfig.
package openai

import "github.com/javahongxi/llm-api/providers/openaicompat"

// New returns the OpenAI provider.
func New() *openaicompat.Provider {
	return &openaicompat.Provider{
		ProviderName: "openai",
		Base:         "https://api.openai.com/v1",
		EnvKeys:      []string{"OPENAI_API_KEY"},
		Embeddings:   true,
	}
}
