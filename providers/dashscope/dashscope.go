// Package dashscope provides Alibaba Cloud DashScope (Model Studio) via its
// OpenAI-compatible mode endpoint. Matches litellm's DashScopeChatConfig, which
// subclasses OpenAIGPTConfig and resolves the compatible-mode base URL.
package dashscope

import "github.com/javahongxi/llm-api/providers/openaicompat"

// New returns the DashScope provider.
func New() *openaicompat.Provider {
	return &openaicompat.Provider{
		ProviderName: "dashscope",
		Base:         "https://dashscope.aliyuncs.com/compatible-mode/v1",
		EnvKeys:      []string{"DASHSCOPE_API_KEY"},
		Embeddings:   true,
	}
}
