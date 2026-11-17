// Command completion shows a basic non-streaming chat completion. The provider
// is inferred from the model name (or set a "provider/model" prefix). Credentials
// come from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
//
//	go run ./examples/completion
package main

import (
	"context"
	"fmt"
	"log"

	llmapi "github.com/javahongxi/llm-api"
)

func main() {
	ctx := context.Background()
	messages := []llmapi.Message{
		llmapi.SystemMessage("You are a helpful assistant."),
		llmapi.UserMessage("用一句话介绍 Go 语言。"),
	}

	// Swap the model to try other providers, e.g. "anthropic/claude-opus-5-5",
	// "gemini/gemini-4-carbon", "deepseek/deepseek-v4-flash", "dashscope/qwen4-max".
	resp, err := llmapi.Completion(ctx, "dashscope/qwen4-max", messages,
		llmapi.WithTemperature(0.7),
		llmapi.WithMaxTokens(256),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Printf("usage: %+v\n", resp.Usage)
}
