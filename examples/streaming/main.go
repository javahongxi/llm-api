// Command streaming shows a streaming chat completion, printing tokens as they
// arrive. Works across providers; the SDK normalizes each backend's SSE format
// to unified OpenAI-shaped chunks.
//
//	go run ./examples/streaming
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	llmapi "github.com/javahongxi/llm-api"
)

func main() {
	ctx := context.Background()
	messages := []llmapi.Message{
		llmapi.UserMessage("写一首关于秋天的短诗。"),
	}

	stream, err := llmapi.CompletionStream(ctx, "dashscope/qwen4-max", messages)
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}
	fmt.Fprintln(os.Stdout)
}
