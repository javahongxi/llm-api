// Command embedding shows creating embeddings. OpenAI and DashScope use the
// OpenAI-shaped /embeddings endpoint; Gemini uses batchEmbedContents — the SDK
// normalizes both to the same response.
//
//	go run ./examples/embedding
package main

import (
	"context"
	"fmt"
	"log"

	llmapi "github.com/javahongxi/llm-api"
)

func main() {
	ctx := context.Background()

	resp, err := llmapi.Embedding(ctx, "dashscope/text-embedding-v3",
		[]string{"Hello world", "你好世界"},
		llmapi.WithDimensions(256),
	)
	if err != nil {
		log.Fatal(err)
	}

	for _, item := range resp.Data {
		fmt.Printf("index=%d dim=%d first=%v\n", item.Index, len(item.Embedding), item.Embedding[0])
	}
}
