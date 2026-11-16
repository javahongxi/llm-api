// Command router shows load balancing, retries and fallbacks across multiple
// backend deployments grouped under one public model name. Credentials come from
// the environment (DASHSCOPE_API_KEY, DEEPSEEK_API_KEY).
//
//	go run ./examples/router
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	llmapi "github.com/javahongxi/llm-api"
	"github.com/javahongxi/llm-api/router"
)

func main() {
	ctx := context.Background()

	// Two different backends behind the public name "chat"; round-robin spreads
	// calls across them. If both fail, fall back to the "backup" group.
	r := router.New([]router.Deployment{
		{ModelName: "chat", Model: "dashscope/qwen3.7-plus", APIKey: "os.environ/DASHSCOPE_API_KEY"},
		{ModelName: "chat", Model: "deepseek/deepseek-v4-flash", APIKey: "os.environ/DEEPSEEK_API_KEY"},
		{ModelName: "backup", Model: "dashscope/qwen3.7-plus", APIKey: "os.environ/DASHSCOPE_API_KEY"},
	}, router.Settings{
		NumRetries: 2,
		Strategy:   router.RoundRobin,
		Timeout:    60 * time.Second,
		Fallbacks:  map[string][]string{"chat": {"backup"}},
	})

	messages := []llmapi.Message{llmapi.UserMessage("Reply with exactly: OK")}
	for i := 0; i < 4; i++ {
		resp, err := r.Completion(ctx, "chat", messages, llmapi.WithMaxTokens(20))
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("call %d: model=%s content=%q\n", i+1, resp.Model, resp.Choices[0].Message.Content)
	}
}
