// Command tools shows a function-calling round trip: define a tool, let the
// model call it, execute it locally, and feed the result back. The tool-call
// shape is unified across providers.
//
//	go run ./examples/tools
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	llmapi "github.com/javahongxi/llm-api"
)

func main() {
	ctx := context.Background()

	weatherTool := llmapi.NewFunctionTool("get_weather", "Get the current weather for a city",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "City name"},
			},
			"required": []string{"city"},
		})

	messages := []llmapi.Message{
		llmapi.UserMessage("北京今天天气怎么样？"),
	}

	resp, err := llmapi.Completion(ctx, "gpt-4o", messages,
		llmapi.WithTools(weatherTool),
		llmapi.WithToolChoice("auto"),
	)
	if err != nil {
		log.Fatal(err)
	}

	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) == 0 {
		fmt.Println(msg.Content)
		return
	}

	// Append the assistant's tool-call message, then answer each tool call.
	messages = append(messages, msg)
	for _, call := range msg.ToolCalls {
		fmt.Printf("model called %s(%s)\n", call.Function.Name, call.Function.Arguments)
		result := runTool(call.Function.Name, call.Function.Arguments)
		messages = append(messages, llmapi.ToolMessage(call.ID, result))
	}

	final, err := llmapi.Completion(ctx, "gpt-4o", messages, llmapi.WithTools(weatherTool))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(final.Choices[0].Message.Content)
}

// runTool is a stub local implementation of the tool.
func runTool(name, arguments string) string {
	var args struct {
		City string `json:"city"`
	}
	_ = json.Unmarshal([]byte(arguments), &args)
	if name == "get_weather" {
		return fmt.Sprintf(`{"city":%q,"temperature":"22°C","condition":"sunny"}`, args.City)
	}
	return `{"error":"unknown tool"}`
}
