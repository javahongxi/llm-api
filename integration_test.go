//go:build integration

// Real-provider integration tests. These make actual network calls and are
// excluded from the default unit-test run; enable them with:
//
//	go test -tags integration -run TestReal -v .
//
// Credentials are read from the environment (DASHSCOPE_API_KEY,
// DEEPSEEK_API_KEY, OPENAI_API_KEY). A test is skipped when its key is absent.
package llmapi

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

const dashScopeBase = "https://dashscope.aliyuncs.com/compatible-mode/v1"

func realCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func requireEnv(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("%s not set; skipping real call", key)
	}
	return v
}

func TestRealDashScopeCompletion(t *testing.T) {
	key := requireEnv(t, "DASHSCOPE_API_KEY")
	resp, err := Completion(realCtx(t), "dashscope/qwen-plus",
		[]Message{UserMessage("Reply with exactly: OK")},
		WithAPIKey(key), WithMaxTokens(20), WithTemperature(0))
	if err != nil {
		t.Fatalf("completion failed: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		t.Fatalf("empty response: %+v", resp)
	}
	t.Logf("content=%q usage=%+v", resp.Choices[0].Message.Content, resp.Usage)
}

func TestRealDashScopeStream(t *testing.T) {
	key := requireEnv(t, "DASHSCOPE_API_KEY")
	stream, err := CompletionStream(realCtx(t), "dashscope/qwen-plus",
		[]Message{UserMessage("Count from 1 to 5, separated by spaces.")},
		WithAPIKey(key), WithMaxTokens(40), WithTemperature(0))
	if err != nil {
		t.Fatalf("stream open failed: %v", err)
	}
	defer stream.Close()

	var content strings.Builder
	var chunks int
	for stream.Next() {
		chunks++
		ch := stream.Current()
		if len(ch.Choices) > 0 {
			content.WriteString(ch.Choices[0].Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if chunks == 0 || content.Len() == 0 {
		t.Fatalf("no streamed content (chunks=%d)", chunks)
	}
	t.Logf("chunks=%d content=%q", chunks, content.String())
}

func TestRealDashScopeEmbedding(t *testing.T) {
	key := requireEnv(t, "DASHSCOPE_API_KEY")
	resp, err := Embedding(realCtx(t), "dashscope/text-embedding-v3",
		[]string{"hello", "world"}, WithAPIKey(key))
	if err != nil {
		t.Fatalf("embedding failed: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(resp.Data))
	}
	for i, d := range resp.Data {
		if len(d.Embedding) == 0 {
			t.Fatalf("embedding %d is empty", i)
		}
	}
	t.Logf("dims=%d usage=%+v", len(resp.Data[0].Embedding), resp.Usage)
}

func TestRealDashScopeToolCalling(t *testing.T) {
	key := requireEnv(t, "DASHSCOPE_API_KEY")
	tools := []Tool{NewFunctionTool("get_weather", "Get the current weather for a city",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "city name"},
			},
			"required": []string{"city"},
		})}
	resp, err := Completion(realCtx(t), "dashscope/qwen-plus",
		[]Message{UserMessage("What is the weather in Beijing? Use the get_weather tool.")},
		WithAPIKey(key), WithTools(tools...), WithToolChoice("auto"), WithMaxTokens(200))
	if err != nil {
		t.Fatalf("tool completion failed: %v", err)
	}
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) == 0 && msg.Content == "" {
		t.Fatalf("expected tool calls or content, got neither: %+v", msg)
	}
	for _, tc := range msg.ToolCalls {
		t.Logf("tool_call name=%s args=%s", tc.Function.Name, tc.Function.Arguments)
	}
	t.Logf("finish=%s content=%q tool_calls=%d", resp.Choices[0].FinishReason, msg.Content, len(msg.ToolCalls))
}

func TestRealDeepSeekCompletion(t *testing.T) {
	key := requireEnv(t, "DEEPSEEK_API_KEY")
	resp, err := Completion(realCtx(t), "deepseek/deepseek-chat",
		[]Message{UserMessage("Reply with exactly: OK")},
		WithAPIKey(key), WithMaxTokens(20), WithTemperature(0))
	if err != nil {
		t.Fatalf("completion failed: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		t.Fatalf("empty response: %+v", resp)
	}
	t.Logf("content=%q usage=%+v", resp.Choices[0].Message.Content, resp.Usage)
}

func TestRealDeepSeekStream(t *testing.T) {
	key := requireEnv(t, "DEEPSEEK_API_KEY")
	stream, err := CompletionStream(realCtx(t), "deepseek/deepseek-chat",
		[]Message{UserMessage("Count from 1 to 5, separated by spaces.")},
		WithAPIKey(key), WithMaxTokens(40), WithTemperature(0))
	if err != nil {
		t.Fatalf("stream open failed: %v", err)
	}
	defer stream.Close()

	var content strings.Builder
	var chunks int
	for stream.Next() {
		chunks++
		ch := stream.Current()
		if len(ch.Choices) > 0 {
			content.WriteString(ch.Choices[0].Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if chunks == 0 || content.Len() == 0 {
		t.Fatalf("no streamed content (chunks=%d)", chunks)
	}
	t.Logf("chunks=%d content=%q", chunks, content.String())
}

// TestRealOpenAICompatViaDashScope exercises the OpenAI-compatible base pointed
// at DashScope, using OPENAI_API_KEY (which holds a DashScope key in this setup).
func TestRealOpenAICompatViaDashScope(t *testing.T) {
	key := requireEnv(t, "OPENAI_API_KEY")
	resp, err := Completion(realCtx(t), "openai/qwen-plus",
		[]Message{UserMessage("Reply with exactly: OK")},
		WithAPIKey(key), WithAPIBase(dashScopeBase), WithMaxTokens(20), WithTemperature(0))
	if err != nil {
		t.Fatalf("completion failed: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		t.Fatalf("empty response: %+v", resp)
	}
	t.Logf("content=%q usage=%+v", resp.Choices[0].Message.Content, resp.Usage)
}
