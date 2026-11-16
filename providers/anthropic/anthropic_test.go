package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/javahongxi/llm-api/schema"
)

func TestChatRequestSystemHoistAndDefaults(t *testing.T) {
	p := New()
	c := &schema.CallConfig{}
	messages := []schema.Message{
		schema.SystemMessage("be brief"),
		schema.UserMessage("hi"),
	}
	body, err := p.ChatRequest("claude-3-5-sonnet-latest", messages, c)
	if err != nil {
		t.Fatal(err)
	}
	m := body.(map[string]any)
	if m["system"] != "be brief" {
		t.Errorf("expected system hoisted, got %v", m["system"])
	}
	if m["max_tokens"] != defaultMaxTokens {
		t.Errorf("expected default max_tokens %d, got %v", defaultMaxTokens, m["max_tokens"])
	}
	msgs := m["messages"].([]map[string]any)
	if len(msgs) != 1 || msgs[0]["role"] != "user" {
		t.Fatalf("expected single user message (system removed), got %+v", msgs)
	}
}

func TestChatRequestToolsAndChoice(t *testing.T) {
	p := New()
	temp := 0.5
	c := &schema.CallConfig{
		Temperature: &temp,
		Tools: []schema.Tool{schema.NewFunctionTool("get_weather", "weather",
			map[string]any{"type": "object"})},
		ToolChoice: "required",
	}
	body, _ := p.ChatRequest("claude-3-5-sonnet-latest", []schema.Message{schema.UserMessage("hi")}, c)
	m := body.(map[string]any)
	if m["temperature"] != 0.5 {
		t.Errorf("temperature not mapped: %v", m["temperature"])
	}
	tools := m["tools"].([]map[string]any)
	if len(tools) != 1 || tools[0]["name"] != "get_weather" {
		t.Fatalf("tools not mapped: %+v", tools)
	}
	if _, ok := tools[0]["input_schema"]; !ok {
		t.Error("expected input_schema on tool")
	}
	tc := m["tool_choice"].(map[string]any)
	if tc["type"] != "any" {
		t.Errorf("required should map to any, got %v", tc)
	}
}

func TestChatRequestToolResultMerging(t *testing.T) {
	p := New()
	messages := []schema.Message{
		schema.UserMessage("weather?"),
		{Role: "assistant", ToolCalls: []schema.ToolCall{{ID: "t1", Type: "function", Function: schema.FunctionCall{Name: "get_weather", Arguments: `{"city":"BJ"}`}}}},
		schema.ToolMessage("t1", `{"temp":22}`),
	}
	body, _ := p.ChatRequest("claude-3-5-sonnet-latest", messages, &schema.CallConfig{})
	msgs := body.(map[string]any)["messages"].([]map[string]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	// assistant tool_use block
	assistantBlocks := msgs[1]["content"].([]any)
	tu := assistantBlocks[0].(map[string]any)
	if tu["type"] != "tool_use" || tu["name"] != "get_weather" {
		t.Fatalf("expected tool_use block, got %+v", tu)
	}
	input := tu["input"].(map[string]any)
	if input["city"] != "BJ" {
		t.Errorf("expected parsed input city=BJ, got %+v", input)
	}
	// tool result becomes user-role tool_result
	if msgs[2]["role"] != "user" {
		t.Fatalf("tool result should be user role, got %v", msgs[2]["role"])
	}
	tr := msgs[2]["content"].([]any)[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != "t1" {
		t.Fatalf("expected tool_result block, got %+v", tr)
	}
}

func TestParseChatTextAndUsage(t *testing.T) {
	p := New()
	body := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-latest",
		"content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`)
	resp, err := p.ParseChat("claude-3-5-sonnet-latest", body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "Hello world" {
		t.Errorf("expected concatenated text, got %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("end_turn should map to stop, got %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage not mapped: %+v", resp.Usage)
	}
}

func TestParseChatToolUse(t *testing.T) {
	p := New()
	body := []byte(`{
		"id":"msg_2","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"BJ"}}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}
	}`)
	resp, _ := p.ParseChat("claude-3-5-sonnet-latest", body)
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("expected tool_calls finish, got %q", resp.Choices[0].FinishReason)
	}
	tc := resp.Choices[0].Message.ToolCalls[0]
	if tc.ID != "toolu_1" || tc.Function.Name != "get_weather" {
		t.Fatalf("tool call not mapped: %+v", tc)
	}
	var args map[string]any
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	if args["city"] != "BJ" {
		t.Errorf("expected arguments city=BJ, got %v", args)
	}
}

func TestStreamDecoderTextAndTool(t *testing.T) {
	p := New()
	d := p.NewStreamDecoder("claude-3-5-sonnet-latest")

	events := []struct{ typ, data string }{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-latest","usage":{"input_tokens":8}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"BJ\"}"}}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}

	var content string
	var toolArgs string
	var toolName string
	var finish string
	var usage *schema.Usage
	for _, ev := range events {
		chunks, err := d.Decode(ev.typ, []byte(ev.data))
		if err != nil {
			t.Fatal(err)
		}
		for _, ch := range chunks {
			for _, choice := range ch.Choices {
				content += choice.Delta.Content
				if choice.FinishReason != "" {
					finish = choice.FinishReason
				}
				for _, tc := range choice.Delta.ToolCalls {
					if tc.Function.Name != "" {
						toolName = tc.Function.Name
					}
					toolArgs += tc.Function.Arguments
					if tc.Index != 0 {
						t.Errorf("first tool call should have index 0, got %d", tc.Index)
					}
				}
			}
			if ch.Usage != nil {
				usage = ch.Usage
			}
		}
	}
	if !d.Done() {
		t.Error("decoder should be done after message_stop")
	}
	if content != "Hi" {
		t.Errorf("expected content Hi, got %q", content)
	}
	if toolName != "get_weather" || toolArgs != `{"city":"BJ"}` {
		t.Errorf("tool not assembled: name=%q args=%q", toolName, toolArgs)
	}
	if finish != "tool_calls" {
		t.Errorf("expected tool_calls finish, got %q", finish)
	}
	if usage == nil || usage.PromptTokens != 8 || usage.CompletionTokens != 12 {
		t.Errorf("usage not assembled: %+v", usage)
	}
}
