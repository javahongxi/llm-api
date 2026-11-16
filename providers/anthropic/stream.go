package anthropic

import (
	"encoding/json"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// NewStreamDecoder returns a stateful decoder for the Anthropic SSE event stream
// (message_start / content_block_start / content_block_delta / message_delta /
// message_stop). It maps Anthropic content-block indices to OpenAI tool-call
// indices and accumulates usage, mirroring litellm's Anthropic streaming handler.
func (p *Provider) NewStreamDecoder(model string) schema.StreamDecoder {
	return &streamDecoder{
		model:            model,
		created:          time.Now().Unix(),
		toolIndexByBlock: map[int]int{},
		blockType:        map[int]string{},
	}
}

type streamDecoder struct {
	model            string
	id               string
	respModel        string
	created          int64
	toolIndexByBlock map[int]int
	blockType        map[int]string
	nextTool         int
	inputTokens      int
	outputTokens     int
	done             bool
}

type wireEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		ID    string `json:"id"`
		Model string `json:"model"`
		Role  string `json:"role"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"message"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
		Text string `json:"text"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		Thinking    string `json:"thinking"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (d *streamDecoder) Decode(eventType string, data []byte) ([]*schema.ModelResponseStream, error) {
	var ev wireEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, schema.NewBadRequestError("unable to parse anthropic stream event: "+err.Error(), "anthropic", d.model)
	}
	kind := ev.Type
	if kind == "" {
		kind = eventType
	}

	switch kind {
	case "message_start":
		d.id = ev.Message.ID
		d.respModel = ev.Message.Model
		d.inputTokens = ev.Message.Usage.InputTokens
		return []*schema.ModelResponseStream{d.chunk(schema.Delta{Role: "assistant"}, "", nil)}, nil

	case "content_block_start":
		d.blockType[ev.Index] = ev.ContentBlock.Type
		if ev.ContentBlock.Type == "tool_use" {
			toolIdx := d.nextTool
			d.nextTool++
			d.toolIndexByBlock[ev.Index] = toolIdx
			return []*schema.ModelResponseStream{d.chunk(schema.Delta{
				ToolCalls: []schema.ToolCall{{
					Index:    toolIdx,
					ID:       ev.ContentBlock.ID,
					Type:     "function",
					Function: schema.FunctionCall{Name: ev.ContentBlock.Name, Arguments: ""},
				}},
			}, "", nil)}, nil
		}
		return nil, nil

	case "content_block_delta":
		switch ev.Delta.Type {
		case "text_delta":
			return []*schema.ModelResponseStream{d.chunk(schema.Delta{Content: ev.Delta.Text}, "", nil)}, nil
		case "thinking_delta":
			return []*schema.ModelResponseStream{d.chunk(schema.Delta{ReasoningContent: ev.Delta.Thinking}, "", nil)}, nil
		case "input_json_delta":
			toolIdx := d.toolIndexByBlock[ev.Index]
			return []*schema.ModelResponseStream{d.chunk(schema.Delta{
				ToolCalls: []schema.ToolCall{{
					Index:    toolIdx,
					Function: schema.FunctionCall{Arguments: ev.Delta.PartialJSON},
				}},
			}, "", nil)}, nil
		}
		return nil, nil

	case "message_delta":
		if ev.Usage.OutputTokens > 0 {
			d.outputTokens = ev.Usage.OutputTokens
		}
		finish := schema.MapFinishReason(ev.Delta.StopReason)
		usage := &schema.Usage{
			PromptTokens:     d.inputTokens,
			CompletionTokens: d.outputTokens,
			TotalTokens:      d.inputTokens + d.outputTokens,
		}
		return []*schema.ModelResponseStream{d.chunk(schema.Delta{}, finish, usage)}, nil

	case "message_stop":
		d.done = true
		return nil, nil
	}
	// ping, content_block_stop and unknown events are bookkeeping.
	return nil, nil
}

func (d *streamDecoder) Done() bool { return d.done }

func (d *streamDecoder) chunk(delta schema.Delta, finish string, usage *schema.Usage) *schema.ModelResponseStream {
	m := d.respModel
	if m == "" {
		m = d.model
	}
	return &schema.ModelResponseStream{
		ID:      d.id,
		Object:  "chat.completion.chunk",
		Created: d.created,
		Model:   m,
		Choices: []schema.StreamChoice{{Index: 0, Delta: delta, FinishReason: finish}},
		Usage:   usage,
	}
}
