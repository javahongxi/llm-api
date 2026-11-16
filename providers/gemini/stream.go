package gemini

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// NewStreamDecoder returns a decoder for the Gemini streamGenerateContent SSE
// stream (alt=sse). Each data payload is a partial generateContent response.
func (p *Provider) NewStreamDecoder(model string) schema.StreamDecoder {
	return &streamDecoder{
		model:   model,
		id:      "chatcmpl-" + strconv.FormatInt(time.Now().UnixNano(), 36),
		created: time.Now().Unix(),
	}
}

type streamDecoder struct {
	model   string
	id      string
	created int64
	done    bool
}

func (d *streamDecoder) Decode(eventType string, data []byte) ([]*schema.ModelResponseStream, error) {
	var r geminiResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, schema.NewBadRequestError("unable to parse gemini stream chunk: "+err.Error(), "gemini", d.model)
	}

	chunk := &schema.ModelResponseStream{
		ID:      d.id,
		Object:  "chat.completion.chunk",
		Created: d.created,
		Model:   d.model,
	}
	for _, cand := range r.Candidates {
		delta, finish := candidateToDelta(cand)
		chunk.Choices = append(chunk.Choices, schema.StreamChoice{
			Index:        cand.Index,
			Delta:        delta,
			FinishReason: finish,
		})
		if finish != "" {
			d.done = true
		}
	}
	if r.UsageMetadata.TotalTokenCount > 0 {
		chunk.Usage = &schema.Usage{
			PromptTokens:     r.UsageMetadata.PromptTokenCount,
			CompletionTokens: r.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      r.UsageMetadata.TotalTokenCount,
		}
	}
	if len(chunk.Choices) == 0 && chunk.Usage == nil {
		return nil, nil
	}
	return []*schema.ModelResponseStream{chunk}, nil
}

func (d *streamDecoder) Done() bool { return d.done }

// candidateToDelta extracts incremental text, tool calls and reasoning.
func candidateToDelta(cand candidate) (schema.Delta, string) {
	var text, reasoning strings.Builder
	var toolCalls []schema.ToolCall
	for i, part := range cand.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			args, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, schema.ToolCall{
				Index: i,
				ID:    part.FunctionCall.Name + "-" + strconv.Itoa(i),
				Type:  "function",
				Function: schema.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				},
			})
		case part.Thought:
			reasoning.WriteString(part.Text)
		default:
			text.WriteString(part.Text)
		}
	}
	finish := schema.MapFinishReason(cand.FinishReason)
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	}
	return schema.Delta{
		Content:          text.String(),
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
	}, finish
}
