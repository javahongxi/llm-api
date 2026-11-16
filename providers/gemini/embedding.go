package gemini

import (
	"encoding/json"
	"strings"

	"github.com/javahongxi/llm-api/schema"
)

// EmbeddingEndpoint builds the batchEmbedContents URL.
func (p *Provider) EmbeddingEndpoint(c *schema.CallConfig, model string) string {
	return p.modelBase(c, model) + ":batchEmbedContents"
}

// EmbeddingRequest builds the Gemini batchEmbedContents body. Each input string
// becomes one embedContent request scoped to "models/{model}".
func (p *Provider) EmbeddingRequest(model string, input []string, c *schema.CallConfig) (any, error) {
	model = strings.TrimPrefix(model, "models/")
	requests := make([]map[string]any, 0, len(input))
	for _, text := range input {
		req := map[string]any{
			"model":   "models/" + model,
			"content": map[string]any{"parts": []any{map[string]any{"text": text}}},
		}
		requests = append(requests, req)
	}
	body := map[string]any{"requests": requests}
	for k, v := range c.ExtraBody {
		body[k] = v
	}
	return body, nil
}

// ParseEmbedding converts a batchEmbedContents response to the unified shape.
func (p *Provider) ParseEmbedding(model string, body []byte) (*schema.EmbeddingResponse, error) {
	var r struct {
		Embeddings []struct {
			Values []float64 `json:"values"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, schema.NewBadRequestError("unable to parse gemini embedding response: "+err.Error(), p.Name(), model)
	}
	resp := &schema.EmbeddingResponse{Object: "list", Model: model}
	for i, e := range r.Embeddings {
		resp.Data = append(resp.Data, schema.EmbeddingData{
			Object:    "embedding",
			Embedding: e.Values,
			Index:     i,
		})
	}
	return resp, nil
}
