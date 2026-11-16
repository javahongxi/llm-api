package llmapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// defaultClient is shared across calls that do not supply their own client.
var defaultClient = &http.Client{Timeout: 600 * time.Second}

func httpClient(c *schema.CallConfig) *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	if c.Timeout > 0 {
		return &http.Client{Timeout: c.Timeout}
	}
	return defaultClient
}

// newRequest builds a POST request with the JSON body, provider headers and any
// extra headers/query parameters.
func newRequest(ctx context.Context, c *schema.CallConfig, endpoint string, headers map[string]string, body any) (*http.Request, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, schema.NewBadRequestError("unable to encode request: "+err.Error(), c.Provider, "")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
	if err != nil {
		return nil, schema.NewBadRequestError("unable to build request: "+err.Error(), c.Provider, "")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	for k, v := range c.ExtraHeaders {
		req.Header.Set(k, v)
	}
	if len(c.ExtraQuery) > 0 {
		q := req.URL.Query()
		for k, v := range c.ExtraQuery {
			q.Set(k, v)
		}
		req.URL.RawQuery = q.Encode()
	}
	return req, nil
}

// doRequest executes a non-streaming call and returns the raw response body.
// HTTP error statuses are mapped to unified errors via the provider.
func doRequest(ctx context.Context, c *schema.CallConfig, p schema.Provider, model, endpoint string, headers map[string]string, body any) ([]byte, error) {
	req, err := newRequest(ctx, c, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	resp, err := httpClient(c).Do(req)
	if err != nil {
		return nil, transportError(ctx, err, p.Name(), model)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, schema.NewConnectionError("unable to read response: "+err.Error(), p.Name(), model)
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, p.MapError(resp.StatusCode, respBody, model)
	}
	return respBody, nil
}

// doStreamRequest executes a streaming call and returns the open response. The
// caller owns closing the body. Error statuses are mapped before returning.
func doStreamRequest(ctx context.Context, c *schema.CallConfig, p schema.Provider, model, endpoint string, headers map[string]string, body any) (*http.Response, error) {
	req, err := newRequest(ctx, c, endpoint, headers, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := httpClient(c).Do(req)
	if err != nil {
		return nil, transportError(ctx, err, p.Name(), model)
	}
	if resp.StatusCode >= http.StatusBadRequest {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, p.MapError(resp.StatusCode, respBody, model)
	}
	return resp, nil
}

// transportError classifies a client.Do failure as a timeout or connection error.
func transportError(ctx context.Context, err error, provider, model string) error {
	if ctx.Err() == context.DeadlineExceeded || errors.Is(err, context.DeadlineExceeded) {
		return schema.NewTimeoutError(err.Error(), provider, model)
	}
	var netErr interface{ Timeout() bool }
	if errors.As(err, &netErr) && netErr.Timeout() {
		return schema.NewTimeoutError(err.Error(), provider, model)
	}
	return schema.NewConnectionError(err.Error(), provider, model)
}
