# llm-api

A Go port of [litellm](https://github.com/BerriAI/litellm): one unified,
OpenAI-shaped API to call many LLM providers. Call `Completion`,
`CompletionStream` or `Embedding` with a model name and messages; the package
resolves the provider, translates the request, and normalizes the response and
errors to a single shape.

It ships in two forms, mirroring litellm:

- **SDK** — `github.com/javahongxi/llm-api`, a stdlib-only Go library.
- **Proxy gateway** — an OpenAI-compatible HTTP server (`cmd/llm-api`) that
  routes a public `model_list` to backend providers, like `litellm --config`.

## Supported providers

| Provider  | Prefix        | Chat | Stream | Tools | Embeddings | Default API base                                    |
|-----------|---------------|:----:|:------:|:-----:|:----------:|-----------------------------------------------------|
| OpenAI    | `openai/`     |  ✓   |   ✓    |   ✓   |     ✓      | `https://api.openai.com/v1`                         |
| Anthropic | `anthropic/`  |  ✓   |   ✓    |   ✓   |     –      | `https://api.anthropic.com`                         |
| Gemini    | `gemini/`     |  ✓   |   ✓    |   ✓   |     ✓      | `https://generativelanguage.googleapis.com`         |
| DeepSeek  | `deepseek/`   |  ✓   |   ✓    |   ✓   |     –      | `https://api.deepseek.com/v1`                       |
| DashScope | `dashscope/`  |  ✓   |   ✓    |   ✓   |     ✓      | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

The provider is resolved like litellm's `get_llm_provider`: an explicit
`WithProvider(...)` wins, then a `provider/model` prefix, then inference from the
model name (`claude*` → anthropic, `gemini*` → gemini, `deepseek*` → deepseek,
`qwen*`/`qwq*`/`qvq*`/`text-embedding-v*` → dashscope, otherwise openai).

## Install

```bash
go get github.com/javahongxi/llm-api
```

Requires Go 1.27+. The SDK core has no third-party dependencies; only the proxy
config loader uses `gopkg.in/yaml.v3`.

## SDK usage

### Chat completion

```go
package main

import (
	"context"
	"fmt"

	llmapi "github.com/javahongxi/llm-api"
)

func main() {
	resp, err := llmapi.Completion(context.Background(), "gpt-4o",
		[]llmapi.Message{llmapi.UserMessage("Hello!")},
		llmapi.WithAPIKey("sk-..."), // or set OPENAI_API_KEY in the environment
		llmapi.WithTemperature(0.7),
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
}
```

Any provider works with the same call — just change the model string:

```go
llmapi.Completion(ctx, "anthropic/claude-3-5-sonnet-latest", messages)
llmapi.Completion(ctx, "gemini/gemini-1.5-pro", messages)
llmapi.Completion(ctx, "deepseek/deepseek-chat", messages)
llmapi.Completion(ctx, "dashscope/qwen-plus", messages)
```

### Streaming

`CompletionStream` returns a `bufio.Scanner`-style iterator of unified chunks:

```go
stream, err := llmapi.CompletionStream(ctx, "gpt-4o", messages)
if err != nil {
	panic(err)
}
defer stream.Close()

for stream.Next() {
	chunk := stream.Current()
	if len(chunk.Choices) > 0 {
		fmt.Print(chunk.Choices[0].Delta.Content)
	}
}
if err := stream.Err(); err != nil {
	panic(err)
}
```

### Tool / function calling

```go
tools := []llmapi.Tool{
	llmapi.NewFunctionTool("get_weather", "Get the weather", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"required": []string{"city"},
	}),
}

resp, _ := llmapi.Completion(ctx, "gpt-4o",
	[]llmapi.Message{llmapi.UserMessage("Weather in Beijing?")},
	llmapi.WithTools(tools...),
	llmapi.WithToolChoice("auto"),
)

for _, tc := range resp.Choices[0].Message.ToolCalls {
	fmt.Println(tc.Function.Name, tc.Function.Arguments)
}
```

Tool calls and tool results are translated to each provider's native shape
(Anthropic `tool_use`/`tool_result`, Gemini `functionCall`/`functionResponse`).

### Embeddings

```go
resp, err := llmapi.Embedding(ctx, "text-embedding-3-small",
	[]string{"hello", "world"},
	llmapi.WithDimensions(256),
)
if err != nil {
	panic(err)
}
fmt.Println(resp.Data[0].Embedding)
```

`input` accepts a `string`, `[]string` or `[]any` of strings.

### Options

All sampling parameters, credentials and HTTP behaviour are supplied as
functional options (the Go analogue of litellm's `completion(**kwargs)`):

`WithAPIKey`, `WithAPIBase`, `WithProvider`, `WithHTTPClient`, `WithTimeout`,
`WithTemperature`, `WithTopP`, `WithTopK`, `WithMaxTokens`,
`WithMaxCompletionTokens`, `WithN`, `WithStop`, `WithTools`, `WithToolChoice`,
`WithParallelToolCalls`, `WithResponseFormat`, `WithReasoningEffort`,
`WithPresencePenalty`, `WithFrequencyPenalty`, `WithSeed`, `WithLogprobs`,
`WithTopLogprobs`, `WithUser`, `WithExtraBody`, `WithExtraHeaders`,
`WithExtraQuery`, `WithDimensions`, `WithEncodingFormat`.

Only parameters you set are sent to the provider (litellm's `non_default_params`
behaviour). `WithExtraBody` merges provider-specific passthrough fields.

### Credentials

If `WithAPIKey` is not given, the key is read from the provider's environment
variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`), `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`.

> Note: a key only authenticates against its own provider. For example, a
> DashScope key placed in `OPENAI_API_KEY` will not work against
> `api.openai.com`; use `dashscope/qwen-plus`, or point `openai/...` at the
> DashScope compatible-mode endpoint via `WithAPIBase`.

### Unified errors

Every provider's HTTP errors are mapped to a single `*llmapi.Error` with a
`Kind` mirroring litellm's exception classes (`AuthenticationError`,
`RateLimitError`, `TimeoutError`, `ContextWindowExceeded`,
`ContentPolicyViolationError`, `InternalServerError`, ...). Use the helpers to
branch:

```go
_, err := llmapi.Completion(ctx, model, messages)
switch {
case llmapi.IsRateLimit(err):
	// back off and retry
case llmapi.IsAuthentication(err):
	// fix the key
case llmapi.IsContextWindowExceeded(err):
	// shorten the input
}
```

Available helpers: `AsError`, `IsAuthentication`, `IsRateLimit`, `IsTimeout`,
`IsContextWindowExceeded`, `IsContentPolicyViolation`, `IsServerError`,
`IsRetryable`.

## Proxy gateway

The proxy exposes an OpenAI-compatible API and routes each public model name to
a configured backend, exactly like the litellm proxy.

```bash
go run ./cmd/llm-api --config config.example.yaml
# or build a binary:
go build -o llm-api ./cmd/llm-api
./llm-api --config config.yaml --port 4000
```

Flags: `--config` (YAML or JSON, default `config.yaml`), `--port`, `--host`.

Configuration (`config.example.yaml`):

```yaml
server:
  host: 0.0.0.0
  port: 4000

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
```

`api_key`/`api_base` support `os.environ/VAR` references so secrets stay out of
the file. Endpoints:

- `POST /v1/chat/completions` (and `/chat/completions`) — supports `stream: true`
- `POST /v1/embeddings` (and `/embeddings`)
- `GET /v1/models` (and `/models`)
- `GET /health`

Point any OpenAI client at the gateway:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet","messages":[{"role":"user","content":"hi"}]}'
```

## Examples

Runnable examples live in [`examples/`](examples):

```bash
OPENAI_API_KEY=sk-... go run ./examples/completion
OPENAI_API_KEY=sk-... go run ./examples/streaming
OPENAI_API_KEY=sk-... go run ./examples/tools
OPENAI_API_KEY=sk-... go run ./examples/embedding
```

## Architecture

```
schema/            unified types, errors, CallConfig, Provider/StreamDecoder interfaces (no deps)
internal/sse/      Server-Sent Events reader
providers/
  openaicompat/    shared OpenAI-compatible base
  openai/ deepseek/ dashscope/   thin wrappers over openaicompat
  anthropic/       request/response transform + stateful stream decoder
  gemini/          request/response transform + stream + embeddings
(root) llmapi      provider registry, model routing, HTTP orchestration, high-level API
proxy/             OpenAI-compatible gateway + config loader
cmd/llm-api/       gateway binary
```

The `Provider` interface is a pure transformer (no I/O); the root package owns
HTTP, credential resolution and streaming. This keeps providers testable in
isolation and avoids import cycles.

## Development

```bash
go build ./...
go vet ./...
go test ./...   # unit tests use httptest; no real API calls
```

## Scope

This port focuses on the core calling surface: chat completion (with streaming),
tool calling, embeddings and unified error mapping across the five providers
above. litellm's Router (retries/fallbacks/load-balancing), cost tracking and
proxy auth/budgets are out of scope for now.