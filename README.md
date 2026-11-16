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

| Provider  | Prefix       | Chat | Stream | Tools | Embeddings | Default API base                                    |
|-----------|--------------|:----:|:------:|:-----:|:----------:|-----------------------------------------------------|
| OpenAI    | `openai/`    |  ✓  |   ✓   |  ✓   |     ✓     | `https://api.openai.com/v1`                         |
| Anthropic | `anthropic/` |  ✓  |   ✓   |  ✓   |     –      | `https://api.anthropic.com`                         |
| Gemini    | `gemini/`    |  ✓  |   ✓   |  ✓   |     ✓     | `https://generativelanguage.googleapis.com`         |
| DeepSeek  | `deepseek/`  |  ✓  |   ✓   |  ✓   |     –      | `https://api.deepseek.com/v1`                       |
| DashScope | `dashscope/` |  ✓  |   ✓   |  ✓   |     ✓     | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

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
llmapi.Completion(ctx, "deepseek/deepseek-v4-flash", messages)
llmapi.Completion(ctx, "dashscope/qwen3.7-plus", messages)
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
> `api.openai.com`; use `dashscope/qwen3.7-plus`, or point `openai/...` at the
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

## Router (retries / fallbacks / load balancing)

The `router` package is the Go analogue of litellm's Router. Group multiple
backend deployments under one public model name; calls are load-balanced across
healthy deployments, retried on transient errors with exponential backoff, and
fall back to other groups when a group is exhausted. Failing deployments are
cooled down temporarily.

```go
import "github.com/javahongxi/llm-api/router"

r := router.New([]router.Deployment{
	{ModelName: "gpt-4o", Model: "openai/gpt-4o", APIKey: "os.environ/OPENAI_API_KEY"},
	{ModelName: "gpt-4o", Model: "openai/gpt-4o", APIKey: "...", APIBase: "https://backup/v1", Weight: 2},
	{ModelName: "claude", Model: "anthropic/claude-3-5-sonnet-latest", APIKey: "os.environ/ANTHROPIC_API_KEY"},
}, router.Settings{
	NumRetries: 3,
	Strategy:   router.SimpleShuffle, // or RoundRobin / LeastBusy / LatencyBased
	Timeout:    60 * time.Second,
	Fallbacks:  map[string][]string{"gpt-4o": {"claude"}},
})

resp, err := r.Completion(ctx, "gpt-4o", messages)
// r.CompletionStream(...) and r.Embedding(...) work the same way.
```

- **Strategies**: `simple-shuffle` (weighted random, default), `round-robin`,
  `least-busy` (fewest in-flight), `latency-based-routing` (lowest observed
  latency).
- **Retries**: `NumRetries` applies to retryable errors (rate limit, timeout,
  connection, 5xx). A `RetryPolicy` overrides the count per error class, mirroring
  litellm (`RateLimitErrorRetries`, `BadRequestErrorRetries`, ...).
- **Cooldown**: after `AllowedFails` failures a deployment is skipped for
  `CooldownTime`.
- **Fallbacks**: ordered group list tried when the primary group fails.

## Cost tracking

The `cost` package is the Go analogue of litellm's cost calculator and spend
tracking. The SDK automatically computes a USD `cost` for every completion and
embedding response from token usage and a model price table (0 for unknown
models):

```go
resp, _ := llmapi.Completion(ctx, "gpt-4o", messages)
fmt.Println(resp.Cost) // e.g. 3.5e-05 (USD)
```

Prices are USD per token. A built-in table covers common models; override or
extend it programmatically:

```go
import "github.com/javahongxi/llm-api/cost"

cost.Register("gpt-4o", cost.ModelCost{InputCostPerToken: 2.5e-6, OutputCostPerToken: 1e-5})
unit := cost.Cost("gpt-4o", 1000, 500) // 0.0075
```

The Router accumulates spend across successful calls, broken down by model:

```go
report := r.Spend() // cost.SpendReport{TotalSpend, ByModel, ByKey}
```

The proxy exposes this at `GET /spend` and accepts a `model_cost` price table in
its config. Spend can also be attributed to a caller key via
`r.SpendTracker().AddForKey(key, model, amount)` (used by proxy auth/budgets).

## Proxy gateway

The proxy exposes an OpenAI-compatible API and routes each public model name to
a configured backend, exactly like the litellm proxy.

```bash
# Quickest: one-command start with the shipped template (export the provider
# keys it references first; extra args are forwarded to the binary):
#   ./run.sh  [--port 8888 --host 127.0.0.1 ...]

# Start directly from the shipped template (no secrets in the file — all keys
# are os.environ/VAR references):
go run ./cmd/llm-api --config config.example.yaml

# Or copy the template to your own config.yaml and edit it freely.
# config.yaml is git-ignored, so real keys you put there stay out of git:
cp config.example.yaml config.yaml
LLM_API_MASTER_KEY=... OPENAI_API_KEY=... ./llm-api   # reads config.yaml by default
# or build a binary:
go build -o llm-api ./cmd/llm-api
./llm-api --config config.yaml
```

Flags: `--config` (YAML or JSON, default `config.yaml`), `--port`, `--host`.

Configuration (`config.example.yaml`):

```yaml
server:
  host: 127.0.0.1
  port: 8888

router_settings:
  num_retries: 3
  routing_strategy: simple-shuffle
  fallbacks:
    - gpt-4o: [qwen3.7-plus]

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: qwen3.7-plus
    litellm_params:
      model: dashscope/qwen3.7-plus
      api_key: os.environ/DASHSCOPE_API_KEY
```

Multiple `model_list` entries sharing a `model_name` form one load-balanced
group; `router_settings` controls retries, strategy, cooldown and fallbacks
(see [`config.example.yaml`](config.example.yaml)).

`api_key`/`api_base` support `os.environ/VAR` references so secrets stay out of
the file. Endpoints:

- `POST /v1/chat/completions` (and `/chat/completions`) — supports `stream: true`
- `POST /v1/embeddings` (and `/embeddings`)
- `GET /v1/models` (and `/models`)
- `GET /spend` — accumulated spend tracked by the router (master key when auth on)
- `POST /key/generate`, `POST /key/update`, `POST /key/delete`, `GET /key/info`, `GET /key/list` — virtual-key management (master key)
- `GET /health`

Point any OpenAI client at the gateway:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-plus","messages":[{"role":"user","content":"hi"}]}'
```

Streaming responses are returned as Server-Sent Events; pass `"stream": true`
and disable curl buffering with `-N`:

```bash
curl -N http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-plus","messages":[{"role":"user","content":"hi"}],"stream":true}'

# data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"},...}]}
# data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":" there"},...}]}
# ...
# data: [DONE]
```

## Authentication & budgets

The `auth` package is the Go analogue of litellm's proxy auth/budgets. Set a
`master_key` in `general_settings` to enable virtual-key auth; without it the
proxy runs in open mode (no auth, fully backward compatible).

```yaml
general_settings:
  master_key: os.environ/LLM_API_MASTER_KEY   # or a literal sk-... key
```

When enabled, data endpoints require a proxy-issued key
(`Authorization: Bearer <key>` or `x-api-key`). The master key bypasses all
checks and administers the `/key/*` endpoints and `/spend`. Mint and manage keys
with the master key:

```bash
# generate a key with a $10 budget, 60 rpm and a model allowlist
curl http://127.0.0.1:8888/key/generate \
  -H 'Authorization: Bearer sk-master' -H 'Content-Type: application/json' \
  -d '{"key_alias":"team-a","max_budget":10,"rpm_limit":60,"models":["gpt-4o"]}'

# call the gateway with the issued key
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Authorization: Bearer sk-<issued>' -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

Each key carries:

- `max_budget` (USD) — enforced against accumulated per-key spend; exceeding it
  returns a 400 `Budget has been exceeded` error. `budget_duration` optionally
  resets spend on a rolling window.
- `rpm_limit` / `tpm_limit` — sliding 60-second request/token rate limits (429).
- `models` — allowlist of public model names (403 when not permitted; empty = all).
- `duration` — key expiry (401 once expired).

Spend is computed per call (see [Cost tracking](#cost-tracking)) and attributed
to the calling key, so `/spend` reports a `spend_by_key` breakdown. Management
endpoints: `POST /key/generate`, `POST /key/update`, `POST /key/delete`,
`GET /key/info?key=...`, `GET /key/list`. Durations accept Go strings (`"24h"`,
`"1h30m"`) or a number of seconds.

Programmatic use:

```go
import "github.com/javahongxi/llm-api/auth"

store := auth.NewStore("sk-master")
key, _ := store.Generate(auth.GenerateRequest{MaxBudget: 10, RPMLimit: 60})
k, err := store.Authorize(key.Token, "gpt-4o") // unified error on failure
store.AddSpend(key.Token, 0.0025)
```

## Examples

Runnable examples live in [`examples/`](examples):

```bash
OPENAI_API_KEY=sk-... go run ./examples/completion
OPENAI_API_KEY=sk-... go run ./examples/streaming
OPENAI_API_KEY=sk-... go run ./examples/tools
OPENAI_API_KEY=sk-... go run ./examples/embedding
DASHSCOPE_API_KEY=... DEEPSEEK_API_KEY=... go run ./examples/router
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
router/            load balancing, retries/backoff, cooldown, fallbacks, spend
cost/              model price table, per-call cost, spend tracking
auth/              virtual API keys, budgets, rpm/tpm limits, model allowlists
proxy/             OpenAI-compatible gateway + config loader + key management
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

This port covers the core calling surface — chat completion (with streaming),
tool calling, embeddings and unified error mapping across the five providers
above — plus the Router (load balancing, retries/backoff, cooldown and
fallbacks), cost tracking (per-call cost and accumulated spend) and proxy
auth/budgets (virtual keys, per-key spend/budget enforcement, rpm/tpm rate
limits and model allowlists).