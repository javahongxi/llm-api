// Command llm-api runs the OpenAI-compatible proxy gateway (the Go analogue of
// `litellm --config config.yaml`). It loads a model_list config and serves
// /v1/chat/completions, /v1/embeddings and /v1/models, routing each public model
// name to its configured backend provider.
package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/javahongxi/llm-api/proxy"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to the proxy config file (YAML or JSON)")
	port := flag.Int("port", 0, "override the listen port from the config")
	host := flag.String("host", "", "override the listen host from the config")
	flag.Parse()

	cfg, err := proxy.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("llm-api: %v", err)
	}
	if *port != 0 {
		cfg.Server.Port = *port
	}
	if *host != "" {
		cfg.Server.Host = *host
	}

	srv := proxy.New(cfg)
	fmt.Printf("llm-api proxy listening on http://%s (%d models)\n", cfg.Addr(), len(cfg.ModelList))
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("llm-api: %v", err)
	}
}
