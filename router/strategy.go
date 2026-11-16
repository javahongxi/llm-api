package router

import (
	"os"
	"strings"
	"time"

	llmapi "github.com/javahongxi/llm-api"
)

// deployment wraps a configured Deployment with mutable runtime state used for
// load balancing and cooldown. All fields are guarded by Router.mu.
type deployment struct {
	Deployment
	inflight      int64
	fails         int
	cooldownUntil time.Time
	latencyEWMA   float64
	hasLatency    bool
}

// deploymentOptions builds the credential/routing options for a deployment.
// Secrets of the form "os.environ/VAR" are resolved from the environment.
func deploymentOptions(d *deployment) []llmapi.Option {
	var opts []llmapi.Option
	if key := resolveSecret(d.APIKey); key != "" {
		opts = append(opts, llmapi.WithAPIKey(key))
	}
	if base := resolveSecret(d.APIBase); base != "" {
		opts = append(opts, llmapi.WithAPIBase(base))
	}
	if d.CustomLLMProvider != "" {
		opts = append(opts, llmapi.WithProvider(d.CustomLLMProvider))
	}
	return opts
}

func resolveSecret(value string) string {
	if rest, ok := strings.CutPrefix(value, "os.environ/"); ok {
		return os.Getenv(rest)
	}
	return value
}

// pick selects a deployment from the group using the configured strategy.
// Deployments in cooldown are skipped; if all are cooling down, the full set is
// used as a last resort so a single-deployment group can still retry.
func (r *Router) pick(group string, deps []*deployment) *deployment {
	now := time.Now()
	healthy := make([]*deployment, 0, len(deps))
	for _, d := range deps {
		if d.cooldownUntil.Before(now) {
			healthy = append(healthy, d)
		}
	}
	if len(healthy) == 0 {
		healthy = deps
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	switch r.settings.Strategy {
	case RoundRobin:
		idx := r.rr[group]
		r.rr[group] = idx + 1
		return healthy[idx%uint64(len(healthy))]
	case LeastBusy:
		best := healthy[0]
		for _, d := range healthy[1:] {
			if d.inflight < best.inflight {
				best = d
			}
		}
		return best
	case LatencyBased:
		best := healthy[0]
		for _, d := range healthy[1:] {
			if latencyOf(d) < latencyOf(best) {
				best = d
			}
		}
		return best
	default: // SimpleShuffle (weighted random)
		return r.weightedPick(healthy)
	}
}

// latencyOf treats an unmeasured deployment as zero latency so it gets tried.
func latencyOf(d *deployment) float64 {
	if !d.hasLatency {
		return 0
	}
	return d.latencyEWMA
}

func (r *Router) weightedPick(deps []*deployment) *deployment {
	total := 0.0
	for _, d := range deps {
		total += weightOf(d)
	}
	if total <= 0 {
		return deps[r.rng.Intn(len(deps))]
	}
	target := r.rng.Float64() * total
	for _, d := range deps {
		target -= weightOf(d)
		if target <= 0 {
			return d
		}
	}
	return deps[len(deps)-1]
}

func weightOf(d *deployment) float64 {
	if d.Weight <= 0 {
		return 1
	}
	return d.Weight
}

// begin marks a request as in-flight on the deployment.
func (r *Router) begin(d *deployment) {
	r.mu.Lock()
	d.inflight++
	r.mu.Unlock()
}

// end records the outcome: decrements in-flight, updates latency on success and
// applies cooldown on repeated failure.
func (r *Router) end(d *deployment, elapsed time.Duration, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if d.inflight > 0 {
		d.inflight--
	}
	if err == nil {
		d.fails = 0
		secs := elapsed.Seconds()
		if !d.hasLatency {
			d.latencyEWMA = secs
			d.hasLatency = true
		} else {
			d.latencyEWMA = 0.5*d.latencyEWMA + 0.5*secs
		}
		return
	}
	d.fails++
	if d.fails >= r.settings.AllowedFails {
		d.cooldownUntil = time.Now().Add(r.settings.CooldownTime)
		d.fails = 0
	}
}
