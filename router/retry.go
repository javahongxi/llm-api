package router

import (
	"context"
	"time"

	"github.com/javahongxi/llm-api/schema"
)

// retriesFor returns how many retries are allowed for err: an explicit
// RetryPolicy override wins, then DefaultRetries, then NumRetries for generally
// retryable errors (non-retryable errors get 0).
func (r *Router) retriesFor(err error) int {
	if p := r.settings.RetryPolicy; p != nil {
		if v := policyForKind(p, kindOf(err)); v != nil {
			return *v
		}
		if p.DefaultRetries != nil {
			return *p.DefaultRetries
		}
	}
	if schema.IsRetryable(err) {
		return r.settings.NumRetries
	}
	return 0
}

func kindOf(err error) schema.ErrorKind {
	if e, ok := schema.AsError(err); ok {
		return e.Kind
	}
	return ""
}

func policyForKind(p *RetryPolicy, kind schema.ErrorKind) *int {
	switch kind {
	case schema.KindBadRequestError:
		return p.BadRequestErrorRetries
	case schema.KindAuthenticationError:
		return p.AuthenticationErrorRetries
	case schema.KindTimeoutError:
		return p.TimeoutErrorRetries
	case schema.KindRateLimitError:
		return p.RateLimitErrorRetries
	case schema.KindContentPolicyViolation:
		return p.ContentPolicyViolationErrorRetries
	case schema.KindInternalServerError:
		return p.InternalServerErrorRetries
	case schema.KindServiceUnavailableError:
		return p.ServiceUnavailableErrorRetries
	case schema.KindNotFoundError:
		return p.NotFoundErrorRetries
	}
	return nil
}

// backoff sleeps for an exponentially increasing delay with jitter before the
// next retry. It returns false if ctx is cancelled while waiting.
func (r *Router) backoff(ctx context.Context, attempt int) bool {
	shift := attempt
	if shift > 20 {
		shift = 20
	}
	delay := r.settings.RetryBackoffBase * time.Duration(1<<uint(shift))
	if delay > r.settings.MaxRetryBackoff || delay <= 0 {
		delay = r.settings.MaxRetryBackoff
	}
	r.mu.Lock()
	jitter := time.Duration(r.rng.Float64() * float64(delay) * 0.25)
	r.mu.Unlock()
	delay += jitter

	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
