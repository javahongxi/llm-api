package cost

import "sync"

// SpendTracker accumulates LLM spend over time, broken down by model and by an
// optional caller key (e.g. a proxy virtual key or user). It is safe for
// concurrent use and mirrors litellm's spend tracking.
type SpendTracker struct {
	mu      sync.Mutex
	total   float64
	byModel map[string]float64
	byKey   map[string]float64
}

// NewSpendTracker creates an empty tracker.
func NewSpendTracker() *SpendTracker {
	return &SpendTracker{byModel: map[string]float64{}, byKey: map[string]float64{}}
}

// Add records spend for a model.
func (s *SpendTracker) Add(model string, amount float64) {
	s.AddForKey("", model, amount)
}

// AddForKey records spend for a model attributed to key (empty key = unattributed).
func (s *SpendTracker) AddForKey(key, model string, amount float64) {
	if amount <= 0 {
		return
	}
	s.mu.Lock()
	s.total += amount
	s.byModel[model] += amount
	if key != "" {
		s.byKey[key] += amount
	}
	s.mu.Unlock()
}

// Total returns the accumulated spend in USD.
func (s *SpendTracker) Total() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}

// SpendReport is a snapshot of accumulated spend.
type SpendReport struct {
	TotalSpend float64            `json:"total_spend"`
	ByModel    map[string]float64 `json:"spend_by_model"`
	ByKey      map[string]float64 `json:"spend_by_key,omitempty"`
}

// Report returns a consistent snapshot of the tracker.
func (s *SpendTracker) Report() SpendReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	report := SpendReport{
		TotalSpend: s.total,
		ByModel:    make(map[string]float64, len(s.byModel)),
		ByKey:      make(map[string]float64, len(s.byKey)),
	}
	for m, v := range s.byModel {
		report.ByModel[m] = v
	}
	for k, v := range s.byKey {
		report.ByKey[k] = v
	}
	return report
}
