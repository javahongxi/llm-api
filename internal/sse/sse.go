// Package sse implements a minimal Server-Sent Events reader used to consume
// streaming chat completions. It yields one Event per blank-line-delimited
// block, exposing the SSE "event" type and the joined "data" payload. It is
// intentionally small and dependency-free (stdlib only).
package sse

import (
	"bufio"
	"io"
	"strings"
)

// Event is a single SSE event. Type is the "event:" field (empty when absent,
// as in OpenAI-style streams). Data is the "data:" field(s) joined by newlines.
type Event struct {
	Type string
	Data string
}

// Scanner reads SSE events from a stream. Usage mirrors bufio.Scanner:
//
//	sc := sse.NewScanner(resp.Body)
//	for sc.Scan() {
//	    ev := sc.Event()
//	    ...
//	}
//	if err := sc.Err(); err != nil { ... }
type Scanner struct {
	r         *bufio.Reader
	data      []string
	eventType string
	cur       Event
	err       error
	eof       bool
}

// NewScanner creates an SSE scanner over r.
func NewScanner(r io.Reader) *Scanner {
	return &Scanner{r: bufio.NewReaderSize(r, 64*1024)}
}

// Scan advances to the next event, returning true when one is available. It
// returns false at end-of-stream or on error.
func (s *Scanner) Scan() bool {
	for {
		if s.eof {
			if len(s.data) > 0 {
				s.cur = Event{Type: s.eventType, Data: strings.Join(s.data, "\n")}
				s.data = nil
				s.eventType = ""
				return true
			}
			return false
		}

		line, err := s.r.ReadString('\n')
		if err != nil {
			s.eof = true
			if err != io.EOF {
				s.err = err
			}
			if line != "" {
				s.processLine(strings.TrimRight(line, "\r\n"))
			}
			continue
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if len(s.data) > 0 {
				s.cur = Event{Type: s.eventType, Data: strings.Join(s.data, "\n")}
				s.data = nil
				s.eventType = ""
				return true
			}
			s.eventType = ""
			continue
		}
		s.processLine(line)
	}
}

// Event returns the most recent event produced by Scan.
func (s *Scanner) Event() Event { return s.cur }

// Err returns the first non-EOF error encountered while reading.
func (s *Scanner) Err() error { return s.err }

func (s *Scanner) processLine(line string) {
	if strings.HasPrefix(line, ":") {
		return // comment / keep-alive
	}
	field, value, _ := strings.Cut(line, ":")
	value = strings.TrimPrefix(value, " ")
	switch field {
	case "data":
		s.data = append(s.data, value)
	case "event":
		s.eventType = value
	}
}
