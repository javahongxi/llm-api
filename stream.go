package llmapi

import (
	"net/http"

	"github.com/javahongxi/llm-api/internal/sse"
	"github.com/javahongxi/llm-api/schema"
)

// Stream iterates over unified streaming chunks. Usage mirrors bufio.Scanner:
//
//	stream, err := llmapi.CompletionStream(ctx, model, messages, opts...)
//	if err != nil { ... }
//	defer stream.Close()
//	for stream.Next() {
//	    chunk := stream.Current()
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
//	if err := stream.Err(); err != nil { ... }
type Stream struct {
	resp    *http.Response
	scanner *sse.Scanner
	decoder schema.StreamDecoder
	pending []*schema.ModelResponseStream
	current *schema.ModelResponseStream
	err     error
	closed  bool
}

func newStream(resp *http.Response, decoder schema.StreamDecoder) *Stream {
	return &Stream{
		resp:    resp,
		scanner: sse.NewScanner(resp.Body),
		decoder: decoder,
	}
}

// Next advances to the next chunk, returning true when one is available. It
// returns false at end-of-stream or on error (inspect Err).
func (s *Stream) Next() bool {
	if s.err != nil || s.closed {
		return false
	}
	for {
		if len(s.pending) > 0 {
			s.current = s.pending[0]
			s.pending = s.pending[1:]
			return true
		}
		if s.decoder.Done() {
			s.Close()
			return false
		}
		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				s.err = schema.NewConnectionError("stream read error: "+err.Error(), "", "")
			}
			s.Close()
			return false
		}
		ev := s.scanner.Event()
		chunks, err := s.decoder.Decode(ev.Type, []byte(ev.Data))
		if err != nil {
			s.err = err
			s.Close()
			return false
		}
		s.pending = chunks
	}
}

// Current returns the most recent chunk produced by Next.
func (s *Stream) Current() *schema.ModelResponseStream { return s.current }

// Err returns the first error encountered while streaming, if any.
func (s *Stream) Err() error { return s.err }

// Close releases the underlying HTTP response body. It is safe to call multiple
// times and is invoked automatically at end-of-stream.
func (s *Stream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	return s.resp.Body.Close()
}
