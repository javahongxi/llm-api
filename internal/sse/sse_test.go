package sse

import (
	"strings"
	"testing"
)

func TestScannerOpenAIStyle(t *testing.T) {
	input := "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n"
	sc := NewScanner(strings.NewReader(input))

	var got []Event
	for sc.Scan() {
		got = append(got, sc.Event())
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 events, got %d: %+v", len(got), got)
	}
	if got[0].Data != `{"a":1}` || got[1].Data != `{"b":2}` || got[2].Data != "[DONE]" {
		t.Fatalf("unexpected data: %+v", got)
	}
}

func TestScannerTypedEvents(t *testing.T) {
	input := "event: message_start\ndata: {\"type\":\"message_start\"}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
	sc := NewScanner(strings.NewReader(input))

	var got []Event
	for sc.Scan() {
		got = append(got, sc.Event())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 events, got %d", len(got))
	}
	if got[0].Type != "message_start" || got[1].Type != "content_block_delta" {
		t.Fatalf("unexpected event types: %+v", got)
	}
}

func TestScannerMultilineDataAndComments(t *testing.T) {
	input := ": keep-alive\ndata: line1\ndata: line2\n\n"
	sc := NewScanner(strings.NewReader(input))
	if !sc.Scan() {
		t.Fatal("expected an event")
	}
	if sc.Event().Data != "line1\nline2" {
		t.Fatalf("expected joined multiline data, got %q", sc.Event().Data)
	}
	if sc.Scan() {
		t.Fatal("expected end of stream")
	}
}

func TestScannerNoTrailingBlankLine(t *testing.T) {
	input := "data: {\"x\":1}"
	sc := NewScanner(strings.NewReader(input))
	if !sc.Scan() {
		t.Fatal("expected a final event even without trailing blank line")
	}
	if sc.Event().Data != `{"x":1}` {
		t.Fatalf("unexpected data: %q", sc.Event().Data)
	}
}
