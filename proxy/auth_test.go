package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const testMasterKey = "sk-master-test"

// newAuthServer builds a gateway with auth enabled (master key set) routing to
// the given backend. Model "my-gpt" maps to openai/gpt-4o.
func newAuthServer(t *testing.T, backendURL string) *httptest.Server {
	t.Helper()
	cfg := &Config{
		ModelList: []ModelEntry{
			{ModelName: "my-gpt", Params: Params{Model: "openai/gpt-4o", APIKey: "x", APIBase: backendURL + "/v1"}},
		},
		GeneralSettings: GeneralSettings{MasterKey: testMasterKey},
	}
	return httptest.NewServer(New(cfg).Handler())
}

func postAuth(t *testing.T, url, body, token string) (*http.Response, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, data
}

func getAuth(t *testing.T, url, token string) (*http.Response, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, data
}

// generateKey mints a virtual key via /key/generate using the master key.
func generateKey(t *testing.T, srvURL, body string) string {
	t.Helper()
	resp, data := postAuth(t, srvURL+"/key/generate", body, testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("key/generate status %d: %s", resp.StatusCode, data)
	}
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	token, _ := out["key"].(string)
	if token == "" {
		t.Fatalf("no key in response: %s", data)
	}
	return token
}

const chatBody = `{"model":"my-gpt","messages":[{"role":"user","content":"hi"}]}`

func TestProxyAuthRequired(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	// No key -> 401.
	resp, data := postAuth(t, srv.URL+"/v1/chat/completions", chatBody, "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("missing key: expected 401, got %d: %s", resp.StatusCode, data)
	}
	// Invalid key -> 401.
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, "sk-bogus")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("invalid key: expected 401, got %d: %s", resp.StatusCode, data)
	}
	// Valid virtual key -> 200.
	token := generateKey(t, srv.URL, `{}`)
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("valid key: expected 200, got %d: %s", resp.StatusCode, data)
	}
	// Master key bypass -> 200.
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("master key: expected 200, got %d: %s", resp.StatusCode, data)
	}
}

func TestProxyAuthModelAllowlist(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	token := generateKey(t, srv.URL, `{"models":["other-model"]}`)
	resp, data := postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("disallowed model: expected 403, got %d: %s", resp.StatusCode, data)
	}
}

func TestProxyAuthBudget(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	// gpt-4o call cost: prompt 1, completion 2 -> 1*2.5e-6 + 2*1e-5 = 2.25e-5.
	token := generateKey(t, srv.URL, `{"max_budget":0.00003}`)
	resp, data := postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first call: expected 200, got %d: %s", resp.StatusCode, data)
	}
	// Spend now 2.25e-5 < 3e-5, second call still allowed (spend becomes 4.5e-5).
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("second call: expected 200, got %d: %s", resp.StatusCode, data)
	}
	// Third call: spend 4.5e-5 >= budget 3e-5 -> 400 budget exceeded.
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("over budget: expected 400, got %d: %s", resp.StatusCode, data)
	}
	if !strings.Contains(string(data), "Budget has been exceeded") {
		t.Errorf("expected budget message: %s", data)
	}
}

func TestProxyAuthRPM(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	token := generateKey(t, srv.URL, `{"rpm_limit":1}`)
	resp, data := postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first call: expected 200, got %d: %s", resp.StatusCode, data)
	}
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("over rpm: expected 429, got %d: %s", resp.StatusCode, data)
	}
}

func TestProxyKeyManagement(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	// Admin endpoints require the master key.
	resp, data := postAuth(t, srv.URL+"/key/generate", `{}`, "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("key/generate without master: expected 401, got %d: %s", resp.StatusCode, data)
	}
	token := generateKey(t, srv.URL, `{"key_alias":"team-a","max_budget":5}`)
	// A virtual key cannot administer.
	resp, data = postAuth(t, srv.URL+"/key/generate", `{}`, token)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("key/generate with virtual key: expected 401, got %d: %s", resp.StatusCode, data)
	}

	// info
	resp, data = getAuth(t, srv.URL+"/key/info?key="+token, testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("key/info: expected 200, got %d: %s", resp.StatusCode, data)
	}
	var info map[string]any
	_ = json.Unmarshal(data, &info)
	if info["info"].(map[string]any)["key_alias"] != "team-a" {
		t.Errorf("unexpected info: %s", data)
	}

	// list
	resp, data = getAuth(t, srv.URL+"/key/list", testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("key/list: expected 200, got %d: %s", resp.StatusCode, data)
	}
	var list map[string]any
	_ = json.Unmarshal(data, &list)
	if list["total_count"].(float64) != 1 {
		t.Errorf("expected 1 key: %s", data)
	}

	// update budget
	resp, data = postAuth(t, srv.URL+"/key/update", `{"key":"`+token+`","max_budget":9}`, testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("key/update: expected 200, got %d: %s", resp.StatusCode, data)
	}
	var updated map[string]any
	_ = json.Unmarshal(data, &updated)
	if updated["max_budget"].(float64) != 9 {
		t.Errorf("budget not updated: %s", data)
	}

	// delete
	resp, data = postAuth(t, srv.URL+"/key/delete", `{"keys":["`+token+`"]}`, testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("key/delete: expected 200, got %d: %s", resp.StatusCode, data)
	}
	// deleted key no longer authenticates
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("deleted key: expected 401, got %d: %s", resp.StatusCode, data)
	}
}

func TestProxySpendAuthAndAttribution(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	srv := newAuthServer(t, backend.URL)
	defer srv.Close()

	// /spend requires master key when auth enabled.
	resp, data := getAuth(t, srv.URL+"/spend", "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("/spend without master: expected 401, got %d: %s", resp.StatusCode, data)
	}

	token := generateKey(t, srv.URL, `{}`)
	resp, data = postAuth(t, srv.URL+"/v1/chat/completions", chatBody, token)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat: expected 200, got %d: %s", resp.StatusCode, data)
	}
	resp, data = getAuth(t, srv.URL+"/spend", testMasterKey)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/spend: expected 200, got %d: %s", resp.StatusCode, data)
	}
	var report map[string]any
	_ = json.Unmarshal(data, &report)
	byKey, _ := report["spend_by_key"].(map[string]any)
	if byKey == nil || byKey[token] == nil {
		t.Fatalf("expected per-key spend attribution: %s", data)
	}
}

func TestProxyOpenModeCompat(t *testing.T) {
	backend := newBackend(t)
	defer backend.Close()
	// No master key -> open mode, no auth required, admin endpoints open.
	srv := newTestServer(t, backend.URL)
	defer srv.Close()

	resp, data := postAuth(t, srv.URL+"/v1/chat/completions", chatBody, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("open mode chat: expected 200, got %d: %s", resp.StatusCode, data)
	}
	resp, data = getAuth(t, srv.URL+"/spend", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("open mode /spend: expected 200, got %d: %s", resp.StatusCode, data)
	}
	// Generating a key activates auth; afterwards admin endpoints need the master.
	resp, data = postAuth(t, srv.URL+"/key/generate", `{}`, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("open mode key/generate: expected 200, got %d: %s", resp.StatusCode, data)
	}
	resp, data = getAuth(t, srv.URL+"/spend", "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("after key exists /spend: expected 401, got %d: %s", resp.StatusCode, data)
	}
}
