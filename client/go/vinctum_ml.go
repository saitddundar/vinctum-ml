// Package vinctumml provides a Go client for the Vinctum ML API.
package vinctumml

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// NodeMetrics matches the Python NodeMetrics schema.
type NodeMetrics struct {
	TotalEvents   int     `json:"total_events"`
	Successes     int     `json:"successes"`
	Failures      int     `json:"failures"`
	Timeouts      int     `json:"timeouts"`
	Reroutes      int     `json:"reroutes"`
	CircuitOpens  int     `json:"circuit_opens"`
	AvgLatencyMs  float64 `json:"avg_latency_ms"`
	MinLatencyMs  float64 `json:"min_latency_ms"`
	MaxLatencyMs  float64 `json:"max_latency_ms"`
	P95LatencyMs  float64 `json:"p95_latency_ms"`
	TotalBytes    int     `json:"total_bytes"`
	AvgBytesPerOp float64 `json:"avg_bytes_per_op"`
	FailureRate   float64 `json:"failure_rate"`
	Uptime        float64 `json:"uptime"`
}

type ScoreRequest struct {
	NodeID  string      `json:"node_id"`
	Metrics NodeMetrics `json:"metrics"`
}

type ScoreResponse struct {
	NodeID     string  `json:"node_id"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
}

type AnomalyRequest struct {
	NodeID          string      `json:"node_id"`
	Metrics         NodeMetrics `json:"metrics"`
	EventsPerMinute float64     `json:"events_per_minute"`
}

type AnomalyResponse struct {
	NodeID       string  `json:"node_id"`
	IsAnomaly    bool    `json:"is_anomaly"`
	AnomalyScore float64 `json:"anomaly_score"`
}

type RouteRequest struct {
	Nodes []ScoreRequest `json:"nodes"`
}

type RouteResponse struct {
	Scores    []ScoreResponse `json:"scores"`
	BestNode  string          `json:"best_node"`
	RouteScore float64        `json:"route_score"`
}

type HealthResponse struct {
	Status       string          `json:"status"`
	ModelsLoaded map[string]bool `json:"models_loaded"`
}

// Health checks the API health status.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/health", nil)
	if err != nil {
		return nil, err
	}

	var resp HealthResponse
	if err := c.do(req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Score scores a single node for route quality.
func (c *Client) Score(ctx context.Context, nodeID string, metrics NodeMetrics) (*ScoreResponse, error) {
	body := ScoreRequest{NodeID: nodeID, Metrics: metrics}
	var resp ScoreResponse
	if err := c.post(ctx, "/score", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// DetectAnomaly checks if a node is anomalous.
func (c *Client) DetectAnomaly(ctx context.Context, nodeID string, metrics NodeMetrics, eventsPerMin float64) (*AnomalyResponse, error) {
	body := AnomalyRequest{NodeID: nodeID, Metrics: metrics, EventsPerMinute: eventsPerMin}
	var resp AnomalyResponse
	if err := c.post(ctx, "/anomaly", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ScoreRoute scores multiple nodes and picks the best route.
func (c *Client) ScoreRoute(ctx context.Context, nodes []ScoreRequest) (*RouteResponse, error) {
	body := RouteRequest{Nodes: nodes}
	var resp RouteResponse
	if err := c.post(ctx, "/route", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *Client) post(ctx context.Context, path string, body any, result any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	return c.do(req, result)
}

func (c *Client) do(req *http.Request, result any) error {
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		var errBody map[string]any
		json.NewDecoder(resp.Body).Decode(&errBody)
		return fmt.Errorf("API error %d: %v", resp.StatusCode, errBody["detail"])
	}

	return json.NewDecoder(resp.Body).Decode(result)
}
