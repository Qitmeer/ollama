package server

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"
)

func FixBlobs(dir string) error {
	return fixBlobs(dir)
}

func (s *Scheduler) UnloadAllRunners() {
	s.unloadAllRunners()
}

func (m *Manifest) Digest() string {
	return m.digest
}

func (m *Manifest) FI() os.FileInfo {
	return m.fi
}

func Pull(ctx context.Context, req *api.PullRequest) (int, interface{}, error) {
	name := model.ParseName(cmp.Or(req.Model, req.Name))
	if !name.IsValid() {
		return 0, nil, errors.New(errtypes.InvalidModelNameErrMsg)
	}

	name, err := getExistingName(name)
	if err != nil {
		return 0, nil, err
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		pctx, cancel := context.WithCancel(ctx)
		defer cancel()

		if err := PullModel(pctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- err
		}
	}()

	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			if r.Status == "success" {
				return http.StatusOK, r, nil
			}
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			if errorMsg, ok := r["error"].(string); ok {
				return status, r, errors.New(errorMsg)
			} else {
				return status, r, errors.New("unexpected error format in progress response")
			}
		default:
			return http.StatusInternalServerError, r, errors.New("unexpected progress response")
		}
	}
	return http.StatusInternalServerError, nil, errors.New("unexpected end of progress response")
}

func Generate(ctx context.Context, req *api.GenerateRequest, sched *Scheduler) (int, interface{}, error) {
	useStream := false
	req.Stream = &useStream
	checkpointStart := time.Now()
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		// Ideally this is "invalid model name" but we're keeping with
		// what the API currently returns until we can change it.
		return http.StatusNotFound, nil, fmt.Errorf("model '%s' not found", req.Model)
	}

	// We cannot currently consolidate this into GetModel because all we'll
	// induce infinite recursion given the current code structure.
	name, err := getExistingName(name)
	if err != nil {
		return http.StatusNotFound, nil, fmt.Errorf("model '%s' not found", req.Model)
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			return http.StatusNotFound, nil, fmt.Errorf("model '%s' not found", req.Model)
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			return http.StatusBadRequest, nil, err
		default:
			return http.StatusInternalServerError, nil, err
		}
	}

	// expire the runner
	if req.Prompt == "" && req.KeepAlive != nil && int(req.KeepAlive.Seconds()) == 0 {
		sched.expireRunner(m)

		return http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		}, nil
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		return http.StatusBadRequest, nil, fmt.Errorf("raw mode does not support template, system, or context")
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}
	if req.Think != nil && *req.Think {
		caps = append(caps, model.CapabilityThinking)
		// TODO(drifkin): consider adding a warning if it's false and the model
		// doesn't support thinking. It's not strictly required, but it can be a
		// hint that the user is on an older qwen3/r1 model that doesn't have an
		// updated template supporting thinking
	}

	r, m, opts, err := scheduleRunner(ctx, name.String(), caps, req.Options, req.KeepAlive, sched)
	if errors.Is(err, errCapabilityCompletion) {
		return http.StatusBadRequest, nil, fmt.Errorf("%q does not support generate", req.Model)
	} else if err != nil {
		return http.StatusInternalServerError, nil, err
	}

	checkpointLoaded := time.Now()

	// load the model
	if req.Prompt == "" {
		return http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		}, nil
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		return http.StatusBadRequest, nil, fmt.Errorf("this model only supports one image while more than one image requested")
	}

	images := make([]llm.ImageData, len(req.Images))
	for i := range req.Images {
		images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
	}

	prompt := req.Prompt
	if !req.Raw {
		tmpl := m.Template
		if req.Template != "" {
			tmpl, err = template.Parse(req.Template)
			if err != nil {
				return http.StatusInternalServerError, nil, err
			}
		}

		var values template.Values
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			var msgs []api.Message
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			} else if m.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: m.System})
			}

			if req.Context == nil {
				msgs = append(msgs, m.Messages...)
			}

			for _, i := range images {
				imgPrompt := ""
				msgs = append(msgs, api.Message{Role: "user", Content: fmt.Sprintf("[img-%d]"+imgPrompt, i.ID)})
			}

			values.Messages = append(msgs, api.Message{Role: "user", Content: req.Prompt})
		}

		values.Think = req.Think != nil && *req.Think
		values.IsThinkSet = req.Think != nil

		var b bytes.Buffer
		if req.Context != nil {
			slog.Warn("the context field is deprecated and will be removed in a future version of Ollama")
			s, err := r.Detokenize(ctx, req.Context)
			if err != nil {
				return http.StatusInternalServerError, nil, err
			}
			b.WriteString(s)
		}

		if err := tmpl.Execute(&b, values); err != nil {
			return http.StatusInternalServerError, nil, err
		}

		prompt = b.String()
	}

	var thinkingState *thinking.Parser
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	if req.Think != nil && *req.Think && openingTag != "" && closingTag != "" {
		thinkingState = &thinking.Parser{
			OpeningTag: openingTag,
			ClosingTag: closingTag,
		}
	}

	ch := make(chan any)
	go func() {
		// TODO (jmorganca): avoid building the response twice both here and below
		var sb strings.Builder
		defer close(ch)
		if err := r.Completion(ctx, llm.CompletionRequest{
			Prompt:  prompt,
			Images:  images,
			Format:  req.Format,
			Options: opts,
		}, func(cr llm.CompletionResponse) {
			res := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Response:  cr.Content,
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
			}

			if thinkingState != nil {
				thinking, content := thinkingState.AddContent(cr.Content)
				res.Thinking = thinking
				res.Response = content
			}

			if _, err := sb.WriteString(cr.Content); err != nil {
				ch <- gin.H{"error": err.Error()}
			}

			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

				if !req.Raw {
					tokens, err := r.Tokenize(ctx, prompt+sb.String())
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
					res.Context = tokens
				}
			}

			ch <- res
		}); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var r api.GenerateResponse
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.GenerateResponse:
				sbThinking.WriteString(t.Thinking)
				sbContent.WriteString(t.Response)
				r = t
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				return http.StatusInternalServerError, nil, fmt.Errorf("%s", msg)
			default:
				return http.StatusInternalServerError, nil, errors.New("unexpected response")
			}
		}

		r.Thinking = sbThinking.String()
		r.Response = sbContent.String()

		return http.StatusOK, r, nil
	}
	return http.StatusInternalServerError, nil, errors.New("unexpected response")
}

func scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration, sched *Scheduler) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	model, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
	}

	if slices.Contains(model.Config.ModelFamilies, "mllama") && len(model.ProjectorPaths) > 0 {
		return nil, nil, nil, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	if err := model.CheckCapabilities(caps...); err != nil {
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	opts, err := modelOptions(model, requestOpts)
	if err != nil {
		return nil, nil, nil, err
	}

	runnerCh, errCh := sched.GetRunner(ctx, model, opts, keepAlive)
	var runner *runnerRef
	select {
	case runner = <-runnerCh:
	case err = <-errCh:
		return nil, nil, nil, err
	}

	return runner.llama, model, &opts, nil
}
