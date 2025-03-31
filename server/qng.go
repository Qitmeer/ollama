package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/models/mllama"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
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

	model, err := GetModel(name.String())
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
		sched.expireRunner(model)

		return http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		}, nil
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		return http.StatusBadRequest, nil, errors.New("raw mode does not support template, system, or context")
	}

	caps := []Capability{CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, CapabilityInsert)
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

	isMllama := checkMllamaModelFamily(model)
	if isMllama && len(req.Images) > 1 {
		return http.StatusBadRequest, nil, errors.New("this model only supports one image: more than one image sent")
	}

	images := make([]llm.ImageData, len(req.Images))
	for i := range req.Images {
		if isMllama && len(model.ProjectorPaths) > 0 {
			data, opts, err := mllama.Preprocess(bytes.NewReader(req.Images[i]))
			if err != nil {
				return http.StatusInternalServerError, nil, errors.New("error processing image")
			}

			ar, ok := opts["aspectRatioIndex"].(int)
			if !ok {
				return http.StatusInternalServerError, nil, errors.New("error processing image")
			}

			buf := new(bytes.Buffer)
			err = binary.Write(buf, binary.LittleEndian, data)
			if err != nil {
				return http.StatusInternalServerError, nil, errors.New("error processing image")
			}

			images[i] = llm.ImageData{ID: i, Data: buf.Bytes(), AspectRatioID: ar}
		} else {
			images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
		}
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
				if isMllama {
					imgPrompt = "<|image|>"
				}
				msgs = append(msgs, api.Message{Role: "user", Content: fmt.Sprintf("[img-%d]"+imgPrompt, i.ID)})
			}

			values.Messages = append(msgs, api.Message{Role: "user", Content: req.Prompt})
		}

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

	slog.Debug("generate request", "images", len(images), "prompt", prompt)

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
				Model:      req.Model,
				CreatedAt:  time.Now().UTC(),
				Response:   cr.Content,
				Done:       cr.Done,
				DoneReason: cr.DoneReason,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
			}

			if _, err := sb.WriteString(cr.Content); err != nil {
				ch <- gin.H{"error": err.Error()}
			}

			if cr.Done {
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

	var gr api.GenerateResponse
	var sb strings.Builder
	for rr := range ch {
		switch t := rr.(type) {
		case api.GenerateResponse:
			sb.WriteString(t.Response)
			gr = t
		case gin.H:
			msg, ok := t["error"].(string)
			if !ok {
				msg = "unexpected error format in response"
			}
			return http.StatusInternalServerError, nil, errors.New(msg)
		default:
			return http.StatusInternalServerError, nil, errors.New("unexpected response")
		}
	}
	gr.Response = sb.String()
	return http.StatusOK, gr, nil
}

func scheduleRunner(ctx context.Context, name string, caps []Capability, requestOpts map[string]any, keepAlive *api.Duration, sched *Scheduler) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	model, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
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
