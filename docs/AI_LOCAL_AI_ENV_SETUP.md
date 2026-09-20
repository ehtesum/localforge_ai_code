# Local AI Development Environment for Windows 11

This document defines a production-ready local AI development environment optimized for a Windows 11 laptop with:

- Ryzen 7 4800HS
- 24 GB RAM
- GTX 1650 4 GB

Target stack:

- Ollama
- Roo Code
- VS Code
- Qwen3:8b
- Qwen2.5-Coder:7b

## Goals

- Deliver a reproducible local environment for AI-assisted coding and inference
- Provide native Windows setup plus optional Docker compose deployment
- Configure Roo Code to talk to Ollama as the LLM backend
- Include startup, health check, model management, and rollback artifacts
- Optimize for low VRAM and CPU-first inference

## Recommended Architecture

1. Native Ollama service running locally on `http://127.0.0.1:11434`
2. Roo Code configured to use Ollama as its model host
3. VS Code workspace with recommended settings and extensions
4. Optional Docker Compose deployment for isolated service execution

## Native Installation Steps

### 1. Install prerequisites

1. Install Windows 11 updates and GPU drivers.
   - Use NVIDIA GeForce Experience or the NVIDIA website.
   - Confirm CUDA support for GTX 1650.

2. Install Docker Desktop for Windows (optional).
   - Use WSL2 backend.
   - Enable `Expose daemon on tcp://localhost:2375` only if needed by other tools.

3. Install Ollama CLI.
   - Recommended: download from https://ollama.com/install.
   - Verify with:
     ```powershell
     ollama --version
     ```

4. Install Roo Code.
   - If Roo Code is distributed as a native installer or ZIP, place its executable in `%USERPROFILE%\bin` or a folder on `PATH`.
   - There is no fixed package name; follow the latest Roo Code installation instructions.

5. Install VS Code.
   - Use the stable VS Code release.
   - Add the workspace folder for this project.

### 2. Configure Ollama for the local environment

Use Ollama's service mode and local pull workflow.

```powershell
ollama serve --host 127.0.0.1 --port 11434
```

Pull the preferred models with low-memory settings.

```powershell
ollama pull qwen3:8b --device cpu --precision int4
ollama pull qwen2.5-coder:7b --device cpu --precision int4
```

> Note: On a GTX 1650 with 4 GB VRAM, full GPU inference for 7B and 8B models is unlikely to fit reliably. Use CPU inference with `int4` quantization for the most stable local experience.

### 3. Configure Roo Code for Ollama

Place the Roo Code configuration in `roocode/roocode-ollama-config.json`.
This file points Roo Code to the Ollama host and defines the two local models.

### 4. Configure VS Code settings

A workspace-level VS Code config is included at `.vscode/settings.json` and `.vscode/extensions.json`.
Use VS Code to open this repository and accept the recommended extensions.

## Optional Docker Compose Deployment

A `docker-compose.yml` file is included for isolated deployment.
The Compose stack runs:

- `ollama` service on port `11434`
- `roocode` service on port `9999`

This option is useful for sandboxed environments and consistent service restarts.

## Startup Scripts

Scripts are provided in `scripts/`:

- `install_native.py` — bootstrap native Windows requirements
- `start_services.py` — launch Ollama and Roo Code in the background
- `stop_services.py` — stop Ollama and Roo Code processes
- `healthcheck.py` — verify service readiness
- `manage_models.py` — pull, remove, list, and inspect Ollama models
- `rollback.py` — remove a local model and stop services for recovery
- `train_model.py` — fine-tune a model from repository training data with resumable checkpoints

## Health Checks

A dedicated health check script tests:

- Ollama API reachability
- Roo Code HTTP endpoint availability
- Model list and status

Use it before development sessions to ensure the stack is healthy.

## Model Management

The `manage_models.py` script supports:

- `pull qwen3:8b`
- `pull qwen2.5-coder:7b`
- `remove <model>`
- `list`

These should be run from PowerShell with the repository root as the working directory.

## Performance Optimization

### Key tuning decisions for 24 GB RAM / GTX 1650 4 GB

- Prefer CPU inference for 7B and 8B models on this hardware.
- Use quantized model representations (`int4` or `fp16`) when available.
- Keep batch size at `1`.
- Limit prompt length and response length to reduce memory pressure.
- Use Ollama's service process instead of repeated CLI runs.
- If GPU is used, reserve no more than 3.2 GB of VRAM and avoid multiple models loaded simultaneously.

### Suggested Ollama invocation

```powershell
ollama serve --host 127.0.0.1 --port 11434 --no-logs
```

For model execution, use:

```powershell
ollama run qwen2.5-coder:7b --device cpu --precision int4 --prompt "<your prompt>"
```

## Performance Expectations

- `Qwen2.5-Coder:7b` on CPU + int4: 0.8–1.5 tokens/sec depending on prompt shape.
- `Qwen3:8b` on CPU + int4: 0.5–1 tokens/sec.
- GPU inference on GTX 1650 may be unstable for these sizes; if it works, expect 1–2 tokens/sec and high VRAM pressure.
- Model warm-up times may be 30–60 seconds on first execution.
- Keep only one large model loaded at a time.

## Troubleshooting

### Ollama service fails to start

- Verify `ollama` binary is on `PATH`.
- Confirm port `11434` is free.
- Check driver and CUDA compatibility for the GTX 1650.
- Run `ollama serve --host 127.0.0.1 --port 11434` manually and inspect errors.

### Model pull fails

- Use CPU mode instead of GPU mode.
- Delete corrupted cache in `ollama_data` before retrying.
- Confirm there is sufficient disk space.

### Roo Code cannot connect to Ollama

- Confirm Ollama is reachable at `http://127.0.0.1:11434`.
- Verify Roo Code config points to the same URL.
- Check `healthcheck.py` output for connectivity errors.

## Rollback and Recovery

### Rollback bad model changes

Use the rollback script to stop services and remove a local Ollama model:

```powershell
python .\scripts\rollback.py qwen3:8b
python .\scripts\rollback.py qwen2.5-coder:7b
```

For a model-only removal without service restart, use:

```powershell
python .\scripts\manage_models.py remove qwen3:8b
python .\scripts\manage_models.py remove qwen2.5-coder:7b
```

### Rollback service startup

Stop services:

```powershell
python .\scripts\stop_services.py
```

Restart with minimal mode:

```powershell
python .\scripts\start_services.py
```

### Rollback to Docker sandbox

If native mode is unstable, switch to Docker compose:

```powershell
docker compose up -d
```

## File map

- `docker-compose.yml`
- `.vscode/settings.json`
- `.vscode/extensions.json`
- `roocode/roocode-ollama-config.json`
- `scripts/install_native.py`
- `scripts/start_services.py`
- `scripts/stop_services.py`
- `scripts/healthcheck.py`
- `scripts/manage_models.py`
- `scripts/train_model.py`
