# LocalForge AI Code — Local AI Coding Assistant

A self-hosted AI coding assistant powered by fine-tuned Qwen2.5-Coder models. Runs entirely on your local machine with a browser-based IDE, interactive CLI, and VS Code extension.

## Architecture

```
localforge_ai_code/
├── src/ai_env/
│   ├── server.py          HTTP server + JSON API (ThreadingHTTPServer)
│   ├── agent.py           Qwen2.5-Coder local inference engine
│   ├── ollama_agent.py    Ollama HTTP API agent backend
│   ├── workspace.py       Git integration, file operations, system snapshots
│   ├── crawler.py         Code context crawler for project understanding
│   ├── data.py            Training data preparation pipeline
│   ├── train.py           LoRA fine-tuning on custom code datasets
│   ├── cli.py             Interactive terminal interface
│   ├── web.py             Web UI bootstrap
│   ├── model_loader.py    Model loading & checkpoint management
│   └── web_ui/            Static frontend (HTML/CSS/JS chat interface)
├── extension/             VS Code extension (TypeScript)
├── checkpoints/           Fine-tuned model weights
├── training_data/         Custom training datasets
├── start.sh               Linux startup script
├── Dockerfile             Container build definition
└── docker-compose.yml     Multi-service orchestration
```

**Data flow**: User prompt → Agent (local Qwen or Ollama) → Workspace context injection → Code diff generation → Web UI inline preview.

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (GTX 1650 minimum) or CPU-only mode
- CUDA 12.1 toolkit (for GPU acceleration)

### Installation

```bash
cd localforge_ai_code
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# For GPU inference (recommended):
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### Running

```bash
# Linux launcher (default headless server)
./start.sh

# Python manual launch
python launch.py --no-browser

# CLI mode
python -m ai_env --cli

# With Ollama backend
python launch.py --backend ollama
```

Open **http://127.0.0.1:7861** after launch.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (GTX 1650) | 8 GB (RTX 3060+) |
| Disk | 15 GB | 30 GB |
| OS | Linux / Windows | Ubuntu 22.04+ / Windows 11 |

## 🚧 Works to be Done / Future Roadmap

- [ ] **Tree-sitter AST Indexing**: Implement repository-wide AST parsing and dependency graph indexing for precise multi-file context retrieval.
- [ ] **Fill-in-the-Middle (FIM) Engine**: Integrate lightweight local models (e.g. Qwen2.5-Coder-1.5B) for real-time inline code completion.
- [ ] **Ollama Status Widget**: Add real-time model downloading and progress telemetry directly inside the Web UI topbar.
- [ ] **Interactive Git Diff Approval**: Require visual user confirmation before applying multi-file code modifications.
- [ ] **Container Hardening**: Add non-root user execution constraints to `Dockerfile` for safe sandbox execution.
