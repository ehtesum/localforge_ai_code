# 💻 LocalForge AI Code — Private On-Device Coding Assistant

## 💡 What is this in simple words?

Tools like GitHub Copilot and Cursor send your private code files across the internet to big cloud servers.

**LocalForge** is your own **100% private, self-hosted coding AI**:
- Runs fine-tuned coding models (like **Qwen2.5-Coder** or models running in **Ollama**) entirely on your computer's GPU or CPU.
- Works as a **browser IDE**, an **interactive terminal (CLI)**, or as a **VS Code extension**.
- Understands your project files, suggests code fixes, generates diffs, and writes functions—with **zero code leaving your machine**.

---

## 📁 What does each file & script do?

| File / Folder | What it actually does |
| :--- | :--- |
| **`launch.py`** | **The main launcher!** Starts the LocalForge AI server and optionally pops open the web interface in your browser. |
| **`src/ai_env/server.py`** | The backend HTTP server that routes requests between the UI, VS Code, and the local AI model. |
| **`src/ai_env/agent.py`** | The AI brain using PyTorch & Hugging Face to run Qwen2.5-Coder on your NVIDIA graphics card. |
| **`src/ai_env/ollama_agent.py`** | Alternative AI brain that talks to your locally running Ollama server (`http://localhost:11434`). |
| **`src/ai_env/workspace.py`** | Reads files in your project, checks git status, and safely applies code diffs. |
| **`src/ai_env/crawler.py`** | Scans your codebase to find related functions and classes to give the AI context before it answers. |
| **`src/ai_env/cli.py`** | Interactive command-line chat (talk to your coding assistant right in PowerShell). |
| **`src/ai_env/train.py`** | Allows you to fine-tune (LoRA) the AI model on your own private code patterns. |
| **`extension/`** | The VS Code extension plugin source code. |

---

## 🚀 How to Run It (Step-by-Step)

### Option 1: Web Interface (Easiest)

1. Open PowerShell and go to this directory:
   ```powershell
   cd c:\Users\user\Desktop\projects\localforge_ai_code
   ```
2. Activate your virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Start the application:
   ```powershell
   python launch.py
   ```
   This will launch the LocalForge server and open your browser to the web chat interface.

---

### Option 2: Command-Line Interface (Terminal Chat)

If you just want to ask coding questions inside your terminal:
```powershell
python -m ai_env --cli
```
You can type questions like:
- *"Explain what launch.py does"*
- *"Write a Python script to resize images"*

---

### Option 3: Using with Ollama

If you already have Ollama installed with a model like `qwen2.5-coder` or `llama3.2`:
```powershell
python launch.py --backend ollama --model qwen2.5-coder
```
LocalForge will connect to your local Ollama instance and use it for fast code completions.

---

## Technical Limitations

- **Consumer VRAM Ceilings**: Loading 7B parameter models unquantized requires >14 GB VRAM; consumer GPUs (4–8 GB VRAM) require 4-bit AWQ or GGUF quantization, constraining concurrent active context to 8,192 tokens.
- **Multi-File Symbol Resolution**: The workspace crawler parses single-file Tree-sitter AST nodes accurately, but cross-file transitive type dependencies across complex monorepos require manual context pinning.
- **Inline Keystroke Autocomplete Latency**: Real-time ghost-text autocomplete requires sub-100 ms latency; on CPU-only setups, inference times of 200–500 ms limit practical usage to chat and function generation modes.

---

## Future Roadmap

- [ ] **LSP (Language Server Protocol) Daemon Bridge**: Connect directly to native language servers (`gopls`, `pyright`, `tsserver`) to ingest exact compile-time type diagnostics.
- [ ] **vLLM / FlashAttention-2 Paged KV-Cache**: Integrate a high-throughput local inference backend with continuous batching and PagedAttention for sub-50 ms completion latency.
- [ ] **Iterative Test-Execution Loop**: Implement an autonomous agent mode that executes unit tests, captures stack traces, and iterates on code diffs until tests pass.
- [ ] **Repository Embedding Graph**: Build an offline LanceDB / sqlite-vec vector store caching symbol embeddings across git revisions.

