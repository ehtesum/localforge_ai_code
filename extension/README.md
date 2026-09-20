# AI Coder - Local Coding Agent & VSCode Extension

Use your fine-tuned Qwen2.5-Coder model as a coding agent (like Claude) and integrate it into VSCode.

## Part 1: Coding Agent (Interactive Chat)

### Quick Start

```bash
# Interactive chat mode
python scripts/agent.py

# Single prompt mode
python scripts/agent.py --prompt "Write a Python function to sort a list"
```

### Interactive Mode

Commands in interactive mode:

- `help` - Show help
- `clear` - Clear conversation history
- `quit` - Exit

### Examples

```
You: Write a Python function that validates email addresses
Assistant: [generates code]

You: How do I handle errors in that function?
Assistant: [explains error handling]

You: Refactor it to use regex
Assistant: [shows refactored version]
```

### Single Prompt Mode

```bash
python scripts/agent.py --prompt "Debug this code: [code here]"
python scripts/agent.py --prompt "Explain what this does" --max-tokens 500
```

## Part 2: VSCode Extension

### Setup

1. Navigate to the extension directory:

```bash
cd extension
```

2. Install dependencies:

```bash
npm install
```

3. Compile TypeScript:

```bash
npm run compile
```

4. Load into VSCode:
   - Open VSCode
   - Press `Ctrl+Shift+D` (Run and Debug)
   - Click "Run Extension"
   - This opens a new VSCode window with the extension loaded

### Usage in VSCode

**Keyboard Shortcut:**

- Press `Ctrl+Shift+A` (or `Cmd+Shift+A` on Mac) to ask the AI coder

**Commands (via Command Palette - `Ctrl+Shift+P`):**

- `Ask AI Coder` - General coding question
- `Explain Code` - Explain selected code (right-click or select text first)
- `Debug Code` - Find issues in selected code
- `Generate Code` - Generate code from description

**Right-Click Context Menu:**

- Select code and right-click to see "Explain Code" or "Debug Code" options

### Example Workflow in VSCode

1. Select a function in your code
2. Right-click → "Explain Code"
3. Result appears in the "AI Coder" output panel
4. Press `Ctrl+Shift+A` to ask follow-up questions

## Advanced Options

### Training Data

Prepare data from web snapshots for training:

```bash
python scripts/prepare_web_data.py \
  --urls https://en.wikipedia.org/wiki/Artificial_intelligence \
         https://en.wikipedia.org/wiki/Machine_learning
```

### Training the Model

```bash
python scripts/train_model.py --num-train-epochs 10
```

The model auto-saves checkpoints. Stop and restart anytime to resume.

### Customizing the Agent

Edit the system prompt in `src/ai_env/agent.py`:

```python
SYSTEM_PROMPT = """Your custom instructions here..."""
```

## Architecture

```
┌─────────────────────────────────────┐
│   VSCode Extension (TypeScript)     │
│  - Commands & UI Integration        │
└────────────┬────────────────────────┘
             │ calls
             ▼
┌─────────────────────────────────────┐
│   Python Agent (agent.py)           │
│  - Conversation History             │
│  - System Prompts                   │
│  - Multi-turn Context               │
└────────────┬────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────┐
│   Fine-tuned Qwen2.5-Coder Model    │
│   (in ./checkpoints)                │
└─────────────────────────────────────┘
```

## Troubleshooting

**"Module not found" error in VSCode:**

- Make sure Python and the venv are in PATH
- Edit extension to use full Python path: `/path/to/.venv/bin/python`

**Extension not showing commands:**

- Run `npm run compile` first
- Reload the extension window with `Ctrl+Shift+P` → "Reload Window"

**Model loading slow:**

- First load downloads the model (~7GB for full Qwen2.5-Coder-7B)
- After that it's cached and loads instantly
- Use a smaller model if needed: `--model-name Qwen/Qwen2.5-Coder-1.5B`

## Next Steps

- [ ] Extend with web search integration
- [ ] Add code generation presets (test writing, docs, refactoring)
- [ ] Create API server for remote access
- [ ] Add streaming responses
- [ ] Deploy to VSCode Marketplace (public version)
