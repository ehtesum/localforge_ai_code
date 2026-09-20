from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
ROOCODE_PORT = 9999

DEFAULT_MODELS = ["qwen2.5-coder:7b", "qwen3:8b"]
DEFAULT_MODEL = "qwen2.5-coder:7b"

ROO_CONFIG_PATH = ROOT / "roocode" / "roocode-ollama-config.json"
DOCKER_COMPOSE_PATH = ROOT / "docker-compose.yml"
OLLAMA_DATA_PATH = ROOT / "ollama_data"
