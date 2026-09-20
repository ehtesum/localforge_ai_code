import json
from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_MODEL, DEFAULT_MODELS, ROO_CONFIG_PATH
from .utils import write_json


DEFAULT_ROO_CODE_CONFIG: Dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "defaultModel": DEFAULT_MODEL,
        "models": {
            "qwen3:8b": {
                "name": "Qwen3 8B",
                "precision": "int4",
                "device": "cpu",
                "context": 4096,
                "temperature": 0.2,
                "maxTokens": 1024,
            },
            "qwen2.5-coder:7b": {
                "name": "Qwen2.5 Coder 7B",
                "precision": "int4",
                "device": "cpu",
                "context": 4096,
                "temperature": 0.15,
                "maxTokens": 1024,
            },
        },
    },
    "editor": {
        "suggestions": True,
        "formatOnSave": True,
        "lintOnSave": True,
    },
    "backend": {
        "healthCheckUrl": "http://127.0.0.1:11434/v1/models",
        "timeoutSeconds": 10,
    },
}


def ensure_roocode_config() -> Path:
    if not ROO_CONFIG_PATH.exists():
        write_json(ROO_CONFIG_PATH, DEFAULT_ROO_CODE_CONFIG)
    return ROO_CONFIG_PATH


def read_roocode_config() -> Dict[str, Any]:
    if not ROO_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Roo Code config not found at {ROO_CONFIG_PATH}")
    return json.loads(ROO_CONFIG_PATH.read_text(encoding="utf-8"))
