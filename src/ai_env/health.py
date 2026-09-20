import urllib.request
from typing import Dict

from .config import OLLAMA_HOST, OLLAMA_PORT, ROOCODE_PORT


HEALTH_ENDPOINTS = {
    "Ollama": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1/models",
    "Roo Code": f"http://127.0.0.1:{ROOCODE_PORT}/health",
}


def check_url(name: str, url: str) -> Dict[str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.status
            content = response.read().decode("utf-8", errors="ignore")
        return {"name": name, "ok": "true", "status": str(status), "body": content}
    except Exception as exc:
        return {"name": name, "ok": "false", "error": str(exc)}


def run_health_check() -> Dict[str, Dict[str, str]]:
    results = {}
    for name, url in HEALTH_ENDPOINTS.items():
        results[name] = check_url(name, url)
    return results
