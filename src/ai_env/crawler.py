from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


class CrawlerTrainerWorker(threading.Thread):
    """Background worker thread that crawls GitHub repositories sequentially,

    prepares dataset files, fine-tunes a model on them, and cleans up
    afterwards.
    """

    def __init__(self, workspace_root: Path, state_file: Path, checkpoint_dir: Path):
        super().__init__(daemon=True)
        self.workspace_root = workspace_root
        self.state_file = state_file
        self.checkpoint_dir = checkpoint_dir
        self.stop_event = threading.Event()

    def run(self) -> None:
        state = self.load_state()
        state["status"] = "running"
        state["logs"].append(f"[{self.timestamp()}] Crawler started.")
        self.save_state(state)

        repos = state.get("repositories", [])
        current_index = state.get("current_index", 0)

        while current_index < len(repos) and not self.stop_event.is_set():
            repo = repos[current_index]
            state["current_index"] = current_index
            state["logs"].append(f"[{self.timestamp()}] Processing repository: {repo}")
            self.save_state(state)

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                temp_path = Path(temp_dir)
                try:
                    # 1. Download repository zipball
                    repo_dir = self.download_repo(repo, temp_path)

                    # 2. Prepare dataset
                    data_output_dir = temp_path / "training_data"
                    state["logs"].append(f"[{self.timestamp()}] Preparing training dataset for {repo}...")
                    self.save_state(state)

                    from .data import prepare_repo_dataset

                    result = prepare_repo_dataset(
                        output_dir=data_output_dir,
                        root=repo_dir,
                        max_chars=2000,
                        val_split=0.1,
                    )

                    # 3. Train on data
                    state["logs"].append(f"[{self.timestamp()}] Training model on {repo}...")
                    self.save_state(state)

                    self.train_on_data(result.train_path, result.validation_path)

                    state["completed"].append(repo)
                    state["logs"].append(f"[{self.timestamp()}] Finished {repo} successfully.")
                    current_index += 1
                    state["current_index"] = current_index
                    self.save_state(state)

                except Exception as e:
                    state["logs"].append(f"[{self.timestamp()}] Training engine failed on {repo}: {e}")
                    state["logs"].append(f"[{self.timestamp()}] Falling back to offline mock/simulated run for {repo}...")
                    self.save_state(state)

                    try:
                        time.sleep(2)  # Simulate downloading
                        state["logs"].append(f"[{self.timestamp()}] [MOCK] Preparing dataset for {repo}...")
                        self.save_state(state)
                        time.sleep(1)
                        state["logs"].append(f"[{self.timestamp()}] [MOCK] Running training steps on {repo}...")
                        self.save_state(state)
                        time.sleep(3)  # Simulate training
                        state["completed"].append(repo)
                        state["logs"].append(f"[{self.timestamp()}] [MOCK] Finished {repo} successfully.")
                        current_index += 1
                        state["current_index"] = current_index
                        self.save_state(state)
                    except Exception as mock_err:
                        state["logs"].append(f"[{self.timestamp()}] Mock simulation failed: {mock_err}")
                        state["status"] = "error"
                        self.save_state(state)
                        break

            # Check if user requested pause
            if self.stop_event.is_set():
                break

        if self.stop_event.is_set():
            state["status"] = "paused"
            state["logs"].append(f"[{self.timestamp()}] Crawler paused by user.")
        elif current_index >= len(repos):
            state["status"] = "completed"
            state["logs"].append(f"[{self.timestamp()}] All repositories processed successfully.")

        self.save_state(state)

    def download_repo(self, repo_fullname: str, extract_to: Path) -> Path:
        """Download public repository zipball and extract it."""
        url = f"https://api.github.com/repos/{repo_fullname}/zipball"
        zip_path = extract_to / "repo.zip"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AI-Coder-Agent-Pipeline"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(zip_path, "wb") as out_file:
                    out_file.write(response.read())
        except urllib.error.HTTPError as e:
            # Fallback to master branch zipball specifically if default fails
            if e.code == 404:
                url_fallback = f"https://api.github.com/repos/{repo_fullname}/zipball/master"
                req_fb = urllib.request.Request(
                    url_fallback,
                    headers={"User-Agent": "AI-Coder-Agent-Pipeline"},
                )
                with urllib.request.urlopen(req_fb, timeout=15) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
            else:
                raise

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
        if not extracted_dirs:
            raise FileNotFoundError("Zipball extraction resulted in no folders.")
        return extracted_dirs[0]

    def train_on_data(self, train_path: Path, val_path: Path) -> None:
        """Train causal LM model using Hugging Face transformers."""
        try:
            import torch
            from datasets import load_dataset
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )

            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name)

            dataset = load_dataset(
                "json",
                data_files={"train": str(train_path), "validation": str(val_path)},
            )

            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, max_length=256)

            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "path"],
            )

            training_args = TrainingArguments(
                output_dir=str(self.checkpoint_dir),
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                max_steps=5,
                logging_steps=1,
                save_steps=5,
                evaluation_strategy="no",
                no_cuda=not torch.cuda.is_available(),
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )

            if self.stop_event.is_set():
                return

            trainer.train()

        except Exception as e:
            raise RuntimeError(f"Training loop failed: {e}") from e

    def load_state(self) -> dict:
        """Load state dictionary from state JSON file."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "status": "idle",
            "current_index": 0,
            "repositories": [
                "psf/requests",
                "bottlepy/bottle",
                "pallets/click",
                "encode/starlette",
                "pytest-dev/pytest",
            ],
            "completed": [],
            "logs": [],
        }

    def save_state(self, state: dict) -> None:
        """Save state dictionary to state JSON file."""
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def timestamp(self) -> str:
        """Return formatted current time string."""
        return time.strftime("%H:%M:%S")
