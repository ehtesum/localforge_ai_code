"""
AI Coder Agent - Interactive coding assistant using fine-tuned Qwen2.5-Coder
Maintains conversation history and provides code-specific prompting.
"""
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = """You are an expert AI coder assistant. You help write, debug, and explain code.
CRITICAL: You must write down your thinking process before providing the final answer. Wrap your thinking process in <thought>...</thought> tags, for example:
<thought>
I need to write a quicksort function.
1. Define base case...
2. Partition the array...
3. Recursively sort...
</thought>
After the closing </thought> tag, provide your final response to the user. Keep the final response clean, well-structured, and concise. Do not repeat the thoughts in the final response."""


class CodingAgent:
    def __init__(self, checkpoint_dir="./checkpoints", model_name="Qwen/Qwen2.5-Coder-7B"):
        """Initialize the coding agent with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_context_length = 4096
        self.tokens_used = 0
        self.total_requests = 0
        self.workspace_root: Path | None = None
        self.last_response_source = "local model"
        self.last_response_text = ""
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print("[AGENT] Loading model...")
        
        model_path = None
        if os.path.exists(self.checkpoint_dir):
            subdirs = [d for d in os.listdir(self.checkpoint_dir) if d.startswith("checkpoint-")]
            if subdirs:
                latest_checkpoint = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
                model_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
                print(f"[AGENT] Using checkpoint: {latest_checkpoint}")
            else:
                if any(os.scandir(self.checkpoint_dir)):
                    model_path = self.checkpoint_dir
                    print(f"[AGENT] Using base model directory")
                else:
                    print(f"[AGENT] Checkpoint directory is empty: {self.checkpoint_dir}")
        else:
            print(f"[AGENT] Checkpoint directory not found: {self.checkpoint_dir}")

        if model_path is None:
            print(f"[AGENT] Falling back to remote model name: {self.model_name}")
            model_path = self.model_name
        
        def load_tokenizer(path: str) -> None:
            try:
                return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception as primary_error:
                print("[AGENT] Primary tokenizer load failed, retrying with use_fast=False...")
                try:
                    return AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        use_fast=False,
                    )
                except Exception as fallback_error:
                    print(f"[AGENT] Fallback tokenizer load failed: {fallback_error}")
                    raise primary_error

        try:
            self.tokenizer = load_tokenizer(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"[AGENT] Model loaded on {self.device}")
        except Exception as e:
            print(f"[AGENT] Error loading model: {e}")
            raise
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_context(self) -> str:
        """Build conversation context with history."""
        context_parts = [f"System: {SYSTEM_PROMPT}\n"]
        
        # Add conversation history
        for msg in self.conversation_history[-20:]:  # Keep last 20 messages
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}\n")
            elif msg["role"] == "assistant":
                context_parts.append(f"Assistant: {msg['content']}\n")
        
        return "".join(context_parts) + "Assistant: "
    
    def generate(self, user_input: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate response from the model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.add_to_history("user", user_input)
        
        context = self.get_context()
        
        # Tokenize
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the new part
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_response[len(context):].strip()

        prompt_tokens = input_ids.size(1)
        total_tokens = output[0].size(0)
        self.tokens_used += total_tokens
        self.total_requests += 1
        self.last_response_text = response
        self.last_response_source = "local model"

        self.add_to_history("assistant", response)
        return response

    # Ensure the model_loader import exists
    def complete(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> str:
        """Use the loaded transformers model directly for single-turn code completion."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        from .model_loader import generate_text
        return generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=self.device,
        )

    def get_status(self) -> dict:
        """Return current agent / model status metadata."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None,
            "tokens_used": self.tokens_used,
            "conversation_turns": len(self.conversation_history),
            "total_requests": self.total_requests,
            "workspace_root": str(self.workspace_root or Path.cwd()),
            "last_response_source": self.last_response_source,
            "last_response_text": self.last_response_text,
        }

    def set_workspace_root(self, root_path: str | Path) -> None:
        """Set the root directory that the web UI may browse and edit."""
        self.workspace_root = Path(root_path).resolve()

    def reset_history(self) -> None:
        """Clear conversation history and usage counters."""
        self.conversation_history = []
        self.tokens_used = 0
        self.total_requests = 0

    def chat(self):
        """Interactive chat loop."""
        print("\n[AGENT] AI Coder Agent Ready!")
        print("[AGENT] Commands: 'help' for help, 'clear' to clear history, 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("[AGENT] Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("[AGENT] Conversation history cleared.\n")
                    continue
                
                if user_input.lower() == "help":
                    print("""
[AGENT] Commands:
  help     - Show this message
  clear    - Clear conversation history
  quit     - Exit the agent
  
Examples:
  'Write a Python function to sort a list'
  'Debug this code: ...'
  'Explain what this does: ...'
  'How do I handle errors in JavaScript?'
                    """)
                    continue
                
                print("\n[AGENT] Generating response...")
                response = self.generate(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n[AGENT] Interrupted. Type 'quit' to exit or continue.\n")
            except Exception as e:
                print(f"[AGENT] Error: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Coder Agent - Interactive coding assistant")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B", help="Base model name")
    parser.add_argument("--prompt", help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    agent = CodingAgent(args.checkpoint_dir, args.model_name)
    agent.load_model()
    
    if args.prompt:
        # Single prompt mode
        response = agent.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        print(f"\nAssistant: {response}\n")
    else:
        # Interactive mode
        agent.chat()


if __name__ == "__main__":
    main()
