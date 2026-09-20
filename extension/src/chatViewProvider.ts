import * as vscode from "vscode";
import { request as httpRequest } from "http";
import { request as httpsRequest } from "https";
import { URL } from "url";

export class AICoderChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "ai-coder-chat-view";
  private _view?: vscode.WebviewView;

  constructor(private readonly _extensionUri: vscode.Uri) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (data) => {
      switch (data.type) {
        case "sendMessage": {
          await this._handleSendMessage(data.value);
          break;
        }
        case "resetChat": {
          await this._handleResetChat();
          break;
        }
        case "checkStatus": {
          await this._handleCheckStatus();
          break;
        }
      }
    });

    // Run initial status check
    this._handleCheckStatus();
  }

  private getEndpoint(): string {
    const config = vscode.workspace.getConfiguration("aiCoder");
    return config.get<string>("endpoint") || "http://127.0.0.1:7861";
  }

  private requestJson<T>(
    method: "GET" | "POST",
    pathname: string,
    body?: Record<string, unknown>
  ): Promise<T> {
    const endpoint = this.getEndpoint();
    const url = new URL(pathname, endpoint);
    const isHttps = url.protocol === "https:";
    const request = isHttps ? httpsRequest : httpRequest;
    return new Promise((resolve, reject) => {
      const payload = body ? JSON.stringify(body) : "";
      const req = request(
        {
          method,
          hostname: url.hostname,
          port: url.port || (isHttps ? 443 : 80),
          path: url.pathname + (url.search || ""),
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(payload).toString(),
          },
          timeout: 60_000,
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (c) => chunks.push(c));
          res.on("end", () => {
            const text = Buffer.concat(chunks).toString("utf-8");
            try {
              resolve(JSON.parse(text) as T);
            } catch (err) {
              reject(new Error(`Invalid JSON: ${text.slice(0, 200)}`));
            }
          });
        }
      );
      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy(new Error("Request timed out"));
      });
      if (payload) {
        req.write(payload);
      }
      req.end();
    });
  }

  private async _handleSendMessage(message: string) {
    if (!this._view) return;

    // Send user message to Webview log
    this._view.webview.postMessage({ type: "addMessage", role: "user", text: message });

    try {
      this._view.webview.postMessage({ type: "setLoading", value: true });
      const data = await this.requestJson<{ response: string; error?: string }>(
        "POST",
        "/api/chat",
        { message, max_tokens: 1024, temperature: 0.7 }
      );
      if (data.error) {
        this._view.webview.postMessage({
          type: "addMessage",
          role: "assistant",
          text: `Error: ${data.error}`,
        });
      } else {
        this._view.webview.postMessage({
          type: "addMessage",
          role: "assistant",
          text: data.response || "(empty response)",
        });
      }
    } catch (err) {
      this._view.webview.postMessage({
        type: "addMessage",
        role: "assistant",
        text: `Error contacting backend: ${(err as Error).message}. Make sure Python Web UI is running on ${this.getEndpoint()}`,
      });
    } finally {
      this._view.webview.postMessage({ type: "setLoading", value: false });
    }
  }

  private async _handleResetChat() {
    if (!this._view) return;
    try {
      await this.requestJson<any>("POST", "/api/reset");
      this._view.webview.postMessage({ type: "clearChat" });
      this._view.webview.postMessage({
        type: "addMessage",
        role: "assistant",
        text: "Conversation history cleared.",
      });
    } catch (err) {
      vscode.window.showErrorMessage(`Failed to reset chat: ${(err as Error).message}`);
    }
  }

  private async _handleCheckStatus() {
    if (!this._view) return;
    try {
      const status = await this.requestJson<any>("GET", "/api/status");
      this._view.webview.postMessage({
        type: "status",
        online: true,
        modelName: status.model_name || "—",
        device: status.device || "—",
      });
    } catch (err) {
      this._view.webview.postMessage({
        type: "status",
        online: false,
        modelName: "offline",
        device: "—",
      });
    }
  }

  private _getHtmlForWebview(webview: vscode.Webview): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Coder Chat</title>
  <style>
    body {
      background-color: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      margin: 0;
      padding: 12px;
      display: flex;
      flex-direction: column;
      height: 100vh;
      box-sizing: border-box;
    }
    
    .status-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 8px;
      background-color: var(--vscode-editor-lineHighlightBackground);
      border-radius: 4px;
      font-size: 0.85em;
      margin-bottom: 12px;
    }
    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 6px;
    }
    .status-online { background-color: var(--vscode-testing-iconPassedColor, #28a745); }
    .status-offline { background-color: var(--vscode-testing-iconFailedColor, #dc3545); }

    .chat-container {
      flex: 1;
      overflow-y: auto;
      margin-bottom: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding-bottom: 8px;
    }

    .message {
      padding: 8px 10px;
      border-radius: 6px;
      max-width: 90%;
      word-wrap: break-word;
      line-height: 1.4;
    }
    .message.user {
      align-self: flex-end;
      background-color: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .message.assistant {
      align-self: flex-start;
      background-color: var(--vscode-editor-lineHighlightBackground);
      border: 1px solid var(--vscode-widget-border, rgba(255, 255, 255, 0.1));
    }
    .message strong {
      display: block;
      font-size: 0.8em;
      opacity: 0.8;
      margin-bottom: 4px;
    }

    .composer {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    textarea {
      width: 100%;
      min-height: 60px;
      max-height: 150px;
      background-color: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border: 1px solid var(--vscode-input-border, transparent);
      border-radius: 4px;
      padding: 8px;
      box-sizing: border-box;
      resize: vertical;
      font-family: inherit;
    }
    textarea:focus {
      outline: 1px solid var(--vscode-focusBorder);
    }

    .button-row {
      display: flex;
      gap: 8px;
    }
    button {
      flex: 1;
      padding: 8px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background-color: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    button.primary:hover {
      background-color: var(--vscode-button-hoverBackground);
    }
    button.secondary {
      background-color: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
    }
    button.secondary:hover {
      background-color: var(--vscode-button-secondaryHoverBackground);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .loading-dots span {
      animation: blink 1.4s infinite both;
      font-weight: bold;
    }
    .loading-dots span:nth-child(2) { animation-delay: .2s; }
    .loading-dots span:nth-child(3) { animation-delay: .4s; }
    @keyframes blink {
      0% { opacity: .2; }
      20% { opacity: 1; }
      100% { opacity: .2; }
    }
  </style>
</head>
<body>
  <div class="status-bar">
    <div>
      <span id="status-dot" class="status-dot status-offline"></span>
      <span id="status-text">Checking...</span>
    </div>
    <div id="status-model" style="opacity: 0.8; font-size: 0.9em;">—</div>
  </div>

  <div id="chat-container" class="chat-container">
    <div class="message assistant">
      <strong>Assistant</strong>
      Welcome! Ask your local coding assistant anything.
    </div>
  </div>

  <div class="composer">
    <textarea id="chat-input" placeholder="Type a message... (Ctrl+Enter to send)"></textarea>
    <div class="button-row">
      <button id="reset-btn" class="secondary">Reset</button>
      <button id="send-btn" class="primary">Send</button>
    </div>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    const chatContainer = document.getElementById("chat-container");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const statusModel = document.getElementById("status-model");

    let isOffline = true;

    // Periodically poll backend status
    setInterval(() => {
      vscode.postMessage({ type: "checkStatus" });
    }, 10000);

    sendBtn.addEventListener("click", sendMessage);
    resetBtn.addEventListener("click", () => {
      vscode.postMessage({ type: "resetChat" });
    });

    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendMessage();
      }
    });

    function sendMessage() {
      const text = chatInput.value.trim();
      if (!text) return;
      vscode.postMessage({ type: "sendMessage", value: text });
      chatInput.value = "";
    }

    window.addEventListener("message", (event) => {
      const msg = event.data;
      switch (msg.type) {
        case "addMessage": {
          const div = document.createElement("div");
          div.className = "message " + msg.role;
          div.innerHTML = "<strong>" + (msg.role === "user" ? "You" : "Assistant") + "</strong>" + escapeHtml(msg.text).replace(/\\n/g, "<br>");
          chatContainer.appendChild(div);
          chatContainer.scrollTop = chatContainer.scrollHeight;
          break;
        }
        case "clearChat": {
          chatContainer.innerHTML = "";
          break;
        }
        case "setLoading": {
          sendBtn.disabled = msg.value;
          chatInput.disabled = msg.value;
          if (msg.value) {
            const loadingDiv = document.createElement("div");
            loadingDiv.id = "loading-bubble";
            loadingDiv.className = "message assistant";
            loadingDiv.innerHTML = "<strong>Assistant</strong><div class='loading-dots'>Thinking<span>.</span><span>.</span><span>.</span></div>";
            chatContainer.appendChild(loadingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
          } else {
            const bubble = document.getElementById("loading-bubble");
            if (bubble) bubble.remove();
          }
          break;
        }
        case "status": {
          if (msg.online) {
            statusDot.className = "status-dot status-online";
            statusText.textContent = "Online (" + msg.device + ")";
            statusModel.textContent = msg.modelName;
          } else {
            statusDot.className = "status-dot status-offline";
            statusText.textContent = "Offline";
            statusModel.textContent = "—";
          }
          break;
        }
      }
    });

    function escapeHtml(text) {
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>`;
  }
}
