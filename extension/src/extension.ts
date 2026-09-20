import * as vscode from "vscode";
import { request as httpRequest } from "http";
import { request as httpsRequest } from "https";
import { URL } from "url";
import { AICoderChatViewProvider } from "./chatViewProvider";
import { AICoderCompletionProvider } from "./completionProvider";

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

const DEFAULT_ENDPOINT = "http://127.0.0.1:7861";

interface ChatResponse {
  response: string;
  status?: Record<string, unknown>;
  response_source?: string;
  error?: string;
}

interface StatusResponse {
  device?: string;
  model_name?: string;
  loaded?: boolean;
  workspace_root?: string;
  error?: string;
}

function getEndpoint(): string {
  const config = vscode.workspace.getConfiguration("aiCoder");
  return config.get<string>("endpoint") || DEFAULT_ENDPOINT;
}

function requestJson<T>(
  method: "GET" | "POST",
  pathname: string,
  body?: Record<string, unknown>
): Promise<T> {
  const endpoint = getEndpoint();
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
            reject(new Error(`Invalid JSON from ${url}: ${text.slice(0, 200)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy(new Error(`Request to ${url} timed out`));
    });
    if (payload) {
      req.write(payload);
    }
    req.end();
  });
}

async function checkStatus(): Promise<StatusResponse | null> {
  try {
    return await requestJson<StatusResponse>("GET", "/api/status");
  } catch (err) {
    outputChannel.appendLine(
      `[AI Coder] Cannot reach ${getEndpoint()}/api/status: ${(err as Error).message}`
    );
    return null;
  }
}

function updateStatusBar(status: StatusResponse | null) {
  if (!statusBarItem) return;
  if (!status || status.error) {
    statusBarItem.text = "$(error) AI Coder: offline";
    statusBarItem.tooltip = "AI Coder Agent not reachable. Run 'python launch.py' to start it.";
    statusBarItem.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.errorBackground"
    );
    return;
  }
  const model = status.model_name || "model";
  const backend = status.device || "?";
  statusBarItem.text = `$(check) AI Coder: ${backend} (${model})`;
  statusBarItem.tooltip =
    `AI Coder Agent\n` +
    `Endpoint: ${getEndpoint()}\n` +
    `Model:    ${model}\n` +
    `Backend:  ${backend}\n` +
    `Loaded:   ${status.loaded ? "yes" : "no"}`;
  statusBarItem.backgroundColor = undefined;
  statusBarItem.command = "ai-coder.ask";
}

async function ask(prompt: string): Promise<void> {
  if (!outputChannel) return;
  outputChannel.clear();
  outputChannel.show(true);
  outputChannel.appendLine(`[AI Coder] > ${prompt}`);
  outputChannel.appendLine("[AI Coder] contacting backend...");

  try {
    const data = await requestJson<ChatResponse>("POST", "/api/chat", {
      message: prompt,
      max_tokens: 1024,
      temperature: 0.2,
    });
    if (data.error) {
      outputChannel.appendLine(`[ERROR] ${data.error}`);
      vscode.window.showErrorMessage(`AI Coder: ${data.error}`);
      return;
    }
    outputChannel.appendLine("");
    outputChannel.appendLine(data.response || "(empty response)");
  } catch (err) {
    const message = (err as Error).message;
    outputChannel.appendLine(`[ERROR] ${message}`);
    vscode.window.showErrorMessage(
      `AI Coder request failed. Is the backend running on ${getEndpoint()} ?`
    );
  }
}

function selectedTextOrEmpty(): string {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return "";
  return editor.document.getText(editor.selection);
}

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("AI Coder");
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.show();
  context.subscriptions.push(outputChannel, statusBarItem);

  // Register Webview View Sidebar Chat Provider
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      AICoderChatViewProvider.viewType,
      new AICoderChatViewProvider(context.extensionUri)
    )
  );

  // Register Inline Auto-Completion Provider
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      new AICoderCompletionProvider()
    )
  );

  // Periodically poll the backend so the status bar reflects reality.
  const pollStatus = async () => updateStatusBar(await checkStatus());
  pollStatus();
  const timer = setInterval(pollStatus, 10_000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-coder.ask", async () => {
      const prompt = await vscode.window.showInputBox({
        placeHolder: "Ask the AI coder anything...",
        prompt: "Enter your coding question",
      });
      if (prompt) {
        await ask(prompt);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-coder.explainCode", async () => {
      const code = selectedTextOrEmpty();
      if (!code) {
        vscode.window.showWarningMessage("Select some code first.");
        return;
      }
      await ask(`Explain this code:\n\`\`\`\n${code}\n\`\`\``);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-coder.debugCode", async () => {
      const code = selectedTextOrEmpty();
      if (!code) {
        vscode.window.showWarningMessage("Select some code first.");
        return;
      }
      await ask(
        `Find any bugs and explain how to fix them:\n\`\`\`\n${code}\n\`\`\``
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-coder.generateCode", async () => {
      const prompt = await vscode.window.showInputBox({
        placeHolder: "e.g., Write a function that validates email addresses",
        prompt: "Describe the code you want to generate",
      });
      if (prompt) {
        await ask(`Write code to: ${prompt}`);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-coder.openWebUI", async () => {
      await vscode.env.openExternal(vscode.Uri.parse(getEndpoint()));
    })
  );
}

export function deactivate() {
  if (statusBarItem) {
    statusBarItem.dispose();
  }
}
