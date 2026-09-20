import * as vscode from "vscode";
import { request as httpRequest } from "http";
import { request as httpsRequest } from "https";
import { URL } from "url";

export class AICoderCompletionProvider implements vscode.InlineCompletionItemProvider {
  private getEndpoint(): string {
    const config = vscode.workspace.getConfiguration("aiCoder");
    return config.get<string>("endpoint") || "http://127.0.0.1:7861";
  }

  private requestJson<T>(
    method: "POST",
    pathname: string,
    body: Record<string, unknown>
  ): Promise<T> {
    const endpoint = this.getEndpoint();
    const url = new URL(pathname, endpoint);
    const isHttps = url.protocol === "https:";
    const request = isHttps ? httpsRequest : httpRequest;
    return new Promise((resolve, reject) => {
      const payload = JSON.stringify(body);
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
          timeout: 4_000, // short timeout for autocompletions to avoid UI lags
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
      req.write(payload);
      req.end();
    });
  }

  public async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList> {
    const config = vscode.workspace.getConfiguration("aiCoder");
    const enabled = config.get<boolean>("enableAutocomplete") ?? true;
    if (!enabled) {
      return [];
    }

    // Extract prefix context
    const offset = document.offsetAt(position);
    const start = Math.max(0, offset - 2000);
    const prefix = document.getText(new vscode.Range(
      document.positionAt(start),
      position
    ));

    if (!prefix.trim()) {
      return [];
    }

    try {
      const model = config.get<string>("autocompleteModel") || "qwen2.5-coder:7b";
      const data = await this.requestJson<{ response: string; error?: string }>(
        "POST",
        "/api/complete",
        {
          prompt: prefix,
          max_tokens: 64,
          temperature: 0.2,
          model: model,
        }
      );

      if (token.isCancellationRequested || !data.response || data.error) {
        return [];
      }

      const completionText = data.response;
      const item = new vscode.InlineCompletionItem(
        completionText,
        new vscode.Range(position, position)
      );
      return [item];
    } catch (err) {
      // Fail silently to avoid interrupting the editor flow
      return [];
    }
  }
}
