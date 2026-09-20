const apiRoutes = {
  status: "/api/status",
  files: "/api/files",
  file: "/api/file",
  chat: "/api/chat",
  reset: "/api/reset",
  diff: "/api/diff",
  commitPreview: "/api/commit-preview",
  upload: "/api/upload",
  sessions: "/api/sessions",
  deleteSession: "/api/sessions/delete",
};

let selectedFile = null;
let editor = null;
let allFiles = [];
let openTabs = [{ id: "overview", name: "Agent Overview", type: "overview" }];
let activeTabId = "overview";

let currentSessionId = null;
let allSessions = [];

const modeMap = {
  py: "python",
  js: "javascript",
  ts: "javascript",
  jsx: "javascript",
  tsx: "javascript",
  html: "xml",
  htm: "xml",
  css: "css",
  json: "javascript",
  md: "markdown",
  markdown: "markdown",
  sh: "shell",
  bash: "shell",
  yml: "yaml",
  yaml: "yaml",
  xml: "xml",
};

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const suffixes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / 1024 ** i).toFixed(1)} ${suffixes[i]}`;
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function showToast(message, type = "success") {
  const container = document.getElementById("toasts");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

function appendMessage(role, text, meta, thinkingTimeSec = null) {
  const container = document.getElementById("messages");
  const message = document.createElement("div");
  message.className = `message ${role}`;
  
  const title = document.createElement("strong");
  title.textContent = role === "user" ? "You" : "Assistant";
  message.appendChild(title);

  const body = document.createElement("div");

  if (role === "assistant" && typeof text === "string") {
    let thoughtText = "";
    let replyText = text;

    const thoughtStart = text.indexOf("<thought>");
    const thoughtEnd = text.indexOf("</thought>");

    if (thoughtStart !== -1 && thoughtEnd !== -1 && thoughtEnd > thoughtStart) {
      thoughtText = text.substring(thoughtStart + 9, thoughtEnd).trim();
      replyText = (text.substring(0, thoughtStart) + text.substring(thoughtEnd + 10)).trim();
    }

    if (thoughtText) {
      const details = document.createElement("details");
      details.className = "thought-accordion";
      
      const summary = document.createElement("summary");
      summary.className = "thought-header";
      
      const timeStr = thinkingTimeSec !== null ? `for ${thinkingTimeSec} seconds` : "";
      summary.innerHTML = `
        <span class="thought-header-left">
          <span class="thought-icon">🧠</span>
          <span>Thought ${timeStr}</span>
        </span>
        <span class="thought-toggle-icon">▶</span>
      `;
      
      const pre = document.createElement("pre");
      pre.className = "thought-body";
      pre.textContent = thoughtText;
      
      details.appendChild(summary);
      details.appendChild(pre);
      body.appendChild(details);
    }

    // Render remaining reply as Markdown
    const replyDiv = document.createElement("div");
    replyDiv.className = "reply-markdown";
    if (window.marked && typeof window.marked.parse === "function") {
      replyDiv.innerHTML = window.marked.parse(replyText);
      // Highlight code blocks
      if (window.hljs) {
        replyDiv.querySelectorAll("pre code").forEach((block) => {
          window.hljs.highlightElement(block);
        });
      }
    } else {
      replyDiv.textContent = replyText;
    }
    body.appendChild(replyDiv);
  } else {
    body.textContent = text;
  }

  message.appendChild(body);

  if (meta) {
    const note = document.createElement("div");
    note.className = "message-meta";
    note.textContent = meta;
    message.appendChild(note);
  }

  container.appendChild(message);
  container.scrollTop = container.scrollHeight;
}

function setEditorPath(path) {
  const header = document.getElementById("editor-path");
  header.textContent = path ? `Editing: ${path}` : "No file selected";
}

function setSelectedFileElement(path) {
  document.querySelectorAll(".file-item").forEach((item) => {
    item.classList.toggle("selected", item.dataset.path === path);
  });
}

function showEditor(active) {
  const emptyState = document.getElementById("editor-empty");
  if (active) {
    emptyState.style.display = "none";
  } else {
    emptyState.style.display = "flex";
  }
}

function renderFileList(files = allFiles) {
  const search = document.getElementById("file-search").value.trim().toLowerCase();
  const fileList = document.getElementById("file-list");
  fileList.innerHTML = "";
  const filtered = files.filter((file) => file.path.toLowerCase().includes(search));
  if (filtered.length === 0) {
    fileList.textContent = "No files found in workspace.";
    return;
  }

  const sorted = filtered.sort((a, b) => {
    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
    return a.path.localeCompare(b.path, undefined, { numeric: true, sensitivity: "base" });
  });

  for (const file of sorted) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "file-item";
    item.dataset.path = file.path;
    item.textContent = `${file.is_dir ? "[Dir]" : "[File]"} ${file.path}`;
    item.disabled = file.is_dir;
    item.addEventListener("click", () => openFile(file.path));
    fileList.appendChild(item);
  }
}

function fileMode(path) {
  const extension = path.split(".").pop()?.toLowerCase() || "";
  return modeMap[extension] || "text/plain";
}

function updateResponseSidebar(status) {
  document.getElementById("response-source").textContent = status.last_response_source || "local model";
  const snippet = status.last_response_text?.slice(0, 180) || "No response yet.";
  document.getElementById("response-snippet").textContent = snippet;
}

async function refreshStatus() {
  try {
    const status = await request(apiRoutes.status);
    document.getElementById("model-name").textContent = status.model_name || "—";
    document.getElementById("device-name").textContent = status.device || "—";
    document.getElementById("model-loaded").textContent = status.loaded ? "Yes" : "No";
    document.getElementById("tokens-used").textContent = status.tokens_used ?? 0;
    document.getElementById("chat-turns").textContent = status.conversation_turns ?? 0;
    document.getElementById("cpu-count").textContent = status.resources?.cpu_count ?? "—";
    document.getElementById("disk-free").textContent = formatBytes(status.resources?.disk_free || 0);
    document.getElementById("root-path").textContent = status.workspace_root || "—";
    document.getElementById("landing-status-text").textContent = status.loaded ? "Services ready" : "Model unavailable";
    document.getElementById("landing-root-text").textContent = status.workspace_root || "—";
    updateResponseSidebar(status);
  } catch (error) {
    document.getElementById("landing-status-text").textContent = "Offline";
    document.getElementById("landing-root-text").textContent = "—";
    showToast(error.message, "error");
  }
}

async function refreshFiles() {
  try {
    const data = await request(apiRoutes.files);
    allFiles = data.files || [];
    document.getElementById("file-count").textContent = `${allFiles.length} items`;
    renderFileList(allFiles);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderTabs() {
  const tabBar = document.getElementById("tab-bar");
  tabBar.innerHTML = "";

  if (openTabs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "tab-empty";
    empty.textContent = "No file open. Pick a file from the Explorer or press Ctrl+N to create one.";
    tabBar.appendChild(empty);
    return;
  }

  for (const tab of openTabs) {
    const tabEl = document.createElement("div");
    tabEl.className = `tab ${tab.id === activeTabId ? "active" : ""}`;
    
    const label = document.createElement("span");
    label.textContent = tab.name;
    label.addEventListener("click", () => switchTab(tab.id));
    tabEl.appendChild(label);

    if (tab.id !== "overview") {
      const closeBtn = document.createElement("span");
      closeBtn.className = "tab-close";
      closeBtn.textContent = "×";
      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        closeTab(tab.id);
      });
      tabEl.appendChild(closeBtn);
    }

    tabBar.appendChild(tabEl);
  }
}

async function switchTab(tabId) {
  activeTabId = tabId;
  renderTabs();

  const overviewContainer = document.getElementById("overview-container");
  const editorContainer = document.getElementById("editor-container");

  if (tabId === "overview") {
    overviewContainer.style.display = "block";
    editorContainer.style.display = "none";
    selectedFile = null;
    setSelectedFileElement(null);
    setEditorPath("");
    await refreshDiff();
  } else {
    overviewContainer.style.display = "none";
    editorContainer.style.display = "block";
    
    const tab = openTabs.find(t => t.id === tabId);
    if (tab && tab.type === "file") {
      selectedFile = tab.path;
      setSelectedFileElement(selectedFile);
      
      try {
        const payload = await request(`${apiRoutes.file}?path=${encodeURIComponent(selectedFile)}`);
        editor.setValue(payload.content);
        editor.setOption("mode", fileMode(selectedFile));
        editor.setOption("readOnly", false);
        document.getElementById("save-file-button").disabled = false;
        document.getElementById("file-editor").removeAttribute("disabled");
        showEditor(true);
        setEditorPath(selectedFile);
        await refreshDiff();
      } catch (error) {
        showToast(error.message, "error");
      }
    }
  }
}

function closeTab(tabId) {
  const index = openTabs.findIndex(t => t.id === tabId);
  if (index === -1) return;

  openTabs.splice(index, 1);

  if (activeTabId === tabId) {
    if (openTabs.length > 0) {
      const nextActiveIndex = Math.min(index, openTabs.length - 1);
      switchTab(openTabs[nextActiveIndex].id);
    } else {
      switchTab("overview");
    }
  } else {
    renderTabs();
  }
}

async function openFile(path) {
  const existingTab = openTabs.find(t => t.id === path);
  if (!existingTab) {
    const filename = path.split("/").pop();
    openTabs.push({ id: path, name: filename, type: "file", path: path });
  }
  await switchTab(path);
}

async function saveFile() {
  if (!selectedFile) return;
  const content = editor.getValue();
  
  if (window.BrainVisualizer) {
    window.BrainVisualizer.triggerFileSave();
  }

  try {
    await request(apiRoutes.file, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: selectedFile, content }),
    });
    showToast(`Saved ${selectedFile}`);
    appendMessage("assistant", `Saved file: ${selectedFile}`);
    await refreshDiff();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";

  const temperature = parseFloat(document.getElementById("temperature").value);
  const maxTokens = parseInt(document.getElementById("max-tokens").value, 10);
  document.getElementById("send-button").disabled = true;

  // Trigger visual sparks
  if (window.BrainVisualizer) {
    window.BrainVisualizer.triggerChatPrompt();
    window.BrainVisualizer.triggerThinkingStart();
  }

  const messagesContainer = document.getElementById("messages");
  const thinkingIndicator = document.createElement("div");
  thinkingIndicator.id = "thinking-indicator";
  thinkingIndicator.className = "thinking-indicator";
  thinkingIndicator.innerHTML = `
    <div class="thinking-spinner"></div>
    <span>Thinking... (0s)</span>
  `;
  messagesContainer.appendChild(thinkingIndicator);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    const span = thinkingIndicator.querySelector("span");
    if (span) {
      span.textContent = `Thinking... (${elapsed}s)`;
    }
  }, 1000);

  try {
    const result = await request(apiRoutes.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        message: text, 
        temperature, 
        max_tokens: maxTokens,
        session_id: currentSessionId
      }),
    });

    clearInterval(timerInterval);
    thinkingIndicator.remove();

    if (result.session_id && currentSessionId !== result.session_id) {
      currentSessionId = result.session_id;
    }
    await refreshSessions();

    const elapsedSeconds = Math.max(1, Math.round((Date.now() - startTime) / 1000));

    if (window.BrainVisualizer) {
      window.BrainVisualizer.triggerThinkingEnd(result.response_source);
    }

    appendMessage("assistant", result.response, `Source: ${result.response_source || "local model"}`, elapsedSeconds);
    await refreshStatus();
  } catch (error) {
    clearInterval(timerInterval);
    thinkingIndicator.remove();
    
    if (window.BrainVisualizer) {
      window.BrainVisualizer.triggerThinkingEnd("error");
    }

    appendMessage("assistant", `Error: ${error.message}`);
  } finally {
    document.getElementById("send-button").disabled = false;
  }
}

async function refreshDiff() {
  if (!selectedFile) {
    document.getElementById("diff-output").textContent = "Select a file and save it to show changes.";
    return;
  }

  const diffPayload = await request(`${apiRoutes.diff}?path=${encodeURIComponent(selectedFile)}`);
  document.getElementById("diff-output").textContent = diffPayload.diff || "No diff available.";
  await refreshCommitPreview();
}

async function refreshCommitPreview() {
  const preview = await request(apiRoutes.commitPreview);
  document.getElementById("commit-summary").value = preview.summary || "No preview available.";
}

async function createWorkspaceItem(kind) {
  const modal = document.getElementById("modal");
  const input = document.getElementById("modal-input");
  const title = document.getElementById("modal-title");
  title.textContent = kind === "folder" ? "New folder" : "New file";
  input.value = "";
  modal.dataset.kind = kind;
  modal.hidden = false;
  input.focus();
}

function updateUploadList(files) {
  const list = document.getElementById("upload-file-list");
  list.innerHTML = "";
  for (const file of files) {
    const chip = document.createElement("span");
    chip.className = "upload-chip";
    chip.textContent = file.name;
    list.appendChild(chip);
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const formData = new FormData();
  const targetDir = "uploads";
  formData.append("path", targetDir);
  for (const file of files) {
    formData.append("files", file, file.name);
  }

  try {
    const result = await request(apiRoutes.upload, {
      method: "POST",
      body: formData,
    });
    showToast(`Uploaded ${result.files.length} file(s)`);
    await refreshFiles();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function submitCreateItem() {
  const modal = document.getElementById("modal");
  const input = document.getElementById("modal-input");
  const path = input.value.trim();
  if (!path) return;
  const kind = modal.dataset.kind;
  const targetPath = kind === "folder" ? `${path.replace(/\/+$/g, "")}/.gitkeep` : path;
  await request(apiRoutes.file, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: targetPath, content: "" }),
  });
  modal.hidden = true;
  await refreshFiles();
  if (kind !== "folder") {
    await openFile(targetPath);
  }
}

function closeModal() {
  const modal = document.getElementById("modal");
  modal.hidden = true;
}

async function loadSession(sessionId) {
  try {
    const session = await request(`${apiRoutes.sessions}?id=${encodeURIComponent(sessionId)}`);
    currentSessionId = session.id;
    
    document.getElementById("messages").innerHTML = "";
    
    if (session.messages && session.messages.length > 0) {
      for (const msg of session.messages) {
        appendMessage(msg.role, msg.content);
      }
    } else {
      document.getElementById("messages").innerHTML = `<div class="empty-state small">Ask the agent anything about your code…</div>`;
    }
    
    renderSessionsList();
    toggleHistoryPanel(false);
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function createNewSession() {
  try {
    const session = await request(apiRoutes.sessions, { method: "POST" });
    await loadSession(session.id);
    await refreshSessions();
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function deleteSession(sessionId, e) {
  e.stopPropagation();
  if (!confirm("Delete this chat session?")) return;
  try {
    await request(apiRoutes.deleteSession, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: sessionId }),
    });
    if (currentSessionId === sessionId) {
      currentSessionId = null;
    }
    await refreshSessions();
    if (!currentSessionId && allSessions.length > 0) {
      await loadSession(allSessions[0].id);
    } else if (!currentSessionId) {
      await createNewSession();
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function refreshSessions() {
  try {
    const data = await request(apiRoutes.sessions);
    allSessions = data.sessions || [];
    renderSessionsList();
  } catch (err) {
    console.error("Failed to load sessions:", err);
  }
}

function renderSessionsList() {
  const list = document.getElementById("chat-history-list");
  list.innerHTML = "";
  if (allSessions.length === 0) {
    list.innerHTML = `<div class="empty-state small">No past sessions found.</div>`;
    return;
  }
  for (const s of allSessions) {
    const d = new Date(s.updated_at * 1000);
    const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    const item = document.createElement("div");
    item.className = `chat-session-item ${s.id === currentSessionId ? "active" : ""}`;
    item.onclick = () => loadSession(s.id);
    
    const info = document.createElement("div");
    info.className = "chat-session-info";
    info.innerHTML = `
      <span class="chat-session-title">${s.title}</span>
      <span class="chat-session-date">${dateStr}</span>
    `;
    
    const delBtn = document.createElement("span");
    delBtn.className = "chat-session-delete";
    delBtn.innerHTML = "×";
    delBtn.onclick = (e) => deleteSession(s.id, e);
    
    item.appendChild(info);
    item.appendChild(delBtn);
    list.appendChild(item);
  }
}

function toggleHistoryPanel(show) {
  const panel = document.getElementById("chat-history-panel");
  if (typeof show === 'boolean') {
    panel.hidden = !show;
  } else {
    panel.hidden = !panel.hidden;
  }
}

function attachEvents() {
  document.getElementById("send-button").addEventListener("click", sendMessage);
  
  const historyBtn = document.getElementById("history-button");
  if (historyBtn) historyBtn.addEventListener("click", () => toggleHistoryPanel());
  
  const newChatBtn = document.getElementById("new-chat-button");
  if (newChatBtn) newChatBtn.addEventListener("click", () => createNewSession());
  
  const closeHistoryBtn = document.getElementById("close-history-button");
  if (closeHistoryBtn) closeHistoryBtn.addEventListener("click", () => toggleHistoryPanel(false));

  document.getElementById("refresh-files").addEventListener("click", refreshFiles);
  document.getElementById("save-file-button").addEventListener("click", saveFile);
  document.getElementById("refresh-diff-button").addEventListener("click", refreshDiff);
  document.getElementById("new-file-btn").addEventListener("click", () => createWorkspaceItem("file"));
  document.getElementById("new-folder-btn").addEventListener("click", () => createWorkspaceItem("folder"));
  document.getElementById("modal-cancel").addEventListener("click", closeModal);
  document.getElementById("modal-ok").addEventListener("click", submitCreateItem);
  document.getElementById("upload-trigger-btn").addEventListener("click", () => {
    document.getElementById("upload-input").click();
  });
  document.getElementById("upload-input").addEventListener("change", (event) => {
    const files = Array.from(event.target.files || []);
    updateUploadList(files);
    uploadFiles(files);
    event.target.value = "";
  });
  const dropzone = document.getElementById("upload-dropzone");
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer?.files || []);
    updateUploadList(files);
    uploadFiles(files);
  });
  document.getElementById("launch-workspace-btn").addEventListener("click", () => {
    document.getElementById("workspace-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  document.getElementById("scroll-to-connector-btn").addEventListener("click", () => {
    document.getElementById("connector-section").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  document.getElementById("help-btn").addEventListener("click", () => {
    document.getElementById("help-overlay").hidden = false;
  });
  document.getElementById("help-close").addEventListener("click", () => {
    document.getElementById("help-overlay").hidden = true;
  });
  document.getElementById("file-search").addEventListener("input", () => renderFileList(allFiles));
  document.getElementById("temperature").addEventListener("input", (event) => {
    const value = parseFloat(event.target.value).toFixed(2);
    document.getElementById("temperature-value").textContent = value;
  });
  document.getElementById("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
}

function initEditor() {
  const textarea = document.getElementById("file-editor");
  editor = CodeMirror.fromTextArea(textarea, {
    theme: "material-darker",
    lineNumbers: true,
    indentUnit: 2,
    tabSize: 2,
    lineWrapping: true,
    mode: "text/plain",
    autoCloseBrackets: true,
    matchBrackets: true,
    viewportMargin: Infinity,
    extraKeys: { "Ctrl-S": () => saveFile(), "Cmd-S": () => saveFile() },
  });
}

async function init() {
  attachEvents();
  initEditor();
  renderTabs();
  if (window.BrainVisualizer) {
    window.BrainVisualizer.init();
  }
  await refreshStatus();
  await refreshFiles();
  await refreshCommitPreview();
  await refreshSessions();
  if (allSessions.length > 0) {
    await loadSession(allSessions[0].id);
  } else {
    await createNewSession();
  }
  if (!currentSessionId) {
    appendMessage("assistant", "Ready. Use the chat panel to ask your custom AI coding agent and browse files on the right.");
  }
}

window.addEventListener("load", init);
