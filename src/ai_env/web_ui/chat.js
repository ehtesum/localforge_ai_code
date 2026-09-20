/**
 * Neural Network Brain Visualization for AI Coding Agent.
 * Renders an interactive 2D canvas representing the project components as a neural net,
 * with pulsing nodes, hover tooltips, and glowing "connection sparks" traversing the synapses.
 */

(function () {
  const THEME = {
    bg: "#080c1c",
    grid: "rgba(148, 163, 184, 0.03)",
    line: "rgba(139, 92, 246, 0.15)",
    lineActive: "rgba(6, 180, 212, 0.4)",
    nodeDefault: "rgba(148, 163, 184, 0.2)",
    user: "#3b82f6",
    api: "#8b5cf6",
    agent: "#ec4899",
    llm: "#10b981",
    ollama: "#f59e0b",
    workspace: "#06b6d4",
    git: "#a855f7",
    train: "#f43f5e",
  };

  const NODES = {
    user: { label: "User Interface", desc: "Monitors input prompts and edits code", x: 0.15, y: 0.75, color: THEME.user },
    api: { label: "Server API", desc: "Routes HTTP requests and coordinates views", x: 0.32, y: 0.5, color: THEME.api },
    agent: { label: "Agent Logic", desc: "Main controller, orchestrates tasks & memory", x: 0.5, y: 0.35, color: THEME.agent },
    llm: { label: "Local LLM", desc: "Local Hugging Face transformers model loader", x: 0.72, y: 0.2, color: THEME.llm },
    ollama: { label: "Ollama API", desc: "Connects to offline local Ollama service", x: 0.85, y: 0.38, color: THEME.ollama },
    workspace: { label: "Workspace Explorer", desc: "Safely reads and writes project files", x: 0.65, y: 0.68, color: THEME.workspace },
    git: { label: "Git Engine", desc: "Manages repository changes and diffs", x: 0.82, y: 0.8, color: THEME.git },
    train: { label: "Data & Training", desc: "Prepares training data and fine-tuning datasets", x: 0.28, y: 0.2, color: THEME.train },
  };

  const SYNAPSES = [
    { from: "user", to: "api" },
    { from: "api", to: "agent" },
    { from: "agent", to: "llm" },
    { from: "agent", to: "ollama" },
    { from: "agent", to: "workspace" },
    { from: "workspace", to: "git" },
    { from: "api", to: "workspace" },
    { from: "agent", to: "train" },
    { from: "train", to: "workspace" },
  ];

  let canvas, ctx;
  let dpr = window.devicePixelRatio || 1;
  let width = 0;
  let height = 0;
  let animId = null;

  // State
  let particles = [];
  let sparks = [];
  let hoveredNodeKey = null;
  let isThinking = false;
  let thinkingTimer = 0;

  // Initialize background drift particles
  function initParticles() {
    particles = [];
    for (let i = 0; i < 24; i++) {
      particles.push({
        x: Math.random(),
        y: Math.random(),
        vx: (Math.random() - 0.5) * 0.0006,
        vy: (Math.random() - 0.5) * 0.0006,
        size: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.4 + 0.1,
      });
    }
  }

  // Setup nodes state
  function setupNodes() {
    for (const key in NODES) {
      NODES[key].pulse = 0;
      NODES[key].glow = 0;
      NODES[key].currentRadius = 6;
    }
  }

  function resize() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
  }

  function getPixelPos(node) {
    // Add small responsive padding to layout
    const padX = 25;
    const padY = 25;
    return {
      x: padX + node.x * (width - padX * 2),
      y: padY + node.y * (height - padY * 2),
    };
  }

  // Fire a spark along a synapse
  function fireSpark(fromKey, toKey, color = "#a5b4fc", speed = 0.02, size = 3) {
    sparks.push({
      from: fromKey,
      to: toKey,
      progress: 0,
      speed: speed,
      color: color,
      size: size,
    });
  }

  // Trigger prompt flow sparks: User -> API -> Agent
  function triggerChatPrompt() {
    NODES.user.glow = 1.0;
    fireSpark("user", "api", THEME.user, 0.04, 3.5);
    setTimeout(() => {
      NODES.api.glow = 0.8;
      fireSpark("api", "agent", THEME.api, 0.04, 3.5);
    }, 250);
  }

  // Start thinking animation (pulses Agent & LLM nodes)
  function triggerThinkingStart() {
    isThinking = true;
  }

  // Trigger reply flow sparks: LLM/Ollama -> Agent -> API -> User
  function triggerThinkingEnd(source = "local") {
    isThinking = false;
    const targetSource = source === "ollama" ? "ollama" : "llm";
    NODES[targetSource].glow = 1.0;
    
    // Spark LLM/Ollama -> Agent
    fireSpark(targetSource, "agent", NODES[targetSource].color, 0.035, 3.5);

    setTimeout(() => {
      NODES.agent.glow = 0.9;
      fireSpark("agent", "api", THEME.agent, 0.035, 3.5);
    }, 300);

    setTimeout(() => {
      NODES.api.glow = 0.8;
      fireSpark("api", "user", THEME.api, 0.035, 3.5);
      NODES.user.glow = 1.0;
    }, 600);
  }

  // Trigger file save flow sparks: User -> API -> Workspace -> Git
  function triggerFileSave() {
    NODES.user.glow = 1.0;
    fireSpark("user", "api", THEME.user, 0.04, 3.5);
    
    setTimeout(() => {
      NODES.api.glow = 0.8;
      fireSpark("api", "workspace", THEME.api, 0.04, 3.5);
    }, 200);

    setTimeout(() => {
      NODES.workspace.glow = 0.9;
      fireSpark("workspace", "git", THEME.workspace, 0.04, 3.5);
      NODES.git.glow = 0.8;
    }, 400);

    setTimeout(() => {
      // Echo back
      fireSpark("git", "workspace", THEME.git, 0.04, 3);
    }, 650);

    setTimeout(() => {
      fireSpark("workspace", "api", THEME.workspace, 0.04, 3);
    }, 850);

    setTimeout(() => {
      fireSpark("api", "user", THEME.api, 0.04, 3);
      NODES.user.glow = 1.0;
    }, 1050);
  }

  // Draw background details
  function drawBackground() {
    // Grid Lines
    ctx.strokeStyle = THEME.grid;
    ctx.lineWidth = 1;
    const gridSpacing = 30;
    for (let x = 0; x < width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Drifting particles
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x += 1;
      if (p.x > 1) p.x -= 1;
      if (p.y < 0) p.y += 1;
      if (p.y > 1) p.y -= 1;

      ctx.fillStyle = `rgba(148, 163, 184, ${p.alpha})`;
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Draw synapses (connections)
  function drawSynapses() {
    ctx.lineWidth = 1;
    for (const s of SYNAPSES) {
      const fromPos = getPixelPos(NODES[s.from]);
      const toPos = getPixelPos(NODES[s.to]);

      // Check if synapse is currently active (has spark on it)
      const isActive = sparks.some(spark => spark.from === s.from && spark.to === s.to);

      ctx.beginPath();
      ctx.moveTo(fromPos.x, fromPos.y);
      ctx.lineTo(toPos.x, toPos.y);
      ctx.strokeStyle = isActive ? THEME.lineActive : THEME.line;
      ctx.stroke();
    }
  }

  // Draw traveling sparks
  function drawSparks() {
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i];
      s.progress += s.speed;

      if (s.progress >= 1.0) {
        // Trigger glow on target node
        NODES[s.to].glow = Math.min(NODES[s.to].glow + 0.4, 1.0);
        sparks.splice(i, 1);
        continue;
      }

      const fromPos = getPixelPos(NODES[s.from]);
      const toPos = getPixelPos(NODES[s.to]);

      // Linear interpolation
      const x = fromPos.x + (toPos.x - fromPos.x) * s.progress;
      const y = fromPos.y + (toPos.y - fromPos.y) * s.progress;

      // Draw spark with glow
      ctx.save();
      ctx.shadowBlur = 8;
      ctx.shadowColor = s.color;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(x, y, s.size, 0, Math.PI * 2);
      ctx.fill();
      
      // Minor trail particle
      if (s.progress > 0.08) {
        const tx = fromPos.x + (toPos.x - fromPos.x) * (s.progress - 0.06);
        const ty = fromPos.y + (toPos.y - fromPos.y) * (s.progress - 0.06);
        ctx.fillStyle = s.color + "99";
        ctx.beginPath();
        ctx.arc(tx, ty, s.size * 0.7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  // Draw nodes
  function drawNodes(time) {
    for (const key in NODES) {
      const n = NODES[key];
      const pos = getPixelPos(n);
      const isHovered = (key === hoveredNodeKey);

      // Decaying glow
      if (n.glow > 0.02) n.glow -= 0.012;
      else n.glow = 0;

      // Thinking pulse
      let thinkingOffset = 0;
      if (isThinking && (key === "agent" || key === "llm" || key === "ollama")) {
        thinkingOffset = Math.sin(time * 0.015) * 4;
        n.glow = Math.max(n.glow, 0.4);
      }

      // Breathing scale
      const breathe = Math.sin(time * 0.003 + n.x * 10) * 1.5;
      const r = 6 + breathe + (n.glow * 4) + thinkingOffset + (isHovered ? 2 : 0);
      n.currentRadius = r;

      // Glow Ring
      if (n.glow > 0.01 || isThinking || isHovered) {
        ctx.save();
        ctx.strokeStyle = n.color;
        ctx.globalAlpha = isHovered ? 0.6 : (isThinking ? 0.4 : n.glow * 0.5);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r * (1.5 + n.glow * 0.5), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Main Node Circle
      ctx.save();
      ctx.shadowBlur = isHovered ? 12 : (n.glow > 0 ? 8 : 4);
      ctx.shadowColor = n.color;
      ctx.fillStyle = n.color;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Label text
      ctx.fillStyle = isHovered ? "#ffffff" : "#cbd5e1";
      ctx.font = isHovered ? "bold 10px Inter, sans-serif" : "9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(n.label, pos.x, pos.y + r + 5);
    }
  }

  // Draw tooltip card for hovered node
  function drawTooltip() {
    if (!hoveredNodeKey) return;
    const n = NODES[hoveredNodeKey];
    const pos = getPixelPos(n);

    const pad = 10;
    ctx.font = "bold 10px Inter, sans-serif";
    const labelW = ctx.measureText(n.label).width;
    ctx.font = "9px Inter, sans-serif";
    const descW = ctx.measureText(n.desc).width;
    
    const cardW = Math.max(labelW, descW) + pad * 2;
    const cardH = 34;
    
    // Position helper
    let tx = pos.x - cardW / 2;
    let ty = pos.y - cardH - 12;

    if (tx < 5) tx = 5;
    if (tx + cardW > width - 5) tx = width - cardW - 5;
    if (ty < 5) ty = pos.y + n.currentRadius + 18;

    // Glassmorphic card
    ctx.save();
    ctx.fillStyle = "rgba(11, 16, 32, 0.92)";
    ctx.strokeStyle = n.color + "66";
    ctx.lineWidth = 1;
    ctx.shadowBlur = 6;
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(tx, ty, cardW, cardH, 6);
    } else {
      // Manual rounded-rect fallback for older browsers
      const rr = 6;
      ctx.moveTo(tx + rr, ty);
      ctx.arcTo(tx + cardW, ty, tx + cardW, ty + cardH, rr);
      ctx.arcTo(tx + cardW, ty + cardH, tx, ty + cardH, rr);
      ctx.arcTo(tx, ty + cardH, tx, ty, rr);
      ctx.arcTo(tx, ty, tx + cardW, ty, rr);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Text inside card
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(n.label, tx + pad, ty + 10);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "9px Inter, sans-serif";
    ctx.fillText(n.desc, tx + pad, ty + 22);
  }

  // Main Loop
  function tick(time) {
    if (!ctx) return;
    ctx.fillStyle = THEME.bg;
    ctx.fillRect(0, 0, width, height);

    drawBackground();
    drawSynapses();
    drawSparks();
    drawNodes(time);
    drawTooltip();

    // Spark thinking drift if in thinking state
    if (isThinking) {
      thinkingTimer++;
      if (thinkingTimer % 35 === 0) {
        // Spark drift between Agent & LLM/Ollama
        const target = Math.random() > 0.45 ? "llm" : "ollama";
        if (Math.random() > 0.5) {
          fireSpark("agent", target, THEME.agent, 0.03, 2);
        } else {
          fireSpark(target, "agent", NODES[target].color, 0.03, 2);
        }
      }
    }

    if (animId !== null) cancelAnimationFrame(animId);
    animId = requestAnimationFrame(tick);
  }

  // Mouse move handler for hover checks
  function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let foundKey = null;
    for (const key in NODES) {
      const pos = getPixelPos(NODES[key]);
      const dx = mx - pos.x;
      const dy = my - pos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < NODES[key].currentRadius + 10) {
        foundKey = key;
        break;
      }
    }
    hoveredNodeKey = foundKey;
  }

  function handleMouseLeave() {
    hoveredNodeKey = null;
  }

  // Initialize visualizer
  function init() {
    canvas = document.getElementById("brain-canvas");
    if (!canvas) {
      console.warn("[chat.js] Canvas 'brain-canvas' not found.");
      return;
    }

    ctx = canvas.getContext("2d");
    
    setupNodes();
    initParticles();
    resize();

    window.addEventListener("resize", resize);
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);

    if (animId !== null) cancelAnimationFrame(animId);
    animId = requestAnimationFrame(tick);

    // Initial warm sparks
    setTimeout(() => {
      fireSpark("user", "api", THEME.user, 0.02, 3);
    }, 800);
    setTimeout(() => {
      fireSpark("api", "agent", THEME.api, 0.025, 3);
    }, 1500);
  }

  // Expose API
  window.BrainVisualizer = {
    init: init,
    triggerChatPrompt: triggerChatPrompt,
    triggerThinkingStart: triggerThinkingStart,
    triggerThinkingEnd: triggerThinkingEnd,
    triggerFileSave: triggerFileSave,
    fireCustomSpark: fireSpark,
  };
})();
