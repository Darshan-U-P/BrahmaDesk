const { _electron: electron } = require("playwright");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const assert = require("node:assert/strict");

const projectDir = path.resolve(__dirname, "..");
const artifactsDir = path.join(projectDir, "test-results", "ui");
const baseUrl = "http://127.0.0.1:5173";
const apiUrl = process.env.BRAHMA_API_URL || "http://127.0.0.1:8000";
const tests = [];
let failures = 0;
let passed = 0;
let skipped = 0;
let viteProcess;
let app;
let window;

function log(status, name, detail = "") {
  console.log(`[${status}] ${name}${detail ? ` — ${detail}` : ""}`);
  if (status === "PASS") passed++;
  else if (status === "FAIL") failures++;
  else skipped++;
}

function test(name, fn) { tests.push({ name, fn }); }

function waitForHttp(url, timeoutMs = 45000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const probe = () => {
      const req = http.get(url, res => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) return resolve();
        retry();
      });
      req.setTimeout(1500, () => req.destroy());
      req.on("error", retry);
      function retry() {
        if (Date.now() - started > timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
        } else setTimeout(probe, 500);
      }
    };
    probe();
  });
}

async function screenshot(name) {
  fs.mkdirSync(artifactsDir, { recursive: true });
  await window.screenshot({ path: path.join(artifactsDir, name), fullPage: true });
}

test("Home window renders", async () => {
  await window.getByText("Think clearly.", { exact: false }).waitFor({ timeout: 15000 });
  await window.getByRole("textbox", { name: /Message Brahma/i }).waitFor();
  await screenshot("01-home.png");
});

test("Backend connection indicator is visible", async () => {
  const body = await window.locator("body").innerText();
  assert.match(body, /Local model connected|Model disconnected|Qwen/i);
});

test("Create a new chat", async () => {
  const newChat = window.getByRole("button", { name: /New chat/i }).first();
  await newChat.click();
  await window.waitForTimeout(400);
  await screenshot("02-new-chat.png");
});

test("Thinking budget control is present", async () => {
  const body = await window.locator("body").innerText();
  assert.match(body, /Thinking/i, "Thinking selector label not found");

  // Native <select> option labels do not always appear in body.innerText.
  const selects = window.locator("select");
  let matched = false;
  for (let i = 0; i < await selects.count(); i++) {
    const labels = await selects.nth(i).locator("option").allTextContents();
    const normalized = labels.map(value => value.trim().toLowerCase());
    if (["off", "low", "medium", "high", "max"].every(value => normalized.includes(value))) {
      matched = true;
      break;
    }
  }
  if (!matched) {
    // Also support a custom menu/button-based implementation.
    const trigger = window.getByText(/Thinking\\s+(Off|Low|Medium|High|Max)/i).first();
    await trigger.waitFor({ timeout: 3000 });
    await trigger.click();
    const menuText = (await window.locator("body").innerText()).toLowerCase();
    for (const option of ["off", "low", "medium", "high", "max"]) {
      assert.ok(menuText.includes(option), `Expected reasoning budget option '${option}'`);
    }
  }
  await screenshot("03-thinking-budget.png");
});

test("Settings can be opened", async () => {
  const settings = window.getByRole("button", { name: /Settings/i }).first();
  await settings.click();
  await window.waitForTimeout(250);
  const body = await window.locator("body").innerText();
  assert.match(body, /Sampling|Penalties|General|Display|Settings/i);
  await screenshot("04-settings.png");
});

test("Sampling & Penalties page is reachable", async () => {
  const body = await window.locator("body").innerText();
  const sampling = window.getByText(/Sampling\s*&\s*Penalties/i).first();
  await sampling.waitFor({ timeout: 5000 });
  await sampling.click();
  await window.waitForTimeout(250);
  const updatedBody = await window.locator("body").innerText();
  assert.match(updatedBody, /Temperature/i);
  await screenshot("05-sampling-penalties.png");
});

test("Chat sends a message and receives a response", async () => {
  // Close settings if a close button is present; otherwise the composer may remain available.
  const close = window.getByRole("button", { name: /Close settings|Close/i }).first();
  if (await close.count()) await close.click().catch(() => {});
  const composer = window.getByRole("textbox", { name: /Message Brahma/i });
  await composer.waitFor({ timeout: 5000 });
  await composer.fill("Reply with exactly UI_AUTOMATION_OK");
  const send = window.getByRole("button", { name: /^Send$/i });
  await send.click();
  await window.getByText(/UI_AUTOMATION_OK/i).waitFor({ timeout: 90000 });
  await screenshot("06-chat-response.png");
});

(async () => {
  fs.mkdirSync(artifactsDir, { recursive: true });
  console.log("Brahma Desk Electron UI tests");
  console.log(`Project: ${projectDir}`);
  console.log(`API: ${apiUrl}`);
  try {
    await waitForHttp(`${apiUrl}/api/health`, 5000);
  } catch {
    throw new Error(`Backend is not reachable at ${apiUrl}. Start the backend first.`);
  }

  // Start only Vite; do not use "npm run dev", which launches a second Electron window.
  viteProcess = spawn(process.platform === "win32" ? "npx.cmd" : "npx",
    ["vite", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
    { cwd: projectDir, stdio: "inherit", windowsHide: true });
  try {
    await waitForHttp(baseUrl, 45000);
    app = await electron.launch({ args: [projectDir], cwd: projectDir, timeout: 45000 });
    window = await app.firstWindow();
    window.setDefaultTimeout(10000);

    for (const item of tests) {
      try {
        await item.fn();
        log("PASS", item.name);
      } catch (err) {
        log("FAIL", item.name, err && err.message ? err.message.replace(/\s+/g, " ").slice(0, 500) : String(err));
        try {
          const safe = item.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
          await screenshot(`failure-${safe}.png`);
        } catch {}
      }
    }
  } finally {
    if (app) await app.close().catch(() => {});
    if (viteProcess && !viteProcess.killed) {
      viteProcess.kill();
      if (process.platform === "win32" && viteProcess.pid) {
        spawn("taskkill", ["/pid", String(viteProcess.pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
      }
    }
  }

  console.log("\n------------- UI TEST SUMMARY -------------");
  console.log(`Passed: ${passed} | Failed: ${failures} | Skipped: ${skipped}`);
  console.log(`Screenshots: ${artifactsDir}`);
  process.exitCode = failures ? 1 : 0;
})().catch(err => {
  console.error(`UI test setup failed: ${err.stack || err}`);
  if (viteProcess && !viteProcess.killed) viteProcess.kill();
  if (app) app.close().catch(() => {});
  process.exit(2);
});
