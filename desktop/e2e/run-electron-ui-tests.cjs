const { _electron: electron } = require("playwright");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const assert = require("node:assert/strict");

const desktopDir = path.resolve(__dirname, "..");
const projectDir = path.resolve(desktopDir, "..");
const artifactsDir = path.join(desktopDir, "test-results", "e2e");
const baseUrl = process.env.BRAHMA_VITE_URL || "http://127.0.0.1:5173";
const apiUrl = process.env.BRAHMA_API_URL || "http://127.0.0.1:8000";
const tests = [];
let failures = 0;
let passed = 0;
let viteProcess;
let app;
let window;
let capturedChatRequest = null;

function log(status, name, detail = "") {
  console.log(`[${status}] ${name}${detail ? ` — ${detail}` : ""}`);
  if (status === "PASS") passed++;
  if (status === "FAIL") failures++;
}
function test(name, fn) { tests.push({ name, fn }); }

function waitForHttp(url, timeoutMs = 45000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const probe = () => {
      const req = http.get(url, res => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) return resolve(res.statusCode);
        retry();
      });
      req.setTimeout(1500, () => req.destroy());
      req.on("error", retry);
      function retry() {
        if (Date.now() - started > timeoutMs) return reject(new Error(`Timed out waiting for ${url}`));
        setTimeout(probe, 500);
      }
    };
    probe();
  });
}

async function screenshot(name) {
  fs.mkdirSync(artifactsDir, { recursive: true });
  await window.screenshot({ path: path.join(artifactsDir, name), fullPage: true });
}

// The visible text lives in a <span> inside each <label>; using the span's
// parent label avoids depending on CSS classes or computed accessible names.
function fieldInput(label) {
  return window.getByText(label, { exact: true }).locator("xpath=..").locator('input[type="number"]');
}

function labeledInput(label) {
  return window.getByText(label, { exact: true }).locator("xpath=..").locator("input");
}

async function closeSettingsIfOpen() {
  const dialog = window.getByRole("dialog", { name: /Settings/i });
  if ((await dialog.count()) === 0 || !(await dialog.isVisible().catch(() => false))) return;

  const closeButton = dialog.getByRole("button", { name: "Close settings", exact: true });
  if (await closeButton.count()) {
    await closeButton.click({ force: true }).catch(async () => {
      await dialog.getByRole("button", { name: /Cancel/i }).click({ force: true }).catch(() => {});
    });
  } else {
    await dialog.getByRole("button", { name: /Cancel/i }).click({ force: true }).catch(() => {});
  }
  await dialog.waitFor({ state: "hidden", timeout: 5000 }).catch(() => {});
}

function assertRequestSetting(key, expected) {
  assert.ok(capturedChatRequest, "No POST /api/chat/stream request was captured from the desktop UI");
  assert.equal(capturedChatRequest[key], expected, `UI did not send ${key}=${expected}; got ${JSON.stringify(capturedChatRequest[key])}`);
}

test("Home window renders and composer is available", async () => {
  await window.getByRole("textbox", { name: /Message Brahma/i }).waitFor({ timeout: 20000 });
  await screenshot("01-home.png");
});

test("Desktop shows local model connection state", async () => {
  const body = await window.locator("body").innerText();
  assert.match(body, /Qwen\s*[·|•]?\s*Local|Model disconnected|Local model connected/i);
});

test("Create a new chat", async () => {
  await window.getByRole("button", { name: /New chat/i }).first().click();
  await window.waitForTimeout(300);
  await screenshot("02-new-chat.png");
});

test("Thinking selector exposes all supported choices", async () => {
  const selector = window.getByRole("combobox", { name: "Reasoning budget", exact: true });
  await selector.waitFor();
  const values = await selector.locator("option").evaluateAll(nodes => nodes.map(n => n.value));
  for (const value of ["off", "low", "medium", "high", "max"]) assert.ok(values.includes(value), `Missing thinking option: ${value}`);
});

test("Sampling settings can be edited, saved, and restored from local storage", async () => {
  try {
    await window.getByRole("button", { name: /Settings/i }).first().click();
    await window.getByRole("dialog", { name: /Settings/i }).waitFor();
    await window.getByRole("button", { name: /Sampling\s*&\s*Penalties/i }).click();

    await fieldInput("Temperature").fill("0.65");
    await fieldInput("Top-K").fill("32");
    await fieldInput("Top-P").fill("0.9");
    await fieldInput("Min-P").fill("0.04");
    await labeledInput("Sampler order").fill("top_k;typ_p;top_p;min_p;temperature");
    await window.getByRole("combobox", { name: "Reasoning budget setting", exact: true }).selectOption("medium");
    await window.getByRole("button", { name: /Save settings/i }).click();
    await window.getByRole("dialog", { name: /Settings/i }).waitFor({ state: "hidden" });

    const stored = await window.evaluate(() => ({
      sampling: JSON.parse(localStorage.getItem("brahma-desk-sampling-settings-v1") || "null"),
      reasoning: localStorage.getItem("brahma-desk-reasoning-budget-v1") || localStorage.getItem("brahma-desk-reasoning-budget"),
    }));
    assert.ok(stored.sampling, "Sampling settings were not persisted to localStorage");
    assert.equal(stored.sampling.temperature, 0.65);
    assert.equal(stored.sampling.top_k, 32);
    assert.equal(stored.sampling.top_p, 0.9);
    assert.equal(stored.sampling.min_p, 0.04);
    assert.equal(stored.sampling.sampler_order, "top_k;typ_p;top_p;min_p;temperature");
    assert.equal(stored.reasoning, "medium");

    await window.getByRole("button", { name: /Settings/i }).first().click();
    await window.getByRole("button", { name: /Sampling\s*&\s*Penalties/i }).click();
    assert.equal(await fieldInput("Temperature").inputValue(), "0.65");
    assert.equal(await fieldInput("Top-K").inputValue(), "32");
    assert.equal(await window.getByRole("combobox", { name: "Reasoning budget setting", exact: true }).inputValue(), "medium");
    await screenshot("03-sampling-settings.png");
  } finally {
    // Keep a failed settings assertion from leaving a modal over the composer.
    await closeSettingsIfOpen();
  }
});

test("Real desktop chat sends saved settings to backend and receives model output", async () => {
  capturedChatRequest = null;
  const onRequest = request => {
    if (request.method() !== "POST" || !request.url().endsWith("/api/chat/stream")) return;
    try { capturedChatRequest = request.postDataJSON(); } catch { /* Report missing body in assertions below. */ }
  };
  window.on("request", onRequest);
  try {
    await window.getByRole("combobox", { name: "Reasoning budget", exact: true }).selectOption("medium");
    const composer = window.getByRole("textbox", { name: /Message Brahma/i });
    await composer.fill("Reply with exactly E2E_TEST_RESPONSE_7F3A and nothing else.");
    await window.getByRole("button", { name: /^Send$/i }).click();
    await window.getByText("E2E_TEST_RESPONSE_7F3A", { exact: true }).last().waitFor({ timeout: 120000 });
    assertRequestSetting("temperature", 0.65);
    assertRequestSetting("top_k", 32);
    assertRequestSetting("top_p", 0.9);
    assertRequestSetting("min_p", 0.04);
    assertRequestSetting("reasoning_budget", "medium");
    assertRequestSetting("sampler_order", "top_k;typ_p;top_p;min_p;temperature");
    await screenshot("04-real-chat-response.png");
  } finally {
    window.off("request", onRequest);
  }
});

(async () => {
  fs.mkdirSync(artifactsDir, { recursive: true });
  console.log("Brahma Desk full desktop E2E tests");
  console.log(`Desktop: ${desktopDir}`);
  console.log(`Backend API: ${apiUrl}`);

  const health = await fetch(`${apiUrl}/api/health`, { signal: AbortSignal.timeout(5000) }).catch(() => null);
  if (!health || !health.ok) throw new Error(`Backend is not reachable at ${apiUrl}. Run Test-BrahmaDesk-E2E.ps1 from the project root.`);
  const healthBody = await health.json();
  if (healthBody.backend !== "ok" || healthBody.llama_server !== "ok") {
    throw new Error(`Backend/model server is not healthy: ${JSON.stringify(healthBody)}`);
  }

  // On Windows spawn npx.cmd directly can fail with EINVAL. Use cmd.exe explicitly.
  if (process.platform === "win32") {
    viteProcess = spawn("cmd.exe", ["/d", "/s", "/c", "npx vite --host 127.0.0.1 --port 5173 --strictPort"], {
      cwd: desktopDir, stdio: "inherit", windowsHide: true,
    });
  } else {
    viteProcess = spawn("npx", ["vite", "--host", "127.0.0.1", "--port", "5173", "--strictPort"], {
      cwd: desktopDir, stdio: "inherit",
    });
  }

  try {
    await waitForHttp(baseUrl, 60000);
    app = await electron.launch({ args: [desktopDir], cwd: desktopDir, timeout: 60000 });
    window = await app.firstWindow();
    window.setDefaultTimeout(15000);
    window.on("pageerror", error => console.error(`[RENDERER PAGE ERROR] ${error.message}`));

    for (const item of tests) {
      try {
        await item.fn();
        log("PASS", item.name);
      } catch (err) {
        log("FAIL", item.name, err && err.message ? err.message.replace(/\s+/g, " ").slice(0, 600) : String(err));
        try {
          const safe = item.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
          await screenshot(`failure-${safe}.png`);
        } catch {}
      }
    }
  } finally {
    if (app) await app.close().catch(() => {});
    if (viteProcess && !viteProcess.killed) {
      if (process.platform === "win32" && viteProcess.pid) {
        spawn("taskkill.exe", ["/PID", String(viteProcess.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
      } else {
        viteProcess.kill();
      }
    }
  }

  console.log("\n------------- DESKTOP E2E SUMMARY -------------");
  console.log(`Passed: ${passed} | Failed: ${failures} | Total: ${tests.length}`);
  console.log(`Screenshots/log artifacts: ${artifactsDir}`);
  process.exitCode = failures ? 1 : 0;
})().catch(err => {
  console.error(`E2E setup failed: ${err.stack || err}`);
  if (viteProcess && !viteProcess.killed) viteProcess.kill();
  if (app) app.close().catch(() => {});
  process.exit(2);
});
