
const { app, BrowserWindow, shell, dialog, Menu, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const crypto = require("node:crypto");
const { autoUpdater } = require("electron-updater");

// Keep the same per-user database location in development and packaged builds.
app.setPath("userData", path.join(app.getPath("appData"), "brahma-desk"));

let mainWindow = null;
let backendProcess = null;
let quitting = false;
let stopBackendPromise = null;

const BACKEND_PORT = Number(process.env.BRAHMA_BACKEND_PORT || 8000);
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const SHUTDOWN_TOKEN = crypto.randomBytes(32).toString("hex");
const isDev = !app.isPackaged;

// Prevent multiple app instances from racing to bind the fixed backend port.
const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });
}
function log(message) {
  console.log(`[Brahma Desk] ${message}`);
}


// ---------------------------------------------------------------------------
// Wait for the FastAPI backend
// ---------------------------------------------------------------------------

function waitForBackend(timeoutMs = 150000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    let settled = false;
    let retryTimer = null;

    const finish = (error) => {
      if (settled) return;
      settled = true;

      if (retryTimer) clearTimeout(retryTimer);

      if (error) reject(error);
      else resolve();
    };

    const probe = () => {
      if (settled) return;

      if (Date.now() - started > timeoutMs) {
        finish(
          new Error("Timed out waiting for the Brahma Desk backend.")
        );
        return;
      }

      const req = http.get(
        `${BACKEND_URL}/api/health`,
        (res) => {
          res.resume();

          if (res.statusCode === 200) {
            finish();
          } else {
            scheduleRetry();
          }
        }
      );

      req.setTimeout(1500, () => req.destroy());
      req.on("error", scheduleRetry);
    };

    const scheduleRetry = () => {
      if (settled || retryTimer) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        probe();
      }, 1000);
    };

    probe();
  });
}


// ---------------------------------------------------------------------------
// Start the packaged backend
// ---------------------------------------------------------------------------

function startPackagedBackend() {
  const executable = path.join(
    process.resourcesPath,
    "backend",
    "brahma-backend.exe"
  );

  if (!fs.existsSync(executable)) {
    throw new Error(
      `Packaged backend was not found at:\n${executable}\n\n` +
      "Build backend/dist/brahma-backend.exe before packaging."
    );
  }

  const logPath = path.join(app.getPath("userData"), "backend.log");
  const logStream = fs.createWriteStream(logPath, { flags: "a" });

  backendProcess = spawn(executable, [], {
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      BRAHMA_BACKEND_PORT: String(BACKEND_PORT),
      BRAHMA_SHUTDOWN_TOKEN: SHUTDOWN_TOKEN,
      BRAHMA_DATA_DIR: app.getPath("userData"),
      BRAHMA_PACKAGED_LLAMA_SERVER_EXE: path.join(process.resourcesPath, "llama-runtime", "llama-server.exe"),
    },
  });

  backendProcess.stdout.pipe(logStream);
  backendProcess.stderr.pipe(logStream);

  const child = backendProcess;

  child.on("error", (error) => {
    log(`Backend process error: ${error.message}`);
  });

  child.on("exit", (code, signal) => {
    log(`Backend exited: code=${code}, signal=${signal}`);

    // Never clear a newer backend process because an older child exited.
    if (backendProcess === child) {
      backendProcess = null;
    }
  });

  log(`Backend logs: ${logPath}`);
}


// ---------------------------------------------------------------------------
// Request graceful model shutdown
// ---------------------------------------------------------------------------

function requestModelShutdown(timeoutMs = 15000) {
  return new Promise((resolve) => {
    const req = http.request(
      `${BACKEND_URL}/api/shutdown`,
      {
        method: "POST",
        headers: {
          "X-Brahma-Shutdown-Token": SHUTDOWN_TOKEN,
        },
        timeout: timeoutMs,
      },
      (res) => {
        let body = "";

        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });

        res.on("end", () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            log(`Backend shutdown response: ${body}`);
          } else {
            log(
              `Backend shutdown endpoint returned HTTP ${res.statusCode}: ${body}`
            );
          }

          resolve();
        });
      }
    );

    req.on("error", (error) => {
      log(`Could not request model shutdown: ${error.message}`);
      resolve();
    });

    req.on("timeout", () => {
      req.destroy(new Error("Backend shutdown request timed out."));
    });

    req.end();
  });
}


// ---------------------------------------------------------------------------
// Stop the backend, allowing it to unload its owned model first
// ---------------------------------------------------------------------------

function forceTerminateBackend(child, timeoutMs = 5000) {
  return new Promise((resolve) => {
    if (!child || child.exitCode !== null || !child.pid) {
      resolve();
      return;
    }

    let settled = false;
    let exitTimer = null;

    const finish = () => {
      if (settled) return;
      settled = true;
      if (exitTimer) clearTimeout(exitTimer);
      child.removeListener("exit", onExit);
      resolve();
    };

    const onExit = () => finish();
    child.once("exit", onExit);

    if (process.platform === "win32") {
      log(`Force-stopping backend PID ${child.pid} with taskkill.`);

      const killer = spawn(
        "taskkill.exe",
        ["/PID", String(child.pid), "/T", "/F"],
        {
          windowsHide: true,
          stdio: "ignore",
        }
      );

      killer.once("error", (error) => {
        log(`taskkill failed: ${error.message}; trying child.kill().`);
        try {
          child.kill();
        } catch (killError) {
          log(`child.kill() failed: ${killError.message}`);
        }
      });

      killer.once("close", (code) => {
        log(`taskkill completed with exit code ${code}.`);
        if (child.exitCode !== null) finish();
      });
    } else {
      try {
        child.kill("SIGTERM");
      } catch (error) {
        log(`Backend termination failed: ${error.message}`);
      }
    }

    // Do not let a broken exit notification block application shutdown forever.
    exitTimer = setTimeout(() => {
      if (child.exitCode === null) {
        log(`Backend PID ${child.pid} did not report exit after forced termination.`);
      }
      finish();
    }, timeoutMs);
  });
}

function stopBackend() {
  if (stopBackendPromise) return stopBackendPromise;

  const child = backendProcess;
  if (!child) return Promise.resolve();

  stopBackendPromise = (async () => {
    const alreadyExited = () => child.exitCode !== null;

    if (!alreadyExited()) {
      // The authenticated endpoint stops the model before requesting Uvicorn exit.
      await requestModelShutdown();

      if (!alreadyExited()) {
        log("Waiting for the backend process to exit.");

        await new Promise((resolve) => {
          let finished = false;
          let timer = null;

          const finish = () => {
            if (finished) return;
            finished = true;
            if (timer) clearTimeout(timer);
            child.removeListener("exit", finish);
            resolve();
          };

          child.once("exit", finish);

          timer = setTimeout(async () => {
            if (!alreadyExited()) {
              log("Backend did not exit in time; forcing backend termination.");
              await forceTerminateBackend(child);
            }
            finish();
          }, 5000);

          // Account for a process that exits while the listener is being attached.
          if (alreadyExited()) finish();
        });
      }
    }

    if (backendProcess === child) {
      backendProcess = null;
    }
  })().finally(() => {
    stopBackendPromise = null;
  });

  return stopBackendPromise;
}


ipcMain.handle("app:version", () => app.getVersion());
ipcMain.handle("dialog:select-llama-server", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select llama-server.exe",
    buttonLabel: "Use this runtime",
    properties: ["openFile"],
    filters: [{ name: "llama-server.exe", extensions: ["exe"] }],
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

let updaterState = {
  status: "idle",
  currentVersion: app.getVersion(),
  availableVersion: null,
  percent: 0,
  bytesPerSecond: 0,
  releaseNotes: "",
  message: "Check for updates to see whether a newer version is available.",
};

function publishUpdaterState(patch) {
  updaterState = { ...updaterState, ...patch, currentVersion: app.getVersion() };
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("updates:state", updaterState);
  }
}

ipcMain.handle("updates:get-state", () => updaterState);
ipcMain.handle("updates:check", async () => {
  if (!app.isPackaged) {
    publishUpdaterState({ status: "unavailable", message: "Update checks are available in packaged builds. The development build does not download releases." });
    return updaterState;
  }
  if (updaterState.status === "checking" || updaterState.status === "downloading") return updaterState;
  publishUpdaterState({ status: "checking", message: "Checking the release channel…", percent: 0, availableVersion: null, releaseNotes: "" });
  try {
    await autoUpdater.checkForUpdates();
  } catch (error) {
    publishUpdaterState({ status: "error", message: error.message || "Could not check for updates." });
  }
  return updaterState;
});
ipcMain.handle("updates:download", async () => {
  if (!app.isPackaged) {
    publishUpdaterState({ status: "unavailable", message: "Download a packaged build to test the release download flow." });
    return updaterState;
  }
  if (updaterState.status !== "available") return updaterState;
  publishUpdaterState({ status: "downloading", message: "Downloading update…", percent: 0 });
  try {
    await autoUpdater.downloadUpdate();
  } catch (error) {
    publishUpdaterState({ status: "error", message: error.message || "Could not download the update." });
  }
  return updaterState;
});
ipcMain.handle("updates:install", () => {
  if (!app.isPackaged || updaterState.status !== "downloaded") return false;
  quitting = true;
  autoUpdater.quitAndInstall(false, true);
  return true;
});

// ---------------------------------------------------------------------------
// Create the application window
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#111318",
    title: "Brahma Desk",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  if (isDev) {
    mainWindow.loadURL("http://127.0.0.1:5173");
  } else {
    mainWindow.loadFile(
      path.join(__dirname, "..", "dist", "index.html")
    );
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) {
      shell.openExternal(url);
    }

    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}


// ---------------------------------------------------------------------------
// User-controlled updates: the Update Center owns check/download/install actions.
// ---------------------------------------------------------------------------

function configureUpdater() {
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = false;
  autoUpdater.allowPrerelease = false;

  autoUpdater.on("checking-for-update", () => {
    publishUpdaterState({ status: "checking", message: "Checking the release channel…" });
  });
  autoUpdater.on("update-available", (info) => {
    const notes = typeof info.releaseNotes === "string"
      ? info.releaseNotes
      : Array.isArray(info.releaseNotes)
        ? info.releaseNotes.map((entry) => entry.note || "").filter(Boolean).join("\n\n")
        : "Release notes were not provided for this version.";
    publishUpdaterState({ status: "available", availableVersion: info.version, percent: 0, releaseNotes: notes, message: `Brahma Desk ${info.version} is available to download.` });
  });
  autoUpdater.on("update-not-available", (info) => {
    publishUpdaterState({ status: "up-to-date", availableVersion: info?.version || null, percent: 0, releaseNotes: "", message: "You're running the latest available version." });
  });
  autoUpdater.on("download-progress", (progress) => {
    publishUpdaterState({ status: "downloading", percent: Math.max(0, Math.min(100, Math.round(progress.percent || 0))), bytesPerSecond: progress.bytesPerSecond || 0, message: `Downloading update… ${Math.round(progress.percent || 0)}%` });
  });
  autoUpdater.on("update-downloaded", (info) => {
    publishUpdaterState({ status: "downloaded", availableVersion: info.version, percent: 100, message: `Version ${info.version} is downloaded and ready to install. Save your work before restarting.` });
  });
  autoUpdater.on("error", (error) => {
    log(`Updater error: ${error.message}`);
    publishUpdaterState({ status: "error", message: error.message || "The update service encountered an error." });
  });
}


// ---------------------------------------------------------------------------
// First-launch installation choice
// ---------------------------------------------------------------------------

function installationStatePath() {
  return path.join(app.getPath("appData"), "brahma-desk-install-state.json");
}

function readInstallationState() {
  try {
    return JSON.parse(fs.readFileSync(installationStatePath(), "utf8"));
  } catch {
    return {};
  }
}

function writeInstallationState(state) {
  const target = installationStatePath();
  fs.writeFileSync(target, JSON.stringify(state, null, 2), "utf8");
}

async function chooseInstallationMode() {
  if (!app.isPackaged) return true;

  const version = app.getVersion();
  const state = readInstallationState();
  if (state.promptedVersion === version) return true;

  const choice = await dialog.showMessageBox({
    type: "question",
    title: "Set up Brahma Desk",
    message: "Choose how Brahma Desk should start.",
    detail:
      "Standard installation keeps your existing conversations, profiles, settings, and downloaded models. " +
      "Clean installation removes Brahma Desk user data and managed downloaded models, then starts fresh setup with a new recommended model download.",
    buttons: ["Standard installation", "Clean installation", "Cancel setup"],
    defaultId: 0,
    cancelId: 2,
    noLink: true,
  });

  if (choice.response === 2) return false;

  if (choice.response === 1) {
    const confirmation = await dialog.showMessageBox({
      type: "warning",
      title: "Confirm clean installation",
      message: "Permanently delete Brahma Desk's local data?",
      detail:
        "This deletes local conversations, profiles, settings, memories, model registry/configuration, and models downloaded into Brahma Desk's managed model folder. " +
        "It does not delete GGUF files in unrelated folders or files outside Brahma Desk's data directory. This cannot be undone.",
      buttons: ["Delete data and continue", "Go back"],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
    });
    if (confirmation.response !== 0) return false;

    const dataDir = path.resolve(app.getPath("userData"));
    const expectedDataDir = path.resolve(path.join(app.getPath("appData"), "brahma-desk"));
    if (dataDir !== expectedDataDir || dataDir === path.parse(dataDir).root) {
      throw new Error("Refusing to clean an unexpected data directory: " + dataDir);
    }

    // Electron owns a lockfile inside userData for the lifetime of this process.
    // Never remove the active userData directory: Windows can reject unlinking
    // that lockfile with EBUSY, which prevents the app window from opening.
    // Clear the directory's contents while preserving Electron's live lockfile.
    fs.mkdirSync(dataDir, { recursive: true });
    for (const entry of fs.readdirSync(dataDir, { withFileTypes: true })) {
      if (entry.name.toLowerCase() === "lockfile") {
        log("Preserving Electron's active user-data lockfile during clean installation.");
        continue;
      }
      const entryPath = path.join(dataDir, entry.name);
      try {
        fs.rmSync(entryPath, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      } catch (error) {
        throw new Error(`Could not clear Brahma Desk data item "${entry.name}": ${error.message}`);
      }
    }
    fs.writeFileSync(
      path.join(dataDir, "clean-install-pending.json"),
      JSON.stringify({ createdAt: new Date().toISOString(), version }),
      "utf8"
    );
    log("Clean installation selected; application data was reset while preserving Electron's active lockfile.");
  }

  writeInstallationState({ promptedVersion: version, selectedMode: choice.response === 1 ? "clean" : "standard" });
  return true;
}


// ---------------------------------------------------------------------------
// Application startup
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  if (!hasSingleInstanceLock) return;

  // This is a focused desktop chat app; remove the default File/Edit/View menu.
  Menu.setApplicationMenu(null);
  try {
    if (!isDev) {
      // The setup choice must run before the backend opens the existing database.
      // This is the first-run UX for MSI installs; it is intentionally separate
      // from the MSI's native wizard.
      const continueStartup = await chooseInstallationMode();
      if (!continueStartup) {
        quitting = true;
        app.quit();
        return;
      }

      // Reuse a responsive local Brahma API instead of starting a second server
      // that would fail to bind to 127.0.0.1:8000. A backend is ready even when
      // no model is configured; model setup is handled inside the UI.
      let backendReady = false;
      try {
        await waitForBackend(1800);
        backendReady = true;
        log("Found an existing responsive Brahma Desk backend.");
      } catch {
        // No backend responded; start the packaged process below.
      }

      if (!backendReady) {
        startPackagedBackend();
        await waitForBackend();
      }
    }

    createWindow();
    configureUpdater();
  } catch (error) {
    log(error.stack || error.message);

    dialog.showErrorBox(
      "Brahma Desk could not start",
      `${error.message}\n\nSee the application logs under:\n${app.getPath("userData")}`
    );

    await stopBackend();
    app.quit();
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0 && !quitting) {
      createWindow();
    }
  });
});


// ---------------------------------------------------------------------------
// Controlled application shutdown
// ---------------------------------------------------------------------------

app.on("before-quit", (event) => {
  if (quitting) return;

  event.preventDefault();
  quitting = true;

  stopBackend()
    .catch((error) => {
      log(`Error during backend shutdown: ${error.message}`);
    })
    .finally(() => {
      app.quit();
    });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
