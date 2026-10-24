import { useCallback, useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Cpu, Download, HardDrive, RefreshCw, ShieldCheck, Sparkles, XCircle } from "lucide-react";
import "./installer.css";

const API = "http://127.0.0.1:8000";
type Hardware = {
  os: string; os_release: string; architecture: string;
  cpu: { name: string; physical_cores: number | null; logical_cores: number | null };
  ram_gb: number | null; gpus: Array<{ name: string; vram_gb_reported: number | null; vram_note: string }>;
  model_directory: string; model_directory_free_gb: number | null; limitations: string[];
};
type Model = {
  id: string; name: string; repo_id: string; filename: string; family: string;
  parameters_b: number; quantization: string; purpose: string; license: string;
  minimum_ram_gb: number; recommended_ram_gb: number; recommendation: string;
  reasons: string[]; installed: boolean; installed_path: string | null; estimated_weight_gb: number;
};
type Job = { id: string; model_id: string; name: string; status: string; progress: number; created_at?: number; error?: string | null; path?: string };
type Runtime = { executable_path: string; exists: boolean; note: string; saved?: boolean };
type ApiError = { detail?: string };

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json().catch(() => ({})) as T & ApiError;
  if (!response.ok) throw new Error(body.detail || "Request failed (" + response.status + ")");
  return body;
}

const recommendationLabel: Record<string, string> = {
  recommended: "Recommended",
  compatible_with_constraints: "Compatible",
  unsupported: "Not recommended",
  unknown: "Needs review",
};

export default function InstallerWizard() {
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selected, setSelected] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [contextSize, setContextSize] = useState(8192);
  const [mode, setMode] = useState("auto");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<"scan" | "select" | "download" | "configure" | "complete">("scan");

  const refresh = useCallback(async () => {
    setError("");
    try {
      const [catalog, downloads, runtimeStatus] = await Promise.all([
        api<{ hardware: Hardware; models: Model[] }>("/api/installer/catalog"),
        api<{ jobs: Job[] }>("/api/installer/downloads"),
        api<Runtime>("/api/installer/runtime"),
      ]);
      setHardware(catalog.hardware);
      setModels(catalog.models);
      setJobs(downloads.jobs);
      setRuntime(runtimeStatus);
      setSelected(current => current || catalog.models.find(m => m.recommendation === "recommended")?.id || catalog.models[0]?.id || "");
      setStep(current => current === "scan" ? "select" : current);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load installer data.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const timer = window.setInterval(() => {
      void api<{ jobs: Job[] }>("/api/installer/downloads").then(result => setJobs(result.jobs)).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, []);

  const chosen = models.find(model => model.id === selected);
  const latestJob = jobs.filter(job => job.model_id === selected).sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0];
  const activeJob = latestJob && (latestJob.status === "queued" || latestJob.status === "downloading") ? latestJob : undefined;
  const failedJob = latestJob?.status === "failed" ? latestJob : undefined;
  const completedJob = latestJob?.status === "completed" ? latestJob : undefined;

  async function chooseRuntime() {
    setBusy(true); setError(""); setNotice("");
    try {
      const selectedPath = await window.brahmaDesktop?.selectLlamaServer?.();
      if (!selectedPath) return;
      const result = await api<Runtime>("/api/installer/runtime", {
        method: "POST",
        body: JSON.stringify({ executable_path: selectedPath }),
      });
      setRuntime(result);
      setNotice("llama.cpp runtime path saved. The selected model can now be activated if this runtime supports its GGUF and hardware backend.");
    } catch (e) { setError(e instanceof Error ? e.message : "Could not select llama-server.exe."); }
    finally { setBusy(false); }
  }

  async function scanLocalModels() {
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await api<{ discovered: Array<{ filename: string }>; rejected: Array<{ filename: string; reason: string }> }>("/api/installer/scan-local", { method: "POST" });
      await refresh();
      setNotice("Found " + result.discovered.length + " local GGUF model(s)." + (result.rejected.length ? " " + result.rejected.length + " file(s) were skipped." : ""));
    } catch (e) { setError(e instanceof Error ? e.message : "Local model scan failed."); }
    finally { setBusy(false); }
  }

  async function beginDownload() {
    if (!chosen) return;
    setBusy(true); setError(""); setNotice("");
    try {
      await api<Job>("/api/installer/downloads", { method: "POST", body: JSON.stringify({ model_id: chosen.id }) });
      setStep("download");
      setNotice("Download queued. Keep this page open to follow progress.");
      await refresh();
    } catch (e) { setError(e instanceof Error ? e.message : "Download could not start."); }
    finally { setBusy(false); }
  }

  async function validateAndConfigure() {
    if (!chosen) return;
    setBusy(true); setError(""); setNotice("");
    try {
      await api("/api/installer/validate/" + chosen.id, { method: "POST" });
      await api("/api/installer/configure", {
        method: "POST",
        body: JSON.stringify({ model_id: chosen.id, context_size: contextSize, mode }),
      });
      const result = await api<{ status: string; generation_test: string; sample_output: string }>("/api/installer/activate/" + chosen.id, { method: "POST" });
      setNotice("Model activated and generation smoke test passed: " + result.sample_output);
      setStep("complete");
      await refresh();
    } catch (e) { setError(e instanceof Error ? e.message : "Validation or configuration failed."); }
    finally { setBusy(false); }
  }

  return <div className="installer-page">
    <div className="installer-hero">
      <div className="installer-icon"><Sparkles size={23}/></div>
      <div className="installer-hero-copy">
        <div className="installer-eyebrow">BRAHMA DESK · FIRST-RUN SETUP</div>
        <h1>Set up your local AI</h1>
        <p>Check this PC, choose a model that fits, and prepare a local inference configuration.</p>
      </div>
      <div className="installer-hero-actions"><button className="installer-ghost-button" onClick={() => void refresh()} disabled={busy}><RefreshCw size={15}/> Rescan hardware</button><button className="installer-ghost-button" onClick={() => void scanLocalModels()} disabled={busy}><HardDrive size={15}/> Scan local GGUF</button></div>
    </div>

    <div className="installer-steps" aria-label="Setup progress">
      {(["scan", "select", "download", "configure", "complete"] as const).map((item, index) =>
        <div key={item} className={"installer-step " + (step === item ? "current" : (["scan", "select", "download", "configure", "complete"].indexOf(step) > index ? "done" : ""))}>
          <span>{index + 1}</span><small>{({ scan: "Hardware", select: "Model", download: "Download", configure: "Configure", complete: "Review" } as Record<string, string>)[item]}</small>
        </div>
      )}
    </div>

    {error && <div className="installer-alert error"><AlertCircle size={17}/><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><XCircle size={15}/></button></div>}
    {notice && <div className="installer-alert info"><CheckCircle2 size={17}/><span>{notice}</span></div>}

    <section className="installer-section installer-runtime-section">
      <div className="installer-section-heading"><div><span className="installer-kicker">RUNTIME</span><h2>llama.cpp engine</h2></div><span className={"installer-tag " + (runtime?.exists ? "recommended" : "unknown")}>{runtime?.exists ? "Found" : "Needs setup"}</span></div>
      <p className="installer-muted">{runtime?.note || "Checking local runtime…"}</p>
      <div className="installer-runtime-row"><code>{runtime?.executable_path || "Runtime path not available"}</code><button className="installer-ghost-button" onClick={() => void chooseRuntime()} disabled={busy}><HardDrive size={15}/> Choose llama-server.exe</button></div>
      {!runtime?.exists && <p className="installer-runtime-warning">Model downloads and validation can proceed, but activation requires a compatible llama-server.exe. Brahma Desk will not stop or replace a llama.cpp process it does not own.</p>}
    </section>

    <section className="installer-section">
      <div className="installer-section-heading"><div><span className="installer-kicker">STEP 01</span><h2>Your hardware</h2></div><span className="installer-live"><i/> Local scan</span></div>
      {!hardware ? <div className="installer-loading"><RefreshCw size={17} className="installer-spin"/> Scanning CPU, memory, GPU and disk…</div> : <>
        <div className="installer-hardware-grid">
          <article className="installer-metric"><Cpu size={18}/><span>Processor</span><strong>{hardware.cpu.name}</strong><small>{hardware.cpu.logical_cores ?? "?"} logical cores · {hardware.architecture}</small></article>
          <article className="installer-metric"><HardDrive size={18}/><span>System memory</span><strong>{hardware.ram_gb == null ? "Unknown" : hardware.ram_gb + " GB"}</strong><small>Detected physical RAM</small></article>
          <article className="installer-metric"><Download size={18}/><span>Model disk space</span><strong>{hardware.model_directory_free_gb == null ? "Unknown" : hardware.model_directory_free_gb + " GB free"}</strong><small>Available for model downloads</small></article>
        </div>
        <div className="installer-gpu-list"><strong>Graphics adapters</strong>{hardware.gpus.length ? hardware.gpus.map((gpu, index) => <div key={gpu.name + "-" + index}><span>{gpu.name}</span><small>{gpu.vram_gb_reported == null ? "VRAM unknown" : gpu.vram_gb_reported + " GB reported VRAM"}</small></div>) : <p>No GPU information was returned by the operating system.</p>}</div>
        <p className="installer-muted">{hardware.os} {hardware.os_release} · Models saved to <code>{hardware.model_directory}</code></p>
        <p className="installer-disclaimer"><ShieldCheck size={14}/> GPU memory and compatibility are best-effort estimates. The final runtime must be tested with the actual llama.cpp backend.</p>
      </>}
    </section>

    <section className="installer-section">
      <div className="installer-section-heading"><div><span className="installer-kicker">STEP 02</span><h2>Choose a model</h2></div><span className="installer-muted">{models.length} catalog models</span></div>
      <div className="installer-model-grid">
        {models.map(model => <button key={model.id} className={"installer-model-card " + (selected === model.id ? "selected" : "")} onClick={() => { setSelected(model.id); setStep("select"); setNotice(""); }} aria-pressed={selected === model.id}>
          <div className="installer-model-top"><span className={"installer-tag " + model.recommendation}>{recommendationLabel[model.recommendation] ?? model.recommendation}</span>{model.installed && <span className="installer-installed"><CheckCircle2 size={13}/> Installed</span>}</div>
          <strong>{model.name}</strong><span className="installer-model-purpose">{model.purpose}</span>
          <div className="installer-model-meta"><span>{model.parameters_b}B parameters</span><span>{model.quantization}</span><span>~{model.estimated_weight_gb} GB weights</span></div>
          {model.reasons.map(reason => <small className="installer-reason" key={reason}>{reason}</small>)}
          <small className="installer-license">License: {model.license}</small>
        </button>)}
      </div>
      {chosen && <div className="installer-selected-summary"><div><strong>Selected: {chosen.name}</strong><p>Source: <code>{chosen.repo_id}</code></p><p>File: <code>{chosen.filename}</code></p></div>
        {chosen.installed ? <button className="installer-primary" onClick={() => { setStep("configure"); void validateAndConfigure(); }} disabled={busy}><CheckCircle2 size={15}/> Validate & configure</button> : <button className="installer-primary" onClick={() => void beginDownload()} disabled={busy || chosen.recommendation === "unsupported"}><Download size={15}/>{busy ? "Starting…" : "Download model"}</button>}
      </div>}
    </section>

    {(activeJob || failedJob || completedJob) && <section className="installer-section">
      <div className="installer-section-heading"><div><span className="installer-kicker">STEP 03</span><h2>Download status</h2></div></div>
      {activeJob && <div className="installer-download-card"><div className="installer-download-title"><strong>{activeJob.name}</strong><span>{activeJob.progress}%</span><button className="installer-cancel-download" onClick={async () => { try { await api("/api/installer/downloads/" + activeJob.id + "/cancel", { method: "POST" }); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "Could not cancel download."); } }}>Cancel</button></div><div className="installer-progress"><div style={{ width: activeJob.progress + "%" }}/></div><p>Downloading from Hugging Face. Partial files are retained for resume support.</p></div>}
      {failedJob && <div className="installer-alert error"><AlertCircle size={17}/><span>{failedJob.error || "Download failed."}</span><button className="installer-retry" onClick={async () => { try { await api("/api/installer/downloads/" + failedJob.id + "/retry", { method: "POST" }); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "Retry failed."); } }}>Retry</button></div>}
      {completedJob && <div className="installer-download-card success"><CheckCircle2 size={18}/><div><strong>Download complete</strong><p>GGUF header and checksum checks passed. Continue to validate and configure the model.</p><button className="installer-primary" onClick={() => void validateAndConfigure()} disabled={busy}>{busy ? "Validating…" : "Validate & configure"}</button></div></div>}
    </section>}

    {step === "complete" && <section className="installer-complete"><CheckCircle2 size={24}/><div><h2>Model setup checks complete</h2><p>The model is registered, the configuration was applied, and llama.cpp returned text for a smoke-test prompt. This verifies a basic generation path, not model quality or long-context performance.</p><button className="installer-ghost-button" onClick={() => setStep("select")}>Choose another model</button></div></section>}
    <div className="installer-config-controls"><label>Context length<select value={contextSize} onChange={e => setContextSize(Number(e.target.value))}><option value={2048}>2,048 tokens</option><option value={4096}>4,096 tokens</option><option value={8192}>8,192 tokens</option><option value={16384}>16,384 tokens</option></select></label><label>Inference placement<select value={mode} onChange={e => setMode(e.target.value)}><option value="auto">Auto-select</option><option value="cpu">CPU</option><option value="hybrid">CPU + GPU hybrid</option><option value="gpu">GPU</option></select></label></div>
    <footer className="installer-footer">Brahma Desk keeps model files and registry data on this device. Downloads connect to Hugging Face; model licenses and repository availability may change.</footer>
  </div>;
}
