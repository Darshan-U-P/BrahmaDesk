import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";
import InstallerWizard from "./InstallerWizard";
import { Activity, Bot, ChevronDown, CircleHelp, Cpu, Menu, Plus, Send, Settings2, ShieldCheck, Sparkles, SquarePen, Trash2, Pencil, Search, Wifi, WifiOff, Square, RotateCcw, Check, X, Copy, ArrowDown, SlidersHorizontal, RefreshCw, Download, PackageCheck, CircleCheck, AlertCircle, ArrowUpCircle } from "lucide-react";

type Message = { id?: string; role: "user" | "assistant"; content: string };
type Conversation = { id: string; title: string; updated_at: number; preview?: string | null };
type Health = { backend: string; llama_server: string; llama_base_url: string; model: string | null };
type Profile = { id: string; display_name: string; avatar_color: string; is_default: boolean };
type ProfileSettings = {
  user_name: string;
  purpose: string;
  role: string;
  interests: string;
  goals: string;
  response_style: string;
  language: string;
  additional_context: string;
};
const EMPTY_PROFILE_SETTINGS: ProfileSettings = {
  user_name: "", purpose: "", role: "", interests: "", goals: "",
  response_style: "Balanced", language: "English", additional_context: "",
};
const API = "http://127.0.0.1:8000";

type UpdateState = {
  status: "idle" | "checking" | "available" | "up-to-date" | "downloading" | "downloaded" | "error" | "unavailable";
  currentVersion: string;
  availableVersion: string | null;
  percent: number;
  bytesPerSecond?: number;
  releaseNotes: string;
  message: string;
};
declare global {
  interface Window {
    brahmaDesktop?: {
      getVersion: () => Promise<string>;
      updates?: {
        getState: () => Promise<UpdateState>;
        check: () => Promise<UpdateState>;
        download: () => Promise<UpdateState>;
        install: () => Promise<boolean>;
        onState: (callback: (state: UpdateState) => void) => () => void;
      };
    };
  }
}
const INITIAL_UPDATE_STATE: UpdateState = {
  status: "idle", currentVersion: "0.1.0", availableVersion: null, percent: 0,
  releaseNotes: "", message: "Check for updates to see whether a newer version is available.",
};

type SamplingSettings = {
  temperature: number; dynatemp_range: number; dynatemp_exponent: number; top_k: number; top_p: number; min_p: number;
  xtc_probability: number; xtc_threshold: number; typ_p: number; min_keep: number; max_tokens: number; sampler_order: string;
  backend_sampling: boolean; repeat_last_n: number; repeat_penalty: number; presence_penalty: number; frequency_penalty: number;
  dry_multiplier: number; dry_base: number; dry_allowed_length: number; dry_penalty_last_n: number;
};
const DEFAULT_SAMPLING: SamplingSettings = { temperature: 0.8, dynatemp_range: 0, dynatemp_exponent: 1, top_k: 40, top_p: 0.95, min_p: 0.05, xtc_probability: 0, xtc_threshold: 0.1, typ_p: 1, min_keep: 1, max_tokens: 1024, sampler_order: "top_k;typ_p;top_p;min_p;temperature", backend_sampling: false, repeat_last_n: 64, repeat_penalty: 1, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: 0 };
const SAMPLING_STORAGE_KEY = "brahma-desk-sampling-settings-v1";
type ReasoningBudget = "off" | "low" | "medium" | "high" | "max";
const REASONING_STORAGE_KEY = "brahma-desk-reasoning-budget-v1";
function loadReasoningBudget(): ReasoningBudget {
  try {
    const value = localStorage.getItem(REASONING_STORAGE_KEY);
    return value === "off" || value === "low" || value === "medium" || value === "high" || value === "max" ? value : "medium";
  } catch { return "medium"; }
}
const REASONING_OPTIONS: { value: ReasoningBudget; label: string; detail: string }[] = [
  { value: "off", label: "Off", detail: "Disable model thinking mode where supported" },
  { value: "low", label: "Low", detail: "Aim for up to 512 reasoning tokens" },
  { value: "medium", label: "Medium", detail: "Aim for up to 2,048 reasoning tokens" },
  { value: "high", label: "High", detail: "Aim for up to 8,192 reasoning tokens" },
  { value: "max", label: "Max", detail: "No application-level reasoning target" },
];
function loadSamplingSettings(): SamplingSettings { try { const saved = localStorage.getItem(SAMPLING_STORAGE_KEY); return saved ? { ...DEFAULT_SAMPLING, ...JSON.parse(saved) } : DEFAULT_SAMPLING; } catch { return DEFAULT_SAMPLING; } }

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const pattern = /(\[[^\]]+\]\(https?:\/\/[^)\s]+\)|`[^`]+`|\*\*[^*]+\*\*|~~[^~]+~~|\*[^*]+\*)/g;
  const output: ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) output.push(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith("[")) {
      const link = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
      if (link) output.push(<a className="markdown-link" href={link[2]} target="_blank" rel="noreferrer noopener" key={`${keyPrefix}-${index}`}>{link[1]}</a>);
      else output.push(token);
    } else if (token.startsWith("`")) output.push(<code className="inline-code" key={`${keyPrefix}-${index}`}>{token.slice(1, -1)}</code>);
    else if (token.startsWith("**")) output.push(<strong key={`${keyPrefix}-${index}`}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("~~")) output.push(<del key={`${keyPrefix}-${index}`}>{token.slice(2, -2)}</del>);
    else output.push(<em key={`${keyPrefix}-${index}`}>{token.slice(1, -1)}</em>);
    last = match.index + token.length;
    index += 1;
  }
  if (last < text.length) output.push(text.slice(last));
  return output;
}

function MarkdownContent({ content, onNotice }: { content: string; onNotice: (message: string) => void }) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let nextOrderedNumber = 1;

  const isFence = (line: string) => /^\s*```/.test(line);
  const isHeading = (line: string) => /^\s*#{1,6}\s+/.test(line);
  const isBullet = (line: string) => /^\s*[-*+]\s+/.test(line);
  const isNumbered = (line: string) => /^\s*\d+[.)]\s+/.test(line);
  const isRule = (line: string) =>
    /^\s*(?:-{3,}|_{3,}|\*\s*){1,}\s*$/.test(line) &&
    /[-_*]/.test(line.trim());
  const isTableSeparator = (line: string) =>
    /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  const tableCells = (line: string) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    const fence = line.match(/^\s*```([\w+#.-]*)\s*$/);
    if (fence) {
      const language = fence[1] || "code";
      const codeLines: string[] = [];
      i++;

      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
        codeLines.push(lines[i++]);
      }

      if (i < lines.length) i++;

      blocks.push(
        <pre className="code-block" key={`code-${blocks.length}`}>
          <div className="code-header">
            <span>{language}</span>
            <button
              type="button"
              onClick={async () => {
                try {
                  if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
                  await navigator.clipboard.writeText(codeLines.join("\n"));
                  onNotice("Code copied to clipboard.");
                } catch {
                  onNotice("Could not copy code. Check clipboard permissions and try again.");
                }
              }}
            >
              Copy code
            </button>
          </div>
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    if (isRule(line)) {
      nextOrderedNumber = 1;
      blocks.push(<hr className="markdown-rule" key={`hr-${i}`} />);
      i++;
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = tableCells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        rows.push(tableCells(lines[i]));
        i++;
      }
      blocks.push(
        <div className="markdown-table-wrap" key={`table-${blocks.length}`}>
          <table className="markdown-table">
            <thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex}>{renderInline(cell, `table-h-${blocks.length}-${cellIndex}`)}</th>)}</tr></thead>
            <tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, cellIndex) => <td key={cellIndex}>{renderInline(row[cellIndex] ?? "", `table-${blocks.length}-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      );
      nextOrderedNumber = 1;
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      blocks.push(<blockquote className="markdown-quote" key={`quote-${blocks.length}`}>{quoteLines.map((quoteLine, quoteIndex) => <p key={quoteIndex}>{renderInline(quoteLine, `quote-${blocks.length}-${quoteIndex}`)}</p>)}</blockquote>);
      nextOrderedNumber = 1;
      continue;
    }

    const heading = line.match(/^\s*(#{1,6})\s+(.+)$/);
    if (heading) {
      nextOrderedNumber = 1;
      const level = heading[1].length;
      const children = renderInline(heading[2], `h-${i}`);
      const key = `h-${i}`;

      if (level === 1) blocks.push(<h1 key={key}>{children}</h1>);
      else if (level === 2) blocks.push(<h2 key={key}>{children}</h2>);
      else if (level === 3) blocks.push(<h3 key={key}>{children}</h3>);
      else blocks.push(<h4 key={key}>{children}</h4>);

      i++;
      continue;
    }

    if (isBullet(line)) {
      const items: ReactNode[] = [];

      while (i < lines.length) {
        if (isBullet(lines[i])) {
          items.push(
            <li key={`ul-${i}`}>
              {renderInline(
                lines[i].replace(/^\s*[-*+]\s+/, ""),
                `ul-${i}`
              )}
            </li>
          );
          i++;
          continue;
        }

        if (!lines[i].trim() && i + 1 < lines.length && isBullet(lines[i + 1])) {
          i++;
          continue;
        }

        break;
      }

      blocks.push(<ul key={`ul-list-${blocks.length}`}>{items}</ul>);
      continue;
    }

    if (isNumbered(line)) {
      const items: ReactNode[] = [];

      while (i < lines.length) {
        if (isNumbered(lines[i])) {
          items.push(
            <li key={`ol-${i}`}>
              {renderInline(
                lines[i].replace(/^\s*\d+[.)]\s+/, ""),
                `ol-${i}`
              )}
            </li>
          );
          i++;
          continue;
        }

        if (
          !lines[i].trim() &&
          i + 1 < lines.length &&
          isNumbered(lines[i + 1])
        ) {
          i++;
          continue;
        }

        break;
      }

      blocks.push(
        <ol key={`ol-list-${blocks.length}`} start={nextOrderedNumber}>
          {items}
        </ol>
      );
      nextOrderedNumber += items.length;
      continue;
    }

    const paragraph: string[] = [line];
    i++;

    while (
      i < lines.length &&
      lines[i].trim() &&
      !isFence(lines[i]) &&
      !isHeading(lines[i]) &&
      !isBullet(lines[i]) &&
      !isNumbered(lines[i]) &&
      !isRule(lines[i])
    ) {
      paragraph.push(lines[i++]);
    }

    blocks.push(
      <p key={`p-${blocks.length}`}>
        {paragraph.map((part, j) => (
          <span key={j}>
            {j > 0 && <br />}
            {renderInline(part, `p-${blocks.length}-${j}`)}
          </span>
        ))}
      </p>
    );
  }

  return <div className="markdown-content">{blocks}</div>;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activePage, setActivePage] = useState<"chat" | "updates" | "installer">("chat");
  const [updateState, setUpdateState] = useState<UpdateState>(INITIAL_UPDATE_STATE);
  const [updateActionBusy, setUpdateActionBusy] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [renameTarget, setRenameTarget] = useState<Conversation | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState("sampling");
  const [sampling, setSampling] = useState<SamplingSettings>(loadSamplingSettings);
  const [samplingDraft, setSamplingDraft] = useState<SamplingSettings>(loadSamplingSettings);
  const [reasoningBudget, setReasoningBudget] = useState<ReasoningBudget>(loadReasoningBudget);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [activeProfile, setActiveProfile] = useState<Profile | null>(null);
  const [profileBusy, setProfileBusy] = useState(false);
  const [profileDialogOpen, setProfileDialogOpen] = useState(false);
  const [profileDialogMode, setProfileDialogMode] = useState<"create" | "edit">("create");
  const [profileDraft, setProfileDraft] = useState("");
  const [profileSettingsDraft, setProfileSettingsDraft] = useState<ProfileSettings>(EMPTY_PROFILE_SETTINGS);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const chatAreaRef = useRef<HTMLElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const restoreAttemptedRef = useRef(false);

  useEffect(() => {
    const updater = window.brahmaDesktop?.updates;
    if (!updater) {
      setUpdateState({ ...INITIAL_UPDATE_STATE, status: "unavailable", message: "The desktop update service is not available in this browser session." });
      return;
    }
    let active = true;
    void updater.getState().then(state => { if (active) setUpdateState(state); }).catch(() => undefined);
    const unsubscribe = updater.onState(state => setUpdateState(state));
    return () => { active = false; unsubscribe(); };
  }, []);

  async function checkForUpdates() {
    const updater = window.brahmaDesktop?.updates;
    if (!updater || updateActionBusy) return;
    setUpdateActionBusy(true);
    try { setUpdateState(await updater.check()); }
    catch (error) { setUpdateState(v => ({ ...v, status: "error", message: error instanceof Error ? error.message : "Update check failed." })); }
    finally { setUpdateActionBusy(false); }
  }
  async function downloadUpdate() {
    const updater = window.brahmaDesktop?.updates;
    if (!updater || updateActionBusy) return;
    setUpdateActionBusy(true);
    try { setUpdateState(await updater.download()); }
    catch (error) { setUpdateState(v => ({ ...v, status: "error", message: error instanceof Error ? error.message : "Update download failed." })); }
    finally { setUpdateActionBusy(false); }
  }
  async function installUpdate() {
    const updater = window.brahmaDesktop?.updates;
    if (!updater || updateActionBusy) return;
    setUpdateActionBusy(true);
    try {
      const accepted = await updater.install();
      if (!accepted) setNotice("The downloaded update could not be started. Check that this is a packaged build.");
    } catch (error) { setNotice(error instanceof Error ? error.message : "Could not start the update installer."); }
    finally { setUpdateActionBusy(false); }
  }

  const refreshHealth = useCallback(async () => { try { const r = await fetch(`${API}/api/health`); setHealth(await r.json()); } catch { setHealth(null); } }, []);
  const refreshProfiles = useCallback(async () => {
    try {
      const [listResponse, activeResponse] = await Promise.all([
        fetch(`${API}/api/profiles`), fetch(`${API}/api/active-profile`),
      ]);
      if (!listResponse.ok || !activeResponse.ok) throw new Error("Profile API unavailable");
      const list = await listResponse.json();
      const active = await activeResponse.json();
      setProfiles(list.items ?? []);
      setActiveProfile(active);
    } catch (error) { console.error("Could not load profiles", error); }
  }, []);
  async function changeProfile(profileId: string) {
    if (!profileId || profileBusy || profileId === activeProfile?.id) return;
    if (busy) { setNotice("Stop the current response before switching profiles."); return; }
    setProfileBusy(true);
    try {
      const response = await fetch(`${API}/api/active-profile`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile_id: profileId }) });
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? `HTTP ${response.status}`); }
      const selected = await response.json();
      setActiveProfile(selected); setActiveId(null); setMessages([]); shouldAutoScrollRef.current = true; setShowJumpToLatest(false); localStorage.removeItem("brahma-desk-active-conversation");
      await refreshConversations();
      setNotice(`Switched to ${selected.display_name}`);
    } catch (error) { setNotice(`Could not switch profile: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setProfileBusy(false); }
  }
  function openCreateProfile() {
    setProfileDialogMode("create");
    setProfileDraft("");
    setProfileSettingsDraft({ ...EMPTY_PROFILE_SETTINGS });
    setProfileDialogOpen(true);
  }
  async function openEditProfile() {
    if (!activeProfile || profileBusy) return;
    setProfileBusy(true);
    try {
      const [profileResponse, settingsResponse] = await Promise.all([
        fetch(`${API}/api/profiles/${encodeURIComponent(activeProfile.id)}`),
        fetch(`${API}/api/profiles/${encodeURIComponent(activeProfile.id)}/settings`),
      ]);
      // Older backends may not expose profile GET; the selected profile is already known.
      const profileData = profileResponse.ok ? await profileResponse.json() : activeProfile;
      const settingsData = settingsResponse.ok ? await settingsResponse.json() : { settings: {} };
      const saved = settingsData.settings ?? {};
      setProfileDialogMode("edit");
      setProfileDraft(profileData.display_name ?? activeProfile.display_name);
      setProfileSettingsDraft({ ...EMPTY_PROFILE_SETTINGS, ...saved });
      setProfileDialogOpen(true);
    } catch (error) {
      setNotice(`Could not load profile settings: ${error instanceof Error ? error.message : String(error)}`);
    } finally { setProfileBusy(false); }
  }
  async function saveProfile() {
    const displayName = profileDraft.trim();
    if (!displayName || profileBusy) return;
    setProfileBusy(true);
    try {
      let profileId = activeProfile?.id ?? "";
      let created = false;
      if (profileDialogMode === "create") {
        const response = await fetch(`${API}/api/profiles`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ display_name: displayName }),
        });
        if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? `HTTP ${response.status}`); }
        const profile = await response.json();
        profileId = profile.id;
        created = true;
      } else {
        if (!activeProfile) throw new Error("No active profile selected");
        const response = await fetch(`${API}/api/profiles/${encodeURIComponent(activeProfile.id)}`, {
          method: "PATCH", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ display_name: displayName }),
        });
        if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? `HTTP ${response.status}`); }
      }
      const settingsResponse = await fetch(`${API}/api/profiles/${encodeURIComponent(profileId)}/settings`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: profileSettingsDraft }),
      });
      if (!settingsResponse.ok) { const body = await settingsResponse.json().catch(() => ({})); throw new Error(body.detail ?? `Profile saved, but settings could not be saved (HTTP ${settingsResponse.status})`); }
      if (created) {
        const switchResponse = await fetch(`${API}/api/active-profile`, {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ profile_id: profileId }),
        });
        if (!switchResponse.ok) { const body = await switchResponse.json().catch(() => ({})); throw new Error(body.detail ?? `Profile created, but switching failed (HTTP ${switchResponse.status})`); }
        const selected = await switchResponse.json();
        setActiveProfile(selected); setActiveId(null); setMessages([]);
        shouldAutoScrollRef.current = true; setShowJumpToLatest(false);
        localStorage.removeItem("brahma-desk-active-conversation");
      } else {
        setActiveProfile(current => current ? { ...current, display_name: displayName } : current);
      }
      await refreshProfiles(); await refreshConversations();
      setProfileDialogOpen(false); setProfileDraft("");
      setNotice(created ? `Created and switched to ${displayName}` : `Updated ${displayName}`);
    } catch (error) { setNotice(`Could not save profile: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setProfileBusy(false); }
  }
  async function deleteActiveProfile() {
    if (!activeProfile || activeProfile.is_default || profileBusy) return;
    const profile = activeProfile;
    const confirmed = window.confirm(`Delete "${profile.display_name}"? Its conversations, memories, projects, and profile settings will be permanently deleted. This cannot be undone.`);
    if (!confirmed) return;
    setProfileBusy(true);
    try {
      const response = await fetch(`${API}/api/profiles/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? `HTTP ${response.status}`); }
      localStorage.removeItem("brahma-desk-active-conversation");
      setActiveId(null); setMessages([]); setShowJumpToLatest(false);
      await refreshProfiles(); await refreshConversations();
      setNotice(`Deleted ${profile.display_name}. Switched to the default profile if it was active.`);
    } catch (error) { setNotice(`Could not delete profile: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setProfileBusy(false); }
  }
  const refreshConversations = useCallback(async () => { try { const r = await fetch(`${API}/api/conversations`); if (!r.ok) throw new Error(); const data = await r.json(); setConversations(data.items ?? []); } catch (e) { console.error("Could not load conversations", e); } }, []);
  useEffect(() => {
    void refreshHealth();
    void refreshProfiles();
    void (async () => {
      await refreshConversations();
    })();
    const t = window.setInterval(() => void refreshHealth(), 8000);
    return () => window.clearInterval(t);
  }, [refreshHealth, refreshConversations, refreshProfiles]);
  useEffect(() => {
    if (restoreAttemptedRef.current || conversations.length === 0) return;
    restoreAttemptedRef.current = true;
    const savedId = localStorage.getItem("brahma-desk-active-conversation");
    if (savedId && conversations.some((item) => item.id === savedId)) void openConversation(savedId);
  }, [conversations]);
  useEffect(() => {
    if (activeId) localStorage.setItem("brahma-desk-active-conversation", activeId);
  }, [activeId]);
  useEffect(() => {
    const textarea = composerRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
      textarea.style.overflowY = textarea.scrollHeight > 180 ? "auto" : "hidden";
    }
  }, [draft]);
  useEffect(() => {
    const element = chatAreaRef.current;
    if (!shouldAutoScrollRef.current || !element) return;
    element.scrollTo({ top: element.scrollHeight, behavior: busy ? "auto" : "smooth" });
  }, [messages, busy]);
  function handleChatScroll() {
    const element = chatAreaRef.current;
    if (!element) return;
    const distanceToBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    const nearBottom = distanceToBottom < 100;
    shouldAutoScrollRef.current = nearBottom;
    setShowJumpToLatest(!nearBottom && messages.length > 0);
  }
  function jumpToLatest() {
    shouldAutoScrollRef.current = true;
    setShowJumpToLatest(false);
    chatAreaRef.current?.scrollTo({ top: chatAreaRef.current.scrollHeight, behavior: "smooth" });
  }
  async function copyMessage(content: string) {
    try {
      await navigator.clipboard.writeText(content);
      setNotice("Message copied to clipboard.");
    } catch {
      setNotice("Clipboard access was unavailable. Try selecting and copying the text.");
    }
  }

  async function openConversation(id: string) {
    if (busy) return;
    try { const r = await fetch(`${API}/api/conversations/${encodeURIComponent(id)}`); if (!r.ok) throw new Error(`HTTP ${r.status}`); const data = await r.json(); shouldAutoScrollRef.current = true; setShowJumpToLatest(false); setActiveId(id); setMessages((data.messages ?? []).map((m: Message) => ({ id: m.id, role: m.role, content: m.content }))); }
    catch (e) { window.alert(`Could not open conversation: ${e instanceof Error ? e.message : String(e)}`); }
  }
  async function createConversation() {
    if (busy) return;
    try { const r = await fetch(`${API}/api/conversations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: "New chat" }) }); if (!r.ok) throw new Error(`HTTP ${r.status}`); const data = await r.json(); shouldAutoScrollRef.current = true; setShowJumpToLatest(false); setActiveId(data.id); setMessages([]); await refreshConversations(); }
    catch (e) { window.alert(`Could not create conversation: ${e instanceof Error ? e.message : String(e)}`); }
  }
  function renameConversation(item: Conversation) {
    setRenameTarget(item);
    setRenameDraft(item.title);
  }
  async function saveConversationRename() {
    if (!renameTarget || renameBusy) return;
    const title = renameDraft.trim();
    if (!title) { setNotice("Conversation name cannot be empty."); return; }
    if (title.length > 200) { setNotice("Conversation name must be 200 characters or fewer."); return; }
    if (title === renameTarget.title) { setRenameTarget(null); return; }
    setRenameBusy(true);
    try {
      const r = await fetch(`${API}/api/conversations/${encodeURIComponent(renameTarget.id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title })
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${r.status}`);
      }
      setRenameTarget(null);
      await refreshConversations();
      setNotice("Conversation renamed.");
    } catch (e) {
      setNotice(`Could not rename conversation: ${e instanceof Error ? e.message : String(e)}`);
    } finally { setRenameBusy(false); }
  }
  async function deleteConversation(item: Conversation) {
    setDeleteTarget(item);
  }
  async function confirmDeleteConversation() {
    if (!deleteTarget) return;
    const item = deleteTarget;
    try {
      const r = await fetch(`${API}/api/conversations/${encodeURIComponent(item.id)}`, { method: "DELETE" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      if (activeId === item.id) { setActiveId(null); setMessages([]); localStorage.removeItem("brahma-desk-active-conversation"); }
      setDeleteTarget(null);
      await refreshConversations();
      setNotice("Conversation deleted.");
    } catch (e) { setNotice(`Could not delete conversation: ${e instanceof Error ? e.message : String(e)}`); }
  }

  async function sendMessage(e?: FormEvent, override?: { prompt: string; prior: Message[]; conversationId: string; existingMessage?: Message }) {
    e?.preventDefault(); const prompt = (override?.prompt ?? draft).trim(); if (!prompt || busy) return;
    let conversationId = override?.conversationId ?? activeId;
    try {
      if (!conversationId) {
        const cr = await fetch(`${API}/api/conversations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: prompt.replace(/\s+/g, " ").slice(0, 60) || "New chat" }) });
        if (!cr.ok) throw new Error(`Could not create chat (HTTP ${cr.status})`);
        const created = await cr.json(); conversationId = created.id; setActiveId(conversationId);
      }
      const prior = override?.prior ?? messages;
      const updated: Message[] = [...prior, override?.existingMessage ? { ...override.existingMessage, content: prompt } : { role: "user", content: prompt }];
      shouldAutoScrollRef.current = true; setShowJumpToLatest(false);
      setMessages([...updated, { role: "assistant", content: "" }]); setDraft(""); setEditingIndex(null); setBusy(true);
      const controller = new AbortController(); abortRef.current = controller;
      const r = await fetch(`${API}/api/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ conversation_id: conversationId, messages: updated, ...sampling, reasoning_budget: reasoningBudget }), signal: controller.signal });
      if (!r.ok || !r.body) throw new Error(`Backend returned HTTP ${r.status}`);
      const reader = r.body.getReader(), decoder = new TextDecoder(); let buffer = "", answer = "";
      while (true) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const chunks = buffer.split("\n\n"); buffer = chunks.pop() ?? "";
        for (const chunk of chunks) { const line = chunk.split("\n").find(x => x.startsWith("data:")); if (!line) continue; const raw = line.slice(5).trim(); if (raw === "[DONE]") continue; const data = JSON.parse(raw); if (data.error) throw new Error(data.error); if (data.conversation_id) { conversationId = data.conversation_id; setActiveId(conversationId); } if (data.delta) { answer += data.delta; setMessages([...updated, { role: "assistant", content: answer }]); } }
      }
      if (conversationId) {
        const saved = await fetch(`${API}/api/conversations/${encodeURIComponent(conversationId)}`);
        if (saved.ok) { const savedData = await saved.json(); setMessages((savedData.messages ?? []).map((m: Message) => ({ id: m.id, role: m.role, content: m.content }))); }
      }
      await refreshConversations();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") { setMessages(current => current.map((m, i) => i === current.length - 1 && m.role === "assistant" && !m.content ? { ...m, content: "Generation stopped." } : m)); }
      else { const message = err instanceof Error ? err.message : String(err); setMessages(current => [...current.filter((m, i) => !(i === current.length - 1 && m.role === "assistant" && !m.content)), { role: "assistant", content: `Connection error: ${message}\n\nCheck that the FastAPI backend and llama.cpp server are running.` }]); }
    } finally { abortRef.current = null; setBusy(false); await refreshConversations(); }
  }
  async function editAndResubmit(index: number) {
    const message = messages[index];
    if (!message || message.role !== "user" || !message.id || !activeId || !editDraft.trim() || busy) return;
    const prompt = editDraft.trim();
    try {
      const r = await fetch(`${API}/api/conversations/${encodeURIComponent(activeId)}/messages/${encodeURIComponent(message.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: prompt }) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const prior = messages.slice(0, index);
      setMessages([...prior, { ...message, content: prompt }]);
      setEditingIndex(null);
      await sendMessage(undefined, { prompt, prior, conversationId: activeId, existingMessage: { ...message, content: prompt } });
    } catch (err) { window.alert(`Could not edit message: ${err instanceof Error ? err.message : String(err)}`); }
  }
  async function regenerateResponse(index: number) {
    if (!activeId || busy) return;
    let userIndex = index - 1;
    while (userIndex >= 0 && messages[userIndex].role !== "user") userIndex--;
    const userMessage = messages[userIndex];
    if (!userMessage?.id) return;
    try {
      const r = await fetch(`${API}/api/conversations/${encodeURIComponent(activeId)}/messages/${encodeURIComponent(userMessage.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: userMessage.content }) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      await sendMessage(undefined, { prompt: userMessage.content, prior: messages.slice(0, userIndex), conversationId: activeId, existingMessage: userMessage });
    } catch (err) { window.alert(`Could not regenerate response: ${err instanceof Error ? err.message : String(err)}`); }
  }
  function stopGeneration() { abortRef.current?.abort(); }
  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendMessage(); } }
  const online = health?.llama_server === "ok";
  const filteredConversations = conversations.filter(c => `${c.title} ${c.preview ?? ""}`.toLowerCase().includes(search.toLowerCase()));

  return <div className="app-shell"><aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}><div className="brand"><div className="brand-mark"><Sparkles size={19}/></div>{sidebarOpen && <div><strong>Brahma Desk</strong><span>LOCAL AI WORKSPACE</span></div>}</div>
    <button className="new-chat" onClick={() => void createConversation()} disabled={busy}><Plus size={17}/>{sidebarOpen && "New chat"}{sidebarOpen && <SquarePen size={15} className="new-chat-right"/>}</button>
    {sidebarOpen && <><div className="side-label">YOUR CHATS</div><div className="conversation-search"><Search size={14}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search conversations" aria-label="Search conversations"/></div><div className="conversation-list">{filteredConversations.map(c => <div className={`conversation-item ${activeId === c.id ? "selected" : ""}`} key={c.id}><button className="conversation-open" onClick={() => void openConversation(c.id)} title={c.title}><span>{c.title}</span></button><div className="conversation-actions"><button title="Rename conversation" onClick={() => void renameConversation(c)}><Pencil size={13}/></button><button title="Delete conversation" onClick={() => void deleteConversation(c)}><Trash2 size={13}/></button></div></div>)}{filteredConversations.length === 0 && <div className="empty-conversations">{search ? "No matching chats" : "Your conversations will appear here"}</div>}</div><div className="side-label">WORKSPACE</div><button className={`nav-item ${activePage === "chat" ? "active" : ""}`} onClick={() => setActivePage("chat")}><Bot size={17}/> Chat {activePage === "chat" && <span className="nav-dot"/>}</button><button className={`nav-item ${activePage === "installer" ? "active" : ""}`} onClick={() => setActivePage("installer")}><Download size={17}/> Model Setup {activePage === "installer" && <span className="nav-dot"/>}</button><button className={`nav-item ${activePage === "updates" ? "active" : ""}`} onClick={() => setActivePage("updates")}><ArrowUpCircle size={17}/> Update Center {updateState.status === "available" || updateState.status === "downloaded" ? <span className="update-nav-dot"/> : null}</button><button className="nav-item" title="Projects planned for a future phase"><Activity size={17}/> Projects <span className="soon">SOON</span></button><button className="nav-item" title="Memory planned for a future phase"><Cpu size={17}/> Memory <span className="soon">SOON</span></button><div className="side-spacer"/><div className="local-card"><div className="local-card-head"><ShieldCheck size={16}/><span>LOCAL-FIRST</span><span className="green-pip"/></div><p>Your prompts are sent to your local llama.cpp server.</p></div></>}
    <button className="nav-item" onClick={() => { setSamplingDraft({ ...sampling }); setSettingsOpen(true); }} title="Sampling and generation settings"><SlidersHorizontal size={17}/>{sidebarOpen && "Settings"}</button><button className="nav-item sidebar-bottom" onClick={() => setSidebarOpen(!sidebarOpen)}><Menu size={17}/>{sidebarOpen && "Collapse sidebar"}</button></aside>
    <main className="main"><header className="topbar"><div className="crumb"><span>Workspace</span><span className="slash">/</span><strong>{activePage === "updates" ? "Update Center" : activePage === "installer" ? "Model Setup" : "Chat"}</strong><ChevronDown size={14}/></div><div className="top-actions"><div className="profile-picker" title="Chats are isolated by profile"><span className="profile-avatar" style={{ backgroundColor: activeProfile?.avatar_color ?? "#7968E8" }}>{(activeProfile?.display_name ?? "D").slice(0, 1).toUpperCase()}</span><select aria-label="Active profile" value={activeProfile?.id ?? ""} onChange={e => void changeProfile(e.target.value)} disabled={profileBusy || profiles.length === 0}>{profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.display_name}</option>)}</select><button className="profile-add" title="Create profile" aria-label="Create profile" onClick={openCreateProfile} disabled={profileBusy}><Plus size={15}/></button><button className="profile-manage" title="Edit active profile" aria-label="Edit active profile" onClick={() => void openEditProfile()} disabled={profileBusy || !activeProfile}><Pencil size={13}/></button><button className="profile-manage danger" title={activeProfile?.is_default ? "The default profile cannot be deleted" : "Delete active profile"} aria-label="Delete active profile" onClick={() => void deleteActiveProfile()} disabled={profileBusy || !activeProfile || activeProfile.is_default}><Trash2 size={13}/></button></div><div className={`connection ${online ? "online" : "offline"}`}>{online ? <Wifi size={14}/> : <WifiOff size={14}/>}<span>{online ? "Local model connected" : "Model disconnected"}</span></div><button className="icon-button" title="Settings"><Settings2 size={18}/></button></div></header>
    <section ref={chatAreaRef} onScroll={handleChatScroll} className={`chat-area ${activePage !== "chat" ? "update-center-area" : messages.length ? "has-messages" : ""}`}>{activePage === "installer" ? <InstallerWizard/> : activePage === "updates" ? <div className="update-center">
      <div className="update-hero"><div className="update-hero-icon"><ArrowUpCircle size={26}/></div><div><div className="dialog-eyebrow">BRAHMA DESK MAINTENANCE</div><h1>Update Center</h1><p>Keep Brahma Desk secure and up to date. Updates are downloaded only when you choose.</p></div><button className="dialog-secondary update-check" onClick={() => void checkForUpdates()} disabled={updateActionBusy || updateState.status === "checking" || updateState.status === "downloading"}><RefreshCw size={15} className={updateState.status === "checking" ? "update-spin" : ""}/>{updateState.status === "checking" ? "Checking…" : "Check for updates"}</button></div>
      <div className="update-version-grid"><article className="update-info-card"><span className="update-card-label">INSTALLED VERSION</span><strong>v{updateState.currentVersion}</strong><span className="update-muted">Current version on this device</span></article><article className="update-info-card"><span className="update-card-label">UPDATE STATUS</span><strong className={`update-status-text ${updateState.status === "error" ? "is-error" : updateState.status === "up-to-date" || updateState.status === "downloaded" ? "is-good" : ""}`}>{updateState.status === "up-to-date" ? "Up to date" : updateState.status === "available" ? "Update available" : updateState.status === "downloading" ? "Downloading" : updateState.status === "downloaded" ? "Ready to install" : updateState.status === "checking" ? "Checking…" : updateState.status === "error" ? "Check failed" : updateState.status === "unavailable" ? "Unavailable in dev" : "Not checked"}</strong><span className="update-muted">{updateState.message}</span></article></div>
      {(updateState.status === "downloading" || updateState.status === "downloaded") && <div className="update-progress-card"><div className="update-progress-heading"><span>{updateState.status === "downloaded" ? "Download complete" : "Downloading update"}</span><strong>{updateState.percent}%</strong></div><div className="update-progress-track"><div style={{ width: `${updateState.percent}%` }}/></div>{updateState.status === "downloaded" && <p>Ready to install. Brahma Desk will restart to apply the update.</p>}</div>}
      {updateState.status === "available" && <div className="update-release-card"><div className="update-release-head"><div><span className="update-card-label">NEW RELEASE</span><h2>Brahma Desk {updateState.availableVersion}</h2></div><button className="dialog-primary" onClick={() => void downloadUpdate()} disabled={updateActionBusy}><Download size={15}/>{updateActionBusy ? "Starting…" : "Download update"}</button></div><p>{updateState.message}</p></div>}
      {updateState.status === "downloaded" && <div className="update-release-card"><div className="update-release-head"><div><span className="update-card-label">READY WHEN YOU ARE</span><h2>Install update</h2></div><button className="dialog-primary" onClick={() => void installUpdate()} disabled={updateActionBusy}><PackageCheck size={15}/>Install and restart</button></div><p>Save your work before continuing. The application will close and reopen with the new version.</p></div>}
      {updateState.status === "up-to-date" && <div className="update-message-card"><CircleCheck size={20}/><div><strong>You're all caught up</strong><p>No newer stable release was found for this installation.</p></div></div>}
      {updateState.status === "error" && <div className="update-message-card error"><AlertCircle size={20}/><div><strong>Could not complete the update check</strong><p>{updateState.message}</p><small>Confirm that this release is published and that the app's update repository is configured.</small></div></div>}
      {updateState.status === "unavailable" && <div className="update-message-card"><AlertCircle size={20}/><div><strong>Release updates need a packaged build</strong><p>{updateState.message}</p><small>Development mode is safe: it will not download or install application updates.</small></div></div>}
      {updateState.releaseNotes && <div className="update-notes-card"><span className="update-card-label">RELEASE NOTES</span><h2>What's new</h2><pre>{updateState.releaseNotes}</pre></div>}
      <div className="update-footnote"><ShieldCheck size={15}/><span>Installation is user-controlled. Checking for updates never installs one automatically.</span></div>
    </div> : messages.length === 0 ? <div className="welcome"><div className="welcome-orb"><Sparkles size={27}/></div><div className="eyebrow"><span className="eyebrow-line"/> YOUR LOCAL AI WORKSPACE <span className="eyebrow-line"/></div><h1>Think clearly.<br/><span>Build anything.</span></h1><p className="welcome-copy">A private, local-first workspace for ideas, research, and building with AI. Your model runs on your machine.</p><div className="suggestions"><button onClick={() => setDraft("Explain how local LLM inference works on my computer.")}><Cpu size={17}/><span><strong>Understand AI</strong><small>Explain local model inference</small></span></button><button onClick={() => setDraft("Help me plan the next milestones for Brahma Desk.")}><Sparkles size={17}/><span><strong>Plan a project</strong><small>Turn an idea into milestones</small></span></button><button onClick={() => setDraft("Write a small TypeScript function and explain how it works.")}><SquarePen size={17}/><span><strong>Write code</strong><small>Build and understand a solution</small></span></button></div><div className="privacy-note"><ShieldCheck size={14}/> Local inference · No cloud model required</div></div> : <div className="message-list">{messages.map((m, i) => <div className={`message-row ${m.role}`} key={m.id ?? `${i}-${m.role}`}><div className={`message-avatar ${m.role}`}>{m.role === "assistant" ? <Sparkles size={16}/> : "D"}</div><div className="message-body"><div className="message-heading"><div className="message-name">{m.role === "assistant" ? "Brahma" : "You"}</div>{!busy && <div className="message-tools">{m.role === "user" && m.id && <button title="Edit message" onClick={() => { setEditingIndex(i); setEditDraft(m.content); }}><Pencil size={13}/> Edit</button>}{m.role === "assistant" && m.content && <button title="Copy response" onClick={() => void copyMessage(m.content)}><Copy size={13}/> Copy</button>}{m.role === "assistant" && i > 0 && messages.slice(0, i).some(previous => previous.role === "user") && <button title="Regenerate response" onClick={() => void regenerateResponse(i)}><RotateCcw size={13}/> Regenerate</button>}</div>}</div>{editingIndex === i ? <div className="edit-message"><textarea value={editDraft} onChange={e => setEditDraft(e.target.value)} rows={3}/><div><button onClick={() => setEditingIndex(null)}><X size={13}/> Cancel</button><button className="edit-save" onClick={() => void editAndResubmit(i)} disabled={!editDraft.trim()}><Check size={13}/> Save & resubmit</button></div></div> : <div className="message-content">{m.content ? <MarkdownContent content={m.content} onNotice={setNotice}/> : (busy && i === messages.length - 1 ? <span className="typing"><i/><i/><i/></span> : "")}</div>}</div></div>)}{showJumpToLatest && <button className="jump-to-latest" onClick={jumpToLatest}><ArrowDown size={14}/> Jump to latest</button>}<div ref={bottomRef}/></div>}</section>
    <footer className="composer-wrap" style={{ display: activePage !== "chat" ? "none" : undefined }}><form className="composer" onSubmit={sendMessage}><textarea ref={composerRef} value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={onKeyDown} placeholder="Message Brahma..." rows={1} disabled={busy} aria-label="Message Brahma"/><div className="composer-bottom"><div className="composer-hints"><span className="model-chip"><span className={`status-pip ${online ? "up" : ""}`}/> Qwen · Local</span><label className="reasoning-control" title={REASONING_OPTIONS.find(option => option.value === reasoningBudget)?.detail ?? "Reasoning budget"}><span>Thinking</span><select aria-label="Reasoning budget" value={reasoningBudget} disabled={busy} onChange={e => { const value = e.target.value as ReasoningBudget; setReasoningBudget(value); localStorage.setItem(REASONING_STORAGE_KEY, value); }}><option value="off">Off</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="max">Max</option></select></label><span className="hint-separator">·</span><span>Shift + Enter for new line</span></div>{busy ? <button className="send-button stop-button" type="button" onClick={stopGeneration}><Square size={14}/><span>Stop</span></button> : <button className="send-button" type="submit" disabled={!draft.trim()}><Send size={16}/><span>Send</span></button>}</div></form><div className="footer-note"><span>AI can make mistakes. Verify important information.</span><span><CircleHelp size={13}/> <span>Phase 2 · Local chat history</span></span></div></footer></main>
    {settingsOpen && <div className="modal-backdrop settings-backdrop" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) setSettingsOpen(false); }}><section className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title"><header className="settings-header"><div><div className="dialog-eyebrow">BRAHMA DESK CONFIGURATION</div><h2 id="settings-title"><Settings2 size={20}/> Settings</h2><p>Control generation, sampling, and repetition behavior for your local model.</p></div><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><X size={18}/></button></header><div className="settings-layout"><nav className="settings-nav" aria-label="Settings sections">{[{id:"general",label:"General",icon:<Settings2 size={17}/>},{id:"display",label:"Display",icon:<Activity size={17}/>},{id:"tools",label:"Tools",icon:<Cpu size={17}/>},{id:"agentic",label:"Agentic",icon:<Sparkles size={17}/>},{id:"import",label:"Import/Export",icon:<Copy size={17}/>},{id:"sampling",label:"Sampling & Penalties",icon:<SlidersHorizontal size={17}/>}].map(t => <button key={t.id} className={settingsTab===t.id?"settings-nav-item active":"settings-nav-item"} onClick={() => setSettingsTab(t.id)}>{t.icon}<span>{t.label}</span></button>)}</nav><div className="settings-content">{settingsTab !== "sampling" ? <div className="settings-placeholder"><SlidersHorizontal size={25}/><h3>{({general:"General",display:"Display",tools:"Tools",agentic:"Agentic","import":"Import/Export"} as Record<string,string>)[settingsTab]}</h3><p>This section is reserved for the next settings milestone. Generation controls are available under Sampling &amp; Penalties.</p><button className="dialog-secondary" onClick={() => setSettingsTab("sampling")}>Open Sampling &amp; Penalties</button></div> : <><div className="settings-section-title"><div><h3>Sampling &amp; Penalties</h3><p>Changes apply to new generations and are saved on this device.</p></div><button className="dialog-secondary compact" onClick={() => setSamplingDraft({...DEFAULT_SAMPLING})}>Reset defaults</button></div><div className="reasoning-budget-card"><div><h4>Reasoning / thinking budget</h4><p>Controls whether the model uses its thinking mode. Token amounts are guidance for compatible models, not a guaranteed hard limit.</p></div><select aria-label="Reasoning budget setting" value={reasoningBudget} onChange={e => { const value = e.target.value as ReasoningBudget; setReasoningBudget(value); localStorage.setItem(REASONING_STORAGE_KEY, value); }}><option value="off">Off — disable thinking</option><option value="low">Low — ~512 tokens</option><option value="medium">Medium — ~2,048 tokens</option><option value="high">High — ~8,192 tokens</option><option value="max">Max — no app target</option></select></div><div className="settings-grid">{[ ["temperature","Temperature",0,2,0.05,"Randomness of generated text. Default 0.8."],["dynatemp_range","Dynamic temperature range",0,2,0.05,"Entropy-based temperature adjustment range."],["dynatemp_exponent","Dynamic temperature exponent",0.1,5,0.1,"Smooths redistribution based on the most probable token."],["top_k","Top-K",0,1000,1,"Keep only the K highest-probability tokens."],["top_p","Top-P",0,1,0.01,"Nucleus sampling cumulative probability threshold."],["min_p","Min-P",0,1,0.01,"Minimum probability relative to the most likely token."],["typ_p","Typical-P",0,1,0.01,"Sorts tokens using entropy difference."],["xtc_probability","XTC probability",0,1,0.01,"Chance of applying XTC token removal; 0 disables it."],["xtc_threshold","XTC threshold",0,1,0.01,"Probability threshold for XTC token removal."],["min_keep","Minimum tokens to keep",1,1000,1,"Minimum candidate tokens retained by samplers."],["max_tokens","Maximum output tokens",-1,131072,1,"Use -1 for no output-token limit."],["repeat_last_n","Repeat penalty last N",-1,4096,1,"Number of recent tokens considered for repetition penalties."],["repeat_penalty","Repetition penalty",0.01,3,0.05,"Controls repeated token sequences; 1 disables the penalty."],["presence_penalty","Presence penalty",-2,2,0.05,"Penalizes tokens based on whether they appeared."],["frequency_penalty","Frequency penalty",-2,2,0.05,"Penalizes tokens based on how often they appeared."],["dry_multiplier","DRY multiplier",0,10,0.1,"Strength of long-range repetition reduction."],["dry_base","DRY base",1,5,0.05,"Base value for DRY repetition penalty."],["dry_allowed_length","DRY allowed length",1,64,1,"Allowed repeated sequence length before DRY penalty."],["dry_penalty_last_n","DRY penalty last N",-1,4096,1,"Recent token window for DRY penalty; 0 uses backend default."]] .map(([key,label,min,max,step,help]) => <label className="setting-field" key={String(key)}><span>{String(label)}</span><input type="number" min={Number(min)} max={Number(max)} step={Number(step)} value={samplingDraft[key as keyof SamplingSettings] as number} onChange={e => setSamplingDraft(v => ({...v,[key]:Number(e.target.value)}))}/><small>{String(help)}</small></label>)}<label className="setting-field setting-wide"><span>Sampler order</span><input value={samplingDraft.sampler_order} onChange={e => setSamplingDraft(v => ({...v,sampler_order:e.target.value}))} placeholder="top_k;typ_p;top_p;min_p;temperature"/><small>Semicolon-separated order. Example: top_k;typ_p;top_p;min_p;temperature</small></label><label className="setting-toggle setting-wide"><input type="checkbox" checked={samplingDraft.backend_sampling} onChange={e => setSamplingDraft(v => ({...v,backend_sampling:e.target.checked}))}/><span><strong>Backend sampling</strong><small>Enable backend-based samplers where supported by the server build.</small></span></label></div></>}</div></div><footer className="settings-footer"><span>Settings are saved in this browser's local storage.</span><div><button className="dialog-secondary" onClick={() => setSettingsOpen(false)}>Cancel</button><button className="dialog-primary" onClick={() => { const next = {...samplingDraft}; setSampling(next); setSamplingDraft(next); localStorage.setItem(SAMPLING_STORAGE_KEY, JSON.stringify(next)); setSettingsOpen(false); setNotice("Sampling settings saved."); }}><Check size={15}/> Save settings</button></div></footer></section></div>}
    {profileDialogOpen && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget && !profileBusy) setProfileDialogOpen(false); }}><section className="dialog-card profile-dialog" role="dialog" aria-modal="true" aria-labelledby="profile-create-title"><div className="dialog-icon">{profileDialogMode === "create" ? <Plus size={20}/> : <Pencil size={20}/>}</div><div className="dialog-eyebrow">PROFILE PERSONALIZATION</div><h2 id="profile-create-title">{profileDialogMode === "create" ? "Create a profile" : "Edit profile"}</h2><p>Personalize Brahma for this identity. Optional fields can be completed later.</p><form onSubmit={(e) => { e.preventDefault(); void saveProfile(); }}><label className="rename-label" htmlFor="new-profile-name">Profile name *</label><input id="new-profile-name" className="secret-input rename-input" autoFocus maxLength={80} value={profileDraft} onChange={(e) => setProfileDraft(e.target.value)} placeholder="e.g. Work, Research, Personal" disabled={profileBusy}/><label className="rename-label" htmlFor="profile-user-name">What should Brahma call you?</label><input id="profile-user-name" className="secret-input rename-input" maxLength={100} value={profileSettingsDraft.user_name} onChange={e => setProfileSettingsDraft(s => ({...s,user_name:e.target.value}))} placeholder="Preferred name" disabled={profileBusy}/><label className="rename-label" htmlFor="profile-purpose">Main purpose</label><select id="profile-purpose" className="secret-input rename-input" value={profileSettingsDraft.purpose} onChange={e => setProfileSettingsDraft(s => ({...s,purpose:e.target.value}))} disabled={profileBusy}><option value="">Choose a purpose</option><option>Personal assistant</option><option>Study and learning</option><option>Software development</option><option>Work and business</option><option>Research and AI</option><option>Writing and creative work</option><option>Other / multiple purposes</option></select><label className="rename-label" htmlFor="profile-role">Occupation or role</label><input id="profile-role" className="secret-input rename-input" maxLength={120} value={profileSettingsDraft.role} onChange={e => setProfileSettingsDraft(s => ({...s,role:e.target.value}))} placeholder="Student, developer, researcher..." disabled={profileBusy}/><label className="rename-label" htmlFor="profile-interests">Interests and expertise</label><textarea id="profile-interests" className="secret-input profile-textarea" rows={2} maxLength={2000} value={profileSettingsDraft.interests} onChange={e => setProfileSettingsDraft(s => ({...s,interests:e.target.value}))} placeholder="Topics, skills, hobbies..." disabled={profileBusy}/><label className="rename-label" htmlFor="profile-goals">Current goals</label><textarea id="profile-goals" className="secret-input profile-textarea" rows={2} maxLength={2000} value={profileSettingsDraft.goals} onChange={e => setProfileSettingsDraft(s => ({...s,goals:e.target.value}))} placeholder="What should Brahma help you achieve?" disabled={profileBusy}/><label className="rename-label" htmlFor="profile-style">Response style</label><select id="profile-style" className="secret-input rename-input" value={profileSettingsDraft.response_style} onChange={e => setProfileSettingsDraft(s => ({...s,response_style:e.target.value}))} disabled={profileBusy}><option>Balanced</option><option>Concise and direct</option><option>Detailed and technical</option><option>Step-by-step guidance</option></select><label className="rename-label" htmlFor="profile-language">Preferred language</label><select id="profile-language" className="secret-input rename-input" value={profileSettingsDraft.language} onChange={e => setProfileSettingsDraft(s => ({...s,language:e.target.value}))} disabled={profileBusy}><option>English</option><option>Kannada</option><option>Hindi</option><option>English + Hindi / Hinglish</option><option>Other</option></select><label className="rename-label" htmlFor="profile-extra">Additional preferences (optional)</label><textarea id="profile-extra" className="secret-input profile-textarea" rows={2} maxLength={3000} value={profileSettingsDraft.additional_context} onChange={e => setProfileSettingsDraft(s => ({...s,additional_context:e.target.value}))} placeholder="Tools, constraints, or preferences..." disabled={profileBusy}/><div className="dialog-actions"><button type="button" className="dialog-secondary" onClick={() => setProfileDialogOpen(false)} disabled={profileBusy}>Cancel</button><button type="submit" className="dialog-primary" disabled={profileBusy || !profileDraft.trim()}>{profileBusy ? "Saving…" : <><Check size={15}/> {profileDialogMode === "create" ? "Create profile" : "Save changes"}</>}</button></div></form></section></div>}
    {notice && <div className="notice-toast" role="status"><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss notification"><X size={14}/></button></div>}
    {renameTarget && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget && !renameBusy) setRenameTarget(null); }}><section className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="rename-title"><div className="dialog-icon"><Pencil size={20}/></div><div className="dialog-eyebrow">CONVERSATION MANAGEMENT</div><h2 id="rename-title">Rename conversation</h2><p>Choose a clear name so you can find this chat later. Your messages will remain unchanged.</p><form onSubmit={(e) => { e.preventDefault(); void saveConversationRename(); }}><label className="rename-label" htmlFor="conversation-name">Conversation name</label><input id="conversation-name" className="secret-input rename-input" autoFocus maxLength={200} value={renameDraft} onChange={(e) => setRenameDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Escape" && !renameBusy) setRenameTarget(null); }} placeholder="Enter conversation name" disabled={renameBusy}/><div className="dialog-actions"><button type="button" className="dialog-secondary" onClick={() => setRenameTarget(null)} disabled={renameBusy}>Cancel</button><button type="submit" className="dialog-primary" disabled={renameBusy || !renameDraft.trim()}>{renameBusy ? "Saving…" : <><Check size={15}/> Save name</>}</button></div></form></section></div>}
    {deleteTarget && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) setDeleteTarget(null); }}><section className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="delete-title"><div className="dialog-icon danger"><Trash2 size={21}/></div><div className="dialog-eyebrow">CONVERSATION MANAGEMENT</div><h2 id="delete-title">Delete this conversation?</h2><p>You’re about to permanently delete <strong>“{deleteTarget.title}”</strong> and all its messages from this device. This action can’t be undone.</p><div className="dialog-actions"><button className="dialog-secondary" onClick={() => setDeleteTarget(null)}>Cancel</button><button className="dialog-danger" onClick={() => void confirmDeleteConversation()}><Trash2 size={15}/> Delete conversation</button></div></section></div>}
  </div>;
}
