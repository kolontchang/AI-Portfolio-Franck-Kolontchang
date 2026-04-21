# ═══════════════════════════════════════════════════════════════
# hunter_gradio_app.py  v2.0 — SVG Edition
# The Hunter · Automated Email Phishing Defense System
# ITAI 2376 · Houston City College · Team 1
#
# HOW TO LAUNCH (from Colab, after notebook training is complete):
#   !pip install -q gradio
#   %run hunter_gradio_app.py
#
# HOW TO LAUNCH (locally):
#   pip install gradio crewai tensorflow scikit-learn litellm python-dotenv
#   python hunter_gradio_app.py
# ═══════════════════════════════════════════════════════════════

import os, re, json, sqlite3, pickle, warnings, logging, io, sys, time
from pathlib import Path
from html import unescape, escape
from datetime import datetime, timezone

import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("hunter_ui")

# ── PATHS & CONFIG ──────────────────────────────────────────────
MODELS_DIR   = Path("models/")
DATA_DIR     = Path("data/")
THREAT_DB    = DATA_DIR / "threat_memory.db"
CONFIG_FILE  = MODELS_DIR / "config.json"

# Mirror notebook constants exactly (fallbacks if config.json absent)
VOCAB_SIZE              = 20_000
MAX_SEQ_LENGTH          = 300
BILSTM_WEIGHT           = 0.65
LOGREG_WEIGHT           = 0.35
CONFIDENCE_LOW          = 0.35
CONFIDENCE_HIGH         = 0.75
TIER_QUARANTINE         = 0.90
TIER_FLAG               = 0.70
DEEP_ANALYSIS_MAX_BOOST = 0.34
MAX_ITERATIONS          = 3
FEATURE_NAMES           = ["char_count","word_count","url_count","exclaim_count","urgent_keyword_count"]

# Load calibrated threshold from saved config (written by notebook)
try:
    with open(CONFIG_FILE) as f:
        _cfg = json.load(f)
    OPTIMAL_THRESHOLD = float(_cfg.get("optimal_threshold", 0.3771))
except Exception:
    OPTIMAL_THRESHOLD = 0.3771  # notebook-reported value

# ── MODEL LOADING ────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODELS_LOADED = False
MODEL_ERROR   = ""
BILSTM = LOGREG = TOKENIZER = SCALER = None

def _load_models():
    global BILSTM, LOGREG, TOKENIZER, SCALER, MODELS_LOADED, MODEL_ERROR
    try:
        BILSTM = tf.keras.models.load_model(str(MODELS_DIR / "bilstm_model.keras"))
        with open(MODELS_DIR / "logreg_model.pkl",  "rb") as f: LOGREG    = pickle.load(f)
        with open(MODELS_DIR / "tokenizer.pkl",     "rb") as f: TOKENIZER = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl",        "rb") as f: SCALER    = pickle.load(f)
        MODELS_LOADED = True
    except Exception as e:
        MODEL_ERROR = str(e)

_load_models()

# ── CREWAI / LLM SETUP ───────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

CREWAI_OK    = False
GROQ_KEY_OK  = bool(os.getenv("GROQ_API_KEY"))
CREWAI_ERROR = ""

try:
    import litellm
    litellm.num_retries    = 5
    litellm.request_timeout = 120
    os.environ.setdefault("LITELLM_RETRY_WAIT", "10")

    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    LLM_MODEL = LLM(model="groq/llama-3.1-8b-instant", temperature=0.0)
    CREWAI_OK = True
except Exception as e:
    CREWAI_ERROR = str(e)

# ── FEATURE EXTRACTION ───────────────────────────────────────────
URGENT_KEYWORDS = {
    "urgent","verify","confirm","suspend","account","click","login",
    "password","bank","limited","immediately","expire","alert",
    "security","update","unusual","access","validate",
}
_URL_RE  = re.compile(r"https?://\S+|www\.\S+", re.I)
_HTML_RE = re.compile(r"<[^>]+>")

def clean_text(raw: str) -> str:
    t = unescape(raw)
    t = _HTML_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_features(text: str) -> dict:
    c = clean_text(text)
    urls  = _URL_RE.findall(c)
    words = c.split()
    urg   = sum(1 for w in words if w.lower().strip(".,!?") in URGENT_KEYWORDS)
    return {
        "cleaned_text":          c,
        "char_count":            len(c),
        "word_count":            len(words),
        "url_count":             len(urls),
        "exclaim_count":         c.count("!"),
        "urgent_keyword_count":  urg,
        "urls_found":            urls[:5],
    }

# ── ENSEMBLE PREDICTION ──────────────────────────────────────────
def ensemble_predict(text: str) -> dict:
    if not MODELS_LOADED:
        return {"error": MODEL_ERROR, "bilstm_score": 0.5,
                "logreg_score": 0.5, "ensemble_score": 0.5}
    feats = extract_features(text)
    c     = feats["cleaned_text"]

    seq    = TOKENIZER.texts_to_sequences([c])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, truncating="post", padding="post")
    bp     = float(BILSTM.predict(padded, verbose=0)[0][0])

    fv     = np.array([[feats[k] for k in FEATURE_NAMES]])
    lp     = float(LOGREG.predict_proba(SCALER.transform(fv))[0][1])
    ens    = BILSTM_WEIGHT * bp + LOGREG_WEIGHT * lp

    return {
        "bilstm_score":   round(bp,  4),
        "logreg_score":   round(lp,  4),
        "ensemble_score": round(ens, 4),
    }

# ── DEEP ANALYSIS (mirrors notebook exactly) ─────────────────────
_PHISH_PATTERNS = [
    (re.compile(r"verify your (account|identity|password)",      re.I), "credential_harvest"),
    (re.compile(r"click\s*(here|below|this link)",               re.I), "clickbait"),
    (re.compile(r"suspend|deactivat|terminat|clos.*your account",re.I), "account_threat"),
    (re.compile(r"urgent|immediate\s*action|immediate\s*response",re.I),"urgency_pressure"),
    (re.compile(r"won|winner|congratulat|prize|reward",          re.I), "reward_lure"),
    (re.compile(r"update|confirm.*payment|billing|credit card",  re.I), "financial_phish"),
]
_STRUCT_FLAGS = [
    ("excessive_urls",     lambda t: t.lower().count("url") > 3),
    ("excessive_urgency",  lambda t: t.count("!") > 5),
    ("short_and_directive",lambda t: len(t.split()) < 50 and "click" in t.lower()),
    ("generic_greeting",   lambda t: bool(re.search(r"dear (customer|user|member|valued)", t, re.I))),
]

def deep_analysis(text: str, current_score: float) -> dict:
    pats  = [n for p, n in _PHISH_PATTERNS if p.search(text)]
    flags = [n for n, fn in _STRUCT_FLAGS if fn(text)]
    boost = min((len(pats) + len(flags)) * 0.034, DEEP_ANALYSIS_MAX_BOOST)
    return {
        "original_score":      round(current_score, 4),
        "boost":               round(boost, 4),
        "adjusted_risk_score": round(min(current_score + boost, 1.0), 4),
        "patterns_found":      pats,
        "flags_found":         flags,
    }

# ── THREAT MEMORY ────────────────────────────────────────────────
def _init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(THREAT_DB)) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS threat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL, risk_score REAL NOT NULL,
            action TEXT NOT NULL, timestamp TEXT NOT NULL)""")

def query_threat_memory(sender: str) -> list:
    _init_db()
    with sqlite3.connect(str(THREAT_DB)) as c:
        rows = c.execute(
            "SELECT risk_score,action,timestamp FROM threat_log "
            "WHERE sender=? ORDER BY id DESC LIMIT 5", (sender,)
        ).fetchall()
    return [{"risk_score": r[0], "action": r[1], "timestamp": r[2]} for r in rows]

def write_threat_memory(sender: str, risk_score: float, action: str):
    _init_db()
    with sqlite3.connect(str(THREAT_DB)) as c:
        c.execute(
            "INSERT INTO threat_log (sender,risk_score,action,timestamp) VALUES (?,?,?,?)",
            (sender, risk_score, action, datetime.now(timezone.utc).isoformat())
        )

def get_all_history() -> list:
    _init_db()
    with sqlite3.connect(str(THREAT_DB)) as c:
        rows = c.execute(
            "SELECT sender,risk_score,action,timestamp FROM threat_log ORDER BY id DESC LIMIT 30"
        ).fetchall()
    return [{"sender":r[0],"risk_score":r[1],"action":r[2],"timestamp":r[3]} for r in rows]

# ── DEFENSE POLICY ───────────────────────────────────────────────
def apply_defense_policy(score: float, is_repeat: bool) -> dict:
    if score >= TIER_QUARANTINE or (is_repeat and score >= TIER_FLAG):
        return {"verdict":"QUARANTINE",
                "label":"Quarantine",
                "action":"Block delivery immediately. Move to quarantine folder. Alert SOC team.",
                "tier":"red",  "score": round(score, 4)}
    elif score >= TIER_FLAG:
        return {"verdict":"FLAG WITH WARNING",
                "label":"Flag with Warning",
                "action":"Deliver with prominent warning banner. Log for SOC review within 4 hours.",
                "tier":"orange","score": round(score, 4)}
    elif score >= OPTIMAL_THRESHOLD:
        return {"verdict":"ALLOW AND LOG",
                "label":"Allow & Log",
                "action":"Deliver. Add to passive audit log for review during next SOC cycle.",
                "tier":"yellow","score": round(score, 4)}
    else:
        return {"verdict":"ALLOW",
                "label":"Allow",
                "action":"Clean email. No action required. Delivered normally.",
                "tier":"green", "score": round(score, 4)}

# ── CREWAI TOOLS (aligned with notebook) ────────────────────────
_STATE = {}

if CREWAI_OK:
    class IngestInput(BaseModel):
        email_body: str = Field(description="Raw email body text")

    class EmailIngestTool(BaseTool):
        name: str = "email_ingest_tool"
        description: str = (
            "Strips HTML, normalizes URLs to [url] token, extracts sender fingerprint, "
            "computes 5 features: char_count, word_count, url_count, exclaim_count, urgent_keyword_count."
        )
        args_schema: type[BaseModel] = IngestInput
        result_as_answer: bool = True

        def _run(self, email_body: str) -> str:
            feats = extract_features(email_body)
            _STATE["features"] = feats
            _STATE["cleaned_text"] = feats["cleaned_text"]
            out = {k: feats[k] for k in FEATURE_NAMES}
            out["cleaned_text"] = feats["cleaned_text"]
            out["feature_names"] = FEATURE_NAMES
            out["sender_hint"] = "unknown"
            return json.dumps(out)

    class ClassifyInput(BaseModel):
        clean_text: str = Field(description="Cleaned email text")

    class PhishingClassifierTool(BaseTool):
        name: str = "phishing_classifier_tool"
        description: str = (
            "BiLSTM (65%) + LogReg (35%) ensemble. Pass only clean_text. "
            "Features are auto-computed. Returns risk_score and confident flag."
        )
        args_schema: type[BaseModel] = ClassifyInput

        def _run(self, clean_text: str) -> str:
            result = ensemble_predict(clean_text)
            score  = result["ensemble_score"]
            result["confident"] = score < CONFIDENCE_LOW or score > CONFIDENCE_HIGH
            result["threshold_used"] = OPTIMAL_THRESHOLD
            _STATE["classifier_result"] = result
            return json.dumps(result)

    class DeepInput(BaseModel):
        clean_text:    str = Field(description="Cleaned email text")
        current_score: str = Field(description="Current risk score string")

    class DeepAnalysisTool(BaseTool):
        name: str = "deep_analysis_tool"
        description: str = (
            "6 regex phishing archetypes + 4 structural red-flags. "
            "Adjusts score by up to 0.34. Call only when score is ambiguous (0.35-0.75)."
        )
        args_schema: type[BaseModel] = DeepInput

        def _run(self, clean_text: str, current_score: str) -> str:
            try:    score = float(current_score)
            except: score = 0.5
            result = deep_analysis(clean_text, score)
            _STATE["deep_result"]    = result
            _STATE["deep_triggered"] = True
            return json.dumps(result)

    class PolicyInput(BaseModel):
        risk_score:  str = Field(description="Risk score string")
        sender_hint: str = Field(description="Sender identifier", default="unknown")

    class DefensePolicyTool(BaseTool):
        name: str = "defense_policy_tool"
        description: str = (
            "Maps risk score to 4-tier action: QUARANTINE (>=0.90), "
            "FLAG WITH WARNING (>=0.70), ALLOW AND LOG (>threshold), ALLOW."
        )
        args_schema: type[BaseModel] = PolicyInput

        def _run(self, risk_score: str, sender_hint: str = "unknown") -> str:
            try:    s = float(risk_score)
            except: s = 0.5
            past      = _STATE.get("threat_history", [])
            is_repeat = any(p["action"] in ("FLAG WITH WARNING","QUARANTINE") for p in past)
            policy    = apply_defense_policy(s, is_repeat)
            _STATE["final_verdict"] = policy
            return json.dumps({"action": policy["verdict"],
                               "explanation": policy["action"],
                               "risk_score": s, "sender": sender_hint})

    class MemInput(BaseModel):
        sender:     str = Field(description="Sender identifier")
        risk_score: str = Field(description="Risk score string")
        action:     str = Field(description="Security action taken")
        mode:       str = Field(description="read, write, or both", default="both")

    class ThreatMemoryTool(BaseTool):
        name: str = "threat_memory_tool"
        description: str = (
            "Reads sender's last 5 verdicts from SQLite, writes new verdict, "
            "flags REPEAT OFFENDER if prior FLAG/QUARANTINE exists."
        )
        args_schema: type[BaseModel] = MemInput

        def _run(self, sender: str, risk_score: str, action: str, mode: str = "both") -> str:
            try:    sc = float(risk_score)
            except: sc = 0.0
            past, repeat = [], False
            if mode in ("read","both"):
                past   = query_threat_memory(sender)
                repeat = any(p["action"] in ("FLAG WITH WARNING","QUARANTINE") for p in past)
                _STATE["threat_history"] = past
                _STATE["is_repeat"]      = repeat
            if mode in ("write","both"):
                write_threat_memory(sender, sc, action)
            return json.dumps({
                "past_incidents":    past,
                "repeat_offender":   repeat,
                "count":             len(past),
                "summary":           f"{sender}: {len(past)} prior. {'REPEAT OFFENDER.' if repeat else 'No prior flags.'}",
            })

    def build_crew(raw_email: str, sender: str):
        ingest_agent = Agent(
            role="Email Ingest Specialist",
            goal="Use the email_ingest_tool to process the raw email body. Pass the entire email body as the email_body parameter. Return the tool output as-is.",
            backstory="You are a senior email security engineer. Your only job is to call email_ingest_tool with the raw email and return its JSON output unchanged. Do not call any other tool.",
            tools=[EmailIngestTool()],
            llm=LLM_MODEL, max_iter=2, max_retry_limit=2,
            respect_context_window=True, allow_delegation=False, verbose=True,
        )
        analyst_agent = Agent(
            role="Phishing Risk Analyst",
            goal=(
                "Call phishing_classifier_tool with the EXACT clean_text from the previous task output. "
                "Do NOT rewrite or invent email text. If the risk_score is between 0.35 and 0.75, "
                "call deep_analysis_tool to refine it. Return the final JSON score."
            ),
            backstory="You are a threat intelligence analyst. You receive structured JSON from the prior agent containing a clean_text field. You MUST pass that exact clean_text to phishing_classifier_tool. The tool auto-computes features internally. Never fabricate text.",
            tools=[PhishingClassifierTool(), DeepAnalysisTool()],
            llm=LLM_MODEL, max_iter=MAX_ITERATIONS, max_retry_limit=2,
            respect_context_window=True, allow_delegation=False, verbose=True,
        )
        soc_agent = Agent(
            role="SOC Orchestrator",
            goal="Issue the security response that best protects the organization and maintain institutional threat memory for repeat-offender detection.",
            backstory="You are the senior SOC lead. Before any verdict you check the sender's threat history — a repeat offender changes the calculus. You apply a calibrated four-tier defense policy and log every verdict to institutional memory.",
            tools=[DefensePolicyTool(), ThreatMemoryTool()],
            llm=LLM_MODEL, max_iter=MAX_ITERATIONS, max_retry_limit=2,
            respect_context_window=True, allow_delegation=False, verbose=True,
        )

        t1 = Task(
            description=(
                "You have exactly ONE tool: email_ingest_tool. "
                "Call email_ingest_tool with the following email body and return "
                "its JSON output unchanged. Do NOT call any other function.\n"
                f"Email:\n{raw_email}"
            ),
            expected_output="Return ONLY the raw JSON string from email_ingest_tool. No commentary.",
            agent=ingest_agent,
        )
        t2 = Task(
            description=(
                "Using the clean_text from the previous task, call phishing_classifier_tool with ONLY "
                "clean_text set to the exact clean_text value from the prior output. Do not invent or change the text. "
                "The tool auto-computes features. If the score is ambiguous (0.35-0.75), use deep_analysis_tool "
                "to refine it before reporting."
            ),
            expected_output="JSON: risk_score, confident, threshold_used, sender_hint.",
            context=[t1], agent=analyst_agent,
        )
        t3 = Task(
            description=f"Check sender '{sender}' threat history, determine security action, log verdict to memory.",
            expected_output="JSON: action, explanation, risk_score, past_incidents, repeat_offender, memory_summary.",
            context=[t2], agent=soc_agent,
        )
        return Crew(
            agents=[ingest_agent, analyst_agent, soc_agent],
            tasks=[t1, t2, t3],
            process=Process.sequential,
            verbose=True,
        )

# ── SVG ICON LIBRARY ─────────────────────────────────────────────
def _svg(path_d, color="#374151", size=20, extra_attrs=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" {extra_attrs}>{path_d}</svg>'
    )

def icon_shield(color="#374151", size=20):
    return _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>', color, size)

def icon_check(color="#16a34a", size=20):
    return _svg('<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>', color, size)

def icon_xmark(color="#dc2626", size=20):
    return _svg('<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>', color, size)

def icon_warning(color="#ea580c", size=20):
    return _svg('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="0.5" fill="{color}"/>', color, size)

def icon_info(color="#ca8a04", size=20):
    return _svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="0.5" fill="{c}"/>', color, size)

def icon_mail(color="#6366f1", size=20):
    return _svg('<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>', color, size)

def icon_cpu(color="#6366f1", size=20):
    return _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>', color, size)

def icon_db(color="#6366f1", size=20):
    return _svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>', color, size)

def icon_terminal(color="#6366f1", size=20):
    return _svg('<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>', color, size)

def icon_spinner(color="#6366f1", size=20):
    return _svg(
        '<line x1="12" y1="2" x2="12" y2="6"/>'
        '<line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/>'
        '<line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/>'
        '<line x1="18" y1="12" x2="22" y2="12"/>'
        '<line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/>'
        '<line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>',
        color, size
    )

# ── GLOBAL CSS ────────────────────────────────────────────────────
CSS = """
:root {
  --red:    #dc2626; --red-bg:    #fef2f2; --red-border:    #fca5a5;
  --orange: #ea580c; --orange-bg: #fff7ed; --orange-border: #fdba74;
  --yellow: #ca8a04; --yellow-bg: #fefce8; --yellow-border: #fde047;
  --green:  #16a34a; --green-bg:  #f0fdf4; --green-border:  #86efac;
  --indigo: #4f46e5; --slate:     #475569;
  --card-bg: #f8fafc; --card-border: #e2e8f0; --radius: 10px;
}
.hunter-card {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: var(--radius); padding: 18px 20px; margin-bottom: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.agent-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--card-border);
}
.agent-header .agent-name {
  font-weight: 700; font-size: 14px; color: #1e293b; flex: 1;
}
.badge {
  font-size: 11px; font-weight: 700; padding: 2px 9px;
  border-radius: 999px; letter-spacing: .4px; text-transform: uppercase;
}
.badge-done   { background: #dcfce7; color: #15803d; }
.badge-wait   { background: #f1f5f9; color: #94a3b8; }
.badge-run    { background: #ede9fe; color: #7c3aed; }
.feat-table   { width: 100%; border-collapse: collapse; font-size: 13px; }
.feat-table td, .feat-table th {
  padding: 6px 10px; text-align: left; border-bottom: 1px solid #f1f5f9;
}
.feat-table th { font-weight: 600; color: #64748b; font-size: 11px; text-transform: uppercase; }
.feat-table td:last-child { font-weight: 600; color: #1e293b; }
.score-row    { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }
.score-label  { width: 160px; color: #64748b; flex-shrink: 0; }
.score-bar    { flex: 1; background: #e2e8f0; border-radius: 999px; height: 8px; overflow: hidden; }
.score-fill   { height: 100%; border-radius: 999px; transition: width .4s; }
.score-val    { width: 55px; text-align: right; font-weight: 700; color: #1e293b; font-size: 13px; }
.verdict-big  { border-radius: 12px; padding: 22px 24px; margin-bottom: 16px; }
.verdict-big .v-header { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.verdict-big .v-label  { font-size: 22px; font-weight: 800; }
.verdict-big .v-action { font-size: 13px; color: #374151; border-left: 3px solid currentColor; padding-left: 10px; margin: 10px 0; }
.verdict-big .v-score  { font-size: 13px; color: #64748b; margin-top: 6px; }
.verdict-red    { background: var(--red-bg);    border: 1px solid var(--red-border); }
.verdict-orange { background: var(--orange-bg); border: 1px solid var(--orange-border); }
.verdict-yellow { background: var(--yellow-bg); border: 1px solid var(--yellow-border); }
.verdict-green  { background: var(--green-bg);  border: 1px solid var(--green-border); }
.repeat-banner {
  background: #7f1d1d; color: #fca5a5; border-radius: 8px;
  padding: 10px 14px; font-size: 13px; font-weight: 600; margin-top: 10px;
  display: flex; align-items: center; gap: 8px;
}
.risk-gauge-wrap { margin: 14px 0 6px; }
.risk-gauge-bar  { background: #e2e8f0; border-radius: 999px; height: 14px; overflow: hidden; position: relative; }
.risk-gauge-fill { height: 100%; border-radius: 999px; }
.risk-gauge-ticks { display: flex; justify-content: space-between; margin-top: 4px; }
.risk-gauge-ticks span { font-size: 10px; color: #94a3b8; }
.hist-table      { width: 100%; border-collapse: collapse; font-size: 13px; }
.hist-table th   { background: #f8fafc; padding: 6px 10px; text-align: left; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; border-bottom: 2px solid #e2e8f0; }
.hist-table td   { padding: 7px 10px; border-bottom: 1px solid #f1f5f9; color: #334155; }
.hist-table tr:hover td { background: #f8fafc; }
.verdict-pill    { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
.pill-QUARANTINE         { background: #fee2e2; color: #991b1b; }
.pill-FLAG\ WITH\ WARNING { background: #ffedd5; color: #9a3412; }
.pill-ALLOW\ AND\ LOG    { background: #fef9c3; color: #854d0e; }
.pill-ALLOW              { background: #dcfce7; color: #166534; }
.trace-block { background: #0f172a; color: #94a3b8; border-radius: 10px; padding: 18px; font-family: 'Fira Code', 'Courier New', monospace; font-size: 12px; line-height: 1.7; overflow-x: auto; white-space: pre-wrap; }
.trace-agent  { color: #a78bfa; font-weight: 700; }
.trace-action { color: #34d399; }
.trace-obs    { color: #fbbf24; }
.trace-final  { color: #60a5fa; }
.status-bar   { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 8px; }
.status-ok    { background: #f0fdf4; border: 1px solid #86efac; color: #15803d; }
.status-warn  { background: #fffbeb; border: 1px solid #fde047; color: #854d0e; }
.status-err   { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
.loading-pulse { color: #94a3b8; font-size: 13px; padding: 30px; text-align: center; }
footer { display: none !important; }
"""

# ── HTML BUILDERS ─────────────────────────────────────────────────
def _progress_html(pct: int, color: str) -> str:
    return (
        f'<div class="risk-gauge-wrap">'
        f'<div class="risk-gauge-bar">'
        f'<div class="risk-gauge-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<div class="risk-gauge-ticks">'
        f'<span>0.0&nbsp;ALLOW</span>'
        f'<span>{OPTIMAL_THRESHOLD:.3f}</span>'
        f'<span>0.70</span>'
        f'<span>0.90</span>'
        f'<span>1.0&nbsp;QUARANTINE</span>'
        f'</div></div>'
    )

TIER_COLORS = {
    "red":    "#dc2626", "orange": "#ea580c",
    "yellow": "#ca8a04", "green":  "#16a34a",
}

def loading_card(msg: str = "Waiting for upstream agent...") -> str:
    return (
        f'<div class="hunter-card loading-pulse">'
        f'{icon_spinner("#6366f1", 18)}'
        f'<span style="margin-left:8px;">{escape(msg)}</span></div>'
    )

def pending_card(msg: str = "Pending — run the pipeline first.") -> str:
    return f'<div class="hunter-card" style="color:#94a3b8;padding:30px;text-align:center;">{escape(msg)}</div>'

def system_status_html() -> str:
    rows = []
    # Models
    if MODELS_LOADED:
        rows.append(f'<div class="status-bar status-ok">{icon_check("#16a34a",16)}&nbsp;ML models loaded (BiLSTM + LogReg + Tokenizer + Scaler)</div>')
    else:
        rows.append(f'<div class="status-bar status-err">{icon_xmark("#dc2626",16)}&nbsp;ML models NOT found — run the full notebook first to train and save models. <code>Error: {escape(MODEL_ERROR[:80])}</code></div>')
    # Groq API key
    if GROQ_KEY_OK:
        rows.append(f'<div class="status-bar status-ok">{icon_check("#16a34a",16)}&nbsp;GROQ_API_KEY detected — full LLM agent reasoning will run</div>')
    else:
        rows.append(f'<div class="status-bar status-warn">{icon_warning("#ca8a04",16)}&nbsp;GROQ_API_KEY not set — pipeline runs in direct mode (no LLM trace). Add key to .env or Colab Secrets.</div>')
    # CrewAI
    if CREWAI_OK:
        rows.append(f'<div class="status-bar status-ok">{icon_check("#16a34a",16)}&nbsp;CrewAI framework ready</div>')
    else:
        rows.append(f'<div class="status-bar status-warn">{icon_warning("#ca8a04",16)}&nbsp;CrewAI unavailable: {escape(CREWAI_ERROR[:60])}</div>')
    return "".join(rows)

def feat_card(feats: dict, clf: dict | None, deep: dict | None, deep_trig: bool) -> str:
    score = None
    if deep_trig and deep:
        score = deep["adjusted_risk_score"]
    elif clf:
        score = clf["ensemble_score"]

    badge = '<span class="badge badge-done">Complete</span>'
    out   = []

    # ── Agent 1 card
    out.append(
        f'<div class="hunter-card">'
        f'<div class="agent-header">{icon_mail("#6366f1",20)}'
        f'<span class="agent-name">Agent 1 — Email Ingest Specialist</span>{badge}</div>'
        f'<table class="feat-table"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>'
    )
    labels = {"char_count":"Characters","word_count":"Words","url_count":"URLs found",
              "exclaim_count":"Exclamation marks","urgent_keyword_count":"Urgent keywords"}
    for k, lbl in labels.items():
        out.append(f'<tr><td>{lbl}</td><td>{feats.get(k,"—")}</td></tr>')
    out.append('</tbody></table></div>')

    # ── Agent 2 card
    a2_badge = badge if clf else '<span class="badge badge-wait">Pending</span>'
    out.append(
        f'<div class="hunter-card">'
        f'<div class="agent-header">{icon_cpu("#6366f1",20)}'
        f'<span class="agent-name">Agent 2 — Phishing Risk Analyst</span>{a2_badge}</div>'
    )
    if clf:
        clr_b = _bar_color(clf["bilstm_score"])
        clr_l = _bar_color(clf["logreg_score"])
        clr_e = _bar_color(clf["ensemble_score"])
        out.append(f'<div class="score-row"><span class="score-label">BiLSTM (65% weight)</span><div class="score-bar"><div class="score-fill" style="width:{int(clf["bilstm_score"]*100)}%;background:{clr_b};"></div></div><span class="score-val">{clf["bilstm_score"]:.4f}</span></div>')
        out.append(f'<div class="score-row"><span class="score-label">Logistic Regression (35%)</span><div class="score-bar"><div class="score-fill" style="width:{int(clf["logreg_score"]*100)}%;background:{clr_l};"></div></div><span class="score-val">{clf["logreg_score"]:.4f}</span></div>')
        out.append(f'<div class="score-row"><strong style="width:160px;font-size:13px;">Ensemble</strong><div class="score-bar"><div class="score-fill" style="width:{int(clf["ensemble_score"]*100)}%;background:{clr_e};"></div></div><strong class="score-val">{clf["ensemble_score"]:.4f}</strong></div>')
        if deep_trig and deep:
            out.append(
                f'<div style="margin-top:12px;padding:10px 12px;background:#ede9fe;border-radius:8px;font-size:12px;">'
                f'<strong style="color:#7c3aed;">Deep Analysis Escalation triggered</strong> (score in ambiguous band 0.35–0.75)<br>'
                f'Base: <code>{deep["original_score"]}</code> &nbsp;+&nbsp; Boost: <code>+{deep["boost"]}</code> &nbsp;=&nbsp; '
                f'<strong>Final: <code>{deep["adjusted_risk_score"]}</code></strong><br>'
                f'Patterns: {", ".join(f"<code>{p}</code>" for p in deep["patterns_found"]) or "none"}&nbsp;&nbsp;'
                f'Flags: {", ".join(f"<code>{f}</code>" for f in deep["flags_found"]) or "none"}'
                f'</div>'
            )
    out.append('</div>')
    return "".join(out)

def _bar_color(score: float) -> str:
    if score >= TIER_QUARANTINE: return TIER_COLORS["red"]
    if score >= TIER_FLAG:       return TIER_COLORS["orange"]
    if score >= OPTIMAL_THRESHOLD: return TIER_COLORS["yellow"]
    return TIER_COLORS["green"]

def verdict_card(policy: dict, is_repeat: bool, final_score: float) -> str:
    tier   = policy["tier"]
    color  = TIER_COLORS[tier]
    icons  = {"red": icon_xmark("#dc2626",28), "orange": icon_warning("#ea580c",28),
              "yellow": icon_info("#ca8a04",28), "green": icon_check("#16a34a",28)}
    pct    = int(min(final_score, 1.0) * 100)
    repeat = (
        f'<div class="repeat-banner">{icon_warning("#fca5a5",16)}'
        f'&nbsp;REPEAT OFFENDER — this sender has prior FLAG/QUARANTINE incidents in threat memory. Auto-escalated.</div>'
    ) if is_repeat else ""

    return (
        f'<div class="verdict-big verdict-{tier}">'
        f'<div class="v-header">{icons.get(tier,"")}<span class="v-label" style="color:{color};">{escape(policy["label"])}</span></div>'
        f'{_progress_html(pct, color)}'
        f'<p class="v-action" style="border-color:{color};">{escape(policy["action"])}</p>'
        f'<p class="v-score">Risk Score: <strong>{final_score:.4f}</strong> &nbsp;|&nbsp; '
        f'Threshold: {OPTIMAL_THRESHOLD:.4f} &nbsp;|&nbsp; '
        f'Tiers: QUARANTINE ≥ 0.90 · FLAG ≥ 0.70 · LOG ≥ {OPTIMAL_THRESHOLD:.3f}</p>'
        f'{repeat}'
        f'</div>'
    )

def history_card(history: list, sender: str = "") -> str:
    if not history:
        return (
            f'<div class="hunter-card">'
            f'<div class="agent-header">{icon_db("#6366f1",20)}'
            f'<span class="agent-name">Agent 3 — SOC Orchestrator · Threat Memory</span>'
            f'<span class="badge badge-done">Complete</span></div>'
            f'<div style="color:#16a34a;font-size:13px;padding:8px 0;">'
            f'{icon_check("#16a34a",16)}&nbsp; No prior incidents found for <code>{escape(sender)}</code>. '
            f'First-time sender.</div></div>'
        )
    rows = "".join(
        f'<tr><td><code>{h["risk_score"]:.4f}</code></td>'
        f'<td><span class="verdict-pill pill-{escape(h["action"])}">{escape(h["action"])}</span></td>'
        f'<td>{h["timestamp"][:19]}</td></tr>'
        for h in history
    )
    return (
        f'<div class="hunter-card">'
        f'<div class="agent-header">{icon_db("#dc2626",20)}'
        f'<span class="agent-name">Agent 3 — Threat Memory: REPEAT OFFENDER DETECTED</span>'
        f'<span class="badge badge-done">Complete</span></div>'
        f'<table class="hist-table"><thead><tr><th>Risk Score</th><th>Prior Verdict</th><th>Timestamp (UTC)</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )

def trace_card(raw: str) -> str:
    if not raw.strip():
        return f'<div class="hunter-card" style="color:#94a3b8;font-size:13px;padding:20px;">Agent trace not captured.</div>'
    # Simple colorize
    colored = (raw
        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        .replace("Agent Started",      '<span class="trace-agent">Agent Started</span>')
        .replace("Agent Final Answer", '<span class="trace-final">Agent Final Answer</span>')
        .replace("Using tool:",        '<span class="trace-action">Using tool:</span>')
        .replace("Tool Output:",       '<span class="trace-obs">Tool Output:</span>')
    )
    return (
        f'<div class="hunter-card">'
        f'<div class="agent-header">{icon_terminal("#6366f1",20)}'
        f'<span class="agent-name">CrewAI Agent Reasoning Trace (ReAct Loop)</span>'
        f'<span class="badge badge-done">Complete</span></div>'
        f'<div class="trace-block">{colored[:6000]}</div></div>'
    )

def trace_unavailable_card(reason: str) -> str:
    return (
        f'<div class="hunter-card">'
        f'<div class="agent-header">{icon_terminal("#94a3b8",20)}'
        f'<span class="agent-name">Agent Reasoning Trace</span>'
        f'<span class="badge badge-wait">Unavailable</span></div>'
        f'<div class="status-bar status-warn">{icon_warning("#ca8a04",16)}&nbsp;{escape(reason)}</div>'
        f'<p style="font-size:13px;color:#64748b;">To see the full LLM ReAct loop, set GROQ_API_KEY in your .env or Colab Secrets.</p>'
        f'</div>'
    )

def all_history_card() -> str:
    rows_data = get_all_history()
    if not rows_data:
        return '<div class="hunter-card" style="color:#94a3b8;font-size:13px;padding:20px;">No threat history yet. Run the pipeline on some emails first.</div>'
    rows = "".join(
        f'<tr><td><code>{escape(r["sender"][:40])}</code></td>'
        f'<td><code>{r["risk_score"]:.4f}</code></td>'
        f'<td><span class="verdict-pill pill-{escape(r["action"])}">{escape(r["action"])}</span></td>'
        f'<td>{r["timestamp"][:19]}</td></tr>'
        for r in rows_data
    )
    return (
        f'<div class="hunter-card">'
        f'<table class="hist-table"><thead><tr>'
        f'<th>Sender</th><th>Risk Score</th><th>Verdict</th><th>Timestamp (UTC)</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )

# ── DEMO EMAILS (exact from notebook) ────────────────────────────
DEMO_EMAILS = {
    "Clear Phishing": {
        "sender": "unknown",
        "body": (
            "URGENT: Your account has been suspended! Verify your identity "
            "immediately to avoid permanent deactivation. Click here to confirm "
            "your credentials https://secure-login-verify.com/auth\n"
            "If you do not act within 24 hours, your account will be terminated. "
            "Do not reply to this message. - Security Team"
        )
    },
    "Clean Legitimate": {
        "sender": "alice@company.com",
        "body": (
            "Hi team, just a reminder that our quarterly planning meeting is "
            "scheduled for Thursday at 2:00 PM in Conference Room B. Please "
            "bring your project updates. Looking forward to seeing everyone there. "
            "Let me know if you have any questions. Best, Alice"
        )
    },
    "Borderline Ambiguous": {
        "sender": "newsletter@updates-service.com",
        "body": (
            "Hello, we noticed some activity on your account that requires your "
            "attention. Please log in to review recent changes and confirm your "
            "details are up to date. Your security is important to us. "
            "Visit your account settings to review. Thank you, Support Team"
        )
    },
    "Repeat Sender": {
        "sender": "unknown",
        "body": (
            "URGENT: Your account has been suspended! Verify your identity "
            "immediately to avoid permanent deactivation. Click here to confirm "
            "your credentials https://secure-login-verify.com/auth\n"
            "If you do not act within 24 hours, your account will be terminated. "
            "Do not reply to this message. - Security Team"
        )
    },
}

# ── MAIN PIPELINE (GENERATOR for progressive UI updates) ─────────
def run_pipeline(sender_in: str, email_body: str):
    """Generator: yields (feat_html, verdict_html, history_html, trace_html) progressively."""

    if not email_body.strip():
        err = f'<div class="status-bar status-err">{icon_xmark("#dc2626",16)}&nbsp;Please enter an email body.</div>'
        yield err, err, err, err
        return

    sender = sender_in.strip() or "unknown"
    _STATE.clear()
    _STATE["deep_triggered"] = False

    # Step 1: ingest
    yield loading_card("Agent 1 — Email Ingest Specialist: extracting features..."), \
          loading_card("Waiting for Agent 1..."), \
          loading_card("Waiting for Agent 1..."), \
          loading_card("Waiting for Agent 1...")

    feats = extract_features(email_body)
    _STATE["features"]      = feats
    _STATE["cleaned_text"]  = feats["cleaned_text"]

    yield feat_card(feats, None, None, False), \
          loading_card("Agent 2 — Phishing Risk Analyst: running ensemble classifier..."), \
          loading_card("Waiting for Agent 2..."), \
          loading_card("Waiting for Agent 2...")

    # Step 2: classify
    if MODELS_LOADED:
        clf = ensemble_predict(feats["cleaned_text"])
        _STATE["classifier_result"] = clf
        score = clf["ensemble_score"]

        if CONFIDENCE_LOW <= score <= CONFIDENCE_HIGH:
            yield feat_card(feats, clf, None, False), \
                  loading_card("Agent 2 — Escalating to deep_analysis_tool (ambiguous score)..."), \
                  loading_card("Waiting for Agent 2..."), \
                  loading_card("Waiting for Agent 2...")
            deep = deep_analysis(feats["cleaned_text"], score)
            _STATE["deep_result"]    = deep
            _STATE["deep_triggered"] = True
            final_score = deep["adjusted_risk_score"]
        else:
            deep        = None
            final_score = score
    else:
        # Heuristic fallback
        clf  = None
        deep = deep_analysis(feats["cleaned_text"],
                             min(feats["urgent_keyword_count"] * 0.07 +
                                 feats["url_count"] * 0.06 +
                                 feats["exclaim_count"] * 0.03, 0.70))
        _STATE["deep_triggered"] = True
        final_score = deep["adjusted_risk_score"]

    yield feat_card(feats, clf, _STATE.get("deep_result"), _STATE["deep_triggered"]), \
          loading_card("Agent 3 — SOC Orchestrator: querying threat memory..."), \
          loading_card("Querying SQLite threat memory..."), \
          loading_card("Waiting for Agent 3...")

    # Step 3: threat memory + defense policy
    history   = query_threat_memory(sender)
    is_repeat = any(h["action"] in ("FLAG WITH WARNING","QUARANTINE") for h in history)
    _STATE["threat_history"] = history
    _STATE["is_repeat"]      = is_repeat

    yield feat_card(feats, clf, _STATE.get("deep_result"), _STATE["deep_triggered"]), \
          loading_card("Agent 3 — applying 4-tier defense policy..."), \
          history_card(history, sender), \
          loading_card("Waiting for Agent 3...")

    policy = apply_defense_policy(final_score, is_repeat)
    write_threat_memory(sender, final_score, policy["verdict"])
    _STATE["final_verdict"] = policy

    f_html = feat_card(feats, clf, _STATE.get("deep_result"), _STATE["deep_triggered"])
    v_html = verdict_card(policy, is_repeat, final_score)
    h_html = history_card(history, sender)

    yield f_html, v_html, h_html, loading_card("Capturing LLM reasoning trace...")

    # Step 4: optional LLM trace via CrewAI
    if CREWAI_OK and GROQ_KEY_OK and MODELS_LOADED:
        _STATE.clear(); _STATE["deep_triggered"] = False
        old_out, sys.stdout = sys.stdout, io.StringIO()
        try:
            crew = build_crew(email_body, sender)
            # Retry up to 3x on hallucinated-tool errors (mirrors notebook)
            for attempt in range(3):
                try:
                    crew.kickoff()
                    break
                except Exception as ex:
                    err_s = str(ex)
                    if "tool_use_failed" in err_s or "was not in request.tools" in err_s:
                        if attempt < 2:
                            time.sleep(10)
                            continue
                    raise
            raw_trace = sys.stdout.getvalue()
            t_html = trace_card(raw_trace)
        except Exception as ex:
            t_html = trace_unavailable_card(f"CrewAI error: {str(ex)[:120]}")
        finally:
            sys.stdout = old_out
    elif not MODELS_LOADED:
        t_html = trace_unavailable_card("ML models not loaded — train the notebook first to enable LLM trace.")
    elif not GROQ_KEY_OK:
        t_html = trace_unavailable_card("GROQ_API_KEY not set. Add it to .env or Colab Secrets to enable full agent reasoning.")
    else:
        t_html = trace_unavailable_card("CrewAI not available.")

    yield f_html, v_html, h_html, t_html


# ── GRADIO UI ─────────────────────────────────────────────────────
def _demo_load(key):
    d = DEMO_EMAILS[key]
    return d["sender"], d["body"]

with gr.Blocks(
    title="The Hunter — Phishing Defense Agent",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as app:

    # ── Header
    gr.HTML(f"""
    <div style="padding:16px 0 6px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        {icon_shield("#4f46e5", 32)}
        <div>
          <h1 style="margin:0;font-size:22px;font-weight:800;color:#1e293b;">The Hunter</h1>
          <p style="margin:0;font-size:13px;color:#64748b;">
            Automated Email Phishing Defense System &nbsp;&middot;&nbsp;
            ITAI 2376 &nbsp;&middot;&nbsp; Houston City College &nbsp;&middot;&nbsp; Team 1
          </p>
        </div>
      </div>
      <p style="font-size:12px;color:#94a3b8;margin:4px 0 0;">
        CrewAI &ge;0.102.0 &nbsp;&middot;&nbsp; BiLSTM + LogReg Ensemble &nbsp;&middot;&nbsp; 
        Groq LLaMA&nbsp;3.1&nbsp;8B &nbsp;&middot;&nbsp; SQLite Threat Memory
      </p>
    </div>
    """)

    # System status
    with gr.Accordion("System Status", open=not MODELS_LOADED):
        status_box = gr.HTML(value=system_status_html())

    gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>")

    with gr.Row(equal_height=False):
        # ── LEFT: Input panel
        with gr.Column(scale=2, min_width=320):
            gr.HTML(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                    f'{icon_mail("#4f46e5",18)}'
                    f'<strong style="font-size:15px;color:#1e293b;">Email Input</strong></div>')

            sender_box = gr.Textbox(
                label="Sender Address",
                placeholder="attacker@phishing-domain.ru",
                lines=1,
            )
            email_box = gr.Textbox(
                label="Email Body",
                placeholder="Paste the raw email body here...",
                lines=13,
            )

            gr.HTML('<p style="font-size:12px;color:#64748b;margin:6px 0 8px;"><strong>Quick-load demo emails (from notebook):</strong></p>')
            with gr.Row():
                btn_phish  = gr.Button("Clear Phishing",     variant="secondary", size="sm")
                btn_clean  = gr.Button("Clean Legitimate",   variant="secondary", size="sm")
            with gr.Row():
                btn_ambig  = gr.Button("Borderline Ambiguous", variant="secondary", size="sm")
                btn_repeat = gr.Button("Repeat Sender",        variant="secondary", size="sm")

            run_btn  = gr.Button("Run The Hunter", variant="primary",   size="lg")
            clear_btn= gr.Button("Clear",          variant="secondary", size="sm")

            with gr.Accordion("Setup & Requirements", open=False):
                gr.Markdown("""
**Step 1 — Train models:** Run the full `working_the_hunter_notebook.ipynb` in Colab. Models save automatically to `models/`.

**Step 2 — API key:** Add `GROQ_API_KEY` to Colab Secrets (key icon in sidebar) or a local `.env` file.

**Step 3 — Launch UI (Colab):**
```python
!pip install -q gradio
%run hunter_gradio_app.py
```

**Step 4 — Demo order for best impression:**
1. Clear Phishing → shows high-confidence QUARANTINE
2. Clean Legitimate → shows ALLOW  
3. Borderline Ambiguous → shows autonomous deep_analysis_tool escalation
4. Repeat Sender → shows SQLite repeat-offender detection

**Required packages:**
```
gradio crewai tensorflow scikit-learn imbalanced-learn litellm python-dotenv
```
                """)

        # ── RIGHT: Output tabs
        with gr.Column(scale=3, min_width=480):
            with gr.Tabs():
                with gr.Tab("Verdict"):
                    verdict_out = gr.HTML(value=pending_card("Run the pipeline to see the security verdict."))
                with gr.Tab("Pipeline Analysis"):
                    feat_out    = gr.HTML(value=pending_card("Run the pipeline to see agent outputs."))
                with gr.Tab("Threat Memory"):
                    hist_out    = gr.HTML(value=pending_card("Run the pipeline to query the SQLite threat memory."))
                with gr.Tab("Agent Reasoning Trace"):
                    trace_out   = gr.HTML(value=pending_card("Run the pipeline to see the CrewAI ReAct reasoning loop."))
                with gr.Tab("All Threat History"):
                    all_hist_out = gr.HTML(value=all_history_card())
                    refresh_btn  = gr.Button("Refresh History", variant="secondary", size="sm")

    gr.HTML("""
    <hr style="border:none;border-top:1px solid #e2e8f0;margin:16px 0 10px;">
    <div style="font-size:12px;color:#94a3b8;padding-bottom:8px;">
      <strong>Pipeline:</strong> Agent 1: Email Ingest &nbsp;&rarr;&nbsp;
      Agent 2: Phishing Risk Analyst &nbsp;&rarr;&nbsp; Agent 3: SOC Orchestrator
      &nbsp;&nbsp;&middot;&nbsp;&nbsp;
      <strong>Model:</strong> BiLSTM (65%) + Logistic Regression (35%)
      &nbsp;&nbsp;&middot;&nbsp;&nbsp;
      <strong>Dataset:</strong> Alam 28,747-email dataset &nbsp;&middot;&nbsp;
      Threshold calibrated via Precision-Recall AUC (t={:.4f})
    </div>
    """.format(OPTIMAL_THRESHOLD))

    # ── Wire demo buttons
    for btn, key in [
        (btn_phish,  "Clear Phishing"),
        (btn_clean,  "Clean Legitimate"),
        (btn_ambig,  "Borderline Ambiguous"),
        (btn_repeat, "Repeat Sender"),
    ]:
        btn.click(fn=lambda k=key: _demo_load(k), outputs=[sender_box, email_box])

    # ── Wire run button (generator for progressive updates)
    run_btn.click(
        fn=run_pipeline,
        inputs=[sender_box, email_box],
        outputs=[feat_out, verdict_out, hist_out, trace_out],
    )

    # ── Wire clear
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[sender_box, email_box],
    )

    # ── Wire refresh history
    refresh_btn.click(fn=all_history_card, outputs=[all_hist_out])

# ── LAUNCH ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(share=True, debug=False, server_name="0.0.0.0")
