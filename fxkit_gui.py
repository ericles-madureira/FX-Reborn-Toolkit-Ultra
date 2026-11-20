"""FX Reborn Toolkit Ultra - Graphical User Interface (GUI) & Local API Server.

Designed to work 100% out-of-the-box with the bundled embeddable Python 3.13
(zero external dependencies: uses only `socket`, `threading`, `json`, `ctypes`,
`subprocess`, and `pathlib`, which are all present in `python313.zip`).

Automatically opens as a native desktop app window via Microsoft Edge / Chrome
`--app` mode (or default browser fallback) and provides native Windows `comdlg32`
File/Folder selection dialogs, CPUID hardware detection, binary scanning,
startup CPU-check bypass detection, and one-click Ultra-Optimized binary building.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
if str(ENGINE / "pylibs") not in sys.path:
    sys.path[:0] = [str(ENGINE / "pylibs"), str(ENGINE)]

import fxkit
import installer

# Global job state for asynchronous progress streaming to the GUI
JOB_LOCK = threading.Lock()
JOB_STATE = {
    "running": False,
    "task": None,
    "status": "Pronto",
    "logs": [],
    "last_scan": None,
    "last_gates": [],
    "last_build": None,
    "error": None,
}


def tr(lang: str, pt: str, en: str) -> str:
    """Pick the Portuguese or English text for GUI log messages / Escolhe o texto PT ou EN."""
    return en if lang == "en" else pt


def append_log(msg: str):
    with JOB_LOCK:
        ts = time.strftime("%H:%M:%S")
        JOB_STATE["logs"].append(f"[{ts}] {msg}")
        if len(JOB_STATE["logs"]) > 500:
            JOB_STATE["logs"] = JOB_STATE["logs"][-500:]
        JOB_STATE["status"] = msg


class LogRedirector:
    """Tee stdout/stderr into JOB_STATE['logs'] while running background tasks."""
    def __init__(self, orig):
        self.orig = orig
        self.buf = ""

    def write(self, text: str):
        if self.orig:
            try:
                self.orig.write(text)
            except Exception:
                pass
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.strip()
            if line:
                append_log(line)

    def flush(self):
        if self.orig:
            try:
                self.orig.flush()
            except Exception:
                pass


def detect_host_cpu() -> dict:
    """Detect the current PC's CPU features via CPUID on Windows (or /proc/cpuinfo fallback)."""
    try:
        if os.name == "nt":
            cpuid = installer.cpuid_reader()
            cache = {}
            have = {}
            for name, (leaf, reg, bit) in installer.CPUID_BITS.items():
                if leaf not in cache:
                    cache[leaf] = cpuid(leaf)
                have[name] = bool(cache[leaf][reg] >> bit & 1)
            return {"platform": "Windows x64", "features": have}
    except Exception:
        pass
    # Fallback for Linux / inspection environment
    have = {}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
        for k in installer.CPUID_BITS:
            have[k] = k.lower() in cpuinfo
    except Exception:
        pass
    return {"platform": sys.platform, "features": have}


def native_open_file_dialog(title: str = "Selecionar Executável ou DLL (.exe / .dll)",
                            filter_spec: str = "Executáveis e DLLs (*.exe;*.dll)\0*.exe;*.dll\0Todos os Arquivos (*.*)\0*.*\0\0") -> str:
    """Open native Windows GetOpenFileNameW dialog via ctypes.windll.comdlg32 (no tkinter needed)."""
    if os.name != "nt":
        return ""
    try:
        from ctypes import wintypes

        class OPENFILENAMEW(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", ctypes.c_void_p),
                ("lpTemplateName", wintypes.LPCWSTR),
                ("pvReserved", ctypes.c_void_p),
                ("dwReserved", wintypes.DWORD),
                ("FlagsEx", wintypes.DWORD),
            ]

        buf = ctypes.create_unicode_buffer(4096)
        ofn = OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        ofn.lpstrFilter = filter_spec
        ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
        ofn.nMaxFile = 4096
        ofn.lpstrTitle = title
        # OFN_EXPLORER (0x80000) | OFN_FILEMUSTEXIST (0x1000) | OFN_PATHMUSTEXIST (0x800) | OFN_NOCHANGEDIR (0x8)
        ofn.Flags = 0x00081808
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return buf.value
    except Exception:
        pass
    return ""


def native_select_folder_dialog(title: str = "Selecionar Pasta de Saída") -> str:
    """Open Windows folder picker via PowerShell System.Windows.Forms.FolderBrowserDialog."""
    if os.name != "nt":
        return ""
    try:
        ps_cmd = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.Description = '{title}'; "
            "$f.ShowNewFolderButton = $true; "
            "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.SelectedPath }"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Sta", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=120
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def run_async_job(task_name: str, fn, lang: str = "pt"):
    with JOB_LOCK:
        if JOB_STATE["running"]:
            return False
        JOB_STATE["running"] = True
        JOB_STATE["task"] = task_name
        JOB_STATE["error"] = None
        JOB_STATE["logs"] = []
        JOB_STATE["status"] = tr(lang, f"Iniciando {task_name}...", f"Starting {task_name}...")

    def worker():
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = LogRedirector(old_out)
        sys.stderr = LogRedirector(old_err)
        try:
            fn()
            append_log(tr(lang, f"Tarefa '{task_name}' concluída com sucesso!", f"Task '{task_name}' finished successfully!"))
        except Exception as exc:
            try:
                tb = traceback.format_exc()
            except Exception:
                # The bundled Python has no `unicodedata`, which traceback needs to
                # format lines with accented characters; fall back to a short form.
                tb = f"{type(exc).__name__}: {exc}"
            append_log(tr(lang, f"ERRO: {exc}", f"ERROR: {exc}"))
            with JOB_LOCK:
                JOB_STATE["error"] = f"{exc}\n{tb}"
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            with JOB_LOCK:
                JOB_STATE["running"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return True


HTML_UI = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FX Reborn Toolkit Ultra v3.0 — Emulador Estático & Otimizador AVX2 / FMA / F16C / BMI</title>
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #131b2e;
    --bg-card-alt: #18233c;
    --border: #263554;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --mono: 'Consolas', 'JetBrains Mono', 'Courier New', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-main);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.5;
    padding: 20px 28px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }
  .brand h1 {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .badge-ultra {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .brand p {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 3px;
  }
  .lang-switch { display: flex; gap: 4px; margin: 0 14px 0 auto; }
  .lang-btn {
    font-size: 11px;
    padding: 4px 9px;
    border-radius: 6px;
    background: var(--bg-card);
    color: var(--text-muted);
    border: 1px solid var(--border);
  }
  .lang-btn.active { color: #090d16; background: var(--accent); border-color: var(--accent); }
  .cpu-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    max-width: 520px;
    justify-content: flex-end;
  }
  .cpu-pill {
    font-size: 11px;
    font-family: var(--mono);
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
  }
  .cpu-pill.has { border-color: rgba(34,197,94,0.45); color: #86efac; background: rgba(34,197,94,0.1); }
  .cpu-pill.miss { border-color: rgba(239,68,68,0.45); color: #fca5a5; background: rgba(239,68,68,0.1); }

  .grid-main {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 20px;
  }
  @media (max-width: 1100px) {
    .grid-main { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  }
  .card h2 {
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .field-group { margin-bottom: 14px; }
  .field-group label {
    display: block;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 5px;
  }
  .input-row {
    display: flex;
    gap: 8px;
  }
  input[type="text"] {
    flex: 1;
    background: #090d16;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 9px 12px;
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 13px;
  }
  input[type="text"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  button {
    cursor: pointer;
    border: none;
    border-radius: 8px;
    padding: 9px 15px;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.15s ease;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary {
    background: var(--bg-card-alt);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover:not(:disabled) { background: #223052; border-color: var(--accent); }
  .btn-primary {
    background: var(--accent);
    color: #090d16;
  }
  .btn-primary:hover:not(:disabled) { background: #7dd3fc; }
  .btn-success {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #fff;
    font-size: 14px;
    padding: 11px 20px;
  }
  .btn-success:hover:not(:disabled) { filter: brightness(1.1); }

  .features-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }
  .feat-box {
    background: #0d1424;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    cursor: pointer;
    user-select: none;
  }
  .feat-box:hover { border-color: var(--accent); }
  .feat-box input { margin-top: 3px; accent-color: var(--accent); }
  .feat-box .feat-title { font-size: 13px; font-weight: 700; color: #fff; }
  .feat-box .feat-desc { font-size: 11px; color: var(--text-muted); line-height: 1.25; }
  .feat-box.primary-feat { border-color: rgba(56, 189, 248, 0.5); background: rgba(56, 189, 248, 0.07); }

  .opts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .action-bar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
  }
  .preset-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .preset-btn {
    font-size: 11.5px;
    padding: 5px 10px;
    border-radius: 6px;
    background: #0d1424;
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.35);
  }
  .preset-btn:hover { background: rgba(56, 189, 248, 0.15); }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
  }
  .kpi-card {
    background: #0d1424;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .kpi-card .kpi-val {
    font-size: 20px;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
  }
  .kpi-card .kpi-lbl {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .terminal {
    background: #060911;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    font-family: var(--mono);
    font-size: 12px;
    color: #a7f3d0;
    height: 290px;
    overflow-y: auto;
    white-space: pre-wrap;
  }
  .status-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0d1424;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 10px;
    font-size: 12.5px;
  }
  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    background: var(--success);
  }
  .dot.busy {
    background: var(--warning);
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.35; }
    100% { opacity: 1; }
  }
  table.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    margin-top: 8px;
  }
  table.data-table th, table.data-table td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  table.data-table th { color: var(--text-muted); font-weight: 600; }
  table.data-table td.num { text-align: right; font-family: var(--mono); color: var(--accent); }
  .scroll-table { max-height: 260px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>
      FX Reborn Toolkit
      <span class="badge-ultra">v3.0 ULTRA</span>
    </h1>
    <p data-i18n="subtitle">Reescritor Estático & Emulador AOT de Instruções AVX2, FMA3, F16C, BMI1/2, LZCNT, MOVBE e POPCNT</p>
  </div>
  <div class="lang-switch" title="Idioma / Language">
    <button id="langPt" class="lang-btn" onclick="setLang('pt')">PT-BR</button>
    <button id="langEn" class="lang-btn" onclick="setLang('en')">EN</button>
  </div>
  <div id="cpuPills" class="cpu-pills">
    <span class="cpu-pill" data-i18n="detecting">Detectando CPU...</span>
  </div>
</header>

<div class="grid-main">
  <!-- LEFT COLUMN: CONFIGURATION & ACTIONS -->
  <div>
    <div class="card">
      <h2 data-i18n="card1">1. Selecionar Executável / DLL e Pasta de Saída</h2>
      <div class="field-group">
        <label data-i18n="lblExe">Arquivo Executável ou DLL Original (.exe / .dll):</label>
        <div class="input-row">
          <input type="text" id="exePath" data-i18n-ph="phExe" placeholder="Ex: C:\Games\FF7Rebirth\End\Binaries\Win64\ff7rebirth_.exe">
          <button class="btn-secondary" data-i18n="btnBrowseExe" onclick="browseFile('exePath', 'exe')">Procurar .EXE/.DLL</button>
        </div>
      </div>

      <div class="field-group">
        <label data-i18n="lblOut">Pasta de Saída (onde o novo executável otimizado + fxcompat.dll serão salvos):</label>
        <div class="input-row">
          <input type="text" id="outDir" data-i18n-ph="phOut" placeholder="Ex: C:\Games\FF7Rebirth_Patched">
          <button class="btn-secondary" data-i18n="btnBrowseFolder" onclick="browseFolder('outDir')">Procurar Pasta</button>
        </div>
      </div>

      <div class="field-group">
        <label data-i18n="lblCfg">Configuração Opcional (game.json) e Log de Instruções Aprendidas (fxreborn-missed.log):</label>
        <div class="input-row" style="margin-bottom: 6px;">
          <input type="text" id="configPath" data-i18n-ph="phCfg" placeholder="Opcional: examples\ff7rebirth\game.json (patches de check de CPU e builds conhecidas)">
          <button class="btn-secondary" onclick="browseFile('configPath', 'json')">game.json</button>
        </div>
        <div class="input-row">
          <input type="text" id="learnedPath" data-i18n-ph="phLearned" placeholder="Opcional: fxreborn-missed.log (adiciona endereços extras capturados em runtime)">
          <button class="btn-secondary" onclick="browseFile('learnedPath', 'log')">missed.log</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>
        <span data-i18n="card2">2. Conjuntos de Instruções para Simular / Emular (AOT)</span>
      </h2>
      <div class="preset-bar">
        <button class="preset-btn" data-i18n="presetSandy" onclick="applyPreset('sandy')">Preset: Intel 2ª Ger. (Sandy Bridge i7-2600 / i5-2500)</button>
        <button class="preset-btn" data-i18n="presetIvy" onclick="applyPreset('ivy')">Preset: Intel 3ª Ger. (Ivy Bridge i7-3770)</button>
        <button class="preset-btn" data-i18n="presetFx" onclick="applyPreset('fx')">Preset: AMD FX (FX-8350 / FX-6300)</button>
        <button class="preset-btn" data-i18n="presetAll" onclick="applyPreset('all')">Selecionar Todas</button>
      </div>

      <div class="features-grid">
        <label class="feat-box primary-feat">
          <input type="checkbox" class="feat-chk" value="AVX2" checked>
          <div>
            <div class="feat-title">AVX2 ★</div>
            <div class="feat-desc" data-i18n="dAvx2">Vetores 256-bit Inteiros, Broadcasts, Gathers, Permutes</div>
          </div>
        </label>
        <label class="feat-box primary-feat">
          <input type="checkbox" class="feat-chk" value="FMA" checked>
          <div>
            <div class="feat-title">FMA3 ★</div>
            <div class="feat-desc" data-i18n="dFma">Fused Multiply-Add (VFMADD / VFMSUB / VFNMADD)</div>
          </div>
        </label>
        <label class="feat-box primary-feat">
          <input type="checkbox" class="feat-chk" value="F16C" checked>
          <div>
            <div class="feat-title">F16C ★</div>
            <div class="feat-desc" data-i18n="dF16c">Conversão Half-Float (VCVTPH2PS / VCVTPS2PH SIMD)</div>
          </div>
        </label>
        <label class="feat-box primary-feat">
          <input type="checkbox" class="feat-chk" value="BMI2" checked>
          <div>
            <div class="feat-title">BMI2 ★</div>
            <div class="feat-desc" data-i18n="dBmi2">SHLX, SHRX, SARX, BZHI, RORX, MULX, PDEP, PEXT</div>
          </div>
        </label>
        <label class="feat-box">
          <input type="checkbox" class="feat-chk" value="BMI1" checked>
          <div>
            <div class="feat-title">BMI1</div>
            <div class="feat-desc" data-i18n="dBmi1">ANDN (In-Place), BLSI, BLSMSK, BLSR, BEXTR, TZCNT</div>
          </div>
        </label>
        <label class="feat-box">
          <input type="checkbox" class="feat-chk" value="LZCNT" checked>
          <div>
            <div class="feat-title">LZCNT</div>
            <div class="feat-desc" data-i18n="dLzcnt">Contagem de zeros à esquerda via BSR + XOR</div>
          </div>
        </label>
        <label class="feat-box">
          <input type="checkbox" class="feat-chk" value="MOVBE" checked>
          <div>
            <div class="feat-title">MOVBE</div>
            <div class="feat-desc" data-i18n="dMovbe">Troca Big-Endian em leitura/escrita via BSWAP</div>
          </div>
        </label>
        <label class="feat-box">
          <input type="checkbox" class="feat-chk" value="POPCNT">
          <div>
            <div class="feat-title">POPCNT</div>
            <div class="feat-desc" data-i18n="dPopcnt">Contagem de bits paralela SWAR (p/ CPUs muito antigas)</div>
          </div>
        </label>
      </div>

      <h2 style="margin-top: 14px;" data-i18n="card3">3. Otimizações de Desempenho Ultra</h2>
      <div class="opts-grid">
        <label class="feat-box">
          <input type="checkbox" id="optFusion" checked>
          <div>
            <div class="feat-title" data-i18n="tFusion">Fusão de Básico-Blocos (Cluster Fusion)</div>
            <div class="feat-desc" data-i18n="dFusion">Agrupa instruções consecutivas em uma única chamada CALL/RET (prevista pelo processador), eliminando milhares de desvios e exceções VEH.</div>
          </div>
        </label>
        <label class="feat-box">
          <input type="checkbox" id="optSteal" checked>
          <div>
            <div class="feat-title" data-i18n="tSteal">Roubo Seguro de Vizinho 4-Bytes (Zero-VEH)</div>
            <div class="feat-desc" data-i18n="dSteal">Incorpora a instrução vizinha (seguinte ou anterior) em sites AVX2 de 4 bytes isolados para usar CALL nativo em vez de exceção VEH.</div>
          </div>
        </label>
      </div>

      <div class="action-bar">
        <button class="btn-primary" id="btnScan" data-i18n="btnScan" onclick="startScan()">1. Analisar Executável (Scan)</button>
        <button class="btn-secondary" id="btnGate" data-i18n="btnGate" onclick="startGate()">2. Detectar Trava de CPU (Gate)</button>
        <button class="btn-success" id="btnBuild" data-i18n="btnBuild" onclick="startBuild()">⚡ 3. Gerar Executável Modificado & Otimizado</button>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN: TELEMETRY, LOGS & SCAN REPORT -->
  <div>
    <div class="card">
      <h2 data-i18n="cardStatus">Status de Execução & Console em Tempo Real</h2>
      <div class="status-banner">
        <div>
          <span id="statusDot" class="dot"></span>
          <strong id="statusText" data-i18n="statusReady">Pronto para analisar ou construir</strong>
        </div>
        <span id="taskBadge" style="font-family: var(--mono); font-size: 11px; color: var(--text-muted);">IDLE</span>
      </div>
      <div id="logConsole" class="terminal" data-i18n="welcome">Bem-vindo ao FX Reborn Toolkit Ultra v3.0!
Selecione um executável (.exe ou .dll) à esquerda para escanear todas as instruções (AVX2, FMA3, F16C, BMI1/2, LZCNT, MOVBE, POPCNT) ou gerar diretamente o novo executável otimizado.</div>
    </div>

    <div class="card">
      <h2 data-i18n="cardSummary">Resumo de Instruções Encontradas & Traduzidas</h2>
      <div class="kpi-row">
        <div class="kpi-card">
          <div class="kpi-val" id="kpiTotal">—</div>
          <div class="kpi-lbl" data-i18n="kTotal">Instruções Alvo</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val" id="kpiInPlace">—</div>
          <div class="kpi-lbl" data-i18n="kInPlace">In-Place / Fused</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val" id="kpiJumps">—</div>
          <div class="kpi-lbl" data-i18n="kJumps">Saltos Nativos</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val" id="kpiVeh">—</div>
          <div class="kpi-lbl" data-i18n="kVeh">Exceções VEH</div>
        </div>
      </div>

      <div id="reportContainer" class="scroll-table">
        <table class="data-table">
          <thead>
            <tr>
              <th data-i18n="thInstr">Instrução / Conjunto</th>
              <th style="text-align:right;" data-i18n="thCount">Ocorrências</th>
              <th data-i18n="thStatus">Status</th>
            </tr>
          </thead>
          <tbody id="reportTableBody">
            <tr><td colspan="3" style="color: var(--text-muted); text-align:center; padding: 18px;" data-i18n="emptyTable">Execute "Analisar Executável (Scan)" ou "Gerar Executável" para visualizar o inventário detalhado de instruções encontradas no arquivo.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// ---------------------------------------------------------------------------
// Idiomas da interface / Interface languages (Português do Brasil + English)
// O texto em português fica no próprio HTML; aqui ficam as traduções em inglês
// e os textos usados pelo JavaScript.
// ---------------------------------------------------------------------------
const I18N = {
  en: {
    docTitle: 'FX Reborn Toolkit Ultra v3.0 — Static Emulator & AVX2 / FMA / F16C / BMI Optimizer',
    subtitle: 'Static Rewriter & AOT Emulator for AVX2, FMA3, F16C, BMI1/2, LZCNT, MOVBE and POPCNT instructions',
    detecting: 'Detecting CPU...',
    card1: '1. Select Executable / DLL and Output Folder',
    lblExe: 'Original Executable or DLL file (.exe / .dll):',
    phExe: 'e.g. C:\\Games\\FF7Rebirth\\End\\Binaries\\Win64\\ff7rebirth_.exe',
    btnBrowseExe: 'Browse .EXE/.DLL',
    lblOut: 'Output Folder (where the new optimized executable + fxcompat.dll will be saved):',
    phOut: 'e.g. C:\\Games\\FF7Rebirth_Patched',
    btnBrowseFolder: 'Browse Folder',
    lblCfg: 'Optional Config (game.json) and Learned Instructions Log (fxreborn-missed.log):',
    phCfg: 'Optional: examples\\ff7rebirth\\game.json (CPU-check patches and known builds)',
    phLearned: 'Optional: fxreborn-missed.log (adds extra addresses captured at runtime)',
    card2: '2. Instruction Sets to Simulate / Emulate (AOT)',
    presetSandy: 'Preset: Intel 2nd Gen (Sandy Bridge i7-2600 / i5-2500)',
    presetIvy: 'Preset: Intel 3rd Gen (Ivy Bridge i7-3770)',
    presetFx: 'Preset: AMD FX (FX-8350 / FX-6300)',
    presetAll: 'Select All',
    dAvx2: '256-bit integer vectors, broadcasts, gathers, permutes',
    dFma: 'Fused Multiply-Add (VFMADD / VFMSUB / VFNMADD)',
    dF16c: 'Half-float conversion (VCVTPH2PS / VCVTPS2PH SIMD)',
    dBmi2: 'SHLX, SHRX, SARX, BZHI, RORX, MULX, PDEP, PEXT',
    dBmi1: 'ANDN (in-place), BLSI, BLSMSK, BLSR, BEXTR, TZCNT',
    dLzcnt: 'Leading-zero count via BSR + XOR',
    dMovbe: 'Big-endian swap on load/store via BSWAP',
    dPopcnt: 'Parallel SWAR bit count (for very old CPUs)',
    card3: '3. Ultra Performance Optimizations',
    tFusion: 'Basic-Block Fusion (Cluster Fusion)',
    dFusion: 'Groups consecutive instructions into a single CALL/RET (predicted by the CPU), removing thousands of jumps and VEH exceptions.',
    tSteal: 'Safe 4-Byte Neighbor Stealing (Zero-VEH)',
    dSteal: 'Moves the neighboring instruction (next or previous) into isolated 4-byte AVX2 sites so they use a native CALL instead of a VEH exception.',
    btnScan: '1. Analyze Executable (Scan)',
    btnGate: '2. Detect CPU Check (Gate)',
    btnBuild: '⚡ 3. Build Modified & Optimized Executable',
    cardStatus: 'Execution Status & Live Console',
    statusReady: 'Ready to analyze or build',
    welcome: 'Welcome to FX Reborn Toolkit Ultra v3.0!\nSelect an executable (.exe or .dll) on the left to scan all instructions (AVX2, FMA3, F16C, BMI1/2, LZCNT, MOVBE, POPCNT) or build the new optimized executable directly.',
    cardSummary: 'Found & Translated Instructions Summary',
    kTotal: 'Target Instructions',
    kInPlace: 'In-Place / Fused',
    kJumps: 'Native Jumps',
    kVeh: 'VEH Exceptions',
    thInstr: 'Instruction / Set',
    thCount: 'Occurrences',
    thStatus: 'Status',
    emptyTable: 'Run "Analyze Executable (Scan)" or "Build Executable" to see the detailed inventory of instructions found in the file.',
    pillHas: 'Natively supported by your CPU',
    pillMiss: 'Missing on your CPU (needs to be emulated)',
    alertFile: 'Enter the path of the executable or DLL.',
    alertFileOut: 'Enter the executable file (.exe/.dll) and the output folder!',
    rowMethod: 'Method:',
    rowApplied: 'Applied ✓',
    rowSet: 'Set',
    rowDetected: 'Detected',
    rowSupported: 'Supported ✓',
    rowNone: 'No target instructions found.',
    statusIdle: 'Ready',
    badgeRunning: 'RUNNING:',
    badgeReady: 'READY',
  },
  pt: {
    pillHas: 'Suportado nativamente pela sua CPU',
    pillMiss: 'Ausente na sua CPU (precisa ser emulado)',
    alertFile: 'Informe o caminho do executável ou DLL.',
    alertFileOut: 'Informe o arquivo executável (.exe/.dll) e a pasta de saída!',
    rowMethod: 'Método:',
    rowApplied: 'Aplicado ✓',
    rowSet: 'Conjunto',
    rowDetected: 'Detectado',
    rowSupported: 'Suportado ✓',
    rowNone: 'Nenhuma instrução alvo encontrada.',
    statusIdle: 'Pronto',
    badgeRunning: 'EXECUTANDO:',
    badgeReady: 'PRONTO',
  },
};
let LANG = 'pt';
let LAST_STATE = null;

function t(key) {
  return (I18N[LANG] && I18N[LANG][key]) || I18N.pt[key] || key;
}

function initLang() {
  // Guarda o texto original (português) de cada elemento antes de traduzir
  document.querySelectorAll('[data-i18n]').forEach(el => { el.dataset.pt = el.textContent; });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.dataset.ptPh = el.placeholder; });
  I18N.pt.docTitle = document.title;
  let saved = null;
  try { saved = localStorage.getItem('fxkit-lang'); } catch (e) {}
  const browser = (navigator.language || 'pt').toLowerCase();
  setLang(saved || (browser.startsWith('pt') ? 'pt' : 'en'));
}

function setLang(lang) {
  LANG = (lang === 'en') ? 'en' : 'pt';
  try { localStorage.setItem('fxkit-lang', LANG); } catch (e) {}
  document.documentElement.lang = (LANG === 'en') ? 'en' : 'pt-BR';
  document.title = t('docTitle');
  const busy = LAST_STATE && LAST_STATE.logs && LAST_STATE.logs.length;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (busy && (el.id === 'logConsole' || el.id === 'statusText')) return;  // keep live output
    el.textContent = (LANG === 'pt') ? el.dataset.pt : (I18N.en[el.dataset.i18n] || el.dataset.pt);
  });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    el.placeholder = (LANG === 'pt') ? el.dataset.ptPh : (I18N.en[el.dataset.i18nPh] || el.dataset.ptPh);
  });
  document.getElementById('langPt').classList.toggle('active', LANG === 'pt');
  document.getElementById('langEn').classList.toggle('active', LANG === 'en');
  fetchCpuInfo();
  if (LAST_STATE) renderTables(LAST_STATE);
}
</script>
<script>
let pollTimer = null;

async function fetchCpuInfo() {
  try {
    const res = await fetch('/api/cpu');
    const data = await res.json();
    const container = document.getElementById('cpuPills');
    container.innerHTML = '';
    const keys = ['AVX', 'AVX2', 'FMA', 'F16C', 'BMI1', 'BMI2', 'LZCNT', 'MOVBE', 'POPCNT'];
    keys.forEach(k => {
      const has = data.features && data.features[k];
      const span = document.createElement('span');
      span.className = 'cpu-pill ' + (has ? 'has' : 'miss');
      span.textContent = k + (has ? ' ✓' : ' ✗');
      span.title = has ? t('pillHas') : t('pillMiss');
      container.appendChild(span);
    });
  } catch (e) {}
}

function getSelectedFeatures() {
  const chks = document.querySelectorAll('.feat-chk:checked');
  return Array.from(chks).map(c => c.value);
}

function applyPreset(name) {
  const map = {
    'sandy': ['AVX2', 'FMA', 'F16C', 'BMI1', 'BMI2', 'LZCNT', 'MOVBE'],
    'ivy':   ['AVX2', 'FMA', 'BMI1', 'BMI2', 'LZCNT', 'MOVBE'],
    'fx':    ['AVX2', 'BMI2', 'MOVBE'],
    'all':   ['AVX2', 'FMA', 'F16C', 'BMI1', 'BMI2', 'LZCNT', 'MOVBE', 'POPCNT']
  };
  const active = new Set(map[name] || map['sandy']);
  document.querySelectorAll('.feat-chk').forEach(c => {
    c.checked = active.has(c.value);
  });
}

async function browseFile(inputId, kind) {
  const res = await fetch('/api/browse_file?kind=' + encodeURIComponent(kind));
  const data = await res.json();
  if (data.path) {
    document.getElementById(inputId).value = data.path;
    if (inputId === 'exePath' && !document.getElementById('outDir').value) {
      const parts = data.path.replace(/\//g, '\\').split('\\');
      parts.pop();
      document.getElementById('outDir').value = parts.join('\\') + '\\saida_ultra';
    }
  }
}

async function browseFolder(inputId) {
  const res = await fetch('/api/browse_folder');
  const data = await res.json();
  if (data.path) {
    document.getElementById(inputId).value = data.path;
  }
}

async function startScan() {
  const file = document.getElementById('exePath').value.trim();
  if (!file) { alert(t('alertFile')); return; }
  const feats = getSelectedFeatures();
  await fetch('/api/scan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ file, popcnt: feats.includes('POPCNT'), lang: LANG })
  });
  startPolling();
}

async function startGate() {
  const file = document.getElementById('exePath').value.trim();
  if (!file) { alert(t('alertFile')); return; }
  await fetch('/api/gate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ file, lang: LANG })
  });
  startPolling();
}

async function startBuild() {
  const file = document.getElementById('exePath').value.trim();
  const out = document.getElementById('outDir').value.trim();
  if (!file || !out) {
    alert(t('alertFileOut'));
    return;
  }
  const payload = {
    file,
    out,
    config: document.getElementById('configPath').value.trim(),
    learned: document.getElementById('learnedPath').value.trim(),
    features: getSelectedFeatures(),
    fusion: document.getElementById('optFusion').checked,
    neighbor_steal: document.getElementById('optSteal').checked,
    lang: LANG
  };
  await fetch('/api/build', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  startPolling();
}

function renderTables(state) {
  const tbody = document.getElementById('reportTableBody');
  if (state.last_build) {
    const b = state.last_build;
    const m = b.methods || {};
    const inplace = (m.in_place || 0) + (m.fused_block_member || 0);
    const jumps = (m.native_call || 0) + (m.fused_block_call || 0) + (m.padding_jump || 0) + (m.neighbor_steal_call || 0) + (m.neighbor_steal_back_call || 0);
    const veh = m.native_exception_redirect || 0;
    document.getElementById('kpiTotal').textContent = (b.translated || 0).toLocaleString();
    document.getElementById('kpiInPlace').textContent = inplace.toLocaleString();
    document.getElementById('kpiJumps').textContent = jumps.toLocaleString();
    document.getElementById('kpiVeh').textContent = veh.toLocaleString();

    let rows = '';
    for (const [k, v] of Object.entries(m)) {
      rows += `<tr><td>${t('rowMethod')} <strong>${k}</strong></td><td class="num">${v.toLocaleString()}</td><td><span style="color:var(--success)">${t('rowApplied')}</span></td></tr>`;
    }
    tbody.innerHTML = rows;
    return;
  }
  if (state.last_scan) {
    const s = state.last_scan;
    document.getElementById('kpiTotal').textContent = (s.total_sites || 0).toLocaleString();
    document.getElementById('kpiInPlace').textContent = (s.translatable || 0).toLocaleString();
    document.getElementById('kpiJumps').textContent = Object.keys(s.by_mnemonic || {}).length;
    document.getElementById('kpiVeh').textContent = (s.unsupported_count || 0).toLocaleString();

    let rows = '';
    if (s.by_feature) {
      for (const [f, cnt] of Object.entries(s.by_feature)) {
        rows += `<tr style="background:rgba(56,189,248,0.08)"><td><strong>${t('rowSet')} ${f}</strong></td><td class="num">${cnt.toLocaleString()}</td><td><span style="color:var(--accent)">${t('rowDetected')}</span></td></tr>`;
      }
    }
    if (s.by_mnemonic) {
      for (const [mn, cnt] of Object.entries(s.by_mnemonic)) {
        rows += `<tr><td style="font-family:var(--mono)">${mn}</td><td class="num">${cnt.toLocaleString()}</td><td><span style="color:var(--success)">${t('rowSupported')}</span></td></tr>`;
      }
    }
    tbody.innerHTML = rows || '<tr><td colspan="3">' + t('rowNone') + '</td></tr>';
  }
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const state = await res.json();
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    const badge = document.getElementById('taskBadge');
    const term = document.getElementById('logConsole');

    dot.className = 'dot' + (state.running ? ' busy' : '');
    txt.textContent = state.status || t('statusIdle');
    badge.textContent = state.running ? (t('badgeRunning') + ' ' + (state.task || '')) : t('badgeReady');

    document.getElementById('btnScan').disabled = state.running;
    document.getElementById('btnGate').disabled = state.running;
    document.getElementById('btnBuild').disabled = state.running;

    if (state.logs && state.logs.length) {
      term.textContent = state.logs.join('\n');
      term.scrollTop = term.scrollHeight;
    }
    LAST_STATE = state;
    renderTables(state);

    if (!state.running && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) {}
}

function startPolling() {
  if (!pollTimer) {
    pollTimer = setInterval(pollStatus, 600);
  }
  pollStatus();
}

window.addEventListener('DOMContentLoaded', () => {
  initLang();  // also loads the CPU pills
  // Pre-fill example config if exists
  fetch('/api/defaults').then(r => r.json()).then(d => {
    if (d.default_config) document.getElementById('configPath').value = d.default_config;
  }).catch(() => {});
});
</script>
</body>
</html>
"""


def handle_http_request(conn: socket.socket):
    """Minimal, rock-solid HTTP/1.1 handler using only standard `socket` and `json`."""
    try:
        conn.settimeout(30.0)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        if not data:
            return
        header_part, _, body_part = data.partition(b"\r\n\r\n")
        header_lines = header_part.decode("utf-8", errors="replace").split("\r\n")
        req_line = header_lines[0].split()
        if len(req_line) < 2:
            return
        method, full_path = req_line[0], req_line[1]
        path, _, query = full_path.partition("?")

        content_len = 0
        for line in header_lines[1:]:
            if line.lower().startswith("content-length:"):
                content_len = int(line.split(":", 1)[1].strip())
        while len(body_part) < content_len:
            chunk = conn.recv(4096)
            if not chunk:
                break
            body_part += chunk

        def send_json(obj: dict, status: str = "200 OK"):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            hdr = (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Cache-Control: no-store\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            conn.sendall(hdr + payload)

        def send_html(html_str: str):
            payload = html_str.encode("utf-8")
            hdr = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html; charset=utf-8\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Cache-Control: no-store\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            conn.sendall(hdr + payload)

        if method == "GET" and path == "/":
            send_html(HTML_UI)
            return

        if method == "GET" and path == "/api/cpu":
            send_json(detect_host_cpu())
            return

        if method == "GET" and path == "/api/defaults":
            ex_cfg = ROOT / "examples" / "ff7rebirth" / "game.json"
            send_json({"default_config": str(ex_cfg) if ex_cfg.exists() else ""})
            return

        if method == "GET" and path == "/api/status":
            with JOB_LOCK:
                send_json(dict(JOB_STATE))
            return

        if method == "GET" and path == "/api/browse_file":
            kind = "exe"
            if "kind=json" in query:
                fspec = "Configuração JSON (*.json)\0*.json\0Todos (*.*)\0*.*\0\0"
                title = "Selecionar game.json"
            elif "kind=log" in query:
                fspec = "Arquivo de Log (*.log)\0*.log\0Todos (*.*)\0*.*\0\0"
                title = "Selecionar fxreborn-missed.log"
            else:
                fspec = "Executáveis e DLLs (*.exe;*.dll)\0*.exe;*.dll\0Todos (*.*)\0*.*\0\0"
                title = "Selecionar Executável ou DLL (.exe / .dll)"
            chosen = native_open_file_dialog(title, fspec)
            send_json({"path": chosen})
            return

        if method == "GET" and path == "/api/browse_folder":
            chosen = native_select_folder_dialog()
            send_json({"path": chosen})
            return

        if method == "POST" and path == "/api/scan":
            req = json.loads(body_part.decode("utf-8") or "{}")
            target_file = Path(req.get("file", ""))
            popcnt = bool(req.get("popcnt", False))
            lang = req.get("lang", "pt")

            def job_scan():
                if not target_file.exists():
                    raise FileNotFoundError(tr(lang, f"Arquivo não encontrado: {target_file}", f"File not found: {target_file}"))
                append_log(tr(lang, f"Escaneando {target_file.name}...", f"Scanning {target_file.name}..."))
                res = fxkit.scan_binary_detailed(target_file, include_popcnt=popcnt)
                with JOB_LOCK:
                    JOB_STATE["last_scan"] = res
                    JOB_STATE["last_build"] = None
                append_log(tr(
                    lang,
                    f"Scan concluído: {res.get('total_sites', 0)} instruções encontradas "
                    f"({res.get('translatable', 0)} traduzíveis). Por conjunto: {res.get('by_feature', {})}",
                    f"Scan finished: {res.get('total_sites', 0)} instructions found "
                    f"({res.get('translatable', 0)} translatable). By set: {res.get('by_feature', {})}",
                ))

            run_async_job(tr(lang, "Análise de Executável (Scan)", "Executable Analysis (Scan)"), job_scan, lang)
            send_json({"started": True})
            return

        if method == "POST" and path == "/api/gate":
            req = json.loads(body_part.decode("utf-8") or "{}")
            target_file = Path(req.get("file", ""))
            lang = req.get("lang", "pt")

            def job_gate():
                if not target_file.exists():
                    raise FileNotFoundError(tr(lang, f"Arquivo não encontrado: {target_file}", f"File not found: {target_file}"))
                append_log(tr(lang, f"Procurando verificações de CPU (AVX2/FMA/F16C) em {target_file.name}...",
                              f"Looking for CPU checks (AVX2/FMA/F16C) in {target_file.name}..."))
                gates = fxkit.detect_gates_detailed(target_file)
                with JOB_LOCK:
                    JOB_STATE["last_gates"] = gates
                if not gates:
                    append_log(tr(lang, "Nenhuma trava explícita de inicialização encontrada próxima a strings de erro.",
                                  "No explicit startup check found next to error strings."))
                else:
                    for g in gates:
                        append_log(tr(lang, f"Trava detectada em {g['rva']} ({g['asm']}) perto de \"{g['string']}\"",
                                      f"Check found at {g['rva']} ({g['asm']}) near \"{g['string']}\""))
                        append_log(tr(lang, "Patch sugerido: ", "Suggested patch: ") + json.dumps(g['suggested_patch']))

            run_async_job(tr(lang, "Detecção de Trava de CPU (Gate)", "CPU Check Detection (Gate)"), job_gate, lang)
            send_json({"started": True})
            return

        if method == "POST" and path == "/api/build":
            req = json.loads(body_part.decode("utf-8") or "{}")
            target_file = Path(req.get("file", ""))
            out_dir = Path(req.get("out", ""))
            cfg_str = req.get("config", "")
            learned_str = req.get("learned", "")
            features = set(req.get("features") or fxkit.TRANSLATED_FEATURES)
            fusion = bool(req.get("fusion", True))
            neighbor_steal = bool(req.get("neighbor_steal", True))
            lang = req.get("lang", "pt")

            def job_build():
                if not target_file.exists():
                    raise FileNotFoundError(tr(lang, f"Arquivo não encontrado: {target_file}", f"File not found: {target_file}"))
                if target_file.parent.resolve() == out_dir.resolve():
                    raise ValueError(tr(lang, "Por segurança, escolha uma pasta de saída diferente da pasta onde está o arquivo original.",
                                        "For safety, choose an output folder different from the folder of the original file."))
                config, cdir = fxkit.load_config(cfg_str if cfg_str and Path(cfg_str).exists() else None)
                learned = fxkit.learned_from_log(Path(learned_str), target_file.name) if learned_str and Path(learned_str).exists() else []
                include_popcnt = "POPCNT" in features
                append_log(tr(lang, f"Selecionando/escaneando instruções em {target_file.name} (conjuntos: {sorted(features)})...",
                              f"Selecting/scanning instructions in {target_file.name} (sets: {sorted(features)})..."))
                sites, kind, report = fxkit.choose_sites(
                    target_file, config, cdir, learned, force_scan=False, include_popcnt=include_popcnt
                )
                specs = [
                    p for p in config.get("patches", [])
                    if p.get("target", target_file.name).lower() == target_file.name.lower()
                ]
                patches = fxkit.locate_patches(target_file, specs) if specs else []
                for p in patches:
                    append_log(tr(lang, f"Patch aplicado '{p['name']}' em {p['rva']:#x}: {p['old']} -> {p['new']}",
                                  f"Patch applied '{p['name']}' at {p['rva']:#x}: {p['old']} -> {p['new']}"))
                append_log(tr(lang, f"Construindo executável otimizado para {len(sites)} locais ({kind})...",
                              f"Building optimized executable for {len(sites)} sites ({kind})..."))
                manifest = fxkit.build(
                    target_file, out_dir, sites, patches,
                    enabled_features=features,
                    enable_fusion=fusion,
                    enable_neighbor_steal=neighbor_steal,
                )
                rt = ROOT / "runtime" / "fxcompat.dll"
                if rt.exists():
                    shutil.copy2(rt, out_dir / "fxcompat.dll")
                with JOB_LOCK:
                    JOB_STATE["last_build"] = manifest
                append_log(tr(lang, f"SUCESSO! Arquivo gerado em: {out_dir / target_file.name}",
                              f"SUCCESS! File written to: {out_dir / target_file.name}"))
                append_log(tr(lang, f"Estatísticas de métodos: {manifest.get('methods')}",
                              f"Method statistics: {manifest.get('methods')}"))
                append_log(tr(lang, "Copie o executável gerado E o arquivo fxcompat.dll para a pasta do jogo (faça backup do original!).",
                              "Copy the generated executable AND fxcompat.dll into the game folder (back up the original first!)."))

            run_async_job(tr(lang, "Construção de Executável Otimizado", "Optimized Executable Build"), job_build, lang)
            send_json({"started": True})
            return

        send_json({"error": "Not found"}, "404 Not Found")
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def open_app_window(url: str):
    """Open the GUI as a native-looking desktop app window using Edge or Chrome --app mode on Windows."""
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / r"Microsoft\Edge\Application\msedge.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / r"Microsoft\Edge\Application\msedge.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / r"Google\Chrome\Application\chrome.exe",
            Path(os.environ.get("LocalAppData", "")) / r"Google\Chrome\Application\chrome.exe",
        ]
        for exe in candidates:
            if exe.exists():
                try:
                    subprocess.Popen([str(exe), f"--app={url}", "--window-size=1280,860"])
                    return
                except Exception:
                    pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def launch(port: int = 0) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    actual_port = srv.getsockname()[1]
    srv.listen(16)
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"============================================================")
    print(f"  FX Reborn Toolkit Ultra v3.0 - Interface Gráfica Ativa")
    print(f"  Acesse: {url}")
    print(f"  (Mantenha esta janela aberta enquanto utiliza o programa)")
    print(f"============================================================", flush=True)
    open_app_window(url)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle_http_request, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        print("\nEncerrando FX Reborn Toolkit Ultra GUI...")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(launch())
