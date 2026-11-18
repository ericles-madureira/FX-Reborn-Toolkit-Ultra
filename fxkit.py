"""FX Reborn Toolkit Ultra: make AVX2/FMA/F16C/BMI1/BMI2 games run at maximum speed on older CPUs (Sandy Bridge / Ivy Bridge / AMD FX).

Rewrites AVX2, BMI2, FMA3, BMI1, F16C, LZCNT, MOVBE (and optional POPCNT) instructions in a game's
exe or DLL into ultra-optimized AVX1/SSE4.2/x86-64 code with:
- CALL/RET stub linkage (every stub exit is predicted by the return-stack buffer)
- Multi-Instruction Basic-Block Fusion (one call per contiguous sequence)
- Forward AND backward neighbor stealing (keeps short sites off the slow VEH path)
- Exact FMA3 lowering with a sign-constant pool (zero scratch / small stack slot)
- In-Place ANDN / VPBROADCASTD / VPCMPEQ All-Ones, destination-as-scratch lane splits
- Pooled-constant SIMD F16C (VCVTPH2PS) lowering

Commands:
  gui                                   Opens the graphical user interface (Native Dark Mode Dashboard).
  scan    <exe|dll>                     Scans target instructions, translatability, and CPU features.
  gate    <exe|dll>                     Finds "requires AVX2/FMA/F16C" startup CPU checks and suggests patches.
  build   <exe|dll> -o DIR [options]    Writes the converted file + manifest + fxcompat.dll.
  package --config game.json -o DIR     Makes a player-ready mod folder and zip.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
sys.path[:0] = [str(ENGINE / "pylibs"), str(ENGINE)]

VERSION = "3.0.0-Ultra"
BASELINE = {"X64", "SSE", "SSE2", "CMOV", "CX8", "FPU", "MMX", "FXSR", "INTEL8086", "INTEL186",
            "INTEL286", "INTEL386", "INTEL486", "INTEL8086_ONLY", "CPUID", "SYSCALL", "TSC",
            "MULTIBYTENOP", "PAUSE", "CLFSH", "MSR", "RDTSCP", "XSAVE", "OSXSAVE", "SSE3",
            "CX16", "LAHF_SAHF", "PREFETCHW", "D3NOW_PREFETCH", "PREFETCH"}
TRANSLATED_FEATURES = {"AVX2", "BMI2", "BMI1", "FMA", "F16C", "LZCNT", "MOVBE"}
ALL_SUPPORTED_FEATURES = TRANSLATED_FEATURES | {"POPCNT"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feature_names():
    import iced_x86 as ix
    return {v: k for k, v in vars(ix.CpuidFeature).items() if isinstance(v, int)}


def load_pe(path: Path):
    import pefile
    data = path.read_bytes()
    pe = pefile.PE(data=data, fast_load=True)
    pe.parse_data_directories(directories=[3])
    return pe, data


# ------------------------------------------------------------------- sites
def find_sites(path: Path, include_popcnt: bool = False) -> tuple[list[int], dict]:
    import find_sites as fs
    import discover_code_sites as dcs
    print("Scanning for AVX2/BMI2/FMA/BMI1/F16C/LZCNT/MOVBE code (follows control flow)...", flush=True)
    d = dcs.Discovery(path, include_popcnt=include_popcnt)
    d.strong_roots()
    d.walk()
    d.reloc_roots()
    d.address_walk()
    discovery = d.result()
    discovery["address_root_sources"] = {hex(r): hex(src) for r, (_, src) in d.address_roots.items()}
    pe, _ = load_pe(path)
    return fs.select(pe, discovery)


def learned_from_log(log: Path, module: str | None = None) -> list[int]:
    out = set()
    if log and log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not e.get("rva"):
                continue
            if module and e.get("module"):
                match = e["module"].lower() == module.lower()
            else:
                match = bool(int(e.get("main_module", "0x0"), 16))
            if match:
                out.add(int(e["rva"], 16))
    return sorted(out)


def choose_sites(path: Path, config: dict, config_dir: Path, extra=(), force_scan=False, include_popcnt=False):
    """Known build -> its frozen list; otherwise scan. Returns (sites, kind, report)."""
    digest = sha256(path)
    known = config.get("known_builds", {}).get(digest)
    if known and not force_scan and not include_popcnt:
        data = json.loads((config_dir / known["sites"]).read_text())
        sites = [int(x, 16) for x in data["sites"]]
        kind, report = f"verified ({known.get('label', digest[:12])})", {}
    else:
        suffix = ".popcnt.sites.json" if include_popcnt else ".sites.json"
        cache = ENGINE / "code_cache" / f"{digest}{suffix}"
        if not cache.exists() and not include_popcnt:
            legacy_cache = ENGINE / "code_cache" / f"{digest}.sites.json"
            if legacy_cache.exists():
                cache = legacy_cache
        if cache.exists() and not force_scan:
            data = json.loads(cache.read_text())
        else:
            sites, report = find_sites(path, include_popcnt=include_popcnt)
            data = {"sites": [hex(s) for s in sites], "report": report}
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(data))
        sites, kind, report = [int(x, 16) for x in data["sites"]], "SCANNED (CFG + Reloc Verified)", data.get("report", {})
    added = sorted(set(extra) - set(sites))
    return sorted(set(sites) | set(extra)), kind, {**report, "learned_added": [hex(a) for a in added]}


# ------------------------------------------------------------------- build
def build(path: Path, out_dir: Path, sites: list[int], patches: list[dict],
          enabled_features: set[str] | None = None,
          enable_fusion: bool = True,
          enable_neighbor_steal: bool = True) -> dict:
    """Patch a copy of path into out_dir. Returns the manifest."""
    import build_static_mod as engine
    out_dir.mkdir(parents=True, exist_ok=True)
    engine.STAGING = out_dir
    engine.Builder(
        path,
        [{"rva": hex(r)} for r in sites],
        patches,
        enabled_features=enabled_features,
        enable_fusion=enable_fusion,
        enable_neighbor_steal=enable_neighbor_steal,
    ).finish()
    return json.loads((out_dir / (path.name + ".manifest.json")).read_text())


def locate_patches(path: Path, specs: list[dict]) -> list[dict]:
    import patterns
    pe, data = load_pe(path)
    return patterns.locate_all(pe, data, specs)


def scan_binary_detailed(path: Path, include_popcnt: bool = False) -> dict:
    """Perform a full scan of an EXE/DLL and return structured JSON results for both CLI and GUI."""
    import iced_x86 as ix
    import avx2_lower
    import bmi2_lower
    import sandy_lower

    pe, data = load_pe(path)
    names = feature_names()
    is_dll = bool(pe.FILE_HEADER.Characteristics & 0x2000)
    patched = any(s.Name.rstrip(b"\0") == b".fxcpt" for s in pe.sections)
    digest = sha256(path)

    used = collections.Counter()
    base = pe.OPTIONAL_HEADER.ImageBase
    t0 = time.time()
    for sec in pe.sections:
        if sec.Characteristics & 0x20000000:
            for ins in ix.Decoder(64, sec.get_data()[:sec.Misc_VirtualSize], ip=base + sec.VirtualAddress):
                for f in ins.cpuid_features():
                    used[names.get(f, str(f))] += 1
    sweep_seconds = round(time.time() - t0, 2)

    if patched:
        return {
            "file": str(path),
            "name": path.name,
            "type": "DLL" if is_dll else "EXE",
            "size_mb": round(len(data) / 1e6, 2),
            "sha256": digest,
            "already_patched": True,
            "used_features": dict(used),
        }

    sites, report = choose_sites(path, {}, ROOT, force_scan=False, include_popcnt=include_popcnt)[:2]
    # Re-fetch report properly
    _, _, report = choose_sites(path, {}, ROOT, force_scan=False, include_popcnt=include_popcnt)

    by_mnemonic = collections.Counter()
    by_feature = collections.Counter()
    unsupported = collections.Counter()
    stub_features = set()
    fmt = ix.Formatter(ix.FormatterSyntax.INTEL)

    active_feats = ALL_SUPPORTED_FEATURES if include_popcnt else TRANSLATED_FEATURES

    for rva in sites:
        ins = ix.Decoder(64, pe.get_data(rva, 15), ip=base + rva).decode()
        mnem = fmt.format_mnemonic(ins)
        by_mnemonic[mnem] += 1
        fs = ins.cpuid_features()
        for f in fs:
            fname = names.get(f, str(f))
            if fname in active_feats:
                by_feature[fname] += 1
        try:
            body = (
                bmi2_lower.lower(ins, ins.ip) if ix.CpuidFeature.BMI2 in fs
                else avx2_lower.lower(ins, ins.ip) if ix.CpuidFeature.AVX2 in fs
                else sandy_lower.lower(ins, ins.ip, include_popcnt=include_popcnt)
                if sandy_lower.supports_features(fs, include_popcnt=include_popcnt)
                else None
            )
        except Exception as err:
            body, why = None, str(err)
        else:
            why = "no lowering"
        if body is None:
            unsupported[f"{mnem} ({why[:60]})"] += 1
        else:
            for sub in ix.Decoder(64, body, ip=ins.ip):
                stub_features.update(names.get(f, str(f)) for f in sub.cpuid_features())

    needs = sorted(k for k in used if k not in BASELINE | active_feats)
    stub_needs = sorted(stub_features - BASELINE)
    ok = len(sites) - sum(unsupported.values())

    return {
        "file": str(path),
        "name": path.name,
        "type": "DLL" if is_dll else "EXE",
        "size_mb": round(len(data) / 1e6, 2),
        "sha256": digest,
        "already_patched": False,
        "sweep_seconds": sweep_seconds,
        "total_sites": len(sites),
        "translatable": ok,
        "unsupported_count": sum(unsupported.values()),
        "by_feature": dict(by_feature.most_common()),
        "by_mnemonic": dict(by_mnemonic.most_common()),
        "unsupported": dict(unsupported.most_common()),
        "other_cpu_features": {k: used[k] for k in needs},
        "all_detected_features": dict(used.most_common()),
        "stub_needs": stub_needs,
        "discovery_report": report,
    }


def detect_gates_detailed(path: Path) -> list[dict]:
    """Find AVX2/FMA/F16C startup CPU checks and return suggested patches."""
    import iced_x86 as ix
    import patterns
    pe, data = load_pe(path)
    base = pe.OPTIONAL_HEADER.ImageBase
    fmt = ix.Formatter(ix.FormatterSyntax.INTEL)
    texts = set()
    keywords = (b"AVX2", b"AVX", b"FMA", b"F16C", b"SSE4")
    for kw in keywords:
        for needle in (kw, kw.decode("ascii").encode("utf-16-le")):
            pos = data.find(needle)
            while pos >= 0 and len(texts) < 80:
                unit = 2 if needle[1:2] == b"\0" else 1
                s = pos
                while s >= unit and data[s - unit:s] != b"\0" * unit and data[s - unit] >= 0x20:
                    s -= unit
                e = pos
                while e + unit <= len(data) and data[e:e + unit] != b"\0" * unit:
                    e += unit
                raw = data[s:e]
                try:
                    txt = raw.decode("utf-16-le" if unit == 2 else "latin-1")
                except UnicodeDecodeError:
                    txt = ""
                if 6 <= len(txt) <= 300 and txt.isprintable():
                    texts.add(txt)
                pos = data.find(needle, pos + unit)

    results = []
    for txt in sorted(texts):
        try:
            regions = patterns._regions_near(pe, data, txt)
        except patterns.PatchError:
            continue
        for begin, end in regions:
            prev = None
            for ins in ix.Decoder(64, pe.get_data(begin, end - begin), ip=base + begin):
                is_test = (
                    ins.mnemonic in (ix.Mnemonic.BT, ix.Mnemonic.TEST)
                    and ins.op_count == 2
                    and ins.op1_kind in (ix.OpKind.IMMEDIATE8, ix.OpKind.IMMEDIATE32)
                    and (ins.immediate(1) in (3, 5, 8, 12, 28, 29, 0x20, 0x100, 0x1000, 0x10000000, 0x20000000))
                )
                if prev is not None and prev[0] and ins.flow_control == ix.FlowControl.CONDITIONAL_BRANCH:
                    p = prev[1]
                    rva = p.ip - base
                    raw_bytes = pe.get_data(rva, p.len + ins.len)
                    find_pat = raw_bytes[:-1].hex(" ") + " ??"
                    rep_bytes = bytearray(raw_bytes[:-1])
                    if rep_bytes[-1] in (0x72, 0x73, 0x74, 0x75):
                        rep_bytes[-1] = 0xEB
                    replace_pat = rep_bytes.hex(" ") + " ??"
                    results.append({
                        "string": txt,
                        "function": f"{begin:#x}-{end:#x}",
                        "rva": hex(rva),
                        "asm": f"{fmt.format(p)} ; {fmt.format(ins)}",
                        "bytes": raw_bytes.hex(" "),
                        "suggested_patch": {
                            "name": f"Skip CPU check near '{txt[:40]}'",
                            "target": path.name,
                            "find": find_pat,
                            "replace": replace_pat,
                            "near_string": txt,
                            "expect": 1,
                        }
                    })
                prev = (is_test, ins)
    return results


# -------------------------------------------------------------- CLI commands
def cmd_scan(a):
    info = scan_binary_detailed(Path(a.file), include_popcnt=getattr(a, "popcnt", False))
    if info.get("already_patched"):
        print(f"{info['name']}: This file is already converted (.fxcpt section present). Scan the original instead.")
        return 1

    print(f"{info['name']}: {info['type']}, {info['size_mb']:.1f} MB, sha256 {info['sha256'][:16]}...")
    needs = info["other_cpu_features"]
    stub_needs = info["stub_needs"]
    lines = [
        f"# FX Reborn Toolkit Ultra scan: {info['name']}", "",
        f"- Target instructions found (control flow + reviewed pointer targets): **{info['total_sites']}**",
        f"- Translatable by the engine: **{info['translatable']}**; not yet supported: **{info['unsupported_count']}**",
        f"- Target breakdown by CPU feature: " + (", ".join(f"**{k}**: {v}" for k, v in info["by_feature"].items()) or "none"),
        f"- Byte-sweep-only candidates ignored (may be data): {info['discovery_report'].get('linear_candidates_ignored', 0)}",
        f"- Pointer-reached groups rejected by the automatic review: {len(info['discovery_report'].get('address_reference_rejected', []))}",
        "", "## Other CPU features this program uses (players' CPUs must have these)", "",
        ", ".join(f"{k} ({v})" for k, v in needs.items()) or "none beyond baseline x86-64",
        "", f"Translated code itself uses: {', '.join(stub_needs) or 'baseline only'}",
        "", "## Target instructions by type", "", "| instruction | count |", "|---|---:|"
    ]
    lines += [f"| {m} | {c} |" for m, c in info["by_mnemonic"].items()]
    if info["unsupported"]:
        lines += ["", "## Not yet supported (left as-is; the game closes if it reaches them)", "",
                  "| instruction | count |", "|---|---:|"]
        lines += [f"| {m} | {c} |" for m, c in info["unsupported"].items()]
    verdict = ("Looks convertible." if not info["unsupported"] else
               "Partly convertible: unsupported instructions need engine work or must be unreachable in practice.")
    if any(k in needs for k in ("AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ")):
        verdict += " The program also has optional AVX-512 paths (usually guarded by CPUID)."
    lines += ["", f"**Verdict:** {verdict}"]
    text = "\n".join(lines)
    print(text)
    if a.report:
        Path(a.report).write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved {a.report}")
    return 0


def cmd_gate(a):
    results = detect_gates_detailed(Path(a.file))
    if not results:
        print("No AVX2/FMA/F16C startup check found next to an error message. The game may check CPUID elsewhere "
              "or not at all; try running the converted game and see whether it complains.")
        return 0
    for r in results:
        print(f'\nString: "{r["string"]}"')
        print(f'  Function {r["function"]} @ {r["rva"]}: {r["asm"]}    bytes: {r["bytes"]}')
        print(f'  Suggested patch: {json.dumps(r["suggested_patch"])}')
    return 0


def load_config(path: str | None) -> tuple[dict, Path]:
    if not path:
        return {}, Path.cwd()
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")), p.parent


def cmd_build(a):
    path = Path(a.file)
    out = Path(a.out)
    config, cdir = load_config(a.config)
    learned = learned_from_log(Path(a.learned), path.name) if a.learned else []

    enabled_features = set(TRANSLATED_FEATURES)
    if getattr(a, "features", None):
        enabled_features = {x.strip().upper() for x in a.features.split(",") if x.strip()}
    if getattr(a, "popcnt", False):
        enabled_features.add("POPCNT")

    include_popcnt = "POPCNT" in enabled_features
    sites, kind, report = choose_sites(path, config, cdir, learned, a.scan, include_popcnt=include_popcnt)
    patches = locate_patches(path, [p for p in config.get("patches", [])
                                    if p.get("target", path.name).lower() == path.name.lower()])
    for p in patches:
        print(f"Patch '{p['name']}' at {p['rva']:#x}: {p['old']} -> {p['new']}")
    print(f"{len(sites)} sites ({kind}). Features: {sorted(enabled_features)}. Building Ultra-Optimized Binary...", flush=True)
    manifest = build(
        path, out, sites, patches,
        enabled_features=enabled_features,
        enable_fusion=not getattr(a, "no_fusion", False),
        enable_neighbor_steal=not getattr(a, "no_neighbor_steal", False),
    )
    runtime = Path(a.runtime) if a.runtime else ROOT / "runtime" / "fxcompat.dll"
    if runtime.exists():
        shutil.copy2(runtime, out / "fxcompat.dll")
    else:
        print("Note: runtime/fxcompat.dll not found; compile runtime/fxcompat_static.cpp (see README).")
    print(f"Done: {out / path.name}  methods {manifest['methods']}  skipped {len(manifest['skipped'])}"
          f"  unwind refusals {len(manifest['unwind_refusals'])}")
    print("Put the converted file AND fxcompat.dll next to each other in the game folder.")
    return 0


def cmd_package(a):
    import package_mod
    return package_mod.package(Path(a.config), Path(a.out), Path(a.runtime) if a.runtime else None)


def cmd_gui(a):
    import fxkit_gui
    return fxkit_gui.launch(port=getattr(a, "port", 0))


def main():
    # If launched with no arguments (e.g. double-clicking launch_gui.cmd), open GUI directly!
    if len(sys.argv) == 1:
        import fxkit_gui
        return fxkit_gui.launch()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g_ui = sub.add_parser("gui", help="open the graphical user interface (GUI)")
    g_ui.add_argument("--port", type=int, default=0, help="local port (0 = auto)")

    s = sub.add_parser("scan", help="report how convertible an exe/dll is")
    s.add_argument("file")
    s.add_argument("--report", help="also save the report as Markdown")
    s.add_argument("--popcnt", action="store_true", help="also scan for POPCNT instructions")

    g = sub.add_parser("gate", help="find AVX2/FMA/F16C CPU checks and suggest patches")
    g.add_argument("file")

    b = sub.add_parser("build", help="convert an exe/dll with Ultra optimizations")
    b.add_argument("file")
    b.add_argument("-o", "--out", required=True, help="output folder (never the game folder)")
    b.add_argument("--config", help="game.json with patches and known builds")
    b.add_argument("--learned", help="fxreborn-missed.log from a test run, to add missed sites")
    b.add_argument("--scan", action="store_true", help="scan even if the build is in known_builds")
    b.add_argument("--runtime", help="fxcompat.dll to copy (default runtime/fxcompat.dll)")
    b.add_argument("--features", help="comma-separated features to translate (default: AVX2,BMI2,BMI1,FMA,F16C,LZCNT,MOVBE)")
    b.add_argument("--popcnt", action="store_true", help="also emulate POPCNT instructions via SWAR")
    b.add_argument("--no-fusion", action="store_true", help="disable multi-instruction basic-block fusion")
    b.add_argument("--no-neighbor-steal", action="store_true", help="disable 4-byte neighbor instruction stealing")

    p = sub.add_parser("package", help="make a player-ready mod folder + zip")
    p.add_argument("--config", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--runtime", help="fxcompat.dll to include (default runtime/fxcompat.dll)")

    a = ap.parse_args()
    print(f"FX Reborn Toolkit {VERSION}")
    return {
        "gui": cmd_gui,
        "scan": cmd_scan,
        "gate": cmd_gate,
        "build": cmd_build,
        "package": cmd_package,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())