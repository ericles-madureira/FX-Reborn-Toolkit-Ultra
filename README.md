# FX Reborn Toolkit Ultra 3.0

**Static binary translator for AVX2, FMA3, F16C, BMI1/2, LZCNT & MOVBE on legacy AVX CPUs**

[🇺🇸 English](#english) · [🇧🇷 Português](#português) · [📖 Guia completo em português](LEIA_ME_COMO_USAR.md)

> **Game won't start or crashes with *"illegal instruction"* (`0xC000001D`) on an AMD FX or Intel 2nd/3rd Gen CPU (`i5-2500K`, `i7-2600`, `i7-3770`)?**  
> This toolkit statically rewrites game `.exe` and `.dll` binaries so instructions from newer extensions (**AVX2, FMA3, F16C, BMI1/BMI2, LZCNT, MOVBE**) run natively on CPUs that only support **AVX**. Tested in **Final Fantasy VII Rebirth** on an i7-2600 (16 GB DDR3, RTX 2060 Super), achieving **~30 FPS average and 75–85 FPS in calmer areas**.
>
> **O jogo não abre ou fecha com *"instrução ilegal"* (`0xC000001D`) em um AMD FX ou Intel de 2ª/3ª geração (`i5-2500K`, `i7-2600`, `i7-3770`)?**  
> Este kit reescreve o `.exe`/`.dll` do jogo para que instruções **AVX2, FMA3, F16C, BMI1/BMI2, LZCNT e MOVBE** rodem em processadores que possuem apenas **AVX**. Resultado medido em **Final Fantasy VII Rebirth** (i7-2600, 16 GB DDR3, RTX 2060 Super): **média de ~30 FPS e 75–85 FPS em áreas mais leves**.

*The graphical interface is available in **English and Portuguese** (auto-detects your browser language or switch via the top **PT-BR / EN** buttons).*

---

## Português

### Como usar (Interface Gráfica)

1. Execute **`Open- Iniciar_Interface_GUI.cmd`** (não requer instalação — o Python 3.13 já vem incluído).
2. Na etapa **1**, selecione o `.exe` (ou `.dll`) do jogo e defina uma **pasta de saída diferente** da pasta original do jogo.
3. Na etapa **2**, escolha o *preset* do seu processador (**Intel 2ª geração**, **Intel 3ª geração** ou **AMD FX**).
4. Clique em **1. Analisar** para verificar quais instruções o jogo utiliza e, caso o jogo exiba um aviso bloqueando CPUs sem AVX2, clique em **2. Detectar Trava de CPU**.
5. Clique em **⚡ 3. Gerar Executável**, faça backup dos originais e copie o executável modificado **junto com o `fxcompat.dll`** para a pasta do jogo.

> [!TIP]
> **Seu jogo exige apenas F16C** (como *Ghost of Tsushima* em um i7-2600)? Utilize o **[FX-Reborn-F16C-Ultra](https://github.com/mcirish3/FX-Reborn-F16C-Ultra)**, que é mais simples e não precisa de DLL auxiliar.

Para instruções detalhadas, configurações avançadas e todas as otimizações da versão 3.0, consulte o **[Guia Completo em Português (`LEIA_ME_COMO_USAR.md`)](LEIA_ME_COMO_USAR.md)**.

> [!WARNING]
> **Não utilize em jogos online com anti-cheat** (risco de banimento). Compartilhe apenas esta ferramenta e seus arquivos de configuração — **nunca distribua executáveis ou DLLs de jogos**.

---

## English

**FX Reborn Toolkit** makes modern AVX2-dependent PC games run on CPUs that only support **AVX** (Intel Sandy Bridge / Ivy Bridge and AMD Bulldozer / Piledriver). It is the engine behind [FX Reborn](https://www.nexusmods.com/finalfantasy7rebirth/mods/2477), the *FINAL FANTASY VII REBIRTH* mod that took an AMD FX-8350 from **~1 FPS under Intel SDE emulation to 30–85 FPS**, packaged so modders can bring the same capability to other games.

The toolkit translates **AVX2, FMA3, F16C, BMI1/BMI2, LZCNT, and MOVBE** instructions ahead of time into equivalent AVX/SSE/x86-64 code inside the binary—with no background emulator and no per-instruction exception overhead at runtime.

### Key Features

- **Feasibility scan:** Counts unsupported instructions in an executable or DLL, flags untranslatable forms, and lists baseline CPU requirements.
- **CPU-check finder:** Locates startup feature gates (e.g., *"This CPU does not support AVX2"*) and generates byte patches to bypass them.
- **Update-proof mod packages:** End-user installers build the patched binary locally on their own machine using wildcard byte patterns rather than hardcoded addresses.
- **Runtime safety net (`fxcompat.dll`):** Any instruction missed during static discovery is caught by a Vectored Exception Handler, logged with its module and offset to `fxreborn-missed.log`, and automatically included on the next build.
- **Zero-install distribution:** Bundles a portable Python 3.13 runtime and launcher scripts so end users install nothing.

**Download:** [GitHub Releases](https://github.com/mcirish3/FX-Reborn-Toolkit/releases) or [Nexus Mods](https://www.nexusmods.com/finalfantasy7rebirth/mods/2499) *(includes portable Python 3.13)*.

### What's New in Ultra 3.0

Benchmarked on an Intel Core i7-2600 (Sandy Bridge, AVX-only) running the exact same test binary (`tests/selftest/bench.cpp`):

| Benchmark Workload | Ultra 2.0 | Ultra 3.0 |
| :--- | :---: | :---: |
| FMA accumulate, 4× `ymm` (`231`/`213`) | 9.3 ns/iter | **3.1 ns/iter** |
| FMA scalar `ss` (`231`) | 9.7 ns/iter | **2.3 ns/iter** |
| AVX2 integer (`vpmulld` / `vpaddd` / `vpslld`) | 12.6 ns/iter | **5.8 ns/iter** |
| Same AVX2 integer loop via VEH fallback | — | 1,753 ns/iter |

- **`CALL`/`RET` stub linkage:** Call sites invoke stubs via `CALL` and return via `RET`, allowing the CPU's Return Stack Buffer (RSB) to predict every exit. Version 2.x used `JMP` + `push [lit]; ret`, causing branch mispredictions on every call.
- **Shared constant pool:** Sign masks and F16C tables are stored once at the start of `.fxcpt` and accessed via RIP-relative addressing.
- **Full FMA3 support (`132`/`213`/`231`):** Exact operand semantics with zero stack spills when the destination is a multiplicand, and a compact 4–32 byte slot (instead of a full YMM register) when the destination is the addend.
- **Smarter lane splits:** Reuses the destination register as the primary scratch register whenever the original instruction does not read it.
- **Compact `VCVTPH2PS` lowering:** Reduced from 55/97 instructions down to **25 instructions (`xmm`) / 46 (`ymm`)** using pooled constants.
- **Bidirectional neighbor stealing:** Relocates adjacent RIP-relative and `[rsp+x]` instructions forward or backward to make room for 5-byte `CALL` instructions, drastically reducing slow VEH fallbacks on 4-byte sites.
- **Bugfixes from 2.x:**
  - Fixed packed `231` FMA computing `2*a*b` (`FMADD`) and `0` (`FMSUB`).
  - Fixed scalar `213` `FNMADD` with memory operands corrupting upper lanes, and zero-sign handling in `FNMSUB`/`FMSUBADD`.
  - Fixed negative half-precision values in `VCVTPH2PS`, RIP-relative `VCVTPH2PS`, and `VCVTPS2PH` sites previously left unpatched.
  - Fixed 16-bit `MOVBE` modifying `CF`/`OF` and `POPCNT` leaving `PF` set.

### Verification & Test Coverage

> [!IMPORTANT]
> Only **FINAL FANTASY VII REBIRTH** has been extensively play-tested so far. Furthermore, while the original FX Reborn v6 build was play-tested end-to-end, **Ultra 3.0's output is structurally new** (`CALL`/`RET` stubs, rewritten FMA/F16C lowerings, aggressive neighbor stealing) and has not yet completed a full in-game playthrough. Treat new titles as experimental.

- **37 AVX2 & BMI2 operations (`tests/selftest`):** Integer math, variable shifts, shuffles, permutes, gathers, pack/unpack, SAD, movemask, and `SHLX`/`SHRX`/`SARX`/`RORX`/`BZHI`/`PDEP`/`PEXT`/`MULX`. Verified on a real AMD FX-8350 across 3,000 random and edge-case inputs per operation against a plain-C reference, both as standalone `.exe` and loaded `.dll`.
- **FMA3 / BMI1 / LZCNT / MOVBE (`tests/selftest/fma_*.cpp`):** 20 kernels—including accumulate (`231`) and Horner (`213`) loops—validated across 4,000 inputs each with exact cancellation and signed-zero checks.
- **Native lowering suite (`tests/native`):** 11,005 test cases (~164,000 executions) plus a 38,400-case FMA register-aliasing sweep executed directly on AVX1 hardware from randomized register, flag, stack, and memory states, including forced backward neighbor steals, padding caves, and VEH fallbacks.
- **FF7 Rebirth site coverage (`examples/ff7rebirth`):** 23,171 translation sites analyzed.

```bat
:: Run lowering validation on any AVX CPU (~10 minutes)
python\python.exe tests\native\test_lowering.py

:: Build and run end-to-end self-tests with MinGW-w64 GCC (or use build.cmd with MSVC)
tests\selftest\build_gcc.cmd
```

### Requirements

- **OS:** Windows 64-bit (x64 executables and DLLs).
- **Python (for mod authors):** Bundled in the release download (Python 3.13). If running from source, install Python 3.13 with `iced-x86==1.22.0` and `pefile==2024.8.26` in `engine/pylibs`.
- **C++ Build Tools (optional):** Only required if recompiling `fxcompat.dll` from source; a prebuilt binary is included in `runtime/`.
- **End-user hardware:** Any CPU with **AVX** (all AVX CPUs also include baseline `SSE4.1`, `SSE4.2`, and `POPCNT`).

### Quick Start (CLI)

Always work on a backup copy of the game binaries rather than directly inside the game directory. Below, `fxkit` refers to `fxkit.cmd` (or `py -3.13 fxkit.py`).

```bat
:: 1. Scan the binary for feasibility and required features
fxkit scan "D:\copy\Game.exe" --report scan.md

:: 2. Locate startup CPU feature checks (e.g., AVX2 gates)
fxkit gate "D:\copy\Game.exe"

:: 3. Configure game.json (see schema below), then build and test
fxkit build "D:\copy\Game.exe" -o out --config game.json
:: Copy out\Game.exe and out\fxcompat.dll into a TEST copy of the game folder and launch.

:: 4. If an illegal instruction occurs at runtime, feed the log back into the build
fxkit build "D:\copy\Game.exe" -o out --config game.json --learned "D:\Game\fxreborn-missed.log"

:: 5. Package the mod for distribution
fxkit package --config game.json -o dist
:: Produces dist\<ModName>-<version>.zip
```

End users simply extract the `.zip` and run **`Install or Update.cmd`**, which locates the game via Steam, verifies their CPU capabilities, backs up original files, builds the patched binaries locally, and cleanly reapplies everything after game updates.

### `game.json` Configuration

```json
{
  "mod_name": "My Game AVX Fix",
  "mod_version": "1.0.0",
  "game_title": "My Game",
  "steam_folder": "My Game",
  "targets": [
    "Binaries/Win64/MyGame.exe",
    "Binaries/Win64/SomeEngine.dll"
  ],
  "process": "MyGame.exe",
  "cpu_features": ["AVX", "FMA", "F16C", "BMI1", "LZCNT"],
  "patches": [
    {
      "name": "Skip the AVX2 startup check",
      "target": "MyGame.exe",
      "find": "0f ba e0 08 72 ??",
      "replace": "0f ba e0 08 eb ??",
      "near_string": "does not support a required feature (AVX2)",
      "expect": 1
    }
  ],
  "known_builds": {
    "<sha256_of_original_exe>": {
      "label": "Steam build 12345",
      "sites": "known_sites_12345.json"
    }
  }
}
```

#### Top-Level Fields

| Field | Description |
| :--- | :--- |
| `steam_folder` | Directory name inside `steamapps\common`. |
| `targets` | Relative paths to executables and DLLs requiring translation (check each file with `fxkit scan`). |
| `cpu_features` | Baseline features verified on the player's CPU using `iced-x86` names (`AVX`, `SSE4_1`, `SSE4_2`, `POPCNT`, `AES`, etc.). |

#### Update-Proof Byte Patches (`patches`)

| Field | Description |
| :--- | :--- |
| `find` / `replace` | Hex byte patterns. Use `??` as a wildcard (in `replace`, `??` preserves the original byte). |
| `near_string` | Restricts pattern matching to functions referencing this ASCII or UTF-16 string. |
| `expect` | Exact number of expected matches. Aborts installation if the count differs, preventing bad patches on incompatible updates. |

**`known_builds` (optional):** Maps the SHA-256 hash of an unmodified binary to a verified site list (`{"sites": ["0x1234", ...]}`). Unrecognized game versions are automatically scanned on the user's machine and flagged as `UNVERIFIED`.

### Architecture

1. **Site Discovery (`engine/discover_code_sites.py`, `engine/find_sites.py`):** Traces control flow from the entry point, exports, and exception-table functions. Indirect/pointer-referenced targets are validated for clean decoding and jump-table exclusion. Raw linear-sweep candidates are never patched statically to avoid corrupting embedded data.
2. **Instruction Lowering (`engine/avx2_lower.py`, `engine/bmi2_lower.py`, `engine/bmi2_extra.py`, `engine/sandy_lower.py`):** Translates each unsupported instruction into an AVX/SSE/x86-64 sequence while preserving all non-destination registers and CPU flags.
3. **Static Binary Patching (`engine/build_static_mod.py`):**
   - Sites `>= 5` bytes are replaced with a `CALL` to a stub in a new `.fxcpt` section, ending with `RET`.
   - Shorter sites (`< 5` bytes) are rewritten inline if space permits, fused with adjacent sites, or expanded by relocating a neighboring instruction into the stub; remaining sites route through nearby padding or fall back to the runtime handler.
   - Generates Windows x64 unwind metadata for `.fxcpt` and injects an entry-point wrapper into patched DLLs to register them with the runtime.
4. **Runtime Handler (`runtime/fxcompat_static.cpp`):** A lightweight Vectored Exception Handler (VEH) that redirects only registered illegal-instruction addresses, logs any missed addresses to `fxreborn-missed.log`, and avoids global `CPUID` spoofing or external process injection.

### Limitations & Warnings

> [!WARNING]
> **Anti-cheat & online games:** Modifying game binaries will trigger anti-cheat or file-integrity protections and can result in account bans. Never use this toolkit on online multiplayer games.

- **Unsupported extensions:** **AVX-512** is not translated, and a few rare AVX2 forms remain unimplemented (reported by `fxkit scan`; extensible in `engine/avx2_lower.py`).
- **Performance overhead:** Translated routines are slower than native hardware execution (~1% CPU overhead in *FF7 Rebirth*, though titles heavily reliant on wide integer AVX2 loops will see greater impact).
- **DRM & obfuscation:** Binaries protected by aggressive self-checking DRM or packers may fail to launch after modification.
- **Distribution policy:** State clearly in your mod's README that it patches game executables. Distribute only your configuration files, site lists, and this toolkit—**never redistribute game binaries**.

### Credits & License

Licensed under the **MIT License** (see `LICENSE`).

- **Patrick McCoy (`mcirish3`):** Project direction, architecture, and hardware testing.
- **Code generation & tooling:** Built with AI assistance (OpenAI Codex for FX Reborn v1–v6; Anthropic Claude for the update-proof patcher, toolkit architecture, and extended BMI2 lowerings) and validated on an AMD FX-8350.
- **Dependencies:** [iced-x86](https://github.com/icedland/iced) (MIT), [pefile](https://github.com/erocarrera/pefile) (MIT), and Python 3.13 (PSF License).

### Related Projects

- **[FX-Reborn-F16C-Ultra 3.0](https://github.com/mcirish3/FX-Reborn-F16C-Ultra)** — Lightweight single-binary patcher for games that only require **F16C** (such as *Ghost of Tsushima* on Sandy Bridge).