# FX Reborn Toolkit

**Make AVX2-only PC games run on CPUs that only have AVX** (AMD FX, Piledriver-era APUs and similar). It's the engine behind [FX Reborn](https://www.nexusmods.com/finalfantasy7rebirth/mods/2477), the FINAL FANTASY VII REBIRTH mod that took an AMD FX-8350 from about **1 FPS under Intel SDE emulation to 30–85 FPS**, packaged so other modders can do the same for other games.

The toolkit rewrites a game's AVX2 and BMI2 instructions into equivalent AVX/SSE code, ahead of time, inside the exe or DLL. There's no emulator and no per-instruction exception at run time. It also gives you:

- **A feasibility scan:** how many AVX2/BMI2 instructions the game has, which ones can be translated, and what other CPU features players will still need.
- **CPU-check finder:** locates "This CPU does not support AVX2" checks and suggests a patch for them.
- **Update-proof patches:** players' installers build the patch on their own PC for whatever game version they have. Byte patches are matched by pattern, not by address.
- **A safety net:** instructions the scan missed are logged with their module and address when the game hits them, and added on the next install.
- **One-command player packages:** launchers, your config and a private Python, so players install nothing.

Nothing here is game-specific except the `game.json` you write.

**Download:** [GitHub Releases](https://github.com/mcirish3/FX-Reborn-Toolkit/releases) or [Nexus Mods](https://www.nexusmods.com/finalfantasy7rebirth/mods/2499). The download includes Python 3.13, so there's nothing to install.

## Tested

- **37 different AVX2/BMI2 operations** (`tests/selftest`): integer math, variable shifts, shuffles, permutes, gathers, pack/unpack, SAD, movemask, and SHLX/SHRX/SARX/RORX/BZHI/PDEP/PEXT/MULX.
  - The test program is compiled with AVX2, and its reference answers come from plain C compiled without AVX2.
  - On an FX-8350, which cannot run the original at all, every converted operation matches the reference over 3,000 random and edge-case inputs each.
  - This passed both as an **exe** and as a **DLL** loaded by an unmodified exe.
- **FMA3 / BMI1 / LZCNT / MOVBE** (`tests/selftest/fma_*.cpp`, Ultra 3.0): 20 kernels, including accumulate (231) and Horner (213) loops, checked against a plain-AVX reference over 4,000 inputs each, with exact cancellations and zero signs.
- **Native lowering tests** (`tests/native`, Ultra 3.0): every generated sequence is executed on the test machine (AVX1 is enough) from random register, flag, stack and memory states and compared bit for bit with a Python model of the original instruction: 11,005 cases, ~164,000 runs, plus an exhaustive 38,400-case FMA register-aliasing sweep. The rare site strategies (backward neighbor steal, padding cave, VEH) are also forced and run.
- **FINAL FANTASY VII REBIRTH** (`examples/ff7rebirth`): 23,171 sites.
  - The original FX Reborn v6 output was play-tested. **Ultra 3.0 output is no longer byte-identical to it** (CALL/RET stub linkage, rewritten FMA and F16C lowerings, more neighbor stealing) and has not been play-tested in the game yet.

Only one real game has been play-tested. Treat every new game as experimental.

## Ultra 3.0

Measured on an i7-2600 (Sandy Bridge, AVX only), same exe converted by Ultra 2.0 and 3.0 (`tests/selftest/bench.cpp`):

| Hot loop | 2.0 | 3.0 |
|---|---|---|
| FMA accumulate, 4x ymm (231/213) | 9.3 ns/iter | 3.1 ns/iter |
| FMA scalar ss (231) | 9.7 ns/iter | 2.3 ns/iter |
| AVX2 integer (vpmulld/vpaddd/vpslld) | 12.6 ns/iter | 5.8 ns/iter |
| same loop, site forced through the VEH handler | | 1,753 ns/iter |

- **CALL/RET stub linkage.** Sites call their stub and the stub returns normally, so every exit is predicted by the return-stack buffer. 2.x jumped in and left through `push [lit]; ret`, which mispredicts on every execution.
- **Constant pool** at the start of `.fxcpt` (sign masks, F16C constants), read RIP-relative.
- **FMA3**: exact operand semantics for every 132/213/231 form; zero scratch when the destination is a multiplicand, a 4-32 byte stack slot (not a whole YMM) when it is the addend.
- **Lane splits** use the destination as the first scratch when the instruction does not read it.
- **VCVTPH2PS** with pooled constants: 25 instructions (xmm) / 46 (ymm) instead of 55 / 97.
- **Neighbor stealing** forward and backward, relocating RIP-relative and `[rsp+x]` neighbors, so fewer 4-byte sites fall back to the VEH handler.
- **Fixed 2.x bugs:** packed 231 FMA computed `2*a*b` (FMADD) and `0` (FMSUB); scalar 213 FNMADD with memory corrupted upper lanes; FNMSUB/FMSUBADD zero signs; every negative half in VCVTPH2PS; VCVTPH2PS RIP-relative and all VCVTPS2PH sites failed to lower (left as illegal instructions); 16-bit MOVBE changed CF/OF; POPCNT left PF set.

Run the checks:

```bat
:: lowering tests (any AVX CPU), about 10 minutes
python\python.exe tests\native\test_lowering.py
:: end-to-end self-tests with MinGW-w64 GCC (or build.cmd with MSVC)
tests\selftest\build_gcc.cmd
```

## Requirements

- **Windows x64.**
- **Python:** the toolkit download includes Python 3.13. Running from source needs Python 3.13 plus `iced-x86==1.22.0` and `pefile==2024.8.26` in `engine/pylibs`.
- **Rebuilding `fxcompat.dll`:** needs the Visual Studio C++ build tools. A prebuilt copy is in `runtime/`.
- **Players' CPUs** need AVX, plus whatever else the game uses natively (FMA3, F16C, BMI1 and so on). The scan lists these.

## Quick start

Work on a copy of the game file, never inside the game folder. `fxkit` below means `fxkit.cmd`, or `py -3.13 fxkit.py`.

```bat
:: 1. Is it feasible?
fxkit scan "D:\copy\Game.exe" --report scan.md

:: 2. Does the game refuse to start on non-AVX2 CPUs? Find the check.
fxkit gate "D:\copy\Game.exe"

:: 3. Write game.json (see below), then convert and test.
fxkit build "D:\copy\Game.exe" -o out --config game.json
::    copy out\Game.exe and out\fxcompat.dll into a TEST install of the game and play.

:: 4. If it closes with an illegal-instruction error, the miss is in fxreborn-missed.log:
fxkit build "D:\copy\Game.exe" -o out --config game.json --learned "D:\Game\fxreborn-missed.log"

:: 5. When it works, freeze the tested site list as a known build (optional), then:
fxkit package --config game.json -o dist
::    -> dist\<ModName>-<version>.zip, ready to upload.
```

Players unzip the package and run `Install or Update.cmd`, which:
1. finds the game through Steam,
2. checks their CPU,
3. backs up the originals,
4. builds the patch on their PC,
5. and, after a game update, does all of that again when re-run.

## `game.json`

```json
{
  "mod_name": "My Game AVX Fix",
  "mod_version": "1.0.0",
  "game_title": "My Game",
  "steam_folder": "My Game",
  "targets": ["Binaries/Win64/MyGame.exe", "Binaries/Win64/SomeEngine.dll"],
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
    "<sha256 of the ORIGINAL exe>": { "label": "Steam build 12345", "sites": "known_sites_12345.json" }
  }
}
```

**Top-level fields:**

| Field | What it is |
|---|---|
| `steam_folder` | The folder name under `steamapps\common`. |
| `targets` | Files to convert, relative to that folder. Add DLLs that contain AVX2 code; the scan of each one tells you. |
| `cpu_features` | What the installer requires from the player's CPU. Use the scan's "Other CPU features" list. Names are iced-x86's: `AVX`, `FMA`, `F16C`, `BMI1`, `LZCNT`, `POPCNT`, `SSE4_1`, `SSE4_2`, `MOVBE`, `AES` and so on. |

**`patches`** are byte patches that survive updates:

| Field | What it is |
|---|---|
| `find` / `replace` | Hex bytes. `??` is a wildcard, and in `replace` it keeps the original byte. |
| `near_string` | Only searches inside functions that reference this text, as ASCII or UTF-16. |
| `expect` | The required number of matches. Anything else stops the install, so a changed game is never patched in the wrong place. |

**`known_builds`** is optional. It maps the hash of an original file to a frozen, play-tested site list: `{"sites": ["0x1234", ...]}`, for example taken from a build manifest's `records`. Any other version is scanned on the player's PC and labelled UNVERIFIED.

## How it works

1. **Find sites** (`engine/discover_code_sites.py`, `engine/find_sites.py`):
   - Walks the program's control flow from its entry point, exports and exception-table functions.
   - Code reached only through pointers is accepted after an automatic review: the reference must really point there, the code must decode cleanly, and the spot must not look like a jump table.
   - Bytes found only by a linear sweep are never patched, because they may be data.
2. **Lower** (`engine/avx2_lower.py`, `engine/bmi2_lower.py`, `engine/bmi2_extra.py`, `engine/sandy_lower.py`): each instruction becomes an AVX/SSE/x86-64 sequence that preserves every other register and the flags.
3. **Patch** (`engine/build_static_mod.py`):
   - Sites of 5 bytes or more `call` their replacement code in a new `.fxcpt` section, which returns with `ret`.
   - Shorter ones are rewritten in place when the replacement fits.
   - Otherwise they are fused with an adjacent site, or the next or previous ordinary instruction is moved into the stub so a `call` fits.
   - Otherwise they jump through nearby padding.
   - Otherwise they're left in place, and `fxcompat.dll`'s exception handler redirects them.
   - Unwind data is generated for the new code.
   - For DLLs, a small entry-point wrapper registers the module with the runtime.
4. **Runtime** (`runtime/fxcompat_static.cpp`):
   - A vectored exception handler that only redirects registered illegal-instruction addresses.
   - It logs unregistered ones to `fxreborn-missed.log`.
   - It doesn't spoof CPUID or inject into other processes.

## Limits and warnings

- **AVX-512 isn't translated.** Some AVX2 forms aren't supported yet; the scan lists any it finds, and `engine/avx2_lower.py` is where to add them.
- **Translated code is slower than native AVX2.** On FX Reborn it was about 1% of CPU time, but games that lean heavily on wide integer AVX2 will notice more.
- **Online games and anti-cheat:** modified executables can trip anti-cheat or integrity checks and get accounts banned. Don't use this on games with anti-cheat or online-only components unless the developer allows it.
- **DRM-wrapped executables** may need the DRM's own requirements respected, or may not be convertible at all.
- **Each mod's README should say** that it modifies the game exe. Distribute only your config, site lists and the toolkit, never game files.

## Credits and license

MIT (see `LICENSE`).

- **Patrick McCoy (mcirish3):** project direction and testing.
- **Code** was largely AI-generated (OpenAI Codex for FX Reborn v1–v6; Anthropic Claude for the update-proof patcher, this toolkit and the extra BMI2 lowerings), then tested locally on an FX-8350.
- **Libraries:** iced-x86 and pefile (MIT). Python 3.13 (PSF license) is included in the release download.
