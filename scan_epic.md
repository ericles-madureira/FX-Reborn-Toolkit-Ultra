# FX Reborn Toolkit scan: ff7rebirth_original.exe

- AVX2/BMI2 instructions found (control flow + reviewed pointer targets): **23171**
- Translatable by the engine: **23171**; not yet supported: **0**
- Byte-sweep-only candidates ignored (may be data): 0
- Pointer-reached groups rejected by the automatic review: 0

## Other CPU features this program uses (players' CPUs must have these)

AVX (1904750), AVX512BW (1766), AVX512DQ (3212), AVX512F (17173), AVX512VL (16301), BMI1 (12877), D3NOW (1), F16C (136), FMA (135480), FPU387 (10), LZCNT (18), MOVBE (3), POPCNT (70), RTM (1), SSE4_1 (18757), SSE4_2 (3), SSSE3 (1459), VMX (1), WAITPKG (1)

Translated code itself uses: AVX

## AVX2/BMI2 instructions by type

| instruction | count |
|---|---:|
| shlx | 6493 |
| vbroadcastss | 2287 |
| vextracti128 | 1599 |
| shrx | 1437 |
| vpaddq | 1193 |
| vpaddd | 961 |
| vpmovsxdq | 728 |
| vpand | 679 |
| vpcmpeqd | 517 |
| vpbroadcastd | 466 |
| vpaddw | 435 |
| vpbroadcastq | 375 |
| vpsubw | 310 |
| vpcmpgtd | 302 |
| vgatherqps | 264 |
| vpmovzxwd | 261 |
| vpblendd | 254 |
| vpandn | 238 |
| vpmaddwd | 213 |
| vpsubd | 211 |
| vpabsw | 211 |
| vpxor | 200 |
| vpmovzxbd | 175 |
| vpmovsxwd | 168 |
| vpshufb | 163 |
| vpsrld | 128 |
| vpsrad | 127 |
| vpslld | 126 |
| vpsadbw | 116 |
| vpermd | 115 |
| vpor | 109 |
| vinserti128 | 107 |
| vperm2i128 | 107 |
| vpmaddubsw | 102 |
| vpavgb | 102 |
| vpmulld | 94 |
| vpavgw | 86 |
| sarx | 78 |
| vpgatherdd | 76 |
| vpermq | 74 |
| vpunpcklbw | 73 |
| vpunpckhbw | 69 |
| vpsrlq | 65 |
| vpbroadcastw | 64 |
| vpermps | 64 |
| vpunpcklwd | 62 |
| vpminsd | 60 |
| vpunpckhwd | 59 |
| vpsraw | 55 |
| vpsrldq | 54 |
| vpmaxsd | 53 |
| vbroadcastsd | 50 |
| vpgatherqd | 44 |
| vpbroadcastb | 42 |
| vpackuswb | 40 |
| vpmaxsw | 40 |
| vpmulhw | 38 |
| bzhi | 34 |
| vpsrlw | 31 |
| vpackssdw | 31 |
| vpaddsw | 30 |
| vphaddd | 30 |
| vpackusdw | 28 |
| vpabsd | 24 |
| vpmuldq | 24 |
| vpsignd | 24 |
| vpminsw | 22 |
| vpcmpgtw | 20 |
| vpmovzxdq | 18 |
| vpunpckhqdq | 18 |
| vpmovmskb | 18 |
| vpsllq | 17 |
| vpsubb | 16 |
| vbroadcasti128 | 15 |
| vpunpckldq | 14 |
| vpshufd | 14 |
| vpmovzxbw | 14 |
| vpmullw | 14 |
| vgatherdps | 12 |
| vpshuflw | 12 |
| vpsignw | 12 |
| vpunpckhdq | 11 |
| vpmuludq | 10 |
| vpunpcklqdq | 8 |
| vpsllvd | 7 |
| vpsllw | 5 |
| vpsravd | 4 |
| vpcmpeqb | 4 |
| vpmaskmovd | 3 |
| vpsrlvd | 2 |
| vpblendvb | 2 |
| vphaddsw | 2 |
| vpminub | 1 |
| vpacksswb | 1 |

**Verdict:** Looks convertible. The program also uses AVX-512, which this toolkit does not translate.
