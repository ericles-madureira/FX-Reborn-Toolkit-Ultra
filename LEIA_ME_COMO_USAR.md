# FX Reborn Toolkit Ultra v3.0 — Documentação Completa e Guia de Uso

Bem-vindo ao **FX Reborn Toolkit Ultra v3.0**, o reescritor estático *Ahead-of-Time (AOT)* de instruções de CPU para jogos e aplicativos Windows x64.

Este software analisa executáveis (`.exe`) e bibliotecas (`.dll`), encontra todas as instruções **AVX2 (principalmente)**, **FMA3**, **F16C**, **BMI2**, **BMI1**, **LZCNT**, **MOVBE** e **POPCNT**, e cria um novo executável modificado onde essas instruções são substituídas por código nativo **AVX1 / SSE4.2 / x86-64**, sem emuladores lentos em tempo de execução como o Intel SDE.

## 1. O Que Mudou na v3.0 (vs. v2.0)

### 1.1 Desempenho medido (i7-2600, mesmo `.exe` convertido pelo motor 2.0 e pelo 3.0)

| Loop quente (`tests/selftest/bench.cpp`) | v2.0 | v3.0 | Ganho |
|---|---|---|---|
| FMA acumulação, 4× YMM (formas 231/213) | 9,3 ns/iter | 3,1 ns/iter | **~3,0×** |
| FMA escalar `ss` (forma 231) | 9,7 ns/iter | 2,3 ns/iter | **~4,2×** |
| AVX2 inteiro (`vpmulld`/`vpaddd`/`vpslld`) | 12,6 ns/iter | 5,8 ns/iter | **~2,2×** |

O ganho real em FPS depende de quanto tempo o jogo passa nesses trechos; não foi medido em jogo.

### 1.2 Otimizações novas

1. **Stubs ligados por `CALL`/`RET` (previsão de retorno perfeita):** na v2.0 cada site desviava com `JMP` e o stub saía com `PUSH [endereço]` + `RET` — um `RET` sem `CALL` correspondente, que o preditor de retorno da CPU **erra em toda execução** (~15-20 ciclos cada). Agora o site faz `CALL stub` e o stub termina em `RET` normal. O endereço de retorno fica dentro da função original, então o *unwind* (exceções C++, crash dumps) continua correto.
2. **Pool de constantes na seção `.fxcpt`:** máscaras de sinal e constantes do F16C ficam numa área somente-leitura e são lidas com endereçamento RIP-relativo, em vez de serem recriadas com várias instruções a cada execução.
3. **FMA3 exato e mais enxuto:** quando o destino é multiplicando (formas 132/213), 2-3 instruções sem pilha; quando o destino é a parcela somada (forma 231, o "`acc += a*b`" de laços), a parcela vai para um slot de 4 a 32 bytes na pilha em vez de salvar um YMM inteiro. Em 576 formas representativas: **−40% instruções, −67% acessos de spill à memória**.
4. **Split de lanes AVX2 usando o destino como rascunho:** quando a instrução não lê o próprio destino, ele guarda a metade alta — um YMM a menos salvo/restaurado por execução.
5. **`VCVTPH2PS` (F16C) com constantes do pool:** 55 → 25 instruções (xmm) e 97 → 46 (ymm), 3-4 YMM de rascunho em vez de 5.
6. **Roubo de vizinho para frente E para trás:** sites AVX2 de 4 bytes agora também podem incorporar a instrução *anterior*, e vizinhos RIP-relativos ou `[rsp+x]` são realocados. Menos sites caem no VEH — que é **~300× mais lento** que um stub (medido: 5,8 ns vs 1.753 ns por iteração).
7. **NOPs multi-byte** no preenchimento (uma instrução decodificada em vez de vários `0x90`).

### 1.3 Correções de bugs da v2.0 (podem causar erros visuais/lógicos em jogo)

* **FMA 231 empacotado** (`vfmadd231ps acc, a, b` etc.): a v2.0 calculava `2·a·b` em vez de `acc + a·b`, e `vfmsub231` devolvia **zero**.
* **FMA escalar 213 com memória** (`vfnmadd213ss`): corrompia os bits altos do registrador.
* **FNMSUB / FMSUBADD:** sinal do zero errado em cancelamentos exatos (`-0` em vez de `+0`).
* **`VCVTPH2PS`:** todo half-float **negativo** era convertido errado (o bit de sinal caía no expoente).
* **`VCVTPH2PS` com operando RIP-relativo** e **`VCVTPS2PH`** inteiro: a geração falhava e o site era pulado — o jogo cairia com instrução ilegal.
* **`MOVBE` 16 bits** alterava as flags CF/OF; **`POPCNT`** emulado deixava PF errado.

### 1.4 Como cada caminho é verificado

* `tests/native/test_lowering.py` executa **cada sequência gerada no próprio processador** (basta AVX1) a partir de estados aleatórios de registradores, flags, pilha e memória, e compara bit a bit com um modelo de referência: **11.005 casos / ~164 mil execuções, 0 falhas** (mais uma varredura FMA exaustiva de 38.400 casos).
* `tests/selftest` compila kernels reais com AVX2/FMA/BMI (MSVC ou MinGW-GCC), converte e confere contra uma referência escalar: **37/37 AVX2/BMI2** (exe e DLL) e **20/20 FMA/BMI1/LZCNT/MOVBE** passam no i7-2600, inclusive forçando os caminhos raros (roubo para trás, caverna de padding e VEH).

### 1.5 Otimizações herdadas da v2.0 (continuam ativas)

* `VPBROADCASTD/Q` de memória → `VBROADCASTSS/SD`/`VMOVDDUP` no mesmo tamanho (in-place); `VPCMPEQ x,y,y` → `VCMPPS` TRUE (in-place); `VPBROADCASTW/B` sem pilha; `VPERMQ` com lanes iguais → `VPERM2F128` + `VSHUFPD`.
* `ANDN` in-place (`NOT` + `AND`) quando `dst == src1`.
* Fusão de blocos: até 16 instruções consecutivas num único stub.
* Descoberta de instruções VEX e legacy (`TZCNT`, `LZCNT`, `MOVBE`, `POPCNT` opcional).

## 2. Como Executar e Usar (Passo a Passo)

O pacote já inclui o **Python 3.13 portátil embarcado** na pasta `python/`, além das bibliotecas `iced-x86` e `pefile` pré-instaladas em `engine/pylibs/` e a biblioteca de tempo de execução `runtime/fxcompat.dll` pré-compilada. **Você não precisa instalar nada no seu Windows.**

### Método A: Usando a Interface Gráfica (Recomendado)

1. Extraia o arquivo `FX-Reborn-Toolkit-Ultra-3.0.zip` em uma pasta de sua preferência (ex.: `C:\FX-Reborn-Toolkit-Ultra`).

2. Dê um **duplo-clique** no arquivo **`Open- Iniciar_Interface_GUI.cmd`**.

3. Uma janela gráfica em **Modo Escuro (Dark Mode)** será aberta automaticamente mostrando os recursos que o processador do seu PC possui no topo da tela.

4. No passo **1. Selecionar Executável / DLL e Pasta de Saída**:

   * Clique em **Procurar .EXE/.DLL** e selecione uma **cópia** do executável ou DLL do jogo (ex.: `ff7rebirth_.exe` ou `Game.exe`).

   * Clique em **Procurar Pasta** (ou mantenha a pasta sugerida `saida_ultra`) para definir onde o novo executável otimizado será salvo.

5. No passo **2. Conjuntos de Instruções para Simular / Emular (AOT)**:

   * Escolha um dos **Presets** rápidos no topo (*Intel 2ª Geração Sandy Bridge i7-2600*, *Intel 3ª Geração Ivy Bridge i7-3770*, *AMD FX-8350*) ou marque manualmente as caixas **AVX2**, **FMA3**, **F16C**, **BMI2**, **BMI1**, **LZCNT**, **MOVBE** e **POPCNT**.

6. No passo **3. Otimizações de Desempenho Ultra**:

   * Mantenha ativadas as opções **Fusão de Básico-Blocos (Cluster Fusion)** e **Roubo Seguro de Vizinho 4-Bytes (Zero-VEH)** para máximo FPS.

   * **Importante:** reconverta o jogo com a v3.0 a partir do executável **original**. Executáveis gerados pela v2.0 contêm os bugs de FMA/F16C listados acima.

7. Clique em:

   * **1. Analisar Executável (Scan):** se quiser ver antes quantas instruções de cada tipo existem dentro do arquivo.

   * **2. Detectar Trava de CPU (Gate):** se quiser verificar se o jogo possui mensagens do tipo *"This CPU does not support AVX2"* e obter o patch automático.

   * **⚡ 3. Gerar Executável Modificado & Otimizado:** para construir o novo executável e copiar a `fxcompat.dll` para a pasta de saída escolhida.

8. Quando o console mostrar **SUCESSO!**, faça backup do executável original na pasta do jogo e copie o **novo executável gerado** junto com o arquivo **`fxcompat.dll`** para a pasta do jogo.

### Método B: Usando por Linha de Comando (CLI)

Se preferir usar pelo Prompt de Comando ou PowerShell:

```
:: 1. Abrir a interface gráfica via comando
fxkit.cmd gui

:: 2. Escanear um executável e salvar relatório detalhado em Markdown
fxkit.cmd scan "D:\Backup\Game.exe" --report relatorio_scan.md

:: 3. Procurar travas de verificação de CPU ("CPU does not support AVX2")
fxkit.cmd gate "D:\Backup\Game.exe"

:: 4. Construir o novo executável ultra-otimizado (AVX2 + FMA + F16C + BMI1/2 + LZCNT + MOVBE)
fxkit.cmd build "D:\Backup\Game.exe" -o "D:\Saida_Ultra" --config examples\ff7rebirth\game.json

:: 5. Construir incluindo também emulação de POPCNT (para CPUs muito antigas)
fxkit.cmd build "D:\Backup\Game.exe" -o "D:\Saida_Ultra" --popcnt

:: 6. Reconstruir incorporando endereços extras caso o jogo tenha gerado fxreborn-missed.log
fxkit.cmd build "D:\Backup\Game.exe" -o "D:\Saida_Ultra" --learned "D:\Jogo\fxreborn-missed.log"

```

## 3. Testes (para quem for contribuir no GitHub)

```
:: Testes nativos de cada sequência gerada (qualquer CPU com AVX, ~10 minutos)
python\python.exe tests\native\test_lowering.py

:: Varredura FMA exaustiva (todas as combinações de registradores, ~25 minutos)
python\python.exe tests\native\test_lowering.py fma-full

:: Selftests ponta-a-ponta com MinGW-w64 GCC (MSYS2) ou build.cmd com Visual Studio
tests\selftest\build_gcc.cmd
```

O `build_gcc.cmd` compila os kernels AVX2/BMI2 e FMA/BMI1, converte com o `fxkit`, roda os executáveis convertidos e o benchmark. Num PC sem AVX2 o executável original fecha com "instrução ilegal" e o convertido deve imprimir `ALL ... PASS`.
