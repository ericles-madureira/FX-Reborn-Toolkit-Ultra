# FX Reborn Toolkit Ultra v2.0 — Documentação Completa e Guia de Uso

Bem-vindo ao **FX Reborn Toolkit Ultra v2.0**, a evolução completa e ultra-otimizada do reescritor estático e emulador *Ahead-of-Time (AOT)* de instruções de CPU para jogos e aplicativos Windows x64.

Este software analisa executáveis (`.exe`) e bibliotecas (`.dll`), encontra todas as instruções **AVX2 (principalmente)**, **FMA3**, **F16C**, **BMI2**, **BMI1**, **LZCNT**, **MOVBE** e **POPCNT**, e cria um novo executável modificado onde essas instruções são substituídas ou emuladas diretamente em código nativo **AVX1 / SSE4.2 / x86-64**, sem precisar de emuladores lentos em tempo de execução como o Intel SDE.

## 1. O Que Foi Otimizado Nesta Versão (v2.0 Ultra vs. v0.1.1)

1. **Motor FMA3 Zero-Scratch (Sem Alocação de Pilha em >82% dos Locais):**

   * Na versão anterior, **toda** instrução FMA (`VFMADD`, `VFMSUB`, `VFNMADD`, etc. — mais de 134.000 locais no *FF7 Rebirth*) alocava `96 bytes` na pilha (`RSP`), salvava 3 registradores `YMM` inteiros na memória (`3x VMOVDQU`), fazia a conta e lia os 3 registradores de volta (`3x VMOVDQU`), gerando 6 acessos pesados à memória por instrução.

   * Na versão **Ultra v2.0**, quando o registrador de destino não sobrescreve o operando aditivo antes da soma/subtração (mais de **109.400 locais** no *FF7 Rebirth*), o motor emite apenas as **2 instruções nativas AVX1** (`VMULPS/PD/SS/SD` + `VADDPS/VSUBPS/VADDSUBPS`) diretamente nos registradores originais, com **zero uso de pilha e zero leituras/escritas de memória**. Nos casos restantes onde há colisão de registrador, salva apenas **1 registrador YMM (32B)** em vez de 3 (96B).

2. **Motor AVX2 In-Place Direto (`VPBROADCASTD`, `VPBROADCASTQ`, `VPCMPEQ` All-Ones, `VPBROADCASTW/B`, `VPERMQ`):**

   * **`VPBROADCASTD ymm, m32` e `VPBROADCASTQ ymm, m64`:** Mapeados 1:1 para `VBROADCASTSS ymm, m32` e `VBROADCASTSD ymm, m64` do AVX1, cabendo **100% In-Place** no mesmo tamanho de bytes da instrução original (sem salto para `.fxcpt` e sem usar pilha).

   * **`VPCMPEQB/W/D/Q ymmX, ymmX, ymmX` (Geração de Máscara toda 1s):** Substituído por uma única instrução AVX1 de 5 bytes (`VCMPTRUE_USPS ymmX, ymmX, ymmX`, imediato `0x0F`), cabendo **In-Place** nos locais de 5 bytes sem tocar na pilha.

   * **`VPBROADCASTW` e `VPBROADCASTB`:** Reescritos com `VPSHUFLW` + `VPSHUFD` sem usar registradores temporários nem pilha.

3. **Motor BMI1 `ANDN` In-Place (Zero Salto):**

   * Quando `dst == src1` e `dst != src2` (mais de 2.200 locais no *FF7 Rebirth*), emite `NOT dst; AND dst, src2` (4 a 5 bytes), cabendo **In-Place** dentro dos 5 bytes da instrução original sem desvio para `.fxcpt` e preservando as flags arquiteturais exatas (`OF=0, CF=0, SF, ZF`). Quando `dst != src1` e `dst != src2`, emite `MOV dst, src1; NOT dst; AND dst, src2` sem usar a pilha.

4. **Motor F16C (`VCVTPH2PS`) Vetorizado em SIMD Branchless:**

   * Substitui o antigo loop escalar de 8 iterações (que salvava 7 registradores gerais + `PUSHFQ`/`POPFQ` e executava dezenas de saltos condicionais) por uma conversão **100% vetorizada em SIMD SSE4.1/AVX1** (`VPMOVZXWD`, `VPSLLD`, `VPSRLD`, `VPCMPEQD`, `VPADDD`, `VSUBPS`, `VBLENDVPS`), matematicamente exata para todos os 65.536 valores FP16 (incluindo subnormais, `±0`, `±Inf` e `NaN`) e cerca de **15x mais rápida**.

5. **Fusão de Básico-Blocos (*Multi-Instruction Cluster Fusion*):**

   * Agrupa sequências de instruções alvo consecutivas (até 16 instruções seguidas que não cruzam alvos de salto) em **um único salto `JMP rel32`** para um bloco combinado na seção `.fxcpt`.

   * Elimina dezenas de milhares de saltos repetidos de ida e volta e resolve **2.704 das 3.299 instruções AVX2 de 4 bytes** sem precisar do *Vectored Exception Handler (VEH)*!

6. **Roubo Seguro de Instrução Vizinha (*4-Byte Neighbor Stealing*):**

   * Para instruções AVX2 de 4 bytes isoladas dentro de funções mapeadas em `.pdata` onde não há *code cave* (`INT3`/`NOP`) próximo, incorpora automaticamente a instrução comum seguinte (`rva + 4`) dentro do stub em `.fxcpt`, abrindo 8+ bytes para aplicar um `JMP rel32` nativo em vez de cair na exceção lenta `EXCEPTION_ILLEGAL_INSTRUCTION` do Windows.

7. **Descoberta Completa de Instruções Legacy + VEX (`discover_code_sites.py`) e Emulação de `POPCNT`:**

   * Corrige o filtro do descobridor para capturar tanto instruções codificadas com prefixo `VEX` quanto instruções codificadas como `LEGACY` (`TZCNT`, `LZCNT`, `MOVBE` e opcionalmente `POPCNT` via algoritmo paralelo *SWAR*).

## 2. Como Executar e Usar (Passo a Passo)

O pacote já inclui o **Python 3.13 portátil embarcado** na pasta `python/`, além das bibliotecas `iced-x86` e `pefile` pré-instaladas em `engine/pylibs/` e a biblioteca de tempo de execução `runtime/fxcompat.dll` pré-compilada. **Você não precisa instalar nada no seu Windows.**

### Método A: Usando a Interface Gráfica (Recomendado)

1. Extraia o arquivo `FX-Reborn-Toolkit-Ultra-2.0.zip` em uma pasta de sua preferência (ex.: `C:\FX-Reborn-Toolkit-Ultra`).

2. Dê um **duplo-clique** no arquivo **`Iniciar_Interface_GUI.cmd`**.

3. Uma janela gráfica em **Modo Escuro (Dark Mode)** será aberta automaticamente mostrando os recursos que o processador do seu PC possui no topo da tela.

4. No passo **1. Selecionar Executável / DLL e Pasta de Saída**:

   * Clique em **Procurar .EXE/.DLL** e selecione uma **cópia** do executável ou DLL do jogo (ex.: `ff7rebirth_.exe` ou `Game.exe`).

   * Clique em **Procurar Pasta** (ou mantenha a pasta sugerida `saida_ultra`) para definir onde o novo executável otimizado será salvo.

5. No passo **2. Conjuntos de Instruções para Simular / Emular (AOT)**:

   * Escolha um dos **Presets** rápidos no topo (*Intel 2ª Geração Sandy Bridge i7-2600*, *Intel 3ª Geração Ivy Bridge i7-3770*, *AMD FX-8350*) ou marque manualmente as caixas **AVX2**, **FMA3**, **F16C**, **BMI2**, **BMI1**, **LZCNT**, **MOVBE** e **POPCNT**.

6. No passo **3. Otimizações de Desempenho Ultra**:

   * Mantenha ativadas as opções **Fusão de Básico-Blocos (Cluster Fusion)** e **Roubo Seguro de Vizinho 4-Bytes (Zero-VEH)** para máximo FPS.

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