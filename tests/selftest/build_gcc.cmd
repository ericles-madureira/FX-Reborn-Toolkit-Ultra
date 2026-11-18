@echo off
rem End-to-end self-tests with MinGW-w64 GCC (for example MSYS2 ucrt64: C:\msys64\ucrt64\bin).
rem Builds AVX2/BMI2 and FMA/BMI1 kernels for a CPU that HAS those features, converts the
rem programs with fxkit, then runs the converted copies (they must print "ALL ... PASS").
setlocal
cd /d "%~dp0"
if not defined GCCBIN if exist C:\msys64\ucrt64\bin\g++.exe set GCCBIN=C:\msys64\ucrt64\bin
if defined GCCBIN set PATH=%GCCBIN%;%PATH%
where g++ >nul 2>nul || (echo g++ not found. Install MSYS2 MinGW-w64 or set GCCBIN. & exit /b 2)
set PY=..\..\python\python.exe
set FX=..\..\fxkit.py
if not exist bin mkdir bin

g++ -O2 -mavx2 -mbmi2 -mbmi -include gcc_compat.h -c kernels.cpp -o bin\kernels.o || exit /b 1
g++ -O2 -mavx -c reference.cpp -o bin\reference.o || exit /b 1
g++ -O2 -mavx -c main.cpp -o bin\main.o 2>nul || exit /b 1
g++ -static -o bin\selftest.exe bin\main.o bin\kernels.o bin\reference.o || exit /b 1

g++ -O2 -mavx2 -mfma -mbmi -mbmi2 -mlzcnt -mmovbe -c fma_kernels.cpp -o bin\fma_kernels.o || exit /b 1
g++ -O2 -mavx -ffp-contract=off -c fma_reference.cpp -o bin\fma_reference.o || exit /b 1
g++ -O2 -mavx -ffp-contract=off -c fma_main.cpp -o bin\fma_main.o || exit /b 1
g++ -static -o bin\fmatest.exe bin\fma_main.o bin\fma_kernels.o bin\fma_reference.o || exit /b 1

g++ -O2 -mavx2 -mfma -static bench.cpp -o bin\bench.exe || exit /b 1

g++ -O2 -mavx2 -mbmi2 -mbmi -include gcc_compat.h -DKERNELS_DLL -shared -static -o bin\kernels.dll kernels.cpp -Wl,--out-implib,bin\libkernels.a || exit /b 1
g++ -O2 -mavx -DKERNELS_IMPORT -static -o bin\selftest_dll.exe main.cpp reference.cpp bin\libkernels.a 2>nul || exit /b 1

set FAIL=0
for %%T in (selftest fmatest bench) do (
  "%PY%" "%FX%" build bin\%%T.exe -o bin\converted_%%T >bin\%%T.build.log 2>&1 || (echo build of %%T FAILED, see bin\%%T.build.log & set FAIL=1)
)
"%PY%" "%FX%" build bin\kernels.dll -o bin\converted_dll >bin\kernels.build.log 2>&1 || (echo build of kernels.dll FAILED & set FAIL=1)
copy /y bin\selftest_dll.exe bin\converted_dll\ >nul

echo.
echo === AVX2/BMI2 self-test (exe) ===
bin\converted_selftest\selftest.exe || set FAIL=1
echo.
echo === AVX2/BMI2 self-test (dll loaded by an unmodified exe) ===
bin\converted_dll\selftest_dll.exe || set FAIL=1
echo.
echo === FMA/BMI1/LZCNT/MOVBE self-test ===
bin\converted_fmatest\fmatest.exe || set FAIL=1
echo.
echo === Benchmark (converted) ===
bin\converted_bench\bench.exe
if "%FAIL%"=="1" (echo. & echo SOME CHECKS FAILED & exit /b 1)
echo.
echo ALL END-TO-END CHECKS PASSED
