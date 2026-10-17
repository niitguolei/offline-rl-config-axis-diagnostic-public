@echo off
REM Launch one TD3+BC shard detached from the interactive session.
REM Mirrors run_cql_shard.cmd. Lists travel with '+' and are converted below,
REM because cmd.exe treats a comma as an argument separator -- a bug that has
REM now bitten three times (weights, seeds, and again seeds).
REM
REM Usage: run_td3bc_shard.cmd <tag> <seeds> [envs] [alphas] [lrs]

setlocal
set "REPO=D:\workspace\codex\RL"
set "PY=C:\Users\Gray\.conda\envs\pytorch_env\python.exe"
set "TAG=%~1"
set "SEEDS=%~2"
set "ENVS=hopper_medium_replay-v2"
if not "%~3"=="" set "ENVS=%~3"
set "ALPHAARG="
if not "%~4"=="" set "ALPHAARG=--alphas %~4"
set "LRARG="
if not "%~5"=="" set "LRARG=--lrs %~5"
set "ENVS=%ENVS:+=,%"
set "ALPHAARG=%ALPHAARG:+=,%"
set "LRARG=%LRARG:+=,%"
set "SEEDS=%SEEDS:+=,%"
set "LOG=%REPO%\results\benchmarks\e2b_td3bc_%TAG%.log"

cd /d "%REPO%"

echo [%DATE% %TIME%] shard %TAG% starting seeds=%SEEDS% >> "%LOG%"
"%PY%" experiments\e2b_td3bc.py --no-is --seeds %SEEDS% --tag %TAG% --envs %ENVS% %ALPHAARG% %LRARG% >> "%LOG%" 2>&1
echo [%DATE% %TIME%] shard %TAG% exited with %ERRORLEVEL% >> "%LOG%"
endlocal
