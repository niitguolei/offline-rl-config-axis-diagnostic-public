@echo off
REM Launch one IQL shard detached from the interactive session.
REM
REM Mirrors run_cql_shard.cmd. The IQL arm keeps all 90 of its runs in one file,
REM so a tagged shard would see an empty cache and re-run everything for its
REM seed. The caller therefore copies e2b_runs.jsonl to e2b_runs_<tag>.jsonl
REM first, and the shards are merged back afterwards.
REM
REM Lists travel with '+' and are converted below, because cmd.exe treats a
REM comma as an argument separator.
REM
REM Usage: run_iql_shard.cmd <tag> <seeds> [envs] [taus]

setlocal
set "REPO=D:\workspace\codex\RL"
set "PY=C:\Users\Gray\.conda\envs\pytorch_env\python.exe"
set "TAG=%~1"
set "SEEDS=%~2"
set "ENVS=hopper_medium_replay-v2"
if not "%~3"=="" set "ENVS=%~3"
set "TAUARG="
if not "%~4"=="" set "TAUARG=--taus %~4"
set "ENVS=%ENVS:+=,%"
set "TAUARG=%TAUARG:+=,%"
set "SEEDS=%SEEDS:+=,%"
set "LOG=%REPO%\results\benchmarks\e2b_iql_%TAG%.log"

cd /d "%REPO%"

if "%TAG%"=="s0" (
  set "TAGARG="
) else (
  set "TAGARG=--tag %TAG%"
)

echo [%DATE% %TIME%] shard %TAG% starting seeds=%SEEDS% envs=%ENVS% >> "%LOG%"
"%PY%" experiments\e2b_crossconfig.py --no-is --seeds %SEEDS% %TAGARG% --envs %ENVS% %TAUARG% >> "%LOG%" 2>&1
echo [%DATE% %TIME%] shard %TAG% exited with %ERRORLEVEL% >> "%LOG%"
endlocal
