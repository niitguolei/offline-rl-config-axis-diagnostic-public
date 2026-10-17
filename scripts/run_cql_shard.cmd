@echo off
REM Launch one CQL shard detached from the interactive session.
REM
REM The first attempt ran these under the Codex terminal, so closing the app
REM killed the process tree and the arm stopped after two runs per shard. This
REM wrapper is started with `Start-Process -WindowStyle Hidden`, which gives the
REM shard its own process group and no console attachment, so it is not torn down
REM with the terminal that launched it.
REM
REM (`schtasks` would be the stronger answer -- a task is owned by the Task
REM Scheduler service rather than by this app -- but it is blocked in this
REM environment: even `schtasks /query` fails with "cannot find the path".)
REM
REM Usage: run_cql_shard.cmd <tag> <seeds> [envs] [weights] [lrs]
REM   tag      s0 | s1 | s2   (s0 writes e2b_cql_runs.jsonl, the others _<tag>)
REM   seeds    the seed list for this shard
REM   envs     optional, defaults to the three main domains
REM   weights  optional min_q_weight list, defaults to the runner's own (5,10)
REM   lrs      optional learning-rate list; splitting by lr is how the seed
REM            top-up keeps three workers busy on a hopper-only job
REM
REM Both optional lists are written with '+' instead of ',' and converted back
REM below: cmd.exe treats a comma as an argument separator, so `5,10,20` arrives
REM as three arguments and only the first survives. That silently turned the S3-1
REM extension into a one-level axis that matched the cache and ran nothing.
REM The same applies to `seeds`, which is why it is converted too -- `3,4` used
REM to split and shift every later positional argument by one.

setlocal
set "REPO=D:\workspace\codex\RL"
set "PY=C:\Users\Gray\.conda\envs\pytorch_env\python.exe"
set "TAG=%~1"
set "SEEDS=%~2"
set "ENVS=hopper_medium_replay-v2,walker2d_medium_replay-v2,halfcheetah_medium_replay-v2"
if not "%~3"=="" set "ENVS=%~3"
set "WEIGHTARG="
if not "%~4"=="" set "WEIGHTARG=--weights %~4"
set "LRARG="
if not "%~5"=="" set "LRARG=--lrs %~5"
set "ENVS=%ENVS:+=,%"
set "WEIGHTARG=%WEIGHTARG:+=,%"
set "LRARG=%LRARG:+=,%"
set "SEEDS=%SEEDS:+=,%"
set "LOG=%REPO%\results\benchmarks\e2b_cql_%TAG%.log"

cd /d "%REPO%"

if "%TAG%"=="s0" (
  set "TAGARG="
) else (
  set "TAGARG=--tag %TAG%"
)

echo [%DATE% %TIME%] shard %TAG% starting seeds=%SEEDS% >> "%LOG%"
"%PY%" experiments\e2b_cql.py --no-is --seeds %SEEDS% %TAGARG% --envs %ENVS% %WEIGHTARG% %LRARG% >> "%LOG%" 2>&1
echo [%DATE% %TIME%] shard %TAG% exited with %ERRORLEVEL% >> "%LOG%"
endlocal
