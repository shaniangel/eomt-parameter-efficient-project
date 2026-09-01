@echo off
REM run_smoke.cmd - Run a smoke training run using the pilot LoRA config.
REM Usage:
REM   run_smoke.cmd [DATA_PATH] [DEVICES] [RUN_NAME] [SAVE_DIR]
REM Defaults (if omitted):
REM   DATA_PATH = data\demo_ade
REM   DEVICES   = 1
REM   RUN_NAME  = smoke_lora_run_<RANDOM>
REM   SAVE_DIR  = pilot_logs\<RUN_NAME>

setlocal
set ROOT=%~dp0
set DATA_PATH=%~1
if "%DATA_PATH%"=="" set DATA_PATH=%ROOT%data\demo_ade
set DEVICES=%~2
if "%DEVICES%"=="" set DEVICES=1
set RUN_NAME=%~3
if "%RUN_NAME%"=="" set RUN_NAME=smoke_lora_run_%RANDOM%
set SAVE_DIR=%~4
if "%SAVE_DIR%"=="" set SAVE_DIR=pilot_logs\%RUN_NAME%

REM Ensure we don't accidentally overwrite main checkpoints: disable checkpointing and run WandB offline
python main.py fit -c configs\pilot\eomt_lora_pilot.yaml --trainer.devices %DEVICES% --data.path "%DATA_PATH%" --trainer.logger.init_args.name "%RUN_NAME%" --trainer.logger.init_args.save_dir "%SAVE_DIR%" --trainer.enable_checkpointing False --trainer.logger.init_args.offline True

if %ERRORLEVEL% NEQ 0 (
  echo Training run failed with exit code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo Smoke run completed. Logs saved to %SAVE_DIR%
endlocal
