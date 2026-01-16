@echo off
setlocal enabledelayedexpansion

REM ===================== 配置项 =====================
set "VENV_PATH=.venv"
REM ==================================================

echo === Starting deployment ===

REM 检查并安装 uv
call :install_uv
if errorlevel 1 goto :error

REM 创建虚拟环境
call :create_venv
if errorlevel 1 goto :error

REM 安装依赖
call :install_deps
if errorlevel 1 goto :error

REM 验证安装
call :verify_install
if errorlevel 1 goto :error

REM 生成启动脚本
call :generate_start_script
if errorlevel 1 goto :error

echo.
echo === Deployment completed successfully ===
echo Virtual environment: %CD%\%VENV_PATH%
echo.
echo Usage:
echo   Start (foreground):  start_service.bat
echo   Start (background):  start_service.bat -d
echo   Stop service:        stop_service.bat
echo   Check status:        status_service.bat
echo   View logs:           type service.log
goto :eof

REM ===================== 函数定义 =====================

:check_cuda
    REM 检测 CUDA 版本
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        set "CUDA_GROUP=cpu"
        goto :eof
    )

    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do (
        set "CUDA_VERSION=%%i"
    )

    REM 移除点号，例如 12.4 -> 124
    set "CUDA_VERSION=%CUDA_VERSION:.=%"

    echo Detected CUDA version: %CUDA_VERSION% >&2

    if "%CUDA_VERSION:~0,3%"=="129" (
        set "CUDA_GROUP=cu129"
    ) else if "%CUDA_VERSION:~0,3%"=="128" (
        set "CUDA_GROUP=cu128"
    ) else if "%CUDA_VERSION:~0,3%"=="124" (
        set "CUDA_GROUP=cu124"
    ) else if "%CUDA_VERSION:~0,3%"=="121" (
        set "CUDA_GROUP=cu121"
    ) else if "%CUDA_VERSION:~0,3%"=="118" (
        set "CUDA_GROUP=cu118"
    ) else (
        set "CUDA_GROUP=cpu"
    )
    goto :eof

:install_uv
    REM 检查 uv 是否已安装
    where uv >nul 2>&1
    if not errorlevel 1 (
        echo uv already installed
        goto :eof
    )

    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo Failed to install uv
        exit /b 1
    )

    REM 添加 uv 到当前会话的 PATH
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    goto :eof

:create_venv
    if exist "%VENV_PATH%" (
        echo Virtual environment %VENV_PATH% already exists, skip creation
        goto :eof
    )

    echo Creating uv virtual environment at %VENV_PATH%...
    uv venv "%VENV_PATH%"
    if errorlevel 1 (
        echo Failed to create virtual environment
        exit /b 1
    )
    goto :eof

:install_deps
    echo Activating virtual environment...
    call "%VENV_PATH%\Scripts\activate.bat"
    if errorlevel 1 (
        echo Failed to activate virtual environment
        exit /b 1
    )

    REM 检测 CUDA 版本
    call :check_cuda

    REM 第一步：安装基础依赖（从清华源）
    echo Step 1: Installing base dependencies...
    uv pip install .
    if errorlevel 1 (
        echo Failed to install base dependencies
        exit /b 1
    )

    REM 第二步：安装 PyTorch（从官方 wheel 源）
    echo Step 2: Installing PyTorch for !CUDA_GROUP!...
    if "!CUDA_GROUP!"=="cpu" (
        uv pip install torch torchvision torchaudio numpy --index-url https://download.pytorch.org/whl/!CUDA_GROUP!
    ) else (
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CUDA_GROUP!
    )
    if errorlevel 1 (
        echo Failed to install PyTorch
        exit /b 1
    )
    goto :eof

:verify_install
    echo Verifying installation...
    call "%VENV_PATH%\Scripts\activate.bat"

    python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
    if errorlevel 1 (
        echo Verification failed
        exit /b 1
    )
    goto :eof

:generate_start_script
    echo Generating start and stop scripts...

    REM 生成启动脚本
    (
        echo @echo off
        echo setlocal
        echo.
        echo set "VENV_PATH=.venv"
        echo set "PID_FILE=service.pid"
        echo set "LOG_FILE=service.log"
        echo.
        echo REM 检查服务是否已运行
        echo if exist "%%PID_FILE%%" ^(
        echo     set /p PID=^<"%%PID_FILE%%"
        echo     tasklist /FI "PID eq %%PID%%" 2^>nul ^| find "%%PID%%" ^>nul
        echo     if not errorlevel 1 ^(
        echo         echo Service is already running ^(PID: %%PID%%^)
        echo         exit /b 1
        echo     ^) else ^(
        echo         echo Removing stale PID file...
        echo         del "%%PID_FILE%%"
        echo     ^)
        echo ^)
        echo.
        echo REM 激活虚拟环境
        echo call "%%VENV_PATH%%\Scripts\activate.bat"
        echo.
        echo REM 判断运行模式
        echo if "%%1"=="-d" goto daemon
        echo if "%%1"=="--daemon" goto daemon
        echo.
        echo REM 前台运行
        echo echo Starting service in foreground mode...
        echo echo Press Ctrl+C to stop
        echo python main.py
        echo goto :eof
        echo.
        echo :daemon
        echo echo Starting service in background mode...
        echo start /B python main.py ^>%%LOG_FILE%% 2^>^&1
        echo REM 获取最后启动进程的 PID ^(Windows 需要通过 wmic^)
        echo for /f "tokens=2" %%%%i in ^('wmic process where "name='python.exe' and commandline like '%%%%main.py%%%%'" get processid /format:list 2^>nul ^| find "ProcessId"'^) do set PID=%%%%i
        echo echo %%PID%%^>%%PID_FILE%%
        echo echo Service started ^(PID: %%PID%%^)
        echo echo Log file: %%LOG_FILE%%
        echo echo Use 'stop_service.bat' to stop the service
    ) > start_service.bat

    REM 生成停止脚本
    (
        echo @echo off
        echo setlocal
        echo.
        echo set "PID_FILE=service.pid"
        echo.
        echo if not exist "%%PID_FILE%%" ^(
        echo     echo Service is not running ^(PID file not found^)
        echo     exit /b 1
        echo ^)
        echo.
        echo set /p PID=^<"%%PID_FILE%%"
        echo.
        echo REM 检查进程是否存在
        echo tasklist /FI "PID eq %%PID%%" 2^>nul ^| find "%%PID%%" ^>nul
        echo if errorlevel 1 ^(
        echo     echo Service is not running ^(PID %%PID%% not found^)
        echo     del "%%PID_FILE%%"
        echo     exit /b 1
        echo ^)
        echo.
        echo echo Stopping service ^(PID: %%PID%%^)...
        echo taskkill /PID %%PID%% /T ^>nul 2^>^&1
        echo.
        echo REM 等待进程结束
        echo for /L %%%%i in ^(1,1,10^) do ^(
        echo     timeout /t 1 /nobreak ^>nul
        echo     tasklist /FI "PID eq %%PID%%" 2^>nul ^| find "%%PID%%" ^>nul
        echo     if errorlevel 1 ^(
        echo         echo Service stopped successfully
        echo         del "%%PID_FILE%%"
        echo         exit /b 0
        echo     ^)
        echo ^)
        echo.
        echo REM 强制终止
        echo echo Service did not stop gracefully, forcing termination...
        echo taskkill /F /PID %%PID%% /T ^>nul 2^>^&1
        echo del "%%PID_FILE%%"
        echo echo Service force stopped
    ) > stop_service.bat

    REM 生成状态查询脚本
    (
        echo @echo off
        echo setlocal
        echo.
        echo set "PID_FILE=service.pid"
        echo.
        echo if not exist "%%PID_FILE%%" ^(
        echo     echo Service is not running
        echo     exit /b 1
        echo ^)
        echo.
        echo set /p PID=^<"%%PID_FILE%%"
        echo.
        echo tasklist /FI "PID eq %%PID%%" 2^>nul ^| find "%%PID%%" ^>nul
        echo if not errorlevel 1 ^(
        echo     echo Service is running ^(PID: %%PID%%^)
        echo     echo Memory usage:
        echo     tasklist /FI "PID eq %%PID%%" /FO TABLE
        echo     exit /b 0
        echo ^) else ^(
        echo     echo Service is not running ^(stale PID file^)
        echo     del "%%PID_FILE%%"
        echo     exit /b 1
        echo ^)
    ) > status_service.bat

    echo Scripts generated successfully:
    echo   - start_service.bat  (use -d or --daemon for background mode)
    echo   - stop_service.bat   (stop the service)
    echo   - status_service.bat (check service status)
    goto :eof

:error
    echo.
    echo === Deployment failed ===
    exit /b 1
