@echo off
setlocal

REM ============================
REM 1. 配置代理（如果不用代理，可以删掉这两行）
REM ============================
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890

REM ============================
REM 2. 切换到本脚本所在目录
REM ============================
cd /d "%~dp0"

REM ============================
REM 3. 安装 / 更新依赖库（已安装的会自动跳过）
REM ============================
echo.
echo [INFO] 正在安装/更新所需 Python 库...
py -m pip install --upgrade pip
py -m pip install streamlit yfinance plotly pandas numpy

REM ============================
REM 4. 启动 Streamlit 应用
REM ============================
echo.
echo [INFO] 启动纳斯达克量化决策台...
py -m streamlit run "%~dp0nasdaq_app.py"

echo.
echo [INFO] 程序已退出。如需再次运行，直接双击本 bat 文件。
pause
endlocal
