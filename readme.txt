纳斯达克量化决策台 使用说明

你已经安装好 Python，文件名为 nasdaq_app.py。

========================================

第一次使用（安装依赖）

1）打开终端：
按 Win + R，输入：cmd，然后回车。

2）切换到脚本所在目录，例如（按你实际路径改）：
cd /d C:\Users\17900\Desktop

3）安装依赖（只需要执行一次）：
py -m pip install --upgrade pip
py -m pip install streamlit yfinance plotly pandas numpy

执行完成后，依赖就装好了，后续一般不用再装。

========================================
2. 运行应用（每次打开时）

在终端中，先切到 nasdaq_app.py 所在目录（和上面一样）：
cd /d C:\Users\17900\Desktop

2.1 如果需要代理（例如本地有 Clash，端口 7890）

依次执行：

set http_proxy=http://127.0.0.1:7890

set https_proxy=http://127.0.0.1:7890

py -m streamlit run "%cd%\nasdaq_app.py"

2.2 如果不需要代理

直接执行：

py -m streamlit run "%cd%\nasdaq_app.py"

终端出现类似提示：

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501

然后用浏览器打开：
http://localhost:8501

即可看到量化决策界面。

========================================
3. 可选：一键启动脚本（后续使用）

在 nasdaq_app.py 同一个目录，新建一个文本文件，
把下面内容复制进去，并保存为：run_nasdaq_app.bat

（注意后缀是 .bat，不是 .txt）

------------------- 下面是 bat 内容 -------------------

@echo off
setlocal

REM 如需代理保留，不需要可删掉下面两行
set http_proxy=http://127.0.0.1:7890

set https_proxy=http://127.0.0.1:7890

cd /d "%~dp0"
py -m streamlit run "%~dp0nasdaq_app.py"

pause
endlocal

------------------- bat 内容结束 -------------------

以后只要双击 run_nasdaq_app.bat，就会自动启动应用。