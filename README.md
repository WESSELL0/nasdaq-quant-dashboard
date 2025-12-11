# 纳斯达克量化决策台 使用说明

> 你已经安装好 Python，下载好文件，文件名为 `nasdaq_app.py`。

---

## 1. 第一次使用（安装依赖）

1. 打开终端（Win + R → 输入 `cmd` → 回车）。

2. 切换到脚本所在目录，例如：

   ```bat
   cd /d C:\Users\17900\Desktop
   ```

3. 安装依赖（只需执行一次）：

   ```bat
   py -m pip install --upgrade pip
   py -m pip install streamlit yfinance plotly pandas numpy
   ```

---

## 2. 运行应用（每次打开时）

在终端执行（仍然在 `nasdaq_app.py` 所在目录）：

### 如果需要代理（常见：Clash 等本地端口 7890）

```bat
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890
py -m streamlit run "%cd%\nasdaq_app.py"
```

### 如果不需要代理

```bat
py -m streamlit run "%cd%\nasdaq_app.py"
```

终端看到类似提示：

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

用浏览器打开 `http://localhost:8501` 即可使用。

---

## 3. 可选：一键启动脚本（后续使用）

在 `nasdaq_app.py` 同目录新建 `run_nasdaq_app.bat`，内容如下：

```bat
@echo off
setlocal

REM 如需代理保留，不需要可删掉下面两行
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890

cd /d "%~dp0"
py -m streamlit run "%~dp0nasdaq_app.py"

pause
endlocal
```

以后双击 `run_nasdaq_app.bat` 即可启动应用。
