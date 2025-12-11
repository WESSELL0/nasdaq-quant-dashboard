# 纳斯达克量化决策台 使用说明 (新版)

本说明适用于：

* 第一次使用本项目（自动安装依赖）
* 之后日常打开使用
* Windows 用户

你的运行方式已经被简化为：

* **第一次使用：双击 `first_nasdaq_app.bat`**（自动安装所有依赖并启动）
* **以后使用：双击 `nasdaq_app.bat`**（直接启动，无需再次安装）

---

## 1. 文件说明

确保目录内至少包含：

```
nasdaq_app.py                 # 主应用代码
first_nasdaq_app.bat          # 第一次使用运行（安装依赖 + 启动）
nasdaq_app.bat                # 之后每次直接启动
```

---

## 2. 第一次使用（自动安装依赖）

直接双击：

```
first_nasdaq_app.bat
```

此脚本会自动完成：

1. 检查 Python 是否可用
2. 自动安装：

   * streamlit
   * yfinance
   * pandas
   * numpy
   * plotly
3. 设置代理（如你需要，可在脚本中编辑端口）
4. 自动运行应用：

   ```bat
   py -m streamlit run nasdaq_app.py
   ```

首次启动可能需要数十秒，属于正常情况。

---

## 3. 日常使用（之后每次打开）

双击：

```
nasdaq_app.bat
```

此脚本会：

1. 可选自动设置代理（如不需要，可删掉脚本中的两行 set proxy）
2. 直接启动 Streamlit 应用：

```bat
py -m streamlit run nasdaq_app.py
```

浏览器将自动打开：

```
http://localhost:8501
```

---

## 4. 可选：代理说明（如你在国内需访问 Yahoo Finance）

如你使用 Clash / Surge 等工具，并监听 7890 端口，则脚本中的两行 proxy 会生效：

```bat
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890
```

如你无需代理，可删除这两行。

---

## 5. 常见问题

### Q1：双击 bat 窗口闪退？

右键 bat → 用“以管理员身份运行”，或在终端手动执行：

```bat
first_nasdaq_app.bat
```

查看报错内容。

### Q2：启动后显示无法访问 Yahoo Finance？

本地网络阻拦，请：

* 确保代理开启（如需）
* 确保 bat 中的代理端口与你工具一致

### Q3：网页打不开？

访问：

```
http://localhost:8501
```

如端口被占用，可在脚本中替换端口，例如：

```bat
py -m streamlit run nasdaq_app.py --server.port 8502
```

---

## 6. 完整流程总结

### 第一次（自动安装 + 启动）

```
双击 first_nasdaq_app.bat
```

### 之后每天/每次使用（直接启动）

```
双击 nasdaq_app.bat
```

无需再安装任何依赖，操作最简化。

---

如需，我可以为你：

* 生成两个 bat 文件的最终版本
* 帮你制作“带图的 README”
* 适配云端部署版本

随时告诉我即可。
