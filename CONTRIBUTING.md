# 贡献指南

感谢你帮助改进 W-D Subtitler。

## 开发流程

1. 从主分支创建功能分支。
2. 使用 Python 3.10 或更高版本创建虚拟环境。
3. 安装 `requirements.txt` 中的依赖。
4. 修改代码时保持 UTF-8 无 BOM，注释、日志和界面文本使用简体中文。
5. 不要提交 API Key、媒体样本、模型缓存、日志或断点。
6. 为行为变化补充自动化测试。
7. 提交前执行：

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m compileall -q wd_subtitler tests
.\venv\Scripts\python.exe -m pip check
```

## 提交建议

- 一次提交只处理一个清晰主题；
- 提交信息说明“改了什么”和“为什么”；
- 涉及识别或翻译质量时，说明测试音频特征和观察结果，但不要上传无授权媒体；
- 修改配置字段、断点结构或提示词时同步更新 README 和测试。
