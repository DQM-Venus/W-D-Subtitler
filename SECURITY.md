# 安全说明

## API Key

W-D Subtitler 把可能包含 DeepSeek API Key 的运行配置保存在：

```text
%LOCALAPPDATA%\W-D Subtitler\config.json
```

该路径位于仓库之外。项目根目录的 `config.json` 仍被 `.gitignore` 排除，防止旧版本
配置被误提交。提交代码前请确认：

- 不要提交 `config.json`、`.env`、日志、断点文件或任何密钥文件；
- 不要把 API Key 写入截图、Issue、日志或测试代码；
- 公开仓库只保留不含密钥的 `config.example.json`；
- 如果密钥曾进入 Git 历史，应立即在服务商控制台撤销并重新生成，仅删除当前文件并不安全。

程序断点不会保存 API Key。Key 仍以明文保存在当前 Windows 用户的数据目录中，
因此请保护好本机账户和该文件的访问权限。

## 漏洞报告

如发现安全问题，请不要公开披露可直接利用的细节。建议通过 GitHub 的私密安全报告
功能联系维护者，并提供复现步骤、影响范围和建议修复方式。
