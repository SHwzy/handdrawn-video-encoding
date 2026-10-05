---
name: ark-seedance-api
description: 通过火山方舟 Agent Plan API Key 调用 Seedance，创建、查询和下载视频生成任务。用于本项目需要 Agent Plan 生视频或维护接入脚本时。
---

# Seedance Agent Plan API

使用 `scripts/seedance.py`。先读 [API 要点](references/api.md)，尤其是素材组合、人脸限制及计费。用户要求生成视频时，确认已有明确的素材、提示词和参数后再提交；生成会消耗 Agent Plan 额度。不要在命令参数、日志、源码或对话中暴露 API Key。脚本优先读取进程环境变量 `ARK_API_KEY`，其次读取项目根目录 `password.json` 的 `password` 字段。

命令（从本目录或使用脚本绝对路径）：

```powershell
python scripts/seedance.py create --prompt "镜头缓慢推进" --duration 5 --ratio 16:9
python scripts/seedance.py create --prompt "镜头缓慢推进" --duration 4 --resolution 480p --dry-run
python scripts/seedance.py create --payload task.json
python scripts/seedance.py list --page-size 5
python scripts/seedance.py get cgt-...
python scripts/seedance.py wait cgt-... --timeout 600
python scripts/seedance.py download cgt-... --output output.mp4
python scripts/seedance.py delete cgt-...
```

`--payload` 接受视频生成任务 JSON，适合参考图片、视频、音频与高级参数；脚本默认补入 Seedance 2.5 模型 ID，可用 `--model` 或 JSON 中的 `model` 覆盖。上传本地素材前先按文档确认格式；可在 JSON 中传公网 URL 或支持的 data URI。`create` 只提交一次，不自动重试，避免重复消耗。任务 ID 要保存；`get` 和 `wait` 不产生新的生成任务。成功后尽快下载，输出 URL 有效期有限。

本适配使用用户给定的 `/api/plan/v3` 视频接口。原 Skill 的智能模型路由、偏好存储及后台通知依赖其 Node wrapper/Agent Plan 运行环境，本脚本不承诺这些能力。用低成本模型验证认证后，再测试 Seedance 2.5；不要把视频生成当作无成本连通性探针。
