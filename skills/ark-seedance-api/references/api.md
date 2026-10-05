# API 核对记录（2026-09-30）

- 用户提供的 Agent Plan OpenAI 兼容 Base URL：`https://ark.cn-beijing.volces.com/api/plan/v3`；视频任务路径为 `/contents/generations/tasks`。Anthropic 兼容入口为 `/api/plan`，不能用于本脚本。原 Skill 使用 Bearer Key。参考：[Agent Plan 官方实践](https://docs.volcengine.com/docs/ark/agent-plan-personal-practice-short-video?lang=zh)、[原 Skill 摘录](https://pluginbench.com/skills/skills.volces.com/byted-ark-seedance-skill)。
- 创建：`POST /contents/generations/tasks`，Body 至少含 `model` 与 `content`。`content` 为文本、`image_url`、`video_url`、`audio_url` 等项目数组。创建返回任务 `id`，异步查询后才有视频 URL。字段细节参照[标准视频 API 文档](https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh)，Agent Plan 支持范围须以其实际响应为准。
- 查询：`GET /contents/generations/tasks/{id}`；状态含 `queued`、`running`、`succeeded`、`failed`、`cancelled`。成功后读取 `content.video_url`。[查询任务](https://docs.volcengine.com/docs/ark/get-video-generation-task-api?lang=zh)
- Agent Plan 模型名：`doubao-seedance-2.5`；标准方舟数据面的模型 ID `doubao-seedance-2-5-260628` 不应直接用于 Plan。[官方 ark-cli 资源配置](https://github.com/volcengine/ark-cli/blob/main/skills/arkcli-shared/references/profile-defaults.md)
- Seedance 2.5：`duration` 为 4–30 秒或 `-1`；默认 `-1`。`resolution` 支持 480p、720p、1080p；默认 720p。`ratio` 默认 `adaptive`。首帧、首尾帧及视频编辑/延长场景仅可用 `adaptive`。视频编辑仅可设置 `duration=-1`。[创建任务](https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh)
- 多模态参考最多 30 张图、10 个视频、10 段音频，合计最多 50 个参考素材；首尾帧模式与全模态参考模式互斥。Seedance 2.5 不支持直接输入含真人人脸的参考图/视频。[创建任务](https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh)
- 输出 URL 有效期 24 小时；Seedance 2.5 输出 URL 下载上限 100 次。任务记录仅保留 7 天。[查询任务列表](https://docs.volcengine.com/docs/ark/list-video-generation-tasks-api?lang=zh)
- 2026-09-30 本项目验证：Agent Plan `GET /contents/generations/tasks` 成功；`doubao-seed-2.0-mini` 最小聊天请求返回 `OK`，总计 61 token；Seedance 2.5 使用 `duration=1` 的无效请求返回针对 `doubao-seedance-2-5` 的参数校验错误，未创建视频任务。此验证证明鉴权、端点和模型路由可达，尚未验证实际生成质量或任务完成。

原 `byted-ark-seedance-skill` 是 Agent Plan 专版（`/api/plan/v3`，Agent 层代传 Key、Node wrapper、偏好路由、后台通知）。本地适配只实现直接 API 调用，不能假定原 wrapper 的后台通知或偏好路由可用。[原 Skill 摘录](https://pluginbench.com/skills/skills.volces.com/byted-ark-seedance-skill)
