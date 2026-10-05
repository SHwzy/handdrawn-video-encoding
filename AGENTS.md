# 莹与奏多 V4 制作工程

开始先读 README.md、baseline.json、换角色指南.md、独立工程验证.md。制作镜头时使用 skills/handdrawn-video-encoding/SKILL.md。

唯一入口 render.py；timeline.json 使用0起算30fps左闭右开区间。replay仅重封装当前认可基线，素材修改使用edit。check仅取样运行检查。保留baseline冻结文件，新输出使用新文件名。

莹替换女仆，奏多替换巫女。只换头部身份，保留原曲服饰与画风；身份互换镜头按叙事交换服装。正脸基线在assets/characters_ying_kanata/baselines/。

人物、动作和状态组通过asset_overrides.json与素材替换清单.csv管理。不要按历史名字补回旧人物素材；某些路径只存几何占位或空白初始化视频。两段current_actions是当前完整短镜头素材，更新角色需更新整段。先渲染修改区间，再全量渲染；核对身份、接触、接地、透明边缘、节奏与字幕，分别记录技术验证、视觉检查、用户验收。

不要引入历史未发布成片、旧角色素材、对照网页、API密钥或生成任务缓存。运行不需要远程服务。字体与FFmpeg由目标电脑本地准备，不随发布包分发。
