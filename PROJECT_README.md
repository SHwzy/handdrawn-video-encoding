> V6追加：采用用户清理的首尾帧重新生成奏多抬头，更新后续正面图。1:19特写修正继续保留。V6待用户复核，V4认可记录属于前序版本。

# 莹与奏多 V6 最小制作工程

当前冻结基线：`baseline/current_ying_kanata_v6.mp4`。V4于2026-10-04获用户认可；V5修正1:19奏多特写的右侧裁切，待用户复核。1920×1080，30fps，6900帧，230秒。莹替换女仆，奏多替换巫女；身份互换镜头按叙事交换服装。

仅包含当前采用的素材、实际渲染依赖及当前成片。不含爱音／祥子版人物素材、历史成片、对照网页、生成任务、API密钥、缓存或原电脑环境。

## 迁移与运行

Windows，Python 3.14。先在目标电脑安装 FFmpeg，并让 `ffmpeg` 和 `ffprobe` 可通过 PATH 找到，或把这两个程序放在本目录 `tools/`。字体从目标电脑的 Windows Fonts 准备，软件和字体不随源码分发。

```powershell
python -m pip install -r requirements.txt
python setup_local.py
python verify_release.py
python render.py --mode check
python render.py --mode edit --start 1941 --end 1973 --output output/swap.mp4
python render.py --mode edit --output output/full.mp4
```

`--mode replay --output output/replay.mp4` 原样重封装当前V6文件；素材修改必须使用 `--mode edit`。输出文件已存在时不会覆盖。帧从0开始，区间左闭右开。`check` 是运行取样，不能代替全片视觉验收。

## 素材与目录

`asset_overrides.json` 将旧配方的逻辑键映射到当前莹／奏多素材。逻辑键中的历史角色名只是代码接口；对应旧人物文件没有打包。不要删除映射，也不要仅依据历史文件名更换路径。`project/` 保留配方所需相对结构。`assets/` 包含当前109项覆盖素材及音轨。原曲PV是渲染参考输入，与未发布的旧换角色成片不同。

运行不需要生成服务或API Key。修改人物时保留画布、锚点、alpha、动作视频时序、状态组和特殊遮罩。检查身份、表情、接触点、接地、字幕与切镜；本轮修正包含207–234走路短发、1957身份互换、雨中／门口透明背景以及莹跑步接地。

## 发布范围

`manifests/files.json` 列出发布文件及SHA256；`verify_release.py` 检查清单完整性。字体、FFmpeg、缓存与新输出在gitignore中。原曲、角色与第三方素材的原有权利不因本工程迁移而转移；本包未擅自为其指定开源许可。请按你选定的代码许可和相应素材授权发布。

定位依赖见 `manifests/geometry.json`：几何占位图只保存画布和包围框，不含旧角色图像。`assets/current_actions/` 中两段是当前画面无损 RGB 动作素材（2601–2707、3427–3478），含现有字幕和镜头运动；后续更换这两段角色时需要更新整段动作文件。对应的历史参考视频已替换为空白初始化占位，不用于最终画面。当前角色正脸基线位于 `assets/characters_ying_kanata/baselines/`。

原曲参考帧由渲染器直接从源视频按帧读取，不需要单独携带 reference/f*.jpg 或 f*.png。
