# 手书制作与角色替换工程

可复用的分层制作方法、SKILL、渲染代码和当前莹／奏多工程。莹替换女仆，奏多替换巫女；本项目按作者要求保留原服装和画风，仅改变角色身份特征。

## 下载完整工程与视频

在本仓库 **Releases** 下载 `ying_kanata_v6_minimal.zip`（完整素材与渲染工程，约 1 GB）和校验文件。解压后在解压目录运行，不要直接在这个仅含代码和文档的 Git checkout 中渲染。成片 `ying_kanata_full_review_v6.mp4` 也作为独立 Release 附件提供。

V6 包含用户手工清理首尾帧后重新生成的奏多抬头动画，以及 1:19 特写裁切修正。V6仍待用户视觉复核；此前V4的认可不代表V6已验收。

完整运行步骤见 [工程说明](PROJECT_README.md) 和 [复制与开工](复制与开工.md)。Windows / Python 3.14，需要自行安装 FFmpeg 并准备本机字体。

```powershell
python -m pip install -r requirements.txt
python setup_local.py
python verify_release.py
python render.py --mode check
python render.py --mode edit --output output/full.mp4
```

## 共享 SKILL

- [handdrawn-video-encoding](skills/handdrawn-video-encoding/SKILL.md)：镜头分析、素材规划、分层合成与验证，附参考文档。
- [ark-seedance-api](skills/ark-seedance-api/SKILL.md)：Seedance 接入说明与 Python 脚本。生成需要使用者自己的 `ARK_API_KEY`，已有素材渲染无需远程服务。

仓库与完整发布包不包含未公开的爱音／祥子替换版素材或视频，不含账号密码、API密钥、生成任务或历史对照网页。配方中的历史名字是素材映射接口，不表示分发对应的旧人物素材。

## 文件组织与权利

代码和文档位于 Git 仓库；视频及完整素材位于 Releases，避免普通 Git 的大文件限制。`manifests/files.json` 对应完整 ZIP 解压后的文件，不用于验证这个仅含代码的 checkout。

原曲、角色与第三方素材的权利归其原权利人；工程共享不授予第三方内容的额外权利。本次尚未为代码指定许可，后续由作者选择后补充。
