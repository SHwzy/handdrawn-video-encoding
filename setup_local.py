from pathlib import Path
import os, shutil
P=Path(__file__).resolve().parent
for folder in ['fonts','tools']:(P/folder).mkdir(exist_ok=True)
for name in ['YuGothL.ttc','msgothic.ttc']:
    src=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
    if not (P/'fonts'/name).exists():
        if not src.is_file():raise SystemExit('Missing Windows font: '+name)
        shutil.copy2(src,P/'fonts'/name)
for name in ['ffmpeg','ffprobe']:
    dest=P/'tools'/(name+'.exe')
    if not dest.exists():
        src=shutil.which(name)
        if not src:raise SystemExit('Install FFmpeg or place '+name+'.exe in tools/')
        shutil.copy2(src,dest)
print('Local fonts and FFmpeg ready. Run python render.py --mode check')
