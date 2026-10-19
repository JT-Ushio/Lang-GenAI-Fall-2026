"""Extract classroom figures from a locally obtained source video.

No network requests; normal lecture compilation does not invoke FFmpeg.
Usage: python3 lectures/scripts/extract_lecture_05_video.py --source PATH --ffmpeg PATH
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/lecture_05/video-guide"
SOURCE_URL = "https://www.bilibili.com/video/BV1Z9ec69Ezz"
AUTHOR_URL = "https://github.com/WelchLabs/videos/tree/master/_2025/sora"

FRAMES = [
    ("01_iterative-generation", 95, "从随机状态开始，重复更新", "01／06",
     "先指出输入是随机噪声；网络给出更新所需的预测，经过多次更新才得到结构。",
     "本帧来自视频对 Wan 的介绍，骨干标为 Transformer；本课经典 SD 1.5 使用 U-Net，不能混用架构。"),
    ("02_clip-encoders", 256, "复习 CLIP 的两个编码器", "01",
     "文字和图片各自编码，再比较向量。这让模型能匹配描述，还没有自动获得把噪声画成图的生成网络。",
     "图中的 512 是视频所举 CLIP 检索表示；本课 SD 1.5 条件采用 77×768 的文本序列隐藏状态。"),
    ("03_clip-pairing", 308, "批量图文配对与对比学习", "01／课前复习",
     "按行看文本，按列看图像；配对的目标是让同一概念靠近，其他候选形成区分。",
     "这张表列的是配对的向量，不能直接把每格数字当成已经算好的相似度。"),
    ("04_forward-noise", 520, "加噪让图像结构逐渐消失", "03",
     "沿左到右读：猫的细节越来越不明显，噪声越来越占主导；接着回到课堂的 RGB 加噪代码。",
     "这是视频的过程示意；具体信号和噪声系数以课件公式及代码为准。"),
    ("05_reverse-generation", 548, "训练好后，从噪声朝图像更新", "05／06",
     "沿箭头方向读生成链：当前状态送入网络，再执行采样更新。不同噪声水平重复使用同一套训练好的参数。",
     "多个网络图标表示多次调用，不表示每一步重新训练一个独立网络。"),
    ("06_noise-target", 662, "训练目标是这次人为加入的噪声", "04",
     "先找到黄色 ε，再找紫色网络预测：损失比较这两个量。干净图、时间和噪声可由程序构造训练样本。",
     "本课采用 ε 预测；预测原图或 velocity 的模型需要相应的目标与采样公式。"),
    ("07_two-pixel-space", 748, "两个像素，也能组成一个数据空间", "06／拓展",
     "横轴记录第一个像素，纵轴记录第二个像素；一幅只有两个像素的图像变成平面上的一个点。",
     "二维平面是数据空间类比，不是原图的宽与高；真实图像每个标量都可能贡献一个维度。"),
    ("08_score-field", 946, "箭头场给出局部移动方向", "06／拓展",
     "把螺旋上的黄点看成数据分布的样本，箭头表示在某个噪声水平下可参考的局部方向。",
     "score function 在这里指对数密度的梯度，不是 CLIP 的图文评分；不用本图替代完整采样公式。"),
    ("09_ddpm-paths", 1103, "DDPM 的随机采样路径", "06",
     "每条彩色路径从随机状态出发；受控随机项会让轨迹弯折，最终样本落向学到的结构。",
     "单条轨迹不要求每一步肉眼更清晰，也不需要恢复某张预先指定的原图。"),
    ("10_ddpm-ddim", 1415, "同一训练目标，可以搭配不同采样路径", "06",
     "比较左侧随机折线与右侧较平滑的路径，再说明网络预测和采样算法是两个层次。",
     "DDIM 的确定性需要 η=0 且初始噪声与条件给定；本帧不是两种算法优劣的受控质量评测。"),
    ("11_text-conditioning", 1588, "文字特征进入去噪模型", "09",
     "上方文本编码器输出条件特征，下面的去噪模型在每次更新时读取条件。语言提供内容约束。",
     "视频本帧用 Transformer 作泛化示意；本课用 U-Net 内的交叉注意力讲 SD 1.5，不能照此图认定所有模型相同。"),
    ("12_conditional-distribution", 1644, "类别条件挑选分布中的一部分", "09／10",
     "先把整条螺旋看成数据集合，再想象不同颜色对应不同类别；给定条件改变的是希望生成哪部分样本。",
     "二维类别示意不等同于现实自然语言的全部语义，也不是实际像素坐标。"),
    ("13_cfg-vector", 1806, "CFG：使用两种预测的差异", "10",
     "先读灰色无条件方向，再读黄色条件方向；绿色差向量表示条件带来的偏向，随后按尺度组合。",
     "视频用向量场 f 表示，课堂代码组合的是噪声预测 ε；符号和尺度约定应先对齐，不能机械抄符号。"),
    ("14_guidance-output", 1904, "引导尺度会改变生成外观", "10／11",
     "指出提示词、引导尺度和输出三个位置，讨论条件遵循、颜色、细节和多样性应分别评价。",
     "这张截图本身不能证明所有控制变量相同；严谨比较还要固定模型、采样器、步数和随机种子。"),
]

ANIMATIONS = [
    ("a01_noise-to-generation", 113, 12, "从噪声逐渐出现结构", "观察迭代中的可见结构，不展开 Wan 的视频骨干。", "01_iterative-generation"),
    ("a02_forward-noising", 512, 8, "构造前向加噪链", "沿着图中加噪箭头解释，不把这段动画倒放当成训练好的生成模型。", "04_forward-noise"),
    ("a03_ddpm-ddim", 1408, 8, "DDPM／DDIM 采样路径对照", "路径规则可以不同；本段只解释形态，不据此宣称哪种质量一定更好。", "10_ddpm-ddim"),
    ("a04_cfg-path", 1815, 8, "沿条件偏向更新状态", "暂停在一个箭头处，说明网络给预测、采样器更新状态；接回课件 CFG 小算例。", "13_cfg-vector"),
]


def stamp(t):
    return f"{int(t)//60:02}:{int(t)%60:02}"


def make_index(manifest):
    frames, animations = manifest["frames"], manifest["animations"]
    records = [dict(x, kind="frame") for x in frames] + [dict(x, kind="animation") for x in animations]
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    options = ''.join(f'<option value="{i}">{i+1:02} · {escape(x["title"])}</option>' for i, x in enumerate(records))
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第五讲 · 视频讲解图册</title>
<style>*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f5f9fc;color:#17334a;font:17px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 6px}h2{font-size:24px;margin:10px 0}.meta{font-size:14px;color:#536c80}.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:16px 0}button,select,a.download{font:inherit;padding:7px 12px;border:1px solid #9ab9c1;border-radius:8px;background:white;color:#17334a}button,select{cursor:pointer}select{max-width:100%}.stage{background:#000;border-radius:12px;overflow:hidden;display:grid;place-items:center;min-height:200px}.stage img,.stage video{display:block;max-width:100%;max-height:70vh;height:auto;object-fit:contain}.notes{background:white;border:1px solid #d5e3e9;border-radius:12px;padding:15px 20px;margin-top:12px}.notes p{margin:6px 0}a{color:#287e80}#links{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}.download{font-size:14px;text-decoration:none}.fullscreen{background:#f5f9fc;padding:20px;overflow:auto}.fullscreen .stage{height:70vh}.fullscreen .meta{display:none}@media(max-width:600px){main{padding:12px}h1{font-size:22px}.toolbar{gap:6px}button,select{font-size:15px;padding:6px}.notes{padding:12px}}</style></head><body><main id="deck">
<h1>第五讲 · 视频讲解图册</h1><p class="meta">14 张关键帧＋4 段无声动画。左右键换图；GIF 按需播放，可恢复静态画面。源视频 852×480，GIF 为 640×360。</p>
<div class="toolbar"><button id="previous">← 上一张</button><select id="select">OPTIONS</select><button id="next">下一张 →</button><button id="fullscreen">全屏</button><button id="notes-toggle">隐藏提示</button></div>
<h2 id="title"></h2><div id="stage" class="stage"></div><div class="toolbar" id="animation-controls" hidden><button id="gif-play">播放 GIF</button><button id="gif-stop">恢复静态</button><button id="mp4-play">带暂停控件的 MP4</button></div>
<div class="notes" id="notes"><p id="time" class="meta"></p><p><strong>讲解：</strong><span id="talk"></span></p><p><strong>口径：</strong><span id="caution"></span></p></div><div id="links"></div>
<p class="meta">原视频：<a href="https://www.bilibili.com/video/BV1Z9ec69Ezz" target="_blank" rel="noopener">B站 技术Flow（中文版本）</a>；原作 Welch Labs／3Blue1Brown，<a href="https://www.youtube.com/watch?v=iv-5mZ_9CPY" target="_blank" rel="noopener">英文视频</a>。本图册为课程选帧与讲解提示，并非作者官方 PPT。原画面与中文字幕保留，全部时间对应中文版本。</p>
<p class="meta"><a href="README.md">素材索引与来源说明</a> · 也可直接将 PNG／GIF 插入自己的 PPT；常规讲课不需要联网或 FFmpeg。</p>
</main><script>const items=PAYLOAD,$=id=>document.getElementById(id);let current=0;
function clearStage(){const old=$('stage').querySelector('video');if(old)old.pause();$('stage').replaceChildren()}
function showImage(path){clearStage();const img=document.createElement('img');img.src=path;img.alt=items[current].title;$('stage').append(img)}
function links(record){$('links').replaceChildren();for(const [label,path] of record.kind==='frame'?[['PNG 原图',record.file],['原视频时刻',record.source_url]]:[['GIF 文件',record.gif],['MP4 文件',record.mp4],['原视频时刻',record.source_url]]){const a=document.createElement('a');a.textContent=label;a.href=path;a.className='download';if(!path.startsWith('http'))a.download='';else{a.target='_blank';a.rel='noopener'}$('links').append(a)}}
function show(i){current=(i+items.length)%items.length;const r=items[current];$('select').value=current;$('title').textContent=`${current+1} / ${items.length} · ${r.title}`;$('talk').textContent=r.talk;$('caution').textContent=r.caution||'原视频动画片段；教学解释请与本课代码、采样器约定分别核对。';$('time').textContent=(r.kind==='frame'?`中文视频 ${r.time}；建议对应第 ${r.lecture_section} 节`:`中文视频 ${r.start_time}—${r.end_time}；${r.duration_seconds} 秒，无声`);$('animation-controls').hidden=r.kind!=='animation';showImage(r.kind==='frame'?r.file:r.poster);links(r);location.hash=r.id}
$('previous').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('select').onchange=()=>show(+$('select').value);$('gif-play').onclick=()=>showImage(items[current].gif);$('gif-stop').onclick=()=>showImage(items[current].poster);$('mp4-play').onclick=()=>{clearStage();const v=document.createElement('video');v.src=items[current].mp4;v.controls=true;v.loop=true;v.muted=true;v.playsInline=true;$('stage').append(v);v.play().catch(()=>{})};$('notes-toggle').onclick=()=>{const n=$('notes');n.hidden=!n.hidden;$('notes-toggle').textContent=n.hidden?'显示提示':'隐藏提示'};$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $('deck').requestFullscreen?.()};document.addEventListener('fullscreenchange',()=>$('deck').classList.toggle('fullscreen',!!document.fullscreenElement));document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA','VIDEO'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){e.preventDefault();show(current+1)}if(e.key==='ArrowLeft'){e.preventDefault();show(current-1)}});const initial=items.findIndex(r=>r.id===location.hash.slice(1));show(initial<0?0:initial);</script></body></html>'''.replace('OPTIONS', options).replace('PAYLOAD', payload)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    lines = ["# 第五讲：视频关键帧与短动画", "", "本素材包供教师逐图讲解，图册不是作者官方 PPT。已核对的作者公开视频目录有动画源码与实验笔记，未找到成套 PPT。", "",
             "## 来源与规格", "", f"- 中文视频：[B站 技术Flow]({SOURCE_URL})，BV1Z9ec69Ezz，CID 41943437272，33:51。",
             "- 原作：[Welch Labs／3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY)。", f"- [作者动画源码]({AUTHOR_URL})；只检索了公开材料，不声称所有渠道都没有 PPT。",
             "- 14 张 PNG 直接提取自可访问的 852×480 视频流；保留完整画面与中文字幕，没有放大冒充高清。",
             "- 4 段 GIF：640×360、10 fps，合计 36 秒；另有 852×480、30 fps 无声 MP4，便于暂停。",
             "- 时间戳均对应中文版。未把英文版的章节时间直接套用。", "- 保留原作者与转载来源；没有重新分发完整视频。编译课件不下载媒体、不调用 FFmpeg。", "",
             "## 使用", "", "直接双击 `index.html` 可离线讲解；也可通过课程网站打开。左右方向键换图，GIF 默认静态，点击播放或恢复静态。MP4 有暂停与进度控件。PNG／GIF 可插入自己的 PPT。", "",
             "## 关键帧", "", "| 图片 | 中文版时间 | 建议章节 | 内容 |", "|---|---|---|---|"]
    for x in frames:
        lines.append(f'| [{x["id"]}]({x["file"]}) | [{x["time"]}]({x["source_url"]}) | {x["lecture_section"]} | {x["title"]} |')
    lines += ["", "## 短动画", "", "| 动画 | 区间 | 时长 |", "|---|---|---|"]
    for x in animations:
        lines.append(f'| [{x["title"]}]({x["gif"]}) · [MP4]({x["mp4"]}) | {x["start_time"]}—{x["end_time"]} | {x["duration_seconds"]} 秒 |')
    lines += ["", "## 逐图讲解", ""]
    for x in frames:
        lines += [f'### {x["time"]} · {x["title"]}', "", x["talk"], "", "口径：" + x["caution"], ""]
    lines += ["## 重建", "", "只在需要重新截取时运行，不属于常规课件编译：", "", "```bash", "python3 lectures/scripts/extract_lecture_05_video.py --source /path/to/source.mp4 --ffmpeg /path/to/ffmpeg", "```", "", "输入必须是相同剪辑、相同时间轴的中文版视频；换成英文版时需要重新选时刻。来源与文件校验和保存在 `manifest.json`。"]
    (OUT / "README.md").write_text('\n'.join(lines) + '\n', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg or not args.source.is_file():
        parser.error("A local source video and FFmpeg executable are required.")
    OUT.mkdir(parents=True, exist_ok=True)
    def run(*command):
        subprocess.run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *map(str, command)], check=True)
    manifest = {"source_video": SOURCE_URL, "english_original": "https://www.youtube.com/watch?v=iv-5mZ_9CPY",
                "author_code": AUTHOR_URL, "checked_at": "2026-10-19", "bvid": "BV1Z9ec69Ezz", "cid": 41943437272,
                "native_resolution": [852, 480], "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "scope": "Selected screenshots and short silent excerpts; not an official PPT or a complete transcript.",
                "frames": [], "animations": []}
    for name, t, title, section, talk, caution in FRAMES:
        file = name + '.png'
        run('-ss', t, '-i', args.source, '-frames:v', 1, OUT / file)
        manifest['frames'].append({"id":name,"file":file,"timestamp_seconds":t,"time":stamp(t),"title":title,
                                   "lecture_section":section,"talk":talk,"caution":caution,"source_url":SOURCE_URL+f"/?t={t}"})
    for name, t, duration, title, talk, poster_id in ANIMATIONS:
        gif, mp4 = name + '.gif', name + '.mp4'
        run('-ss',t,'-i',args.source,'-t',duration,'-an','-vf','fps=30,setsar=1',
            '-c:v','libx264','-preset','medium','-crf','21','-movflags','+faststart',OUT/mp4)
        run('-i',OUT/mp4,'-filter_complex',
            '[0:v]fps=10,scale=640:360:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
            '-loop','0',OUT/gif)
        poster = name + '-poster.png'
        run('-ss',t,'-i',args.source,'-frames:v',1,OUT/poster)
        manifest['animations'].append({"id":name,"gif":gif,"mp4":mp4,"poster":poster,"related_frame":poster_id,
                                       "start_seconds":t,"end_seconds":t+duration,"duration_seconds":duration,
                                       "start_time":stamp(t),"end_time":stamp(t+duration),"title":title,"talk":talk,
                                       "caution":"片段中的结构与符号服务于视频讲解；本课固定 SD 1.5 配置和 ε 预测约定。", "source_url":SOURCE_URL+f"/?t={t}"})
    make_index(manifest)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json']
    manifest['files'] = {p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Extracted {len(FRAMES)} PNGs and {len(ANIMATIONS)} GIF/MP4 pairs to {OUT}')


if __name__ == '__main__':
    main()
