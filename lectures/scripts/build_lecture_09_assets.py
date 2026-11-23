"""Build original video experiments. Optional FFmpeg/imageio-ffmpeg is build-only."""
from pathlib import Path
import sys, json, subprocess, shutil, wave, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demos.lecture_09_video import uniform_indices, pixel_motion
import torch

IMG=ROOT/'images/lecture_09';VID=ROOT/'videos/lecture_09';DEMO=ROOT/'demos'
FPS=8;N=96;SR=16000
FONT='/System/Library/Fonts/PingFang.ttc'


def font(size):
    return ImageFont.truetype(FONT,size) if Path(FONT).exists() else ImageFont.load_default()


def frame(index):
    t=index/FPS;im=Image.new('RGB',(640,360),'#f1f6fa');d=ImageDraw.Draw(im)
    d.text((20,14),'操作台视频 · 原创示意动画',font=font(22),fill='#153553')
    d.ellipse((574,55,596,77),fill='#ef4750' if index in [10,11] else '#94b7ac')
    d.text((525,83),'状态灯',font=font(15),fill='#153553')
    d.rectangle((25,285,615,305),fill='#b9c9d5');d.rectangle((65,305,80,355),fill='#8ba2b3')
    d.rectangle((560,305,575,355),fill='#8ba2b3')
    d.ellipse((111,83,175,150),fill='#eac4a0',outline='#153553',width=2)
    d.ellipse((155,110,160,116),fill='#153553');d.line((162,133,174,133),fill='#153553',width=2)
    d.rounded_rectangle((104,153,177,278),radius=20,fill='#3c7b92')
    if t<3:cx=470-120*min(t/2.5,1);cy=255
    elif t<6:cx=350;cy=255
    elif t<9:cx=350-151*min((t-6)/1.2,1);cy=255-94*min((t-6)/1.2,1)
    else:cx=199+151*min((t-9)/1.3,1);cy=161+94*min((t-9)/1.3,1)
    d.line((165,182,cx-14,cy+10),fill='#eac4a0',width=13)
    # Transparent cup with a changing water level. No burned-in action labels or time IDs.
    d.polygon([(cx-19,cy-27),(cx+19,cy-27),(cx+15,cy+25),(cx-15,cy+25)],fill='#ffffff',outline='#27475e')
    fill=max(0,min(1,(t-3.6)/2.1))*.65
    if t>7.3:fill=max(.15,fill-(t-7.3)*.2)
    water_y=cy+22-fill*46
    d.polygon([(cx-14,water_y),(cx+14,water_y),(cx+12,cy+21),(cx-12,cy+21)],fill='#68bce1')
    d.arc((cx+10,cy-14,cx+35,cy+14),-90,90,fill='#27475e',width=3)
    if 3<t<6:
        jx=550-170*min((t-3)/.5,1);jy=205
        d.polygon([(jx-20,jy-45),(jx+25,jy-45),(jx+15,jy+8),(jx-20,jy+8)],fill='#d5e9f2',outline='#27475e')
        d.polygon([(jx-20,jy-45),(jx-42,jy-32),(jx-20,jy-29)],fill='#d5e9f2',outline='#27475e')
        if 3.6<t<5.7:d.line((jx-40,jy-32,cx,cy-21),fill='#4aa6d0',width=4)
    else:
        d.rectangle((530,215,570,282),fill='#d5e9f2',outline='#27475e');d.arc((555,225,589,268),-90,90,fill='#27475e',width=4)
    return im


def ffmpeg_binary():
    executable=shutil.which('ffmpeg')
    if executable:return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(frames,name,audio=None):
    cmd=[ffmpeg_binary(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
         '-pix_fmt','rgb24','-s','640x360','-r',str(FPS),'-i','pipe:0']
    if audio:cmd+=['-i',str(audio),'-c:a','aac','-b:a','96k','-shortest']
    cmd+=['-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(VID/name)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for im in frames:p.stdin.write(np.asarray(im,dtype=np.uint8).tobytes())
    finally:p.stdin.close()
    if p.wait()!=0:raise RuntimeError('FFmpeg encode failed')


def save(name):
    plt.tight_layout();plt.savefig(IMG/name,dpi=160,bbox_inches='tight');plt.close()


def diagram(name,title,boxes,arrows=()):
    fig,ax=plt.subplots(figsize=(11,5));ax.set_xlim(0,11);ax.set_ylim(0,5);ax.axis('off');ax.set_title(title,fontsize=18,pad=15)
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#56778a'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12,color='#153553')
    for a,b,label in arrows:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#3c657a',lw=2))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,label,ha='center',fontsize=11)
    save(name)


def main():
    IMG.mkdir(parents=True,exist_ok=True);VID.mkdir(parents=True,exist_ok=True)
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT);plt.rcParams['font.family']=font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({'font.size':12,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc','text.color':'#153553'})
    frames=[frame(i) for i in range(N)]
    (IMG/'frames').mkdir(exist_ok=True)
    for i,im in enumerate(frames):im.resize((320,180)).save(IMG/'frames'/f'frame_{i:03d}.jpg',quality=85)
    compact=np.stack([np.asarray(im.resize((160,90),Image.Resampling.BILINEAR)) for im in frames])
    np.savez_compressed(IMG/'teaching_clip.npz',frames=compact,timestamps=np.arange(N,dtype=np.float64)/FPS)
    time=np.arange(N*SR//FPS)/SR;rng=np.random.default_rng(9);audio=np.zeros(len(time),dtype=np.float32)
    mask=(time>=1.25)&(time<1.42);audio[mask]=.18*np.sin(2*np.pi*880*time[mask])
    pour=(time>=3.6)&(time<5.7);noise=rng.normal(0,.055,len(time));audio[pour]=noise[pour]
    click=(time>=9.75)&(time<9.95);audio[click]+=.18*np.sin(2*np.pi*1800*time[click])*np.exp(-30*(time[click]-9.75))
    wav=VID/'operation-audio.wav'
    with wave.open(str(wav),'wb') as w:w.setparams((1,2,SR,0,'NONE','not compressed'));w.writeframes(np.rint(audio*32767).astype('<i2').tobytes())
    order=torch.randperm(N,generator=torch.Generator().manual_seed(9)).tolist()
    indices=uniform_indices(N,8).tolist()
    encode(frames,'operation.mp4',wav)
    from demos.lecture_09_extract import decode_teaching_clip
    decoded=decode_teaching_clip(VID/'operation.mp4')
    assert decoded.shape==(96,90,160,3)
    np.savez_compressed(IMG/'teaching_clip.npz',frames=decoded,timestamps=np.arange(N,dtype=np.float64)/FPS)
    encode(list(reversed(frames)),'reversed.mp4')
    encode([frames[i] for i in order],'shuffled.mp4')
    encode([frames[indices[min(i//12,7)]] for i in range(N)],'sparse-eight.mp4')
    subprocess.run([ffmpeg_binary(),'-hide_banner','-loglevel','error','-y','-i',str(VID/'operation.mp4'),
                    '-itsoffset','0.5','-i',str(wav),'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac',
                    '-t','12','-movflags','+faststart',str(VID/'audio-delayed.mp4')],check=True)
    poster=Image.new('RGB',(1280,800),'#f8fafc');d=ImageDraw.Draw(poster)
    for j,i in enumerate(indices):
        x=j%4*320;y=j//4*400;im=frames[i].resize((320,180));poster.paste(im,(x,y+25));d.text((x+12,y+210),f'原视频 {i/FPS:.2f} 秒 · 帧 {i}',font=font(18),fill='#153553')
    poster.crop((0,0,1280,670)).save(IMG/'storyboard.png')
    lamp=Image.new('RGB',(960,240),'#f8fafc');d=ImageDraw.Draw(lamp)
    for j,i in enumerate([8,10,12]):
        lamp.paste(frames[i].resize((320,180)),(j*320,0));d.text((j*320+15,190),f'{i/FPS:.2f} 秒 · 帧 {i}',font=font(20),fill='#153553')
    lamp.save(IMG/'brief-event.png')
    fig,ax=plt.subplots(figsize=(10,3.8));ax.plot(np.arange(N)/FPS,np.isin(np.arange(N),[10,11]),label='短闪真实区间')
    ax.scatter(np.arange(0,N,4)/FPS,np.full(24,.15),label='2 FPS 采样',marker='|',s=180)
    ax.scatter(np.arange(0,N,2)/FPS,np.full(48,.35),label='4 FPS 采样',marker='|',s=140)
    ax.set_xlim(.7,2);ax.set_xlabel('原视频时间 / 秒');ax.set_ylabel('状态示意');ax.legend();ax.set_title('相位与间隔决定短事件是否被采到');save('sampling.png')
    fig,ax=plt.subplots(figsize=(10,3.8));motion=pixel_motion(torch.from_numpy(decoded).permute(0,3,1,2));ax.plot(np.arange(1,N)/FPS,motion)
    ax.set_xlabel('相邻帧后一帧的时间 / 秒');ax.set_ylabel('平均绝对像素差 / 255');ax.set_title('像素变化是低层线索，不等于理解动作');save('motion.png')
    blue='#d7eaf1';green='#dcefe5';orange='#f1dec9'
    diagram('video-language.png','从视频证据到语言答案',[(.2,2,1.7,1.1,'采样帧\n＋真实时间戳',blue),(2.7,2,1.7,1.1,'视觉编码器\n空间特征',blue),(5.2,2,1.9,1.1,'时间组织／压缩\n＋连接器',green),(8,2,2.5,1.1,'语言模型\n问题 → 答案',orange)], [((1.9,2.55),(2.7,2.55),''),((4.4,2.55),(5.2,2.55),''),((7.1,2.55),(8,2.55),'')])
    diagram('tubelets.png','教学视频块：同时保留时间、空间和颜色轴',[(.3,2,2.2,1.2,'8 帧 × 3 × 32 × 32\n[T,C,H,W]',blue),(3.7,2,2.4,1.2,'2 帧 × 8 × 8\n每块 384 个数',green),(7.5,2,2.5,1.2,'4 × 4 × 4 = 64 块\n[64,384]',orange)],[((2.5,2.6),(3.7,2.6),''),((6.1,2.6),(7.5,2.6),'')])
    diagram('qwen-dimensions.png','Qwen2.5-VL-7B：8 帧、224×224 的形状演算',[(.1,2.1,2.1,1.3,'像素 [1,8,3,224,224]\n2 帧 × 14×14 块\n1024 个 tubelet',blue),(3.1,2.1,1.9,1.3,'ViT 隐藏维 1280\n[4,16,16,1280]',blue),(5.9,2.1,2,1.3,'相邻 2×2 合并\n[256,5120]',green),(8.8,2.1,2,1.3,'MLP → LM 空间\n[256,3584]',orange)],[((2.2,2.75),(3.1,2.75),''),((5,2.75),(5.9,2.75),''),((7.9,2.75),(8.8,2.75),'')])
    diagram('architecture-routes.png','三种视频理解接口：时间信息在哪里进入？',[(.2,3.3,2.5,.8,'逐帧编码＋压缩',blue),(3.6,3.3,6.6,.8,'LLaVA-OneVision：帧词元按序进入语言模型',green),(.2,2,2.5,.8,'时空连接器',blue),(3.6,2,6.6,.8,'VideoLLaMA 2：STC 提前融合与压缩时间空间',green),(.2,.7,2.5,.8,'动态采样＋时间位置',blue),(3.6,.7,6.6,.8,'Qwen2.5-VL：双帧块、合并器、绝对时间 MRoPE',green)],[((2.7,3.7),(3.6,3.7),''),((2.7,2.4),(3.6,2.4),''),((2.7,1.1),(3.6,1.1),'')])
    diagram('grounding.png','时间定位：先约定动作区间的标注标准',[(.4,3.2,2.4,.8,'问题：“何时有水流？”',blue),(3.7,3.2,2.6,.8,'预测 [3.0,6.0) 秒',orange),(7.4,3.2,2.7,.8,'参考 [3.6,5.7) 秒',green),(.6,1.2,9.4,.9,'交集 2.1 秒 ÷ 并集 3 秒 = 时间 IoU 0.70',blue)],[((2.8,3.6),(3.7,3.6),''),((6.3,3.6),(7.4,3.6),'')])
    metadata={'fps':FPS,'frame_count':N,'duration':12.,'frames_shape':[N,90,160,3],
              'uniform_eight':indices,'shuffle_order':order,'events':[{'label':'取杯','start':0,'end':3},{'label':'倒水','start':3,'end':6},{'label':'端杯饮水','start':6,'end':9},{'label':'放回杯子','start':9,'end':12}],
              'lamp_interval':[1.25,1.5],'water_flow_interval':[3.6,5.7],'provenance':'Original programmed illustration and analytic sounds. Event boundaries are authored segment labels, not learned annotations.'}
    (DEMO/'lecture_09_assets.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (IMG/'timeline.json').write_text(json.dumps(metadata,ensure_ascii=False)+'\n')
    (IMG/'timeline-data.js').write_text('window.clipMetadata = '+json.dumps(metadata,ensure_ascii=False)+';\n')
    official=Path('/private/tmp/lecture09-mova-source.mp4')
    if official.exists():
        subprocess.run([
            ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(official), '-map', '0:v:0', '-map', '0:a:0',
            '-vf', 'scale=768:-2', '-c:v', 'libx264', '-crf', '24',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', str(VID / 'mova-official-full.mp4'),
        ], check=True)
    manifest=[]
    for path in sorted(list(IMG.rglob('*'))+list(VID.glob('*'))):
        if path.is_file():manifest.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (DEMO/'lecture_09_sources.json').write_text(json.dumps({'checked':'2026-11-23','media':manifest,
        'mova_sample_source':'https://github.com/OpenMOSS/MOVA (first embedded video; complete, classroom encode)',
        'mova_sample_url':'https://gist.github.com/user-attachments/assets/cee573cc-56ce-4987-beef-0b374e1ed3b7'},ensure_ascii=False,indent=2)+'\n')
    print('Built original 12-second clips, figures and provenance.')


if __name__=='__main__':main()
