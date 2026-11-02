"""Rebuild measured plots and offline listening assets. Run from repository root."""
from pathlib import Path
import sys, os, subprocess, json, wave, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lecture07-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demos.lecture_07_speech import read_wav, features, collapse, optimize_ctc, attention_demo, ctc_example
IMG=ROOT/'images/lecture_07'; VID=ROOT/'videos/lecture_07'
IMG.mkdir(parents=True,exist_ok=True); VID.mkdir(parents=True,exist_ok=True)
font=Path('/System/Library/Fonts/PingFang.ttc')
if font.exists():
 font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F8FAFC','axes.facecolor':'#F8FAFC','text.color':'#153553','axes.labelcolor':'#153553'})
def save(name):
 plt.tight_layout();plt.savefig(IMG/name,dpi=155,bbox_inches='tight',pad_inches=.2);plt.close()
def wav(path,x,rate=16000):
 with wave.open(str(path),'wb') as f:
  f.setparams((1,2,rate,0,'NONE','not compressed')); f.writeframes((np.clip(x,-1,1)*32767).astype('<i2').tobytes())
# Chinese master synthesized with macOS Tingting, 175 words/minute.
rate,x=read_wav(VID/'speech-source.wav')
x = x * (0.45 / max(float(x.abs().max()), 1e-6))
wav(VID/'speech.wav',x.numpy())
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(VID/'speech.wav'),'-af','atempo=0.75',str(VID/'speech-slow.wav')],check=True)
_,slow=read_wav(VID/'speech-slow.wav')
fig,axs=plt.subplots(2,1,figsize=(10,4.7))
for a,y,n in zip(axs,[x,slow],['正常语速','慢速，保持音高']):
 a.plot(np.arange(len(y))/rate,y,color='#167D8D',linewidth=.5);a.set_title(f'{n}：{len(y)/rate:.2f} 秒',loc='left');a.set_ylim(-.55,.55);a.set_ylabel('振幅')
axs[-1].set_xlabel('时间 / 秒');save('duration.png')
fig,axs=plt.subplots(2,1,figsize=(10,4.5))
axs[0].plot(np.arange(len(x))/rate,x,color='#167D8D',linewidth=.5);axs[0].set_xlabel('时间 / 秒');axs[0].set_ylabel('振幅')
i=int(np.argmax(np.abs(x.numpy())));start=max(0,i-160);z=x[start:start+320]
axs[1].plot((np.arange(len(z))+start)/rate,z,'.-',color='#D16B36',markersize=2,linewidth=.7);axs[1].set_xlabel('局部时间 / 秒');axs[1].set_ylabel('振幅');save('waveform.png')
fig,ax=plt.subplots(figsize=(10,3.8)); t=np.arange(1000)/16
ax.plot(t,x[start:start+1000],linewidth=.8)
for j,c in zip([0,10,20],['#167D8D','#D16B36','#6975C7']):
 ax.axvspan(j,j+25,alpha=.15,color=c,label=f'窗口从 {j} ms 开始')
ax.set_xlabel('相对时间 / ms');ax.set_ylabel('振幅');ax.legend(loc='upper right');save('window.png')
t=np.arange(400)/16000; sig=np.sin(2*np.pi*300*t)+.5*np.sin(2*np.pi*1000*t)
fig,axs=plt.subplots(2,1,figsize=(9,4.5));axs[0].plot(t*1000,sig);axs[0].set_xlabel('时间 / ms');axs[0].set_ylabel('振幅')
sp=np.abs(np.fft.rfft(sig*np.hanning(400)))**2
axs[1].plot(np.fft.rfftfreq(400,1/16000),sp,color='#D16B36');axs[1].set_xlim(0,2000);axs[1].set_xlabel('频率 / Hz');axs[1].set_ylabel('功率');save('fft.png')
power,mel=features(x)
for matrix,name,ylabel in [(10*torch.log10(power.clamp_min(1e-10)),'spectrogram.png','频率 / Hz'),(mel,'mel.png','梅尔频带编号')]:
 fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(matrix.numpy(),origin='lower',aspect='auto',extent=[0,len(x)/rate,0,8000 if name=='spectrogram.png' else 80],cmap='magma');ax.set_xlabel('时间 / 秒');ax.set_ylabel(ylabel);plt.colorbar(im,ax=ax,label='功率 / dB' if name=='spectrogram.png' else '缩放后的 log-Mel');save(name)
fig,ax=plt.subplots(figsize=(10,3.4))
for row,labels in enumerate([['你','你','空' ,'好','好','空' ],['空' ,'你','你','好','空' ,'空' ],['你','你','你','空' ,'好','好']]):
 for j,v in enumerate(labels):ax.text(j+.5,row+.5,v,ha='center',va='center',fontsize=26,color='#167D8D' if v=='你' else '#D16B36')
ax.set_xlim(0,6);ax.set_ylim(0,3);ax.set_yticks([.5,1.5,2.5],['路径 1','路径 2','路径 3']);ax.set_xticks(np.arange(6)+.5,[f't{i+1}' for i in range(6)]);ax.set_title('三条示意路径均折叠为“你好”，时间位置为手工构造');save('alignment.png')
fig,ax=plt.subplots(figsize=(9,3.5))
rows=[['哈','哈','空' ,'哈','哈'],['哈','空' ,'哈'],['哈','哈']]
for i,(label,row) in enumerate(zip(['原始路径','合并相邻相同','删除空白'],rows)):
 ax.text(-.4,2-i,label,ha='right',va='center',fontsize=16)
 for j,v in enumerate(row):ax.text(j+.2,2-i,v,ha='center',va='center',fontsize=30,color='#167D8D' if v!='空'  else '#8B94A5')
ax.set_xlim(-2.6,5);ax.set_ylim(-.6,2.6);ax.axis('off');save('collapse.png')
probs,p,valid,totals=ctc_example()
fig,ax=plt.subplots(figsize=(9,4));labels=[' '.join('好' if k else '空'  for k in path) for path,_ in valid];ax.barh(labels,[v for _,v in valid],color='#167D8D');ax.set_xlabel('路径概率');ax.set_title(f'折叠为“好”的 6 条路径，概率之和 {p:.3f}');save('ctc-prob.png')
fig,ax=plt.subplots(figsize=(8,4));ax.bar(['空串','好','好好'],[totals[()],totals[(1,)],totals[(1,1)]],color=['#9AA9BC','#167D8D','#D16B36']);ax.set_ylim(0,.8);ax.set_ylabel('文字序列概率');ax.set_title('逐帧最大值为 blank，文字概率最大值却是“好”');save('greedy.png')
probs,path,answer,history=optimize_ctc()
fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].plot([v['step'] for v in history],[v['loss'] for v in history],'o-',color='#167D8D');axs[0].set_xlabel('优化步数');axs[0].set_ylabel('CTC 负对数概率')
axs[1].imshow(probs.T.numpy(),aspect='auto',vmin=0,vmax=1,cmap='Blues');axs[1].set_yticks([0,1,2],['空白','看','一']);axs[1].set_xlabel('时间位置');axs[1].set_title('优化后逐帧概率，目标“看一看”');save('ctc-training.png')
a,_,_=attention_demo();fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(a.numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=1);ax.set_yticks(range(3),['文字位置 1','文字位置 2','文字位置 3']);ax.set_xlabel('声音位置');ax.set_title('手工构造的分数，经 softmax 得到软读取权重');plt.colorbar(im,ax=ax);save('attention.png')
manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in IMG.iterdir() if p.suffix in ('.npz','.json')}
report={'text':'你好，欢迎来到语音与语言对齐课堂。请听同一句话在不同语速下的变化。','voice':'macOS Tingting, zh_CN','rate':rate,'samples':len(x),'duration':len(x)/rate,'slow_duration':len(slow)/rate,'power_shape':list(power.shape),'mel_shape':list(mel.shape),'ctc_probability':float(ctc_example()[1]),'history':history,'decoded':list(answer),'decoded_text':''.join({1:'看',2:'一'}[v] for v in answer),'sources':{'official_figure':'https://raw.githubusercontent.com/openai/whisper/main/approach.png','mel_filters':'https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz','config':'https://huggingface.co/openai/whisper-tiny/raw/main/config.json'},'assets':manifest}
# p may have been shadowed by the manifest comprehension in other Python versions; explicit number.
report['ctc_probability']=float(ctc_example()[1])
(ROOT/'demos/lecture_07_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('sources','assets')},ensure_ascii=False,indent=2))
