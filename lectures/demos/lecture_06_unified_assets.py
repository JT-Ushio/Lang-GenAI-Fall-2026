"""Original diagrams and measured outputs for unified multimodal models."""
import json
from pathlib import Path
import numpy as np
from demos.lecture_06_assets import txt,box,arrow,save,embed
from demos.lecture_06_unified import COLORS,PALETTE

OUT=Path(__file__).resolve().parents[1]/'images/lecture_06/unified'


def figure(name,title,body,foot,height=500):
    OUT.mkdir(parents=True,exist_ok=True)
    save('unified/'+name,title,body,foot,height)


def picture(pixels,x,y,width=200):
    if pixels is None:return box(x,y,width,width,['输出格式不合法'],'#fff0da',19)
    rgb=np.array(PALETTE)[np.array(pixels)]/255
    return embed(rgb.transpose(2,0,1),x,y,width)


def matrix(values,x,y,title,cell=41):
    body=txt(x,y-28,title,22,weight=600)
    labels=['T1','T2','I1','I2','T3','T4']
    for i,label in enumerate(labels):
        body+=txt(x+38+i*cell,y+18,label,16)+txt(x,y+55+i*cell,label,16)
    for row,items in enumerate(values):
        for col,value in enumerate(items):
            body+=f'<rect x="{x+34+col*cell}" y="{y+31+row*cell}" width="{cell-3}" height="{cell-3}" rx="3" fill="{"#307f82" if value else "#e5ebf2"}"/>'
    return body


def prepare_assets(data):
    OUT.mkdir(parents=True,exist_ok=True)
    records=data['records'];report=data['report']
    row=next(r for r in records if r['task']=='理解' and r['left']==0 and r['right']==2)
    b=picture(row['source'],35,125,230)
    b+=txt(35,100,'4×4 图像：左红、右蓝',22)
    for i,line in enumerate([[0,0,0,0],[1,1,3,3],[1,1,3,3],[0,0,0,0]]):
        b+=txt(355,150+50*i,' '.join(str(x) for x in line),30)
    b+=txt(355,105,'固定调色板编号',22)
    b+=box(625,125,340,195,['0：背景 → 词元 14','1：红色 → 词元 15','2：绿色 → 词元 16','3：蓝色 → 词元 17','图像词元与文字词元分区'],size=20)
    b+=txt(35,420,'按行展开为 16 个图像词元，加上图像起止标记，即可进入序列。',22)
    figure('palette-tokens.svg','图像也能进入词元序列：固定调色板小例子',b,'这是可逆的四颜色数组编码；真实模型可使用训练得到的视觉量化器。',490)

    triangular=[[int(j<=i) for j in range(6)] for i in range(6)]
    b=matrix(triangular,40,130,'完全自回归：只能看已知前缀')
    b+=matrix(data['attention'],545,130,'模态分块：图像块内可双向')
    b+=txt(40,478,'行是查询，列是可读取的位置；绿色表示允许，灰色表示屏蔽。',21)
    b+=txt(40,522,'I1、I2 可互相读取；后面的 T3、T4 仍不能泄漏给前面的位置。',21)
    figure('attention-mask.svg','统一上下文不等于所有位置都互相可见',b,'右图仅展示已知条件块的简化规则，不包含 BAGEL 的干净／加噪视觉块隔离。',585)

    colors=['#307f82','#d58c35','#8065aa'];history=report['history'];max_loss=max(h[t] for h in history for t in ['理解','生成','编辑'])
    b='<path d="M75 110V365H630" fill="none" stroke="#9ab9c1" stroke-width="2"/>'
    for task,color in zip(['理解','生成','编辑'],colors):
        points=' '.join(f"{75+h['step']/report['updates']*545:.1f},{365-h[task]/max_loss*235:.1f}" for h in history)
        b+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
    b+=txt(75,400,'0',19)+txt(510,400,f"{report['updates']} 次更新",19)
    for i,(task,color) in enumerate(zip(['理解','生成','编辑'],colors)):
        score=report['training_set_free_running'][task]
        b+=txt(680,155+i*90,task+'：自由生成精确匹配',20,color,600)
        b+=txt(680,190+i*90,f"{score['correct']} / {score['total']}",25,color)
    b+=txt(40,460,'同一套参数；三种任务均参与训练。评估逐步生成，不喂标准答案前缀。',21)
    figure('training-results.svg','同一个微型 Transformer：看图、生成、编辑',b,'曲线为各任务交叉熵；72 个样本全部用于训练，因此这里检验拟合，不是泛化能力。',525)

    draw=next(r for r in records if r['task']=='生成' and r['left']==1 and r['right']==2)
    edit=next(r for r in records if r['task']=='编辑' and r['left']==0 and r['right']==2 and r['side']==0 and r['color']==1)
    b=txt(30,100,'图像 → 文字',23,weight=600)+picture(row['source'],30,125,220)
    caption=row['predicted_caption'];description=f'左：{COLORS[caption[0]]}；右：{COLORS[caption[1]]}' if caption else '格式错误'
    b+=txt(30,390,'模型回答：'+description,21)
    b+=txt(355,100,'文字 → 图像',23,weight=600)+picture(draw['predicted_image'],355,125,220)
    b+=txt(355,390,'条件：左绿、右蓝',21)
    b+=txt(680,100,'图像＋指令 → 图像',23,weight=600)+picture(edit['predicted_image'],680,125,220)
    b+=txt(680,390,'原图左红右蓝；改左为绿',20)
    figure('learned-tasks.svg','实际输出：三个任务共享一套模型参数',b,f"共享 {report['parameters']:,} 参数；固定词表和四颜色图像。选项映射为词元，不代表开放自然语言理解。",465)

    episode=data['episodes']['fault'];frames=[row['source']]+[step['after'] for step in episode['steps']]
    b=''
    for i,pixels in enumerate(frames[:4]):
        x=30+i*245;b+=txt(x,100,'原图' if i==0 else f'动作 {i}',22,weight=600)
        b+=picture(pixels,x,125,200)
        if i:
            step=episode['steps'][i-1]
            b+=txt(x,365,f"实际改{'左' if step['executed_side']==0 else '右'}为{COLORS[step['color']]}",20)
            b+=txt(x,405,'验收：'+('通过' if step['success'] else '未通过'),20)
    b+=txt(30,460,'目标：左变绿、右保持蓝。首次执行错位 → 恢复右侧 → 完成左侧修改。',21)
    figure('agent-trajectory.svg','执行、看结果、纠错：反馈改变下一步动作',b,'规划与验收为规则；看图回答和编辑都调用同一个训练后模型。首次错位由实验注入。',525)

    public={key:value for key,value in data.items() if not key.startswith('_')}
    (OUT/'data.js').write_text('window.UNIFIED_DATA='+json.dumps(public,ensure_ascii=False,separators=(',',':'))+';\n')
    (OUT.parent.parent.parent/'demos/lecture_06_unified_result.json').write_text(json.dumps({
        'report':report,'episodes':{key:{'status':ep['status'],'success':ep['success'],'steps':len(ep['steps'])}
                                  for key,ep in data['episodes'].items()}},ensure_ascii=False,indent=2)+'\n')
