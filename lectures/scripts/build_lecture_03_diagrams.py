"""Deterministic SVG teaching diagrams, no model or network required."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / 'images/lecture_03'
OUT.mkdir(parents=True, exist_ok=True)


def txt(x, y, s, size=21, color='#17334a'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(s)}</text>'


def box(x, y, w, h, lines, color='#e0f0ef'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#609899"/>' + ''.join(txt(x+16,y+30+i*28,s,19) for i,s in enumerate(lines))


def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#477b8b" stroke-width="3" marker-end="url(#a)"/>'


def save(name,title,body,foot='',h=450):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{h}" viewBox="0 0 1100 {h}"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L0 6L8 3Z" fill="#477b8b"/></marker></defs><rect width="1100" height="{h}" rx="18" fill="#f7fafc"/><g font-family="sans-serif">'+txt(35,45,title,28)+body+txt(35,h-25,foot,17)+'</g></svg>'
    (OUT/name).write_text(s)


def main():
    b=txt(35,105,'文本分支',22)+txt(35,265,'图像分支',22)
    for y,names in [(125,[['词／子词向量'],['RNN：传递状态'],['Transformer','按内容聚合信息']]),(285,[['像素数组'],['CNN：局部特征'],['ViT','图像块序列']])]:
        for x,lines in zip([35,290,550],names): b+=box(x,y,215,78,lines)
        b+=arrow(253,y+40,280,y+40)+arrow(508,y+40,540,y+40)
    b+=box(835,195,225,90,['CLIP：图文对齐','共享可比较的空间'],'#ffebd2')+arrow(770,164,825,220)+arrow(770,325,825,265)
    save('routes.svg','两条表示学习路线，在 CLIP 中汇合',b,'路线是本课的叙述顺序；不是所有模型都经历相同阶段，CLIP 图像分支也可用 CNN。')
    b=txt(35,98,'RNN：信息沿状态链传递',22)
    for i,word in enumerate(['小王','把书','递给','小李']):
        x=65+i*250;b+=box(x,120,180,60,[f'状态 h{i+1}'])+txt(x+30,215,word)
        if i<3:b+=arrow(x+185,150,x+240,150)
    b+=txt(35,282,'自注意力：每个位置按相关程度聚合其他位置的信息',22)
    for i,word in enumerate(['小王','把书','递给','小李']):
        x=65+i*250;b+=box(x,340,180,55,[word])
        b+=arrow(x+90,400,600,425)
    b+=box(480,430,260,55,['聚合为“递给”的新表示'])
    save('rnn-attention.svg','从“逐步传递的笔记”到“按需参考上下文”',b,'箭头仅表示信息通路；没有展示真实模型学到的权重，类比不等同于人类阅读机制。',545)
    b=box(35,125,220,125,['局部窗口：3×3','同一组权重','在图像上滑动'])+arrow(265,185,320,185)+box(335,125,205,125,['特征图','记录各位置','对某种模式的响应'])+arrow(550,185,605,185)+box(620,125,200,125,['多层组合','更大的感受野','局部 → 更大结构'])+arrow(830,185,875,185)+box(885,125,180,125,['整图表示','分类或对齐'])
    b+=txt(45,320,'语言类比：局部搭配 → 更大结构；但卷积窗口不是语言学短语边界。',21)
    save('cnn.svg','CNN：同一个局部检测器，寻找不同位置的模式',b,'感受野是某个特征能够受到输入影响的区域；示意不代表每层必然对应可命名的概念。',390)
    b=''
    for y in range(4):
        for x in range(4):
            c=['#adcde1','#78b7aa','#f6c789','#dc8663'][(x+y)%4]
            b+=f'<rect x="{35+x*45}" y="{125+y*45}" width="42" height="42" fill="{c}"/>'
    b+=txt(35,338,'图像切为 4×4 块',18)+arrow(230,210,280,210)+box(295,155,210,105,['每块展平','共享线性投影','加位置向量'])+arrow(515,210,560,210)+box(575,155,225,105,['图像块 token 序列','Transformer','跨块交互'])+arrow(810,210,850,210)+box(865,155,200,105,['汇总向量','图像表示'])
    save('vit.svg','ViT：图像块像 token 一样进入 Transformer',b,'图像块不是天然的词：一个对象可以跨越多个块，一个块也可能包含多个对象。')
    b=box(35,110,210,75,['图像 batch','I₁、I₂、I₃'])+arrow(255,148,310,148)+box(325,110,230,75,['图像编码器','CNN 或 ViT'])+arrow(565,148,620,148)+box(635,110,200,75,['投影＋归一化','图像向量 U'])
    b+=box(35,245,210,75,['对应描述 batch','T₁、T₂、T₃'])+arrow(255,283,310,283)+box(325,245,230,75,['文本编码器','Transformer'])+arrow(565,283,620,283)+box(635,245,200,75,['投影＋归一化','文本向量 V'])
    b+=arrow(845,148,885,202)+arrow(845,283,885,242)+box(895,170,170,110,['两两相似度','3×3 分数表','双向匹配损失'],'#ffebd2')
    save('clip.svg','CLIP：两条分支独立编码，用配对关系共同训练',b,'原始 CLIP 没有在两条分支之间做逐 token 的交叉注意力；它比较整图与整段文本的向量。')
    b=txt(35,110,'每行：一张图像；每列：一段描述。绿色对角线是给定配对。',21)
    vals=[[8,2,1],[1,8,2],[2,1,8]]
    for i in range(3):
        b+=txt(235,195+i*65,f'图像 {i+1}',20)+txt(380+i*150,142,f'文本 {i+1}',20)
        for j in range(3):b+=box(350+j*150,160+i*65,130,53,[str(vals[i][j])], '#bde4d8' if i==j else '#edf0f4')
    b+=txt(35,400,'逐行：这张图匹配哪段文字？     逐列：这段文字匹配哪张图？',21)
    save('contrastive.svg','对比学习：让配对分数相对其他候选更高',b,'这里是人为构造的分数示意，不是真实 CLIP 输出；其他项可能是语义相关的“假负例”。',460)

if __name__=='__main__':
    main()
