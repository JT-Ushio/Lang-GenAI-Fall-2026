"""Pictorial and layered SVG examples; standard library only."""
from build_lecture_03_diagrams import txt, box, arrow, save


def polygon(points, color, stroke='#477b8b'):
    return f'<polygon points="{points}" fill="{color}" stroke="{stroke}" stroke-width="2"/>'


def slab(x, y, w, h, depth, color):
    return (polygon(f'{x},{y} {x+depth},{y-depth} {x+w+depth},{y-depth} {x+w},{y}', '#e8eff5')
            + polygon(f'{x+w},{y} {x+w+depth},{y-depth} {x+w+depth},{y+h-depth} {x+w},{y+h}', '#b6c8da')
            + f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="#477b8b" stroke-width="2"/>')


def cat(x, y, scale=1):
    return f'<g transform="translate({x} {y}) scale({scale})">' + polygon('10,23 8,0 28,14 52,14 72,0 70,23', '#edb56b') + '<ellipse cx="40" cy="36" rx="33" ry="27" fill="#edb56b"/><circle cx="28" cy="31" r="3" fill="#17334a"/><circle cx="52" cy="31" r="3" fill="#17334a"/><path d="M36 41 L44 41 L40 46Z M9 41 L-1 38 M9 48 L-1 51 M71 41 L81 38 M71 48 L81 51" stroke="#17334a" fill="#17334a"/></g>'


def vector(x, y, values):
    s=''
    for i,v in enumerate(values):
        s+=f'<rect x="{x}" y="{y+i*30}" width="145" height="25" rx="4" fill="#e1e9ef"/>'
        s+=f'<rect x="{x}" y="{y+i*30}" width="{145*v}" height="25" rx="4" fill="{["#77b9b2","#f2bd79"][i]}"/>'
        s+=txt(x+155,y+20+i*30,f'{v:.3f}',19)
    return s


def rnn():
    b=txt(35,95,'每读一个词，就改写一次“阅读笔记”；两个方向使用完全相同的更新规则。',22)
    for y,words,states in [(140,['猫','吃','鱼'],[(.762,0),(.707,.462),(.339,.843)]),(360,['鱼','吃','猫'],[(0,.762),(.462,.707),(.843,.339)])]:
        b+=txt(35,y+60,'从零开始',18)
        for i,(word,state) in enumerate(zip(words,states)):
            x=160+i*310
            b+=box(x,y,260,170,[f'读入「{word}」'],'#ffffff')
            b+=txt(x+16,y+64,f'第 {i+1} 步状态',18)+vector(x+16,y+80,state)
            if i<2:b+=arrow(x+265,y+86,x+300,y+86)
    b+=txt(35,585,'相同词语，顺序不同 → 最后留下的向量不同。',25)
    save('rnn-memory.svg','RNN：猫吃鱼 / 鱼吃猫，笔记怎样逐步改变？',b,'数值对应本课固定递推的结果（保留三位小数）；两个维度没有预先指定的语言学含义。',650)


def cnn():
    b=txt(35,95,'把 RGB 图片想成三张叠在一起的数表；一个卷积核要同时看三个通道。',22)
    for i,c in [(2,'#d9e4f7'),(1,'#dceede'),(0,'#f4d9d5')]:
        b+=slab(65+i*22,180-i*22,210,210,15,c)
    for k in range(1,7):
        b+=f'<path d="M{65+k*30} 180 V390 M65 {180+k*30} H275" stroke="#ffffff"/>'
    b+='<rect x="125" y="240" width="90" height="90" fill="#f3c477" fill-opacity=".75" stroke="#bf6c2f" stroke-width="3"/>'
    b+=txt(65,435,'输入：H × W × 3',22)+txt(65,470,'橙色框沿高、宽滑动',20)
    b+=arrow(345,265,400,265)
    for i,c in [(2,'#d9e4f7'),(1,'#dceede'),(0,'#f4d9d5')]:
        b+=slab(425+i*17,220-i*17,90,90,12,c)
    b+=txt(402,365,'一个核：3 × 3 × 3',21)+txt(402,405,'27 次乘法再求和',20)+txt(402,435,'（再加一个偏置）',19)
    b+=arrow(575,265,640,265)
    b+=slab(665,185,155,155,18,'#d4e9e5')
    b+='<rect x="720" y="240" width="24" height="24" fill="#d88a42"/>'
    b+=txt(635,390,'一个位置 → 一个数',21)+txt(635,425,'滑完整图 → 一张特征图',19)
    for i in reversed(range(4)):
        b+=slab(915+i*15,225-i*22,85,120,10,['#daeee8','#eadff1','#fae8cf','#dce7f6'][i])
    b+=txt(865,460,'多个核 → 多张特征图',19)
    b+=txt(35,540,'输入深度是通道，输出深度是检测器数量；图中的“厚度”不是物体的真实三维深度。',21)
    save('cnn-volume.svg','CNN：一个小窗口，穿过所有颜色通道',b,'扩展示意采用 RGB；本课可执行卷积例子使用单通道灰度图，因此核只有 3×3×1。',605)


def clip_pairs():
    b=txt(35,95,'同一件事，可以用图像表达，也可以用文字描述。CLIP 学习两者的对应关系。',22)
    b+=box(35,130,225,150,['图像'],'#ffffff')+cat(100,180)
    b+=box(35,330,225,100,['描述','a photo of a cat'],'#ffffff')
    b+=arrow(270,205,320,205)+arrow(270,380,320,380)
    b+=slab(340,160,205,95,18,'#d7ebe6')+txt(358,201,'图像编码器',23)+txt(358,232,'CNN 或 ViT',20)
    b+=slab(340,340,205,95,18,'#e5def0')+txt(358,381,'文本编码器',23)+txt(358,412,'Transformer',20)
    b+=arrow(575,205,635,205)+arrow(575,380,635,380)
    for y,c in [(175,'#78b7aa'),(350,'#ad91bc')]:
        for i,w in enumerate([45,65,30,55]):
            b+=f'<rect x="655" y="{y+i*20}" width="{w}" height="14" rx="3" fill="{c}"/>'
    b+=txt(630,286,'图像向量',20)+txt(630,462,'文字向量',20)
    b+=arrow(735,210,805,275)+arrow(735,385,805,325)
    b+=box(820,235,245,140,['投影并归一化后','比较向量方向','匹配项得分更高'],'#ffebd2')
    b+=txt(35,525,'像两位用不同方式做笔记的同学：各自整理，再比较是否在说同一件事。',22)
    save('clip-pictorial.svg','CLIP：让图像和文字的“笔记”可以比较',b,'条形只是向量符号，不是真实特征；两条分支独立编码，通过图文配对共同训练。',590)


def clip_candidates():
    b=box(35,130,235,180,['待匹配图片'],'#ffffff')+cat(105,205,1.1)
    b+=arrow(285,220,350,220)+box(370,160,215,120,['图像向量','与每个文字向量','分别计算相似度'])
    labels=['a photo of a cat','a photo of a fish','a photo of a bus']
    for i,(label,score) in enumerate(zip(labels,[.82,.24,.11])):
        y=115+i*125
        b+=box(650,y,400,95,[label],'#ffffff')
        b+=f'<rect x="668" y="{y+49}" width="{score*320}" height="22" rx="5" fill="{"#78b7aa" if i==0 else "#cbd8e4"}"/>'
        b+=txt(955,y+68,f'{score:.2f}',21)
        b+=arrow(595,220,638,y+45)
    b+=txt(35,535,'从已有描述中选最匹配的一项；改变候选描述，就改变了比较任务。',23)
    b+=txt(35,580,'“零样本分类”：用类别描述替代为这组类别另行训练的分类头。',22)
    save('clip-candidates.svg','CLIP 的使用：拿一张图片，与多个描述逐一比较',b,'分数为手工示意，不是真实 CLIP 输出，也不是正确率；CLIP 本身不会在这里生成一句新描述。',645)


def clip_learning():
    b=txt(35,95,'圆点代表图像，方块代表文字；同色代表已知配对。',22)
    for x,title in [(40,'训练前：配对尚未对齐'),(605,'训练后：匹配方向更接近')]:
        b+=box(x,135,445,300,[title],'#ffffff')
    colors=['#bd7441','#3a8e87','#8271ad']
    before=[((100,245),(370,340)),((220,380),(150,310)),((395,240),(290,210))]
    after=[((685,240),(718,260)),((815,365),(846,345)),((940,225),(978,250))]
    for group in [before,after]:
        for (p,q),c in zip(group,colors):
            b+=f'<path d="M{p[0]} {p[1]} L{q[0]} {q[1]}" stroke="{c}" stroke-dasharray="6 5" stroke-width="2"/>'
            b+=f'<circle cx="{p[0]}" cy="{p[1]}" r="12" fill="{c}"/>'
            b+=f'<rect x="{q[0]-11}" y="{q[1]-11}" width="22" height="22" rx="2" fill="{c}"/>'
    b+=arrow(500,285,587,285)
    b+=txt(490,245,'训练',20)
    b+=txt(40,490,'正例：提高已配对图文的相似度。',23)
    b+=txt(40,535,'对比项：让正确配对的分数相对其他候选更高。',23)
    b+=txt(40,580,'两个编码器一起调整；相似图片可能同样适合一段描述，不能把所有非配对项当作语义相反。',20)
    save('clip-learning.svg','CLIP 的训练：让配对表达逐渐对齐',b,'二维位置仅作概念示意，并非真实嵌入投影；实际训练比较高维归一化向量的方向。',645)


def main():
    rnn()
    cnn()
    clip_pairs()
    clip_candidates()
    clip_learning()


if __name__ == '__main__':
    main()
