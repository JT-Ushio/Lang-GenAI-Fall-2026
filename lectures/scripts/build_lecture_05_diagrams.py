"""Original vector diagrams for the diffusion lecture; no network or model needed."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "images/lecture_05"


def txt(x, y, value, size=22, color="#17334a", weight=400):
    content = escape(str(value))
    # SVG subspans avoid missing Unicode subscript-t glyphs in CJK fonts.
    for glyph, letter in {'ₜ':'t', '₀':'0', '₁':'1', '₋':'−', 'ᵤ':'u', '𝚌':'c'}.items():
        content = content.replace(glyph, f'<tspan baseline-shift="sub" font-size="70%">{letter}</tspan>')
    content = content.replace('εθ', 'ε<tspan baseline-shift="sub" font-size="70%">θ</tspan>')
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{content}</text>'


def box(x, y, w, h, lines, color="#e2f1f0", size=21):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{color}" stroke="#9ab9c1"/>' +
            ''.join(txt(x+14, y+30+i*29, s, size) for i, s in enumerate(lines)))


def arrow(x1, y1, x2, y2):
    return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#307f82" stroke-width="3" marker-end="url(#a)"/>'


def save(name, title, body, foot, height=500):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{escape(title)}</title>'
           '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L0 6L8 3Z" fill="#307f82"/></marker></defs>'
           f'<rect width="1000" height="{height}" rx="18" fill="#f5f9fc"/><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">' +
           txt(30,45,title,28,weight=600) + body + txt(30,height-24,foot,17,"#536c80") + '</g></svg>')
    (OUT/name).write_text(svg, encoding="utf-8")


def main():
    b=''
    for i,(title,lines) in enumerate([
        ('第三课 · CLIP',['图像与文字分别编码','比较已有候选的相似度','图文对齐']),
        ('第四课 · 生成式 VLM',['图像＋问题作为条件','语言模型逐词元回答','图像 → 文字']),
        ('第五课 · 条件扩散',['文字＋初始随机噪声','重复更新图像／潜变量','文字 → 图像'])]):
        x=30+i*328
        b+=txt(x,107,title,23,weight=600)+box(x,130,300,135,lines,'#fff0da' if i==2 else '#e2f1f0',20)
    b+=txt(30,337,'对应关系可以帮助生成，但不会自动提供“把向量画成图”的解码器。',22)
    save('course-route.svg','从图文对应，到语言条件下的图像生成',b,'本课讲经典 Stable Diffusion 1.5；用于说明机制与维度，不代表最新模型的统一架构。',410)

    b=txt(30,100,'VAE：先压缩、再重建',23,weight=600)+box(30,130,280,98,['图像 → 连续潜变量','解码器重建图像','本课用于理解潜空间'],'#eaf1fa',20)
    b+=txt(355,100,'GAN：生成与判别共同训练',23,weight=600)+box(355,130,290,98,['噪声 → 生成器 → 图像','判别器提供训练信号','本课只作历史对照'],'#eaf1fa',20)
    b+=txt(690,100,'扩散：逐步生成',23,weight=600)+box(690,130,280,98,['训练：给样本加噪','学习预测噪声等目标','生成：多次更新样本'],'#fff0da',20)
    b+=txt(30,298,'2015：扩散生成研究 → 2020：DDPM／DDIM → 2022：潜空间扩散与 CFG',21)
    b+=txt(30,345,'这些路线可以组合：Stable Diffusion 将 VAE 压缩与条件扩散结合。',21)
    save('generative-routes.svg','生成模型有多条路线：本课聚焦扩散',b,'历史节点用于定位代表工作；DDPM 不是所有扩散生成研究的起点。',410)

    b=box(30,115,280,110,['干净数据 x₀','固定加噪规则','混入标准高斯噪声 ε'],'#eaf1fa')+arrow(320,170,355,170)+box(370,115,260,110,['噪声水平随 t 改变','噪声样本 xₜ','保留的信号越来越少'])+arrow(640,170,675,170)+box(690,115,280,110,['很高噪声水平','主要呈现随机纹理','不能唯一反推原图'],'#fff0da')
    b+=txt(30,305,'xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε',29,'#307f82')
    b+=txt(30,353,'αₜ = 1 − βₜ，ᾱₜ = α₁ × … × αₜ，ε 来自标准正态分布。',22)
    b+=txt(30,398,'系数平方之和为 1；不是把原图与随机图直接按百分比混合。',21)
    save('forward-formula.svg','前向扩散：怎样构造一个指定噪声水平的样本？',b,'代码数组从 0 开始；alpha_bar[0] 已是第一步加噪后，干净原图单独展示。',465)

    b=box(30,115,225,103,['随机取训练图 x₀','随机取时间索引 t','随机采样噪声 ε'],'#eaf1fa',20)+arrow(265,166,292,166)+box(305,115,235,103,['按公式合成 xₜ','输入：xₜ、t、条件 c','同一个网络处理各 t'],size=20)+arrow(550,166,577,166)+box(590,115,195,103,['去噪网络 εθ','预测噪声 ε̂','输出形状同 xₜ'],'#fff0da',20)+arrow(795,166,817,166)+box(830,115,140,103,['和 ε 比较','MSE 损失','更新参数'],size=19)
    b+=txt(30,301,'L = mean((εθ(xₜ, t, c) − ε)²)',29,'#307f82')
    b+=txt(30,351,'监督目标是这次人为采样的噪声；目标可计算，不需要人手标注每一张“去噪图”。',21)
    b+=txt(30,398,'训练常随机抽一个噪声水平；不必每条数据都先跑完整个反向采样链。',21)
    save('noise-objective.svg','学习去噪：给出噪声图、时间和条件，预测噪声',b,'这是 epsilon 预测的常见简化目标；其他扩散模型也可能预测原样本或 velocity。',465)

    b=txt(30,98,'训练：标准答案可构造',23,weight=600)+txt(530,98,'生成：没有那张已知原图',23,weight=600)
    b+=box(30,128,430,115,['训练样本 → 加噪 → 预测噪声','已知 ε，计算损失并反向传播','每次训练更新模型参数'],'#eaf1fa')
    b+=box(530,128,440,115,['随机初始噪声 → 重复采样更新','使用训练好的网络和调度器','每一步主要更新样本，参数固定'],'#fff0da')
    b+=box(30,315,430,95,['已知真实 ε 的代数还原','只检查公式和数值实现','不等于模型已经学会去噪'],size=21)
    b+=box(530,315,440,95,['从纯噪声出发的实际生成','依靠学到的噪声预测和采样规则','不必恢复任何一张指定原图'],size=21)
    save('training-vs-sampling.svg','区分三件事：公式验证、模型训练、从噪声生成',b,'课堂代码分别记录这些结果；不会把已知噪声的反算或倒放加噪动画当成模型生成。',475)

    b=box(30,120,235,104,['当前样本 xₜ','当前时间索引 t','文字／类别条件 c'],'#eaf1fa')+arrow(275,172,307,172)+box(320,120,235,104,['同一套去噪网络','预测当前噪声 ε̂','参数不随步骤重新训练'],'#fff0da',20)+arrow(565,172,597,172)+box(610,120,355,104,['调度器／采样器','结合时间与噪声预测','计算下一状态 xₜ₋₁'])
    b+='<path d="M790 237V290H140V237" stroke="#307f82" fill="none" stroke-width="3" marker-end="url(#a)"/>'
    b+=txt(30,365,'DDPM：常在反向步骤中注入受控随机噪声，最后一步通常不再加噪。',21)
    b+=txt(30,407,'DDIM：η=0 时，给定初始噪声与条件可确定采样路径；η>0 可引入随机性。',21)
    b+=txt(30,449,'减少推理步数要用相应采样规则；不能简单漏掉循环中的任意若干步。',21)
    save('sampling-loop.svg','每次预测由网络给出，怎样迈下一步由采样规则决定',b,'训练时间网格 T 与推理更新次数 K 是不同概念；本课微型例子用 100 步 DDPM。',515)

    b=box(30,105,285,105,['RGB 图像／训练输入','[B,3,512,512]','VAE 编码器'],'#eaf1fa')+arrow(325,158,350,158)+box(365,105,295,105,['后验均值与对数方差','[B,8,64,64]','拆成两组 4 通道'])+arrow(670,158,695,158)+box(710,105,255,105,['采样潜变量 z','[B,4,64,64]','幅值乘 0.18215'],'#fff0da',20)
    b+=txt(30,281,'空间：512 → 64，宽与高各缩小 8 倍，位置数量变为 1/64。',22)
    b+=txt(30,325,'元素：3×512×512 / (4×64×64) = 48；这不是图像文件压缩率。',22)
    b+=box(30,386,425,82,['生成结束：先把潜变量幅值除以 0.18215','VAE 解码器再输出 RGB 像素'],'#eaf1fa',20)+arrow(465,426,500,426)+box(515,386,450,82,['[B,4,64,64] → [B,3,512,512]','4 个潜通道没有固定的 R/G/B/透明度含义'],size=20)
    save('vae-latent.svg','潜空间扩散：先学习压缩，再在较小表示上生成',b,'0.18215 是所选 SD 1.x 的幅值缩放；它不改变 shape，也不是空间缩放的 1/8。',535)

    b=txt(30,98,'文字分支：整段上下文化表示，不是 CLIP 整句检索向量',22,weight=600)
    b+=box(30,123,285,92,['提示词 → CLIP 分词器','[B,77] 编号','77 包含特殊与补齐位置'],'#eaf1fa',20)+arrow(325,169,350,169)+box(365,123,295,92,['CLIP 文本 Transformer','[B,77,768]','提供每个位置的条件特征'],size=20)
    b+=txt(30,291,'生成分支：从潜空间随机噪声出发，反复更新',22,weight=600)
    for i,lines in enumerate([['初始随机潜变量','[B,4,64,64]'],['U-Net 预测噪声','[B,4,64,64]'],['调度器更新潜变量','[B,4,64,64]']]):
        x=30+i*328;b+=box(x,319,300,90,lines,'#fff0da' if i==1 else '#e2f1f0',21)
        if i<2:b+=arrow(x+309,365,x+322,365)
    b+='<path d="M515 228V307" stroke="#307f82" stroke-width="3" marker-end="url(#a)"/>'
    b+=box(700,123,265,92,['时间步 t → 时间嵌入','[B,320] → [B,1280]','进入 U-Net 残差块'],'#eaf1fa',19)
    b+='<path d="M830 228V271H583V307" stroke="#307f82" fill="none" stroke-width="3" marker-end="url(#a)"/>'
    b+='<path d="M830 420V451H175V420" stroke="#307f82" fill="none" stroke-width="3" marker-end="url(#a)"/>'
    b+=arrow(175,454,175,483)+txt(195,480,'迭代结束后解码',17)
    b+=box(30,493,430,105,['最终潜变量 / 0.18215 → VAE 解码器','[B,4,64,64] → [B,3,512,512]'],'#eaf1fa',20)+arrow(470,543,505,543)+box(520,493,445,105,['映射数值范围并转换为图像','通常保存为 HWC 的 uint8 RGB','输出图像，不是语言词表分数'],size=20)
    save('sd-overview.svg','Stable Diffusion 1.5：文字条件、潜空间迭代和像素解码',b,'图中 B 表示每个提示的样本数；CFG 可把 U-Net 前向的实际批量扩为 2B，后面单独展示。',640)

    b=''
    down=[('64×64',320),('32×32',640),('16×16',1280),('8×8',1280)]
    for i,(resolution,channel) in enumerate(down):
        y=105+i*110
        b+=box(30,y,320,78,[f'下行层级：{resolution}',f'[B,{channel},{resolution.replace("×",",")}]'],'#eaf1fa',20)
        b+=box(650,y,320,78,[f'上行层级：{resolution}',f'[B,{channel},{resolution.replace("×",",")}]'],size=20)
        b+=arrow(361,y+39,638,y+39)
        if i<3:
            b+=arrow(190,y+82,190,y+102)+arrow(810,y+102,810,y+82)
    b+=box(375,435,250,78,['瓶颈／中间块','[B,1280,8,8]'],'#fff0da',20)
    b+=arrow(361,474,363,474)+arrow(635,474,638,474)
    b+=txt(30,562,'跳跃连接：把相同分辨率的下行特征送到上行，帮助保留空间信息。',21)
    b+=txt(30,606,'输入／输出均为 [B,4,64,64]；中间通道数改变，输出不直接是 RGB 图片。',21)
    save('unet-dimensions.svg','U-Net：分辨率先降低再升高，跨层保留空间信息',b,'展示代表性层级并省略块内多次残差与拼接；细部实现中拼接会暂时增加通道数。',675)

    b=box(30,115,440,110,['视觉空间位置提供查询 Q','64×64 = 4096 个位置','特征 [B,4096,320]'],'#eaf1fa',22)
    b+=box(530,115,440,110,['文字位置提供 K 和 V','上下文化特征 [B,77,768]','先投影 768 → 320'],size=22)
    b+=arrow(250,235,360,300)+arrow(750,235,640,300)
    b+=box(290,311,420,108,['匹配每个图像位置与文字位置','分数表（省略多头）：4096×77','按权重组合 V，返回视觉计算'],'#fff0da',22)
    b+=txt(30,486,'不同空间位置可以取用不同文字信息；条件不是把文字画成像素贴到图上。',21)
    b+=txt(30,530,'交叉注意力有助于条件交互，但不能仅凭某个权重保证属性、数量或关系正确。',21)
    save('cross-attention.svg','文字怎样影响去噪：视觉查询读取文本条件',b,'以 SD 1.5 的第一个 64×64 层级为例；先用单组 Q/K/V 说明计算，省略多头拆分。',595)

    b=box(30,110,280,95,['同一个噪声状态 xₜ','同一个时间 t','两条分支用同一套权重'],'#eaf1fa',21)
    b+=box(420,95,250,82,['空提示／无条件分支','预测 εᵤ'],size=21)+box(420,225,250,82,['目标提示／条件分支','预测 ε𝚌'],size=21)
    b+=arrow(320,150,408,136)+arrow(320,177,408,266)+arrow(680,136,755,190)+arrow(680,266,755,216)
    b+=box(765,155,205,105,['组合两种预测','交给调度器','计算下一状态'],'#fff0da',21)
    b+=txt(30,380,'ε̂ = εᵤ + s · (ε𝚌 − εᵤ)',30,'#307f82')
    b+=txt(30,426,'常用尺度约定：s=0 无条件，s=1 条件预测，s>1 沿条件差异方向外推。',21)
    b+=txt(30,470,'空提示特征不是全零；较大尺度不保证更好，也可能带来过饱和或伪影。',21)
    b+=txt(30,514,'批量实现：潜变量 [2B,4,64,64]，文本 [2B,77,768] → 拆分并组合回 B。',21)
    save('cfg.svg','CFG：无分类器引导，用条件与无条件预测的差异调节生成',b,'这里采用常见 Diffusers scale 约定；负面提示可更换基准条件，但不是严格的逻辑排除。',580)

    b=box(30,112,440,118,['固定：模型、采样器、步数、随机种子','只改：一个关键语言因素','例：left ↔ right，red ↔ blue'],'#eaf1fa',21)
    b+=box(530,112,440,118,['固定：提示词及其他参数','只改：随机种子','观察同一要求下的输出变化'],size=21)
    b+=txt(30,300,'最小对照 1：a red square to the left / right of a blue circle',23)
    b+=txt(30,347,'最小对照 2：a cat eating a fish / a fish eating a cat',23)
    b+=txt(30,401,'分别标注：对象、属性、数量、关系、文字可读性和两语言的信息等值。',21)
    b+=txt(30,447,'看起来像图片与忠实表达提示词是不同评价维度；高相似度不逐条认证事实。',21)
    save('language-evaluation.svg','语言学实验：控制一个因素，再检查画面变化',b,'这里是实验设计，没有把教师写出的目标要求当成真实 Stable Diffusion 的生成结果。',515)

    print('Generated 12 original lecture 05 concept diagrams:', OUT)


if __name__=='__main__':
    main()
