"""Original, reproducible VLM teaching figures (SVG; no network required)."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "images/lecture_04"
INK = "#17334a"
TEAL = "#307f82"


def txt(x, y, value, size=22, fill=INK, weight=400):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def box(x, y, w, h, lines, fill="#e2f1f0", size=21):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
            f'fill="{fill}" stroke="#9ab9c1"/>' +
            "".join(txt(x + 14, y + 30 + i * 29, s, size)
                    for i, s in enumerate(lines)))


def arrow(x1, y1, x2, y2, color=TEAL):
    return (f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" '
            'stroke-width="3" fill="none" marker-end="url(#arrow)"/>')


def save(name, title, body, foot, height=500):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
           f'viewBox="0 0 1000 {height}" role="img"><title>{escape(title)}</title>'
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
           'refY="3" orient="auto"><path d="M0 0L0 6L8 3Z" fill="#307f82"/>'
           '</marker></defs>'
           f'<rect width="1000" height="{height}" rx="18" fill="#f5f9fc"/>'
           '<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">' +
           txt(30, 45, title, 28, weight=600) + body +
           txt(30, height - 24, foot, 17, "#536c80") + '</g></svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(svg, encoding="utf-8")


def main():
    b = box(30, 85, 395, 325, [], "#fff9ed")
    b += txt(53, 123, "同一张教学场景", 23, weight=600)
    b += '<rect x="80" y="168" width="105" height="105" fill="#e95f59"/>'
    b += '<circle cx="310" cy="220" r="56" fill="#358ac8"/>'
    b += txt(84, 302, "左：红方块", 20) + txt(257, 302, "右：蓝圆形", 20)
    b += box(75, 329, 300, 56, ["展签：1280 年 · 陶制品"], "#fff", 20)
    cards = [("图像描述", "左侧是红色方块，右侧是蓝色圆形。"),
             ("视觉问答", "问：左边什么颜色？答：红色。"),
             ("OCR／文字读取", "读出展签中的年份：1280。"),
             ("双语表达", "The red square is on the left.")]
    for i, (task, answer) in enumerate(cards):
        y = 83 + 85 * i
        b += box(475, y, 495, 73, [task, answer], "#eaf1fa", 20)
    b += arrow(435, 246, 465, 246)
    save("scene-tasks.svg", "图像相同，任务与语言输出可以不同", b,
         "场景与答案由教师设计；用于定义任务，不是任何真实模型的预测。", 465)

    b = txt(30, 96, "CLIP：比较已有候选", 23, weight=600)
    b += box(30, 123, 205, 92, ["图像 → 图像向量", "候选 → 文本向量"])
    b += arrow(245, 169, 285, 169)
    b += box(295, 123, 260, 92, ["计算相似度", "选出更匹配的候选"], "#eaf1fa")
    b += arrow(565, 169, 605, 169)
    b += box(615, 123, 355, 92, ["『红色方块』分数高", "输出受候选集合限制"], "#fff0da")
    b += txt(30, 290, "本课的生成式 VLM：根据图像与问题续写答案", 23, weight=600)
    b += box(30, 318, 205, 92, ["图像特征序列", "问题：左边是什么？"])
    b += arrow(245, 364, 285, 364)
    b += box(295, 318, 260, 92, ["连接器＋语言模型", "每次预测下一个词元"], "#eaf1fa")
    b += arrow(565, 364, 605, 364)
    b += box(615, 318, 355, 92, ["『左边是一个红色方块。』", "直到生成结束标记"], "#fff0da")
    save("clip-to-vlm.svg", "从『哪句话更匹配』到『怎样组织答案』", b,
         "VLM 是广义的视觉语言模型；CLIP 也属此类。本课聚焦能生成文本的 VLM。", 475)

    b = txt(30, 100, "先编码图像，再逐词元生成答案", 22, weight=600)
    b += box(30, 130, 285, 92, ["图像 I ＋ 问题 Q", "始终作为条件上下文"])
    for i, (word, prefix) in enumerate([("a", "起始标记"), ("red", "a"),
                                        ("square", "a red"), ("<eos>", "a red square")]):
        x = 355 + i * 154
        b += box(x, 132, 143, 90, ["预测：" + word, prefix], "#fff0da", 18)
        if i < 3:
            b += arrow(x + 145, 177, x + 153, 177)
    b += arrow(319, 177, 346, 177)
    b += txt(355, 247, "卡片第二行显示已给出的答案前文。", 18)
    b += txt(30, 295, "训练：输入正确前文，在多个位置同时计算预测", 22, weight=600)
    b += txt(30, 350, "已知前文", 20) + txt(30, 425, "目标词元", 20)
    for i, (before, target) in enumerate([("<bos>", "a"), ("a", "red"),
                                          ("red", "square"), ("square", "<eos>")]):
        x = 175 + i * 200
        b += box(x, 320, 177, 51, [before], "#eaf1fa")
        b += arrow(x + 88, 377, x + 88, 395)
        b += box(x, 402, 177, 51, [target], "#fff0da")
    b += txt(30, 507, "p(A | I, Q) = ∏ₜ p(aₜ | I, Q, a₁,…,aₜ₋₁)", 26, TEAL)
    b += txt(30, 543, "t 从 1 到 T：整段答案概率＝每一步条件概率的乘积。", 20)
    save("autoregressive.svg", "自回归生成与教师强制：下一词元从哪里来？", b,
         "训练可并行算各位置，但因果掩码仍禁止读取未来答案；推理通常逐词元展开。", 595)

    b = txt(30, 98, "保留多个位置的表示", 23, weight=600)
    for row in range(3):
        for col in range(3):
            c = ["#f4b2ab", "#f7d2a3", "#a6cde8"][(row + col) % 3]
            b += f'<rect x="{42+col*52}" y="{130+row*52}" width="49" height="49" fill="{c}"/>'
    b += txt(35, 319, "像素／图像块", 20) + arrow(211, 209, 257, 209)
    b += box(270, 146, 235, 125, ["视觉编码器", "局部＋上下文特征", "每块一个向量"])
    b += arrow(515, 209, 557, 209)
    for i in range(5):
        x, y = 580 + i * 16, 153 - i * 11
        b += f'<path d="M{x} {y}l180 0 18 15-180 0Z" fill="#c8e9e6" stroke="#4d8c90"/>'
        b += f'<path d="M{x} {y+15}l180 0 0 32-180 0Z" fill="#e5f3f2" stroke="#4d8c90"/>'
    b += txt(587, 280, "特征序列 N × Dv", 21)
    b += txt(30, 385, "用于匹配：可汇总为整图向量，再做相似度比较。", 22)
    b += txt(30, 428, "用于生成：常保留图像块序列，让答案利用不同区域。", 22)
    save("visual-features.svg", "从一个整图摘要，到一串可被查询的视觉特征", b,
         "特征不是人工物体标签；保留图像块也不保证每一处细节都能被正确读出。", 490)

    b = ""
    rows = [("LLaVA · 拼接前缀", ["视觉特征", "逐块投影", "与文字拼接", "因果语言模型"]),
            ("BLIP-2 · 查询压缩", ["视觉特征", "Q-Former", "少量视觉向量", "语言模型"]),
            ("Flamingo · 层间读取", ["视觉特征", "重采样器", "交叉注意力", "语言模型层"])]
    for r, (label, names) in enumerate(rows):
        y = 128 + r * 147
        b += txt(30, y - 19, label, 23, weight=600)
        for i, name in enumerate(names):
            x = 30 + i * 244
            b += box(x, y, 218, 66, [name], "#fff0da" if i == 1 else "#e2f1f0")
            if i < 3:
                b += arrow(x + 220, y + 33, x + 238, y + 33)
    save("connector-routes.svg", "视觉与语言的连接：三条有代表性的路线", b,
         "简化到连接方式；BLIP-2 可用不同语言模型，Flamingo 在冻结 LM 层间加入新模块。", 570)

    b = box(30, 103, 255, 96, ["RGB 输入：336×336", "块大小：14×14", "24×24＝576 个块"], "#eaf1fa", 20)
    b += arrow(295, 151, 330, 151)
    b += box(342, 103, 265, 96, ["CLIP ViT-L/14", "取中间层的块特征", "[B, 576, 1024]"], size=20)
    b += arrow(617, 151, 650, 151)
    b += box(662, 103, 305, 96, ["MLP 连接器", "1024 → 4096 → 4096", "[B, 576, 4096]"], "#fff0da", 20)
    b += arrow(820, 210, 820, 267)
    b += box(30, 287, 300, 80, ["问题词元 → 嵌入", "[B, L, 4096]"], "#eaf1fa", 20)
    b += arrow(340, 327, 380, 327)
    b += box(392, 287, 255, 80, ["拼接视觉与文字序列", "长度：576 ＋ L"], size=20)
    b += arrow(657, 327, 697, 327)
    b += box(709, 287, 258, 80, ["Vicuna 7B 语言模型", "输出文字词表的分数"], "#fff0da", 20)
    b += '<path d="M820 267L520 267L520 282" fill="none" stroke="#307f82" stroke-width="3" marker-end="url(#arrow)"/>'
    b += txt(30, 432, "视觉块没有被强行翻译成单词；它们直接进入连续的嵌入空间。", 22)
    save("llava-architecture.svg", "读懂一个具体例子：LLaVA-1.5 7B 的数据形状", b,
         "数字对应此公开配置；原始 LLaVA 的连接器是线性层，其他 VLM 的配置可能不同。", 505)

    b = txt(30, 96, "阶段一：先学会连接", 23, weight=600)
    b += txt(535, 96, "阶段二：再学会按要求回答", 23, weight=600)
    for x, label, fills in [(30, "图像—文字配对", ["#e9eef3", "#fff0da", "#e9eef3"]),
                             (535, "图像＋问题＋答案", ["#e9eef3", "#fff0da", "#fff0da"])]:
        b += box(x, 122, 430, 58, [label], "#eaf1fa")
        for i, (name, fill) in enumerate(zip(["视觉编码器", "连接器", "语言模型"], fills)):
            y = 207 + i * 72
            state = "冻结" if fill == "#e9eef3" else "更新参数"
            b += box(x, y, 430, 58, [name + "　·　" + state], fill)
            if i < 2:
                b += arrow(x + 215, y + 60, x + 215, y + 69)
    b += txt(30, 466, "冻结参数 ≠ 停止所有梯度计算：梯度仍可穿过冻结的语言模型传到连接器。", 21)
    save("training-stages.svg", "LLaVA 两阶段训练：哪些模块在学习？", b,
         "这是所选 LLaVA 基线的方案；现代模型也可能解冻视觉分支或使用其他训练阶段。", 525)

    positions = ["图像 1", "图像 2", "问题", "<bos>", "red", "<eos>"]
    b = txt(30, 95, "输入序列", 23, weight=600)
    for i, name in enumerate(positions):
        x = 30 + i * 160
        b += box(x, 122, 148, 56, [name], "#e2f1f0" if i < 2 else "#eaf1fa", 20)
        b += txt(x + 48, 207, f"位置 {i}", 17)
    b += txt(30, 263, "labels（与输入位置对齐）", 23, weight=600)
    for i, name in enumerate(["-100", "-100", "-100", "-100", "red", "<eos>"]):
        x = 30 + i * 160
        b += box(x, 288, 148, 56, [name], "#e9eef3" if i < 4 else "#fff0da", 20)
    b += arrow(590, 180, 710, 279) + arrow(750, 180, 870, 279)
    b += txt(30, 399, "位置 3 的 logits → 预测位置 4 的 red。", 22)
    b += txt(30, 439, "位置 4 的 logits → 预测位置 5 的 <eos>。", 22)
    b += txt(30, 479, "-100 只让该位置不计入损失；它仍可作为被注意的上下文。", 21)
    b += txt(30, 533, "L = − (1/T) ∑ₜ log p(aₜ | I, Q, a₁,…,aₜ₋₁)", 26, TEAL)
    b += txt(30, 569, "t 从 1 到 T：只对需要监督的答案词元求平均。", 20)
    save("answer-loss.svg", "答案区监督：标签向右一位，预测向前对齐", b,
         "这里只省略真实聊天模板与分词；多轮对话需要正确标出每个助手回答区间。", 620)

    b = txt(30, 98, "输入：3 × 8 × 16 像素", 22, weight=600)
    b += '<rect x="30" y="124" width="130" height="105" fill="#e95f59"/>'
    b += '<rect x="160" y="124" width="130" height="105" fill="#358ac8"/>'
    b += txt(65, 261, "左 red", 20) + txt(193, 261, "右 blue", 20)
    b += arrow(300, 177, 343, 177)
    b += box(355, 125, 255, 103, ["局部切块嵌入", "2 个 16 维特征", "投影为 2 × 32"], size=20)
    b += arrow(620, 177, 662, 177)
    b += box(674, 125, 295, 103, ["拼入问题与答案前文", "2 层因果 Transformer", "7 个词元的输出词表"], "#fff0da", 20)
    b += box(30, 334, 435, 75, ["同图换问题：left? → red <eos>", "同图换问题：right? → blue <eos>"], "#eaf1fa", 20)
    b += box(505, 334, 465, 75, ["交换两块颜色，再问 left?", "应输出 blue <eos>"], "#eaf1fa", 20)
    save("mini-vlm.svg", "课堂微型 VLM：真正做一次条件生成训练", b,
         "全部 8 种组合用于拟合；不是自然图像理解、真实预训练 VLM 或独立泛化评测。", 470)

    b = txt(30, 97, "小字与多图：细节增加，序列也会变长", 23, weight=600)
    for x, size, n in [(30, 224, 256), (355, 336, 576), (680, 448, 1024)]:
        b += box(x, 123, 285, 100, [f"{size}×{size} 像素", "14×14 像素一个块", f"{n} 个视觉块"], "#eaf1fa", 21)
    b += txt(30, 294, "简单切块：宽高各增加一倍 → 图像块数量变为四倍。", 22)
    b += txt(30, 339, "完整自注意力的分数表随序列长度的平方增长。", 22)
    b += txt(30, 384, "实践中可用缩放、裁块、动态分辨率、重采样或压缩来控制成本。", 21)
    b += txt(30, 429, "视频还需时间顺序；多张图片的集合不能完整表达运动过程。", 21)
    save("resolution-cost.svg", "分辨率、视觉词元与计算：更清晰需要付出什么？", b,
         "这里只计算整除尺寸的普通网格；实际模型可增加特殊词元、拼图或合并图像块。", 490)

    b = box(30, 100, 445, 286, ["可见证据", "左边：红方块", "右边：蓝圆形", "展签：1280 年", "图中没有汽车"], "#e2f1f0", 23)
    b += box(525, 100, 445, 126, ["可核对的回答", "『展签写着 1280 年。』", "『图中未见汽车。』"], "#eaf1fa", 22)
    b += box(525, 252, 445, 134, ["需要警惕的回答", "『这辆红色汽车……』", "『展品一定来自某位收藏家。』"], "#ffe9e5", 22)
    b += txt(30, 440, "事实正确／关系正确／证据充分／语言自然：分别检查。", 23, weight=600)
    save("grounding-evaluation.svg", "语言很流畅，不意味着回答真的有图像依据", b,
         "上面是教师构造的正反例；不得把示例当作对某个真实模型的测试结果。", 505)

    preprocessing_and_llava_dimensions()
    print(f"Generated 16 original SVG teaching diagrams in {OUT} (plus runtime/case figures)")


def preprocessing_and_llava_dimensions():
    b = txt(30, 95, "空间处理：把图片整理到模型训练时的输入格式", 22, weight=600)
    for x, lines in [(30, ["读取文件", "EXIF 转正／RGB", "[H, W, 3]"]),
                     (355, ["按配置补边与缩放", "LLaVA-1.5：pad", "[336, 336, 3]"]),
                     (680, ["转浮点并除以 255", "uint8 → float32", "数值范围 [0, 1]"])]:
        b += box(x, 122, 285, 104, lines, "#eaf1fa", 20)
    b += arrow(325, 174, 345, 174) + arrow(650, 174, 670, 174)
    b += txt(30, 293, "数值处理：每个通道使用预训练处理器给定的常数", 22, weight=600)
    b += box(30, 323, 600, 119, ["标准化：z = (像素 / 255 − mean) / std",
                                 "R/G/B 均值 ≈ [0.4815, 0.4578, 0.4082]",
                                 "R/G/B 标准差 ≈ [0.2686, 0.2613, 0.2758]"], size=21)
    b += arrow(640, 384, 670, 384)
    b += box(680, 323, 285, 119, ["换轴 HWC → CHW", "加入批量维 B", "[B, 3, 336, 336]"], "#fff0da", 20)
    b += txt(30, 495, "缩放与标准化不是模型推理；标准化后的数可以为负，也可以大于 1。", 22)
    save("preprocess-pipeline.svg", "预处理做了两件事：整理空间尺寸，匹配数值分布", b,
         "常数来自所选 CLIP 处理器；不要随意换成其他模型均值，或把已经 [0,1] 的图再除以 255。", 560)

    b = txt(30, 94, "图像分支：保留块级特征，投影到语言模型的宽度", 22, weight=600)
    rows = [(["预处理像素", "[1,3,336,336]"], ["ViT-L/14 · 24 层", "含 CLS：577×1024"], ["选 -2 层，去 CLS", "[1,576,1024]"], ["两层 MLP", "[1,576,4096]"])]
    for row in rows:
        for i, lines in enumerate(row):
            x = 30 + i * 244
            b += box(x, 122, 218, 90, lines, "#fff0da" if i == 3 else "#e2f1f0", 20)
            if i < 3: b += arrow(x + 221, 168, x + 237, 168)
    b += txt(30, 288, "文字分支：图像占位符定位插入点；其他词元照常查嵌入", 22, weight=600)
    b += box(30, 319, 285, 105, ["原始文字输入：21 项", "20 个普通词元＋<image>", "嵌入：每个词元 4096 维"], "#eaf1fa", 19)
    b += arrow(325, 372, 350, 372)
    b += box(365, 319, 295, 105, ["用 576 个向量替换占位符", "长度：21 − 1 ＋ 576", "[1,596,4096]"], size=20)
    b += '<path d="M870 223V275H512V309" stroke="#307f82" stroke-width="3" fill="none" marker-end="url(#arrow)"/>'
    b += arrow(670, 372, 695, 372)
    b += box(710, 319, 255, 105, ["Vicuna 7B · 32 层", "[1,596,4096]", "各层保持序列形状"], "#fff0da", 20)
    b += box(365, 486, 295, 80, ["文字输出头：4096→32000", "训练 logits [1,596,32000]"], "#eaf1fa", 19)
    b += '<path d="M835 432V526H670" stroke="#307f82" stroke-width="3" fill="none" marker-end="url(#arrow)"/>'
    b += txt(30, 613, "生成下一词元只取最后一个位置：[1,1,32000]；选出词元，再接回序列。", 21)
    save("llava-full-pipeline.svg", "LLaVA-1.5 7B：像素与文字汇合，再输出词表分数", b,
         "B=1；21 项是教学设定的序列长度，非某句自然语言的实测分词；不展示所有聊天模板细节。", 675)

    b = box(30, 110, 290, 97, ["卷积切块：核 14，步幅 14", "3 → 1024 通道", "[B,1024,24,24]"], "#eaf1fa", 20)
    b += arrow(330, 160, 352, 160)
    b += box(365, 110, 295, 97, ["展平空间＋换轴", "576 个块，每块 1024 维", "[B,576,1024]"], size=20)
    b += arrow(670, 160, 692, 160)
    b += box(705, 110, 260, 97, ["加入 CLS＋位置嵌入", "576 ＋ 1 = 577", "[B,577,1024]"], "#fff0da", 20)
    b += txt(30, 271, "一个视觉 Transformer 块（共 24 个）", 23, weight=600)
    b += box(30, 300, 450, 160, ["LayerNorm → 自注意力 → 残差", "16 个头，每头 64 维", "Q/K/V：[B,16,577,64]", "组合输出：[B,577,1024]"], "#eaf1fa", 21)
    b += box(520, 300, 445, 160, ["LayerNorm → 前馈网络 → 残差", "1024 → 4096 → 1024", "中间：[B,577,4096]", "块输出：[B,577,1024]"], "#eaf1fa", 21)
    b += arrow(488, 380, 510, 380)
    b += box(30, 515, 450, 82, ["所选配置 hidden_states[-2]", "24 层模型中对应第 23 个块后的状态"], size=20)
    b += arrow(490, 557, 510, 557)
    b += box(520, 515, 445, 82, ["去掉 CLS，留下 576 个块向量", "[B,576,1024] → 连接器"], "#fff0da", 20)
    save("llava-vision-dimensions.svg", "展开视觉分支：336 像素怎样变成 576 个向量？", b,
         "图中不经过 CLIP 的整图池化与检索投影头；取中间层块特征供 LLaVA 使用。", 660)

    b = box(30, 111, 445, 119, ["MLP 连接器：逐块计算", "Linear 1024→4096 → GELU", "Linear 4096→4096",], "#fff0da", 21)
    b += box(520, 111, 445, 119, ["输入和输出的位置数量相同", "[B,576,1024] → [B,576,4096]", "不是把整图向量复制 576 次"], size=21)
    b += arrow(485, 173, 510, 173)
    b += txt(30, 294, "一个 Vicuna/LLaMA 语言块（共 32 个；S 是拼接后的长度）", 22, weight=600)
    b += box(30, 326, 445, 157, ["RMSNorm → 因果注意力 → 残差", "32 个头，每头 128 维", "Q/K/V：[B,32,S,128]", "输出：[B,S,4096]"], "#eaf1fa", 21)
    b += box(520, 326, 445, 157, ["RMSNorm → 门控前馈 → 残差", "gate/up：4096→11008", "SiLU(gate) × up → down", "down：11008→4096"], "#eaf1fa", 21)
    b += arrow(485, 404, 510, 404)
    b += box(30, 537, 445, 80, ["32 层后再归一化", "隐藏状态：[B,S,4096]"], size=21)
    b += arrow(485, 578, 510, 578)
    b += box(520, 537, 445, 80, ["输出头：4096→32000", "logits：[B,S,32000]"], "#fff0da", 21)
    save("llava-language-dimensions.svg", "展开连接器与语言分支：宽度改变，位置数量怎样保持？", b,
         "MLP=多层感知机；RMSNorm=均方根归一化；SiLU/GELU 为非线性变换。", 680)

    b = txt(30, 98, "原始编号序列：20 个普通词元，加 1 个图像占位符", 22, weight=600)
    b += box(30, 132, 240, 86, ["前 3 个普通词元", "[B,3,4096]"], "#eaf1fa", 21)
    b += box(300, 132, 265, 86, ["<image> → -200", "是插入位置标记"], "#fff0da", 21)
    b += box(595, 132, 370, 86, ["后 17 个普通词元", "[B,17,4096]"], "#eaf1fa", 21)
    b += arrow(430, 229, 430, 297)
    b += box(300, 307, 265, 91, ["576 个视觉块向量", "[B,576,4096]"], size=21)
    b += txt(30, 469, "新嵌入序列：3 ＋ 576 ＋ 17 = 596 个位置", 24, weight=600)
    b += box(30, 500, 240, 80, ["文字前缀", "3 个位置"], "#eaf1fa", 21)
    b += box(300, 500, 265, 80, ["视觉向量", "576 个位置"], size=21)
    b += box(595, 500, 370, 80, ["文字后缀", "17 个位置"], "#eaf1fa", 21)
    b += txt(30, 633, "labels 的图像位置设为 -100；批量对齐的补齐位置还需要 attention mask。", 21)
    save("llava-image-placeholder.svg", "图像占位符不是普通词：先定位，再替换为视觉向量", b,
         "-200 不能直接送入普通 Embedding 查表；本图按原始 LLaVA 实现说明，其他库格式可不同。", 690)


if __name__ == "__main__":
    main()
