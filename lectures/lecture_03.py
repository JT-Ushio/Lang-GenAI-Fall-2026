"""第三讲：从文本与图像编码器走向 CLIP。90 分钟初稿，配套 notes/lecture_03.md。"""
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from execute_util import image, link, text


def main():
    text("# 第三讲 · CLIP：让图像与语言进入可比较的空间")
    text("## 从 RNN 到 Transformer，从 CNN 到 ViT")
    text("上一讲认识了像素、采样点与跨模态任务。本讲进一步学习：模型怎样把原始数据变成表示，再用图文配对关系学习对应。")
    text("**第一条线：文本表示。** 从按顺序积累信息，到按相关程度使用上下文。")
    text("**第二条线：图像表示。** 从局部模式组合，到图像块之间的信息交互。")
    text("**汇合点：CLIP。** 两个编码器分别表示图像与文字，通过对比学习建立可比较的空间。")
    image("images/lecture_03/routes.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    representations()
    recurrent_text()
    runnable_rnn()
    attention_text()
    transformer_text()
    runnable_transformer()
    convolution_images()
    runnable_cnn()
    vision_transformer()
    runnable_vit()
    clip_architecture()
    clip_training()
    runnable_clip_training()
    clip_applications()
    linguistic_limits()
    recap()


def representations():
    text("## 01 · 表示：从数据格式到可学习的特征")
    text("**编码器（encoder）**：把输入转换为向量表示的模型。向量是一列数，训练使这些数对任务有用。")
    text("**特征（feature）**：用于区分或关联输入的信息；可以人工设计，也可以由模型学习。")
    text("**嵌入（embedding）**：输入在连续向量空间中的表示。原始像素已经是数值，但不自动具有适合语义比较的结构。")
    text("**参数（parameter）**：模型中由训练调整的数值，例如权重矩阵。架构规定计算方式，训练目标决定模型被鼓励学会什么。")
    text("### 语言学切入：相同的词，不保证相同的意义")
    text("『猫吃鱼』与『鱼吃猫』包含相同的词，但施事与受事互换。词的身份、线性顺序与结构关系，都可能影响意义。")
    text("**词袋（bag of words）**：只统计词出现多少次，忽略顺序。它可用于一些任务，却无法仅凭这些计数区分上面两句话。")
    first = ["猫", "吃", "鱼"]
    second = ["鱼", "吃", "猫"]
    vocabulary = ["猫", "吃", "鱼"]
    counts1 = [first.count(word) for word in vocabulary]
    counts2 = [second.count(word) for word in vocabulary]
    print("句子一词袋：", counts1)  # @inspect counts1
    print("句子二词袋：", counts2)  # @inspect counts2
    text("**词元（token）**：分词器处理的基本单位，可以是字、词或子词片段，不必等于语言学意义上的词。")
    text("**词元编号（token ID）**：词元在词表中的索引。编号大小本身不表示语义远近。")
    text("**嵌入查表**：用编号取出一个可学习向量。后续编码器再结合上下文更新这个向量。")


def recurrent_text():
    text("## 02 · RNN：用状态记录读到的位置")
    text("**RNN（Recurrent Neural Network，循环神经网络）**：按序列顺序处理输入，每一步同时接收当前词元与上一步状态。")
    text("**隐藏状态（hidden state）**：模型在某一步维护的向量；『隐藏』表示它是内部表示，并不是一个人工给出的标签。")
    image("images/lecture_03/rnn-attention.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("可以把状态类比为一份不断改写的阅读笔记：读入新词，就结合已有笔记更新理解。它不是完整逐字存档，也不保证保存了所有信息。")
    text(r"一个简单 RNN：$h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$。$x_t$ 是当前词向量，$h_{t-1}$ 是前一步状态；同一组权重在各位置重复使用。")
    text("### 小实验：同样的词，换序后最终状态可以改变")
    text("代码行末的 `@inspect` 标记会记录变量值。逐步播放时，拖动变量面板到空白处，观察 word、state1 和 state2；每读入一个词，状态更新一次。")
    embeddings = {"猫": np.array([1., 0.]), "吃": np.array([0.5, 0.5]), "鱼": np.array([0., 1.])}
    state1 = np.zeros(2)  # @inspect state1
    for word in ["猫", "吃", "鱼"]:
        state1 = np.tanh(embeddings[word] + 0.5 * state1)  # @inspect word, @inspect state1
    state2 = np.zeros(2)  # @inspect state2
    for word in ["鱼", "吃", "猫"]:
        state2 = np.tanh(embeddings[word] + 0.5 * state2)  # @inspect word, @inspect state2
    print("猫吃鱼：", np.round(state1, 3))
    print("鱼吃猫：", np.round(state2, 3))
    image("images/lecture_03/rnn-memory.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("结果不同，说明这个运算对顺序敏感。")
    text("### 来龙去脉：顺序处理的优势与负担")
    text("Elman 在 1990 年的工作是简单循环网络用于序列学习的重要代表；RNN 不是从这篇论文才首次出现。")
    text("语言随时间展开，RNN 的逐步更新容易与这种输入方式对应；可处理不同长度的序列，参数也不必随长度增加。")
    text("较远位置的信息需要经过多次状态更新；训练时梯度反复相乘，可能衰减或放大，形成梯度消失或梯度爆炸。")
    text("**梯度**：衡量参数的小变化如何影响损失，是调整参数的依据。梯度消失会让早先位置难以获得有效学习信号，不等于所有 RNN 都完全记不住长句。")
    text("### LSTM：给状态更新增加控制机制")
    text("**LSTM（Long Short-Term Memory，长短期记忆网络）**：1997 年提出的重要循环结构，通过记忆单元与门控机制改善长期信息学习；常见现代版本包含输入门、遗忘门与输出门。")
    text("**门（gate）**：用可学习的数值控制保留、写入或输出多少信息。它是连续计算，不是事先写好的语法规则。")
    text("LSTM 改善了记忆与训练，但仍有按时间递推的依赖。双向 RNN 能利用前后文，但每个方向内部仍逐步处理。")
    link(title="Elman（1990）：Finding Structure in Time", url="https://jeffelman.ucsd.edu/research/publications/")
    link(title="Hochreiter 与 Schmidhuber（1997）：LSTM", url="https://www.bioinf.jku.at/publications/older/2604.pdf")


def attention_text():
    text("## 03 · 注意力：按当前需要聚合相关信息")
    text("早期序列到序列翻译常用编码器压缩源句，再由解码器逐步生成译文。长句的信息若主要依赖固定长度摘要，容易形成瓶颈。")
    text("Bahdanau 等人的注意力翻译工作（2014 年预印本、ICLR 2015）让解码器在每一步对源句各位置分配权重，获取当前需要的信息。")
    text("**注意力（attention）**：计算当前位置与候选位置的相关程度，再按权重组合信息。它首先是一种计算机制，不等同于人的心理注意。")
    text("翻译例子：生成 key 对应的译词时，可更多利用源句中的 key；生成 missing 对应内容时，可转向其他位置。权重不必是一对一词语对齐。")
    text("**交叉注意力（cross-attention）**：查询与被参考的信息来自不同序列，例如译文状态查询源句表示。")
    text("**自注意力（self-attention）**：查询与被参考的信息来自同一序列，例如句中词元结合其他词元更新表示。")
    text("### Q、K、V：查询、匹配依据与被取用的信息")
    text("**Q（Query，查询）**：当前位置用于提出匹配需求的向量。")
    text("**K（Key，键）**：候选位置用于参与匹配的向量。")
    text("**V（Value，值）**：候选位置在被关注后提供的信息向量。")
    text("类比查资料：Q 像检索需求，K 像用于匹配的索引，V 像实际取用的内容。模型中的三者是经线性变换得到的向量，不是人工关键词列表。")
    text(r"缩放点积注意力：$A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$，输出为 $AV$。$d_k$ 是键向量的维数。")
    text("**softmax**：把一组分数转换成非负且总和为 1 的权重。分数相对更大，分到的权重更多。")
    text("### 一次真实计算：用一条查询组合三个候选位置")
    text("三个候选位置依次记作 key、cabinets、windows。下面的数值由教师设定，只帮助读懂矩阵运算，不是模型学出的句法分析。")
    q = np.array([[1., 0.]])  # @inspect q
    keys = np.array([[2., 0.], [0., 1.], [-1., 0.]])  # @inspect keys
    values = np.array([[1., 0.], [0., 1.], [0., 2.]])  # @inspect values
    scores = q @ keys.T / np.sqrt(2)  # @inspect scores
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))  # @inspect weights
    weights = weights / weights.sum(axis=1, keepdims=True)  # @inspect weights
    context = weights @ values  # @inspect context
    print("Q / K / V 形状：", q.shape, keys.shape, values.shape)
    print("匹配分数：", np.round(scores, 3))
    print("注意力权重：", np.round(weights, 3))
    print("组合后的表示：", np.round(context, 3))
    text("注意力可以把较远位置直接纳入计算，但它不会自动保证语法正确。权重大小也不能单独证明模型采用了某条语言学规则。")
    link(title="Bahdanau 等：Neural Machine Translation by Jointly Learning to Align and Translate", url="https://arxiv.org/abs/1409.0473")


def transformer_text():
    text("## 04 · Transformer：以注意力组织上下文")
    text("Vaswani 等人在 2017 年提出 Transformer：原始工作面向机器翻译，包含编码器与解码器，以注意力替代循环结构来组织序列交互。")
    text("**位置表示（positional representation）**：告诉模型词元在序列中的位置。只有词元集合而缺少位置信息，会难以区分角色交换等语序差异。")
    text("**多头注意力（multi-head attention）**：并行进行多组投影与信息聚合，再组合结果；提供多种关联方式，不应直接给每个头指定『主谓头』『指代头』等固定职责。")
    text("**前馈网络（feed-forward network）**：在每个位置上进一步变换特征；同一层通常对各位置使用同一组参数。")
    text("**残差连接（residual connection）**：把一层的输入加到变换结果上，便于信息和梯度传播。")
    text("**层归一化（Layer Normalization）**：按层内特征统计量调整数值尺度，有助于稳定训练。")
    text("一个 Transformer 块不只有注意力：注意力负责位置间的信息交互，前馈网络负责逐位置变换，残差与归一化帮助训练。")
    text("### 编码器、解码器与可见范围")
    text("**双向自注意力**：一个位置可参考前后位置，适合在完整输入上做理解。")
    text("**因果掩码（causal mask）**：屏蔽当前位置之后的内容，避免预测下一个词时偷看答案。掩码决定信息是否可见。")
    text("Transformer 在已知序列上可并行计算多个位置；自回归生成仍通常逐词元产生输出。『用了 Transformer』不等于任何阶段都不需要等待前一步。")
    text("### 用语言学术语描述能力，而不把结构当成保证")
    text("**上下文化表示**：同一个 bank 在 river bank 与 bank account 中，可以因上下文而获得不同表示。")
    text("**长距离依存**：注意力为远距离联系提供计算通路；模型是否学会正确使用，仍取决于训练与评价。")
    text("标准全注意力对 N 个位置构造 N×N 的分数表，长序列会增加计算与存储开销；它并不是没有代价的替换。")
    text("RNN、CNN、Transformer 都仍可在不同场景使用。历史上的改进是在调整信息流、训练难度与计算效率，不是宣布之前的方法完全无用。")
    link(title="Vaswani 等（2017）：Attention Is All You Need", url="https://arxiv.org/abs/1706.03762")


def convolution_images():
    text("## 05 · CNN：从局部模式组合出图像特征")
    text("**CNN（Convolutional Neural Network，卷积神经网络）**：利用局部连接与共享权重提取图像特征。LeNet 系列是手写字符识别的重要代表，1998 年论文系统展示了相关方法。")
    text("把每个像素直接连接到所有输出，参数多，也没有明确利用图像中相邻像素的关系。CNN 把『先看局部、同类模式可出现在不同位置』写进结构。")
    image("images/lecture_03/cnn.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("**卷积核（kernel／filter）**：一小组权重，在不同位置处理局部像素。深度学习中通常使用不翻转卷积核的互相关运算，习惯仍称卷积。")
    image("images/lecture_03/cnn-volume.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("**权重共享**：同一个检测器扫描整张图，不必在每个位置独立学习一套检测器。")
    text("**特征图（feature map）**：记录局部模式在不同位置的响应强弱。多个卷积核可产生多个特征通道。")
    text("### 小实验：一个检测左右变化的核")
    patch = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])  # @inspect patch
    kernel = np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])  # @inspect kernel
    print("局部像素：\n", patch)
    print("卷积核：\n", kernel)
    print("逐元素相乘后求和：", np.sum(patch * kernel))
    print("左右翻转像素后：", np.sum(patch[:, ::-1] * kernel))
    text("同一组权重会对不同方向的明暗变化给出不同响应。本例手工指定权重；训练 CNN 时，权重一般由任务数据学习。")
    text("**非线性变换**：例如 ReLU 把负值变为 0，让多层网络不只是一个大的线性变换。")
    text("**下采样**：用步幅卷积或池化等方式降低空间尺寸，节省计算并汇总局部信息，也可能丢失细节。")
    text("**感受野（receptive field）**：某个特征可以受到输入中哪些区域的影响。多层组合能逐步联系更大范围，CNN 不只会看一小格。")
    text("语言类比：局部搭配可以参与形成更大的结构；视觉中局部边缘与纹理也可参与更复杂的表示。但神经网络层级不等同于明确的语言学词、短语、句子层级。")
    text("### 从 LeNet 到更深的视觉网络")
    text("AlexNet（2012）展示了大规模数据、GPU 训练与深层 CNN 的组合优势；ResNet（2015 年预印本、CVPR 2016）用残差连接帮助训练更深的网络。")
    text("**归纳偏置（inductive bias）**：结构预先偏好的规律。CNN 偏好局部性与权重共享；它们有助于学习，但池化、边界与步幅等会影响平移行为，并非绝对位置不变。")
    link(title="LeCun 等（1998）：Gradient-Based Learning Applied to Document Recognition", url="https://bottou.org/papers/lecun-98h")
    link(title="AlexNet（2012）原论文", url="https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html")
    link(title="ResNet 原论文", url="https://arxiv.org/abs/1512.03385")


def vision_transformer():
    text("## 06 · ViT：把图像块组织成序列")
    text("**ViT（Vision Transformer，视觉 Transformer）**：把图像切成固定大小的块，经投影形成一串向量，再用 Transformer 编码。论文于 2020 年发布预印本、发表于 ICLR 2021。")
    image("images/lecture_03/vit.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("**图像块（patch）**：如 16×16 像素区域，先展平成向量，再用同一线性变换映射到模型的隐藏维度。")
    text("**位置嵌入（position embedding）**：加入每块在图像中的位置信息，避免只知道『有哪些块』却不知道它们怎样排列。")
    text("**分类词元（class token，常记为 [CLS]）**：一个可学习的汇总位置，与图像块交互后用于表示整图。它不是图片里真实存在的物体。")
    text("### 从像素数组到图像块序列")
    pixels = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    patches = pixels.reshape(4, 2, 4, 2, 3).transpose(0, 2, 1, 3, 4).reshape(16, 12)  # @inspect patches
    rng = np.random.default_rng(3)
    projection = rng.normal(size=(12, 6))
    tokens = patches @ projection  # @inspect tokens
    print("8×8 RGB 图像：", pixels.shape)
    print("2×2 图像块，每块 2×2×3 个数：", patches.shape)
    print("投影到 6 维后的图像块序列：", tokens.shape)
    print("首块像素是否对应左上角：", np.array_equal(patches[0], pixels[:2, :2].reshape(-1)))
    text("本例只演示切块和投影，没有运行完整 ViT；随机投影本身不产生语义理解。")
    print("224×224 图像，16×16 块：", (224 // 16) ** 2, "个块")
    print("加入一个汇总位置后：", (224 // 16) ** 2 + 1, "个位置")
    text("### CNN 与 ViT：不同的组织方式")
    text("**CNN**：局部交互与共享检测器较强地写进结构；多层堆叠逐渐扩大可利用的空间范围。")
    text("**ViT**：基础版本在自注意力中让图像块跨位置交互；局部性约束较弱，更依赖数据和训练方案来学习视觉规律。")
    text("原始 ViT 的一个重要背景是大规模预训练；不能由此推出任何数据规模、任何任务下 ViT 都优于 CNN。实际系统也可混合卷积与注意力。")
    text("论文标题『一张图像值 16×16 个词』是一种类比：patch 像 token 一样进入模型，但一个对象可能跨多个块，一个块也可能包含多个对象。")
    link(title="Dosovitskiy 等：An Image is Worth 16x16 Words", url="https://arxiv.org/abs/2010.11929")


def clip_architecture():
    text("## 07 · CLIP：让两条分支对齐")
    text("**CLIP（Contrastive Language–Image Pre-training，对比式语言—图像预训练）**：通过成对的图像与文字训练两个编码器，使匹配内容在共同空间中得到更高相似度。")
    image("images/lecture_03/clip.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    image("images/lecture_03/clip-pictorial.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("**文本分支**：词元编号 → 词元与位置嵌入 → Transformer → 文本汇总表示 → 投影。")
    text("**图像分支**：像素 → 图像编码器 → 整图表示 → 投影。原始 CLIP 比较了改进的 ResNet 与 ViT，不是所有 CLIP 都用 ViT。")
    text("**投影（projection）**：把两条分支的输出映射到相同维度，使它们能做相似度比较；两条分支的内部维度不必相同。")
    text("**归一化（normalization）**：这里把向量长度调整为 1；随后点积就是余弦相似度，重点比较方向。")
    text("### 原始 CLIP 文本编码器的具体选择")
    text("公开原始实现采用 BPE（Byte Pair Encoding，字节对编码）分词，并加入序列起止标记；最大上下文长度是 77 个词元位置，包含特殊标记，不是 77 个单词。")
    text("原始文本 Transformer 使用因果掩码，取文本结束标记所在位置的表示，再做投影。结束位置能利用此前输入，这里最终用于图文匹配，而不是逐词生成回答。")
    text("图像 ViT 分支使用图像块与可学习汇总位置；经过视觉编码后，把汇总向量投影到共同空间。CNN 版本则使用不同的视觉汇总实现。")
    text("两分支独立编码，可以分别预计算。原始 CLIP 没有让每个文字词元与每个图像块通过交叉注意力逐一交流；全局对齐不自动等于精确定位。")
    text("语言学类比：图像与描述像两种表达方式，模型学习哪些表达相互对应。但文本通常只说出画面的一部分，因此不是严格逐项翻译。")
    link(title="CLIP 原论文（ICML 2021）", url="https://proceedings.mlr.press/v139/radford21a.html")
    link(title="原始实现：文本编码、图像编码与相似度", url="https://github.com/openai/CLIP/blob/main/clip/model.py")


def clip_training():
    text("## 08 · 对比学习：在候选中找回正确配对")
    text("**正例（positive pair）**：训练数据中配对的一张图像与一段文字。")
    text("**负例（negative pair）**：被用作对比的其他组合。原始 CLIP 把一个训练批次内其余图文组合视作负例。")
    text("**批次（batch）**：一次共同参与计算的一组样本。N 对图文两两比较，得到 N×N 个分数。")
    image("images/lecture_03/contrastive.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("逐行比较：给定图像，把对应文字排在其他文字之前。")
    text("逐列比较：给定文字，把对应图像排在其他图像之前。")
    text(r"设归一化图像向量为 $u_i$、文字向量为 $v_j$，分数为 $s_{ij}=u_i^\top v_j/\tau$。$\tau$ 是温度，控制分数分布的尖锐程度。")
    text(r"图像到文字的损失：$L_{I\to T}=-\frac{1}{N}\sum_i\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})}$。文字到图像交换行列，最后取两者平均。")
    text("**交叉熵损失（cross-entropy loss）**：在这里惩罚『正确配对分到的概率太低』。直观目标是提高正确项相对其他候选的分数。")
    text("### 小型数值演示：归一化、相似度与双向损失")
    text("下面人为给定三对向量；对角线对应配对编号。这不是已经训练好的 CLIP，数值仅演示其核心匹配计算。")
    image_vectors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    text_vectors = np.array([[1., .2, 0.], [0., 1., .2], [.2, 0., 1.]])
    u = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)
    v = text_vectors / np.linalg.norm(text_vectors, axis=1, keepdims=True)
    similarity = u @ v.T  # @inspect similarity
    logits = similarity / 0.2
    row_exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    row_prob = row_exp / row_exp.sum(axis=1, keepdims=True)  # @inspect row_prob
    col_exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    col_prob = col_exp / col_exp.sum(axis=0, keepdims=True)  # @inspect col_prob
    loss = -(np.log(np.diag(row_prob)).mean() + np.log(np.diag(col_prob)).mean()) / 2  # @inspect loss
    print("图文相似度矩阵：\n", np.round(similarity, 3))
    print("逐行匹配概率：\n", np.round(row_prob, 3))
    print("双向平均损失：", round(float(loss), 4))
    wrong_pairing = np.array([1, 2, 0])
    wrong_loss = -np.log(row_prob[np.arange(3), wrong_pairing]).mean()  # @inspect wrong_loss
    print("保持分数不变却把对应标签错位，行损失：", round(float(wrong_loss), 4))
    text("完整训练还会反向传播，通过梯度更新两分支与投影参数。本例只计算损失，没有进行训练；原始实现还学习相似度的缩放系数。")
    image("images/lecture_03/clip-learning.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("### 配对监督的优点与噪声")
    text("描述文字比单个类别标签更丰富：可包含对象、属性、动作与关系，也省去了为每项视觉属性单独设计固定标签表的步骤。")
    text("**假负例（false negative）**：同一批次另一张图也可能是『一只猫』，却被当作不匹配项。网页描述也可能与图片无关、只描述一部分或包含偏见。")
    text("因此，对比学习是根据训练配对关系优化，不是直接获得了完备、无歧义的语义对应。")


def clip_applications():
    text("## 09 · 从图文匹配到检索与零样本分类")
    image("images/lecture_03/clip-candidates.svg", width=760, style={"maxWidth": "100%", "height": "auto"})
    text("**图文检索**：提前编码图片库；输入一句话，编码成向量，再按相似度找图片。也可以用图片检索文字。")
    text("**零样本分类（zero-shot classification）**：针对当前分类任务，不用该任务的标注图片重新训练分类器，而是把候选类别写成文本，与图像比较。")
    text("例如候选类别为 cat、dog、car，可写成『a photo of a cat』『a photo of a dog』『a photo of a car』，分别编码，再选择与输入图像最相似的一项。")
    text("零样本不代表从未训练，也不保证预训练数据中从未出现相关类别或图像。它描述的是迁移到当前任务时的使用方式。")
    text("**提示模板（prompt template）**：给类别词增加上下文，如『a photo of a …』。模板、措辞与候选集合改变，都可能影响比较结果。")
    text("相似度分数不是经校准的『正确概率』；把候选分数做 softmax 得到的数值还依赖候选集合与缩放方式。最高分也可能只是全部错误候选中最接近的一个。")
    text("**生成模型中的作用**：CLIP 类文本编码器或图文相似度可成为更大系统的组件，但 CLIP 的基本输出是向量与匹配分数；它本身不直接画图，也不直接生成完整问答。")


def linguistic_limits():
    text("## 10 · 从语言学角度检验图文对齐")
    text("**词汇对应**：画面有狗，文字提到 dog；这只是最基本的内容联系。")
    text("**组合意义**：『红色方块在蓝色圆左边』与『蓝色方块在红色圆左边』使用相近词汇，但属性与对象的绑定不同。")
    text("**论元角色**：『猫吃鱼』与『鱼吃猫』交换动作参与者。仅识别猫、鱼和吃的动作还不够。")
    text("**否定与数量**：『有两只猫』、『没有猫』与『一只猫』需要区分；不能只依据 cat 这个词与画面对象相似。")
    text("**指称与语境**：『左边那个人』依赖参照系；『他』还可能依赖图像以外的篇章信息。")
    text("**跨语言与文化**：同一图像可有不同语言和文化背景下的描述。原始 CLIP 的结果不能直接当成所有语言表现一致的证据。")
    text("### 评价思路：控制变量，改变一个关系")
    text("固定一张图，把正确描述与只改变颜色、数量、左右或角色的描述成组比较。这类最小对照能减少模型仅靠关键词命中的机会。")
    text("例如正确描述为『猫吃鱼』，对照包括『鱼吃猫』『猫与鱼同时出现在画面中』。记录真实模型分数与配对差异，再判断它对关系的敏感性。")
    text("本讲提出的是评价设计，没有在这些例子上测量预训练 CLIP，不能把潜在局限写成每个版本都必然失败的结论。")


def recap():
    text("## 本讲回顾 · 结构负责表示，目标塑造对应")
    text("| 结构／方法 | 主要信息通路 | 本讲的直观解释 |\n|---|---|---|\n| RNN／LSTM | 沿时间更新状态 | 不断更新阅读笔记 |\n| Transformer | 按相关程度聚合序列位置 | 结合需要参考上下文 |\n| CNN | 局部连接、共享权重、多层组合 | 同一个检测器扫描不同位置 |\n| ViT | 图像块序列的注意力交互 | 把图像块组织为可交互单元 |\n| CLIP | 双编码器＋双向对比目标 | 让配对图文在候选中互相找回 |")
    text("从 RNN 到 Transformer、从 CNN 到 ViT，改变的是组织信息的方式；CLIP 则用图文配对目标把两条表示路线联系起来。")
    text("课后小练习：为同一张图片写出一条正确描述，以及分别改变颜色、数量、位置、动作角色的四条对照描述，标注每条改变的语言现象。后续实验再比较真实 CLIP 的分数。")


def runnable_rnn():
    text("### 可执行演示 · 一个真正的 PyTorch RNN")
    text("PyTorch 是深度学习计算框架。下面构造随机初始化的循环层，比较同一组词向量按不同顺序输入后的结果。")
    torch.manual_seed(7)
    inputs = torch.tensor([[[1., 0.], [.5, .5], [0., 1.]]])
    rnn = nn.RNN(input_size=2, hidden_size=4, batch_first=True)
    states, final_state = rnn(inputs)  # @inspect states, @inspect final_state
    reversed_states, reversed_final = rnn(inputs.flip(1))  # @inspect reversed_final
    print("输入 [批次, 长度, 特征]：", tuple(inputs.shape))
    print("每步状态：", tuple(states.shape))
    print("最终状态 [层数×方向数, 批次, 隐藏维度]：", tuple(final_state.shape))
    print("原顺序最终状态：", final_state.detach().numpy().round(3))
    print("反顺序最终状态：", reversed_final.detach().numpy().round(3))
    text("batch_first=True 让输入和每步输出以批次为第一维，但最终状态的轴顺序仍不同。换成 nn.LSTM 时，还会返回记忆单元状态。")
    text("本例运行了真实网络层，但未训练，输出差异只能说明信息通路对顺序敏感。")


def runnable_transformer():
    text("### 可执行演示 · 因果掩码让未来词元不可见")
    text("用同一个 Transformer 层编码三个位置，只改动最后一个位置，再观察第一个位置是否受影响。关闭 dropout（训练时随机丢弃部分激活的机制），保证比较不受随机丢弃干扰。")
    torch.manual_seed(11)
    tokens = torch.randn(1, 3, 8)
    changed = tokens.clone()
    changed[0, 2, 0] += 10
    layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True).eval()
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)  # @inspect mask
    print("掩码（True 表示禁止关注）：\n", mask)
    full = layer(tokens)
    full_changed = layer(changed)
    causal = layer(tokens, src_mask=mask)
    causal_changed = layer(changed, src_mask=mask)
    first_change_full = (full[:, 0] - full_changed[:, 0]).abs().max().item()  # @inspect first_change_full
    first_change_causal = (causal[:, 0] - causal_changed[:, 0]).abs().max().item()  # @inspect first_change_causal
    print("输出 shape：", tuple(full.shape))
    print("无掩码，第一个位置最大变化：", round(first_change_full, 6))
    print("有因果掩码，第一个位置最大变化：", round(first_change_causal, 6))
    assert torch.allclose(causal[:, 0], causal_changed[:, 0], atol=1e-6)
    text("因果掩码屏蔽了未来位置，因而改动最后一个位置不会影响第一个位置。这里验证信息可见范围，没有训练模型，也没有测试语义理解。")


def runnable_cnn():
    text("### 可执行演示 · 让卷积核在整张图上滑动")
    pixels = torch.zeros(1, 1, 6, 6)
    pixels[:, :, :, 3:] = 1
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    kernel = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])  # @inspect kernel
    with torch.no_grad():
        conv.weight.copy_(kernel.reshape(1, 1, 3, 3))
    features = conv(pixels)  # @inspect features
    print("输入 [批次, 通道, 高, 宽]：", tuple(pixels.shape))
    print("输入像素：\n", pixels[0, 0].numpy())
    print("输出 shape：", tuple(features.shape))
    print("卷积特征图：\n", features[0, 0].detach().numpy())
    text("6×6 图像经过 3×3 核，步幅为 1、不补边，得到 4×4 特征图。同一个核在每个位置重复使用；明暗边界附近的响应更强。")


def runnable_vit():
    text("### 可执行演示 · 从图像到一个 ViT 编码块")
    torch.manual_seed(13)
    pixels = torch.randn(1, 3, 8, 8)
    patch_embedding = nn.Conv2d(3, 12, kernel_size=2, stride=2)
    patch_tokens = patch_embedding(pixels).flatten(2).transpose(1, 2)
    cls = nn.Parameter(torch.zeros(1, 1, 12))
    position = nn.Parameter(torch.randn(1, 17, 12) * .02)
    sequence = torch.cat([cls, patch_tokens], dim=1) + position
    block = nn.TransformerEncoderLayer(d_model=12, nhead=3, dim_feedforward=24, dropout=0.0, batch_first=True).eval()
    encoded = block(sequence)
    image_vector = encoded[:, 0]  # @inspect image_vector
    print("图像块向量：", tuple(patch_tokens.shape))
    print("加入 CLS 与位置向量：", tuple(sequence.shape))
    print("编码后的所有位置：", tuple(encoded.shape))
    print("提取整图向量：", tuple(image_vector.shape))
    text("卷积核大小与步幅都为 2，实现不重叠切块与共享线性投影；然后将 16 个块和 1 个 CLS 一起送入注意力层。")
    text("这是只含一个编码块的小型结构演示，参数随机初始化；它没有真实预训练 ViT 的视觉识别能力。")


class MiniCLIP(nn.Module):
    """Minimal dual encoder for four color labels; not a pretrained CLIP."""
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 8))
        self.text_encoder = nn.Embedding(4, 8)

    def forward(self, pixels, text_ids):
        u = F.normalize(self.image_encoder(pixels), dim=-1)
        v = F.normalize(self.text_encoder(text_ids), dim=-1)
        return u @ v.T / 0.2


def fit_miniclip(model, pixels, text_ids, steps=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    targets = torch.arange(4)
    history = []
    for step in range(steps + 1):
        logits = model(pixels, text_ids)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
        if step in (0, 20, steps):
            accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
            history.append((step, round(loss.item(), 4), accuracy))
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return history


def runnable_clip_training():
    text("### 可执行演示 · 让四对图文从随机状态开始学习")
    text("用红、绿、蓝、黄四张 8×8 色块图，以及 red、green、blue、yellow 四个单词建立配对，训练一个微型双编码器。只使用 CPU，不下载预训练权重。")
    text("为突出对比学习，本例把图像分支简化为线性层，把每个颜色词作为一个词元并查表；没有自然语言 Transformer，也没有大规模图文预训练。")
    torch.manual_seed(23)
    names = ["red", "green", "blue", "yellow"]
    colors = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
    pixels = colors[:, :, None, None].expand(-1, -1, 8, 8).clone()
    text_ids = torch.arange(4)
    model = MiniCLIP()  # @stepover
    before = model(pixels, text_ids).detach()  # @stepover; @inspect before
    print("图像 batch：", tuple(pixels.shape))
    print("文字词表：", names)
    print("训练前，每张图找到的文字编号：", before.argmax(dim=1).tolist())
    text("每一步：计算两个方向的交叉熵 → 清空旧梯度 → 反向传播 → 更新两分支参数。下面的记录来自实际运行，不是预先写定的成绩。")
    text("```python\n" + inspect.getsource(fit_miniclip) + "\n```")
    history = fit_miniclip(model, pixels, text_ids)  # @stepover; @inspect history
    print("训练记录（更新步数, 双向损失, 图→文训练集准确率）：\n", history)
    after = model(pixels, text_ids).detach()  # @stepover; @inspect after
    print("训练后，相似度矩阵（未除温度）：\n", (after * .2).numpy().round(3))
    print("图→文编号：", after.argmax(dim=1).tolist())
    print("文→图编号：", after.argmax(dim=0).tolist())
    print("正确配对编号：", text_ids.tolist())
    assert history[-1][1] < history[0][1]
    assert torch.equal(after.argmax(dim=1), text_ids)
    assert torch.equal(after.argmax(dim=0), text_ids)
    text("再打乱候选文字的顺序：图像应找到颜色词，而不是固定选择矩阵的对角线。训练时的对角线仅来自配对样本的排列方式。")
    order = torch.tensor([2, 0, 3, 1])  # @inspect order
    shuffled = model(pixels, text_ids[order]).detach()  # @stepover
    predicted_ids = order[shuffled.argmax(dim=1)]  # @inspect predicted_ids
    print("候选顺序：", [names[i] for i in order.tolist()])
    print("恢复成词表编号后的匹配：", predicted_ids.tolist())
    assert torch.equal(predicted_ids, text_ids)
    text("这里只检查四个训练样本是否被记住，不能据此宣称模型能理解新句子、识别真实照片或完成零样本泛化。完整的 CLIP 需要丰富数据、更强编码器与独立评测。")
    text("课堂可改动：调整训练步数、温度或配对标签，再重新编译，比较实际损失与检索结果。训练循环源代码位于本课文件末尾。")


if __name__ == "__main__":
    main()
