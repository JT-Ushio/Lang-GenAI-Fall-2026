"""第四讲：从图文对齐到看图说话。90 分钟；配套 notes/lecture_04.md。"""
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from execute_util import image, interactive, link, text
from demos.lecture_04_minivlm import (
    MiniVLM, VOCAB, IDS, make_data, answer_loss, generate, train_demo,
)
from demos.lecture_04_preprocess import (
    load_configs, load_rgb, pad_to_square, geometry, normalize_rgb,
    prepare_assets, shape_walkthrough,
)

FIGURE_STYLE = {"maxWidth": "min(100%, calc(100vw - 125px))", "height": "auto"}


def main():
    text("# 第四讲 · 从图文对齐到看图说话")
    text("## 视觉语言模型：图像描述、视觉问答与多模态对话")
    text("**课程**：《以语言为中心的生成式AI》｜**对象**：外语＋计算机双学位本科生｜**时长**：45 分钟 × 2 课时。")
    text("第三讲用 CLIP 学习『一张图与哪段文字对应』。第四讲学习把视觉表示交给语言模型，让它根据图像与问题组织答案。")
    text("**本讲目标一**：沿着『像素 → 视觉特征 → 连接器 → 语言模型 → 答案』说明信息怎样流动。")
    text("**本讲目标二**：区分对比学习与下一词元预测，读懂冻结参数、因果掩码和答案区监督。")
    text("**本讲目标三**：运行一个微型 VLM，并分别检查视觉事实、关系、证据与语言表达。")
    text("**第一课时**：任务与生成机制 → 视觉词元 → 三种连接路线 → LLaVA 结构。")
    text("**第二课时**：训练数据与损失 → 微型模型训练 → 评测、幻觉与多语言应用。")
    from_clip_to_vlm()
    tasks_and_language()
    autoregressive_language()
    visual_tokens()
    image_preprocessing_cases()
    runnable_visual_prefix()
    connector_routes()
    runnable_query_compression()
    llava_case()
    llava_module_dimensions()
    training_data_and_stages()
    runnable_frozen_gradient()
    answer_supervision()
    runnable_answer_loss()
    runnable_mini_vlm()
    details_and_video()
    evaluation_and_language()
    recap_and_reading()


def from_clip_to_vlm():
    text("## 01 · 从『匹配』到『生成』")
    text("**VLM（Vision-Language Model，视觉语言模型）**：联合处理视觉与语言的模型。这个名称覆盖图文检索、理解与生成；CLIP 也属于广义 VLM。")
    text("本课主要讨论**生成式 VLM**：输入图像与文字，输出文字；接入大型语言模型时，也常称为 **MLLM（Multimodal Large Language Model，多模态大语言模型）**。")
    image("images/lecture_04/clip-to-vlm.svg", width=760, style=FIGURE_STYLE)
    text("**CLIP 的典型使用**：把图像和候选描述分别编码，比较相似度；候选里可以写很长的句子，但匹配本身不会逐词生成新句子。")
    text("**生成式 VLM 的典型使用**：把图像与问题作为条件，依次产生词元，组成描述、答案或解释。")
    text("**增加一个语言模型**只是结构起点。还需要连接视觉与语言的表示，并用合适数据训练，否则维度接上了也不代表模型学会看图。")
    text("### 从三讲到四讲，沿用与改变的部分")
    text("**沿用**：图像编码器、图像块、向量表示、注意力与预训练。")
    text("**改变**：输出从相似度分数变成文字词表上的概率；训练不再只回答『哪一对匹配』。")
    link(title="教学组织参考：Stanford CS336 · Lecture 17 · Multimodal models", url="https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py")


def tasks_and_language():
    text("## 02 · 同一张图，不同的语言任务")
    image("images/lecture_04/scene-tasks.svg", width=760, style=FIGURE_STYLE)
    text("**图像描述（image captioning）**：生成图像的文字描述，例如『左侧是红色方块，右侧是蓝色圆形』。需要选择哪些信息值得说。")
    text("**视觉问答（VQA，Visual Question Answering）**：结合图像和问题回答，例如『左边是什么颜色？』。问题规定了需要提取的内容。")
    text("**OCR（Optical Character Recognition，光学字符识别）**：识别图片中的文字。VLM 可结合文字、版面与问题作答，但细小文字仍可能读错。")
    text("**视觉定位（visual grounding）**：把语言表达对应到图像区域，例如为『右边的蓝色圆形』返回边界框；是否能输出坐标取决于模型和训练。")
    text("**多模态对话**：保留图像及对话上下文，连续处理问题；系统实际保留哪些图像与前文，需要由输入格式和上下文管理决定。")
    text("### 语言学连接：共同参照与指称")
    text("**指称（reference）**：『它』『左边那个』要与具体对象对应。图像提供可共享的场景，语言提供挑选对象与关系的线索。")
    text("**视角**：本课的『左边』统一指观看图像时的左侧；人物自身的左手、画面的左边与相机的左边可能不同。")
    text("**预设（presupposition）**：『这辆红色汽车是什么牌子？』预设了汽车存在；图中没有车时，模型应指出证据不支持这个前提。")
    text("**双语表达**：改用英文回答不应改变颜色、数量、年代与左右关系。表达自然和信息等值需要分别检查。")
    text("上图的描述与问答都是教师构造的标准答案，用来说明任务，没有调用真实大模型。")


def autoregressive_language():
    text("## 03 · 先补上语言分支：答案怎样逐步生成？")
    text("**解码器式语言模型（decoder-only language model）**：根据已给出的上下文预测下一个词元。『解码』在这里指生成序列，不是文件解压。")
    text("**自回归（autoregressive）**：已生成的内容成为后续生成的条件，例如 a → red → square → 结束。")
    text("**词表（vocabulary）**：模型可输出的词元集合。")
    text("**输出头（output head）**：把当前位置的隐藏向量变成词表中各词元的分数。")
    text("**logits（未归一化分数）**：经 softmax 变成概率。一个概率最高的词元不一定对应正确事实，只表示模型在给定条件下更倾向于输出它。")
    text("**EOS（End of Sequence，序列结束标记）**：告诉生成过程可以停止；实际系统也可使用对话结束标记或长度限制。")
    image("images/lecture_04/autoregressive.svg", width=760, style=FIGURE_STYLE)
    text("**条件生成**：整个答案的概率，是各步下一词元概率的乘积。图中 I 是图像，Q 是问题，A 是答案；从第 1 到第 T 个词元，每一步都以图像、问题和已生成前文为条件。")
    text("**教师强制（teacher forcing）**：训练时给模型正确的答案前文，让它预测下一词元；推理时用的是模型自己的输出，错误可能逐步积累。")
    text("**因果掩码（causal mask）**：预测某位置之后的词元时，只允许参考该位置及以前的内容，禁止偷看未来答案。")
    text("### 代码演示一：哪些位置可以看见？")
    names = ["图像1", "图像2", "问题", "BOS", "red", "EOS"]
    blocked = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)  # @inspect blocked
    visible = (~blocked).int()  # @inspect visible
    print("行：发起查询的位置；列：被读取的位置。1＝允许，0＝屏蔽。")
    print(names)
    print(visible)
    assert visible[3, 4].item() == 0
    text("BOS 所在位置可以读取图像与问题，却不能读取后面的 red。下图可点击每个查询位置，查看它的可见范围。")
    interactive("images/lecture_04/vlm-lab/index.html", height=710, fragment="mask")
    text("这里展示微型模型采用的完整因果序列。真实视觉编码器可先用双向注意力编码图像，其他 VLM 也可能使用不同的图像前缀掩码。")
    link(title="下一词元建模与上下文学习背景：Language Models are Few-Shot Learners", url="https://arxiv.org/abs/2005.14165")


def visual_tokens():
    text("## 04 · 图像分支：从 CLIP 摘要到视觉词元")
    text("第三讲的 CLIP 检索用整图向量比较相似度；图像编码器内部已经有多个位置的特征。生成模型经常取出这些块级特征，保留更多区域信息。")
    image("images/lecture_04/visual-features.svg", width=760, style=FIGURE_STYLE)
    text("**视觉词元（visual token）**：进入后续网络的视觉向量位置。这里的词元是计算序列中的单位，不是语言学意义上的词。")
    text("**连续向量**：常见 VLM 的图像块特征直接以浮点数向量进入语言模型，不必先变成『猫』『鱼』等文字。")
    text("**离散图像词元**：另一些系统用视觉码本把图像表示为编号。本课的 LLaVA 路线和微型演示采用连续特征，不使用视觉码本。")
    text("**块级特征**包含局部与上下文信息，但一个块不一定就是一个物体；一个物体也可能横跨多个块。")
    text("语言类比：只有一句概括，很难回答所有细节；保留多个表示位置，为针对问题读取不同区域提供了计算条件。")


def runnable_visual_prefix():
    text("### 代码演示二：把视觉特征拼进文字序列")
    text("使用 4×4 的小 RGB 图：观看时左半红、右半蓝。这里演示真实的切块嵌入与维度转换，不运行预训练 CLIP。")
    pixels = torch.zeros(1, 3, 4, 4)  # @inspect pixels
    pixels[:, 0, :, :2] = 1.
    pixels[:, 2, :, 2:] = 1.
    hwc_pixels = pixels[0].permute(1, 2, 0)  # @inspect hwc_pixels
    print("便于看图的 HWC（高、宽、通道）：", tuple(hwc_pixels.shape))
    print("模型输入 NCHW（批量、通道、高、宽）：", tuple(pixels.shape))
    torch.manual_seed(9)
    patch_embed = nn.Conv2d(3, 6, kernel_size=2, stride=2)
    feature_map = patch_embed(pixels)  # @inspect feature_map
    patch_features = feature_map.flatten(2).transpose(1, 2)  # @inspect patch_features
    projector = nn.Linear(6, 8)
    visual_prefix = projector(patch_features)  # @inspect visual_prefix
    question_ids = torch.tensor([[1, 2, 3]])
    text_embeddings = nn.Embedding(10, 8)(question_ids)  # @inspect text_embeddings
    combined = torch.cat([visual_prefix, text_embeddings], dim=1)  # @inspect combined
    print("卷积特征图 [B, Dv, H块, W块]：", tuple(feature_map.shape))
    print("视觉块 [B, N, Dv]：", tuple(patch_features.shape))
    print("投影后的视觉序列：", tuple(visual_prefix.shape))
    print("问题的嵌入：", tuple(text_embeddings.shape))
    print("拼接后 [B, N+L, D语言]：", tuple(combined.shape))
    assert combined.shape == (1, 7, 8)
    text("**连接器（connector／projector）**：把视觉特征维度映射到语言模型的嵌入维度。长度可以保持不变，向量维数需要兼容。")
    text("视觉编码器和连接器的权重在本段随机初始化；形状检查证明计算可连接，不证明这些向量已具备语义能力。")
    text("### 交互观察：逐个模块查看像素怎样进入语言计算")
    text("下图采用后面微型模型的两个视觉块；点选模块查看形状，交换左右颜色或换一个问题，观察应回答的内容怎样变化。")
    interactive("images/lecture_04/vlm-lab/index.html", height=710, fragment="flow")


def image_preprocessing_cases():
    text("### VLM 预处理：模型看到的图，可能已经变了")
    text("**预处理（preprocessing）**：在模型前向计算前，整理图像方向、颜色通道、空间尺寸、数值范围和批量格式；本身没有生成语义答案。")
    text("**缩放（resize）**：改变像素尺寸。")
    text("**中心裁剪（center crop）**：保留中心窗口，边缘可能被丢弃。")
    text("**补边（padding）**：扩展画布而不直接剪掉原内容；缩放到固定尺寸后，原内容占用的像素仍可能减少。")
    text("所选 LLaVA-1.5 7B 官方配置是 `image_aspect_ratio='pad'`：先补成正方形，再交给 CLIP 图像处理器；直接使用处理器的中心裁剪是下图的对照。")
    cases = prepare_assets()  # @stepover
    text("### Case A · 横图：物体在左右边缘")
    image("images/lecture_04/preprocess-wide.svg", width=760, style=FIGURE_STYLE)
    text("原图 600×300。短边放大到 336 后变成 672×336，再中心裁成 336×336，只留下原图中部；左右物体可能直接消失。")
    text("先补成 600×600 再缩放到 336×336，可以保留两端。**保住内容**与**保住细节清晰度**是不同目标。")
    text("### Case B · 长文档：年份与编号在上下边缘")
    image("images/lecture_04/preprocess-document.svg", width=760, style=FIGURE_STYLE)
    text("中心裁剪会删掉页眉年份与页脚编号；补边保留全页，但文字变小。高分辨率、局部裁块或单独提供相关区域，是不同的处理选择。")
    text("**直接拉伸到正方形**会改变长宽比例，圆形可能变成椭圆；本案例的补边方式保持内容比例。上图是教学像素经过代码处理的真实结果，没有做模型 OCR。")
    text("### Case C · 灰度、透明图与相机方向")
    gray_rgb = load_rgb(cases["gray"])  # @stepover
    gray_pixel = np.asarray(gray_rgb)[0, 0].tolist()  # @inspect gray_pixel
    print("灰度像素 128 转为 RGB：", gray_pixel)
    text("**灰度 → RGB**：复制为三个通道，符合模型输入格式；不能恢复原本不存在的颜色。")
    rgba_rgb = load_rgb(cases["rgba"])  # @stepover
    transparent_pixel = np.asarray(rgba_rgb)[0, 0].tolist()  # @inspect transparent_pixel
    print("透明红像素合成到白背景：", transparent_pixel)
    text("**Alpha（透明度通道）**：透明图要明确如何合成背景。本例选白色；仅丢掉透明度可能把本来不可见的 RGB 值显示出来。")
    oriented_rgb = load_rgb(cases["exif"])  # @stepover
    corrected_size = oriented_rgb.size  # @inspect corrected_size
    print("JPEG 存储像素为 160×80；按 EXIF 方向转正后：", corrected_size)
    text("**EXIF（可交换图像文件格式／相机元数据）**：可记录显示方向；`ImageOps.exif_transpose` 让实际像素与显示方向一致，避免左右、上下分析发生错位。")
    text("### 可执行代码：文件 → 模型像素张量")
    text("```python\n" + inspect.getsource(load_rgb) + "```")
    image("images/lecture_04/preprocess-pipeline.svg", width=760, style=FIGURE_STYLE)
    text("**标准化（standardization）**：各通道先除以 255，再减去处理器给定均值并除以标准差；这里使用预训练时的常数，不重新估计每张图片的均值。")
    rgb = load_rgb(cases["wide"])  # @stepover
    _, _, processor_config = load_configs()  # @stepover
    background = tuple(int(x * 255) for x in processor_config["image_mean"])  # @inspect background
    square = pad_to_square(rgb, background)  # @stepover
    square_size = square.size  # @inspect square_size
    resized = square.resize((336, 336), Image.Resampling.BICUBIC)
    uint8_hwc = np.asarray(resized, dtype=np.uint8)
    pixel_sample = uint8_hwc[0, 0].tolist()  # @inspect pixel_sample
    rgb01 = uint8_hwc.astype(np.float32) / 255.0
    mean = np.asarray(processor_config["image_mean"], dtype=np.float32)
    std = np.asarray(processor_config["image_std"], dtype=np.float32)
    normalized_hwc = (rgb01 - mean) / std
    normalized_sample = np.round(normalized_hwc[0, 0], 4).tolist()  # @inspect normalized_sample
    object_pixel = uint8_hwc[151, 42].tolist()  # @inspect object_pixel
    object_normalized = np.round(normalized_hwc[151, 42], 4).tolist()  # @inspect object_normalized
    chw = torch.from_numpy(np.ascontiguousarray(normalized_hwc.transpose(2, 0, 1)))
    pixel_values = chw.unsqueeze(0)
    model_input_shape = list(pixel_values.shape)  # @inspect model_input_shape
    print("PIL 尺寸按宽、高：", rgb.size, "→ 补边：", square_size, "→ 缩放：", resized.size)
    print("uint8 像素 HWC：", uint8_hwc.shape, "样本：", pixel_sample)
    print("标准化后的同一像素：", normalized_sample)
    print("红块内一个像素：", object_pixel, "→ 标准化：", object_normalized)
    print("模型输入 BCHW：", model_input_shape, "dtype：", pixel_values.dtype)
    assert model_input_shape == [1, 3, 336, 336]
    assert torch.allclose(chw, normalize_rgb(resized, processor_config))  # @stepover
    text("**HWC**：高、宽、通道。**CHW**：通道、高、宽。**BCHW／NCHW**：再加批量维；这里 B=1。PIL 的 `size` 则是宽、高。")
    text("标准化后的数可以为负或大于 1；显示图片应使用原 RGB 像素，或先反标准化。这里仅查看单个像素和 shape，避免变量面板展开整幅图片。")
    text("**数值陷阱**：已经是 [0,1] 的浮点图不要再次除以 255；RGB 与 BGR（蓝、绿、红顺序）也不能混用。具体步骤始终以目标模型的处理器为准。")
    text("这些操作只整理像素；后面仍需要视觉编码器，才能得到供语言模型使用的特征。")
    link(title="预处理依据：LLaVA 官方 process_images / expand2square", url="https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py")
    link(title="均值、标准差与尺寸：CLIP ViT-L/14-336 处理器配置", url="https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json")


def connector_routes():
    text("## 05 · 来龙去脉：怎样让语言模型使用图像？")
    text("早期图像描述已经使用『视觉编码器＋语言解码器』：例如 Show and Tell 用 CNN 编码图片、用 LSTM 生成描述。VLM 的思想早于对话式大模型。")
    text("后来可以复用较强的视觉预训练和语言预训练，再学习两者之间的连接。下面三种路线帮助理解信息如何进入语言计算。")
    image("images/lecture_04/connector-routes.svg", width=760, style=FIGURE_STYLE)
    text("### 路线 A · LLaVA：投影后与文字拼接")
    text("把视觉块逐个映射到语言维度，作为视觉前缀或插入到指定图像位置；后续文字在自注意力中读取这些位置。")
    text("结构直观，但大量视觉块会占用语言模型的上下文长度。『拼接』是拼接向量序列，不是把图片路径粘到文本里。")
    text("### 路线 B · BLIP-2：少量可学习查询读取视觉特征")
    text("**Q-Former（Querying Transformer，查询 Transformer）**：用可学习的查询向量读取冻结视觉编码器的输出，再把压缩后的表示接入冻结语言模型。")
    text("原论文实验使用 32 个查询。这些是学习得到的向量，不是 32 个自然语言问题；压缩减小后续输入，也可能限制细节保留。")
    text("### 路线 C · Flamingo：在语言模型层间读取视觉信息")
    text("**重采样器（resampler）**：先把数量可变的视觉特征变成固定数量表示，再在冻结的语言模型层间加入可训练的门控交叉注意力等模块。")
    text("**交叉注意力**：语言状态提供 Q（查询），视觉表示提供 K（键）和 V（值）；语言生成过程借此读取视觉内容。")
    text("三种连接方式不构成固定的优劣排名；选择取决于数据、训练方式、视觉细节、序列长度与任务。")
    link(title="Show and Tell（2014 预印本／CVPR 2015）：早期神经图像描述", url="https://arxiv.org/abs/1411.4555")
    link(title="BLIP-2（2023）：Q-Former 与冻结的两类预训练模型", url="https://arxiv.org/abs/2301.12597")
    link(title="Flamingo（2022）：重采样与门控交叉注意力", url="https://arxiv.org/abs/2204.14198")


def runnable_query_compression():
    text("### 代码演示三：4 个视觉位置压缩成 2 个查询输出")
    text("这是一个独立的交叉注意力运算，只演示查询压缩的形状和权重，不是完整 Q-Former 或 Flamingo。")
    torch.manual_seed(12)
    visual_features = torch.randn(1, 4, 8)
    learned_queries = nn.Parameter(torch.randn(1, 2, 8))
    cross_attention = nn.MultiheadAttention(8, 2, dropout=0., batch_first=True)
    compressed, weights = cross_attention(learned_queries, visual_features, visual_features)  # @inspect compressed, @inspect weights
    print("原始视觉特征：", tuple(visual_features.shape))
    print("两条查询的输出：", tuple(compressed.shape))
    print("查询对四个视觉位置的权重：", weights.detach().round(decimals=3))
    assert compressed.shape == (1, 2, 8)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2))
    text("两条查询可以学习不同的信息组合。这里没有训练，权重不能解释成模型已经识别了某个物体。")


def llava_case():
    text("## 06 · 具体读一个模型：LLaVA")
    text("**LLaVA（Large Language and Vision Assistant，大型语言与视觉助手）**：在 2023 年展示了连接视觉编码器与语言模型、再进行视觉指令微调的路线。")
    text("**原始 LLaVA**采用线性投影连接 CLIP 视觉分支与 Vicuna 语言模型；**LLaVA-1.5**的常用基线改用两层 MLP 连接器。本课区分这两个版本。")
    text("**MLP（Multilayer Perceptron，多层感知机）**：由线性层和非线性变换组成的网络；这里用来变换每个视觉块的特征。")
    image("images/lecture_04/llava-architecture.svg", width=760, style=FIGURE_STYLE)
    text("### LLaVA-1.5 7B：从具体维度追踪信息")
    text("**视觉输入**：336×336 RGB 图像；14×14 图像块形成 24×24＝576 个位置。")
    text("**视觉特征**：所选 CLIP ViT-L/14 配置的隐藏维度是 1024；官方实现取指定中间层的块特征，并在 patch 模式下去掉汇总词元。")
    text("**连接器**：1024 → 4096 → 4096，中间使用 GELU 非线性变换；得到语言模型可以接收的 4096 维视觉向量。")
    text("**语言输入**：视觉向量与问题、聊天模板及答案前文的嵌入共同进入 Vicuna 7B；输出头仍预测文字词元。")
    visual_count = (336 // 14) ** 2  # @inspect visual_count
    ordinary_text_length = 20
    raw_input_length = ordinary_text_length + 1
    combined_length = raw_input_length - 1 + visual_count  # @inspect combined_length
    print("20 个普通文字位置＋1 个图像占位符；替换后总长度：", combined_length)
    text("『投影』不等于逐块翻译成单词；『576 个位置』也不等于图片里有 576 个物体。其他版本的尺寸、维度和模板可能不同。")
    link(title="LLaVA 原论文：Visual Instruction Tuning", url="https://arxiv.org/abs/2304.08485")
    link(title="LLaVA-1.5 原论文：Improved Baselines with Visual Instruction Tuning", url="https://arxiv.org/abs/2310.03744")
    link(title="核对视觉特征选择：LLaVA 官方 CLIP encoder 实现", url="https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py")
    link(title="核对维度与连接器：LLaVA-1.5 7B 配置", url="https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json")


def llava_module_dimensions():
    text("### 展开 LLaVA-1.5 7B：每个模块的维度")
    text("下面固定批量 B=1，普通文字位置 20 个；另有一个图像占位符。长度是教学设定，不能用字数代替真实分词器的词元计数。")
    image("images/lecture_04/llava-full-pipeline.svg", width=760, style=FIGURE_STYLE)
    link(title="独立打开 LLaVA 架构总览图（可放大）", url="images/lecture_04/llava-full-pipeline.svg")
    text("### 视觉分支内部：块序列、CLS 与注意力")
    image("images/lecture_04/llava-vision-dimensions.svg", width=760, style=FIGURE_STYLE)
    text("**CLS（分类／汇总词元）**：在 576 个图像块前加一个汇总位置，视觉 Transformer 因而处理 577 个位置；本配置选出中间层后，再去掉这个汇总位置。")
    text("**16 个注意力头**：1024 维分成每头 64 维；视觉注意力仍把输出组合回 1024 维。前馈网络中间扩展为 4096 维，再回到 1024 维。")
    text("**第 -2 项隐藏状态**：官方配置选择倒数第二项；在这个 24 层 CLIP 的隐藏状态序列中，对应第 23 个 Transformer 块后的表示。不是最终整图检索向量。")
    text("### 连接器与语言分支内部")
    image("images/lecture_04/llava-language-dimensions.svg", width=760, style=FIGURE_STYLE)
    text("**连接器 MLP**逐个变换视觉位置：先 1024→4096，经过 GELU 非线性，再 4096→4096；它改变宽度，不把 576 个位置合并成一个。")
    text("**Vicuna 7B**在此配置中有 32 层、32 个注意力头，每头 128 维。语言块输出仍为 `[B,S,4096]`；S 是拼接后位置数。")
    text("**RMSNorm（均方根归一化）**：调整特征尺度。")
    text("**门控前馈网络**：用两条 4096→11008 分支，经 SiLU 非线性和逐元素相乘后，再映射回 4096 维。")
    text("**输出头**把每个隐藏向量映射到 32000 个文字词元的分数；生成下一词元只使用最后一个位置，不能把 4096 维隐藏向量当成词表概率。")
    text("### 文字预处理：模板、分词与图像占位符")
    text("真实问题先按模型要求放入对话模板，再由对应分词器变成编号。模板标记、BOS 和标点也可能占用位置；任意一个中文字符不保证等于一个词元。")
    image("images/lecture_04/llava-image-placeholder.svg", width=760, style=FIGURE_STYLE)
    text("原始 LLaVA 用 `<image>` 和内部 -200 标记定位视觉插入点；普通文字查嵌入，图像标记被视觉向量替换。它不是要直接送进普通词嵌入表的负编号。")
    text("### 可执行维度演示：使用 meta 设备验证形状")
    text("**meta（仅形状模拟设备）**：保存张量的形状与类型，不分配真实参数和像素数据。下面执行切块、投影、拼接与输出头的形状运算，没有下载或运行 7B 模型。")
    text("```python\n" + inspect.getsource(shape_walkthrough) + "```")
    dimensions = shape_walkthrough()  # @stepover
    vision_output_shape = dimensions["drop_cls"]  # @inspect vision_output_shape
    merged_shape = dimensions["combined"]  # @inspect merged_shape
    logits_shape = dimensions["logits"]  # @inspect logits_shape
    generation_shape = dimensions["generation_last_logits"]  # @inspect generation_shape
    text("| 实际形状运算 | shape |\n|---|---|\n" + "\n".join(f"| {stage} | `{shape}` |" for stage, shape in dimensions.items()))
    print("视觉块：", vision_output_shape, "拼接：", merged_shape)
    print("训练输出：", logits_shape, "单步生成输出：", generation_shape)
    assert merged_shape == [1, 596, 4096]
    assert generation_shape == [1, 1, 32000]
    text("`torch.empty_like` 在代码中表示未运行的视觉／语言 Transformer 输出接口；Conv2d、MLP、嵌入与拼接则实际执行 meta 形状运算。维度正确不等于模型会正确回答。")
    text("### 批量预处理：两条问题长度不同，需要补齐序列")
    text("同一大小图像均有 576 个视觉位置；普通文字长度设为 20 与 12，替换占位符后的序列长度分别为 596 与 588。")
    sequence_lengths = torch.tensor([596, 588])  # @inspect sequence_lengths
    attention_mask = torch.arange(596)[None, :] < sequence_lengths[:, None]
    mask_shape = list(attention_mask.shape)  # @inspect mask_shape
    mask_tail = attention_mask[:, -10:].int().tolist()  # @inspect mask_tail
    print("批量补齐后的 attention mask：", mask_shape, "最后 10 项：", mask_tail)
    assert attention_mask.sum(1).tolist() == [596, 588]
    text("**序列补齐（sequence padding）**：把长度对齐，`attention_mask` 标出有效位置。这里的 0 屏蔽补齐项；它与因果掩码、答案 labels 的 -100 是不同机制，也不同于给图像补边。")
    link(title="核对占位符替换与批量补齐：LLaVA 官方 llava_arch.py", url="https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py")
    text("**第一课时小结**：预处理整理像素，视觉编码器形成块级表示；连接器协调宽度并替换图像占位符；语言模型对图像与问题条件下的答案进行预测。")


def training_data_and_stages():
    text("## 07 · 模型怎样学会按图回答？")
    image("images/lecture_04/training-stages.svg", width=760, style=FIGURE_STYLE)
    text("**阶段一：连接／特征对齐**。在所选 LLaVA 基线中冻结视觉编码器和语言模型，用图文配对训练连接器，让视觉表示能服务于语言生成。")
    text("**阶段二：视觉指令微调（visual instruction tuning）**。使用『图像＋问题／指令＋答案』训练，更新连接器和语言模型，视觉分支继续冻结。")
    text("**冻结（freeze）**：不更新所选模块的参数。冻结的模型仍参与前向计算；是否需要保留梯度通路，取决于其前面是否还有需要训练的模块。")
    text("**预训练（pretraining）**：提供原有视觉或语言能力。")
    text("**微调（fine-tuning）**：在此基础上适配任务。这两阶段不是所有 VLM 必须遵循的固定规范。")
    text("### 一条视觉指令数据的结构")
    text('```json\n{"image": "scene.png",\n "question": "只描述左侧物体的颜色和形状。",\n "answer": "左侧是红色方块。"}\n```')
    text("**图像描述数据**给出『图像—句子』配对；**视觉问答／对话数据**再提供任务要求，帮助模型按问题选择内容。")
    text("原始 LLaVA 的一部分指令数据用语言模型根据已有图片描述与边界框信息合成；当时生成这些指令的语言模型没有直接读取图片像素。")
    text("**合成数据核查**：源描述未提到的事实不能随意补入；生成的问题也不能默认包含图中不存在的物体。")
    text("**多语言数据核查**：语言之间保留事实与关系，检查翻译是否引入颜色、数量、指称和文化判断的变化。")
    text("**数据划分**：同一图片的不同问题、语言版本和近重复图应放在同一数据划分中；独立测试不应泄漏到训练。")
    link(title="LLaVA 官方项目：两阶段训练与视觉指令数据", url="https://llava-vl.github.io/")


def runnable_frozen_gradient():
    text("### 代码演示四：语言模型被冻结，连接器仍能学习")
    text("用三个小线性层代表视觉分支、连接器与语言模型，直接观察各模块是否收到参数梯度。只是梯度通路演示，不是训练真实 LLaVA。")
    torch.manual_seed(7)
    vision = nn.Linear(3, 4)
    projector = nn.Linear(4, 6)
    language = nn.Linear(6, 2)
    vision.requires_grad_(False)
    language.requires_grad_(False)
    features = vision(torch.tensor([[1., 0., 0.]]))
    prediction = language(projector(features))
    loss = F.cross_entropy(prediction, torch.tensor([0]))
    loss.backward()
    vision_has_grad = vision.weight.grad is not None  # @inspect vision_has_grad
    projector_grad = float(projector.weight.grad.norm())  # @inspect projector_grad
    language_has_grad = language.weight.grad is not None  # @inspect language_has_grad
    print("阶段一：视觉参数梯度＝", vision_has_grad,
          "连接器梯度范数＝", round(projector_grad, 4), "语言参数梯度＝", language_has_grad)
    assert not vision_has_grad and projector_grad > 0 and not language_has_grad
    text("损失仍通过冻结的 language 运算传回 projector。若把这段 language 前向放进 `torch.no_grad()`，就会切断连接器所需的梯度通路。")
    language.requires_grad_(True)
    projector.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(language(projector(features)), torch.tensor([0]))
    loss2.backward()
    stage2_gradients = {"vision": vision.weight.grad is not None,
                        "projector": projector.weight.grad is not None,
                        "language": language.weight.grad is not None}  # @inspect stage2_gradients
    print("阶段二：", stage2_gradients)  # @inspect stage2_gradients
    assert stage2_gradients == {"vision": False, "projector": True, "language": True}


def answer_supervision():
    text("## 08 · 监督目标：预测答案，不是重新抄写全部输入")
    text("**对比学习**：让配对的图像与文本相对其他候选更接近；上一讲和作业一采用这一目标。")
    text("**生成式监督**：对标准答案中的下一词元计算交叉熵，鼓励模型给正确词元更高概率。")
    text("**答案损失**：对每个正确答案词元的概率取负对数，再在答案区求平均；公式见下图。T 包含需要监督的结束标记，其他输入位置不计入此平均。")
    text("**交叉熵（cross-entropy）**：在这里相当于正确词元概率的负对数；正确词元概率越低，惩罚越大。")
    text("**labels（监督标签）**：与输入位置对齐。")
    text("**-100（忽略标签）**：让图像、用户问题等位置不直接计入答案损失。")
    image("images/lecture_04/answer-loss.svg", width=760, style=FIGURE_STYLE)
    text("**右移对齐**：位置 t 的输出预测 t＋1 的词元。代码用 `logits[:, :-1]` 对齐 `labels[:, 1:]`。")
    text("**忽略标签不等于遮挡上下文**：图像和问题虽然不作为答案目标，仍然可被回答位置读取；labels 和 attention mask 承担不同职责。")
    text("生成式 VLM 可以使用多种训练目标；这里聚焦 LLaVA 风格的答案区自回归监督，不表示所有模型只用一种损失。")


def runnable_answer_loss():
    text("### 代码演示五：检查 -100 与预测位置")
    labels = torch.tensor([[-100, -100, -100, -100, IDS["red"], IDS["<eos>"]]])  # @inspect labels
    logits = torch.zeros(1, 6, len(VOCAB))
    targets = labels[:, 1:]  # @inspect targets
    prediction_positions = torch.where(targets[0] != -100)[0].tolist()  # @inspect prediction_positions
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(VOCAB)), targets.reshape(-1))
    print("参与损失的预测位置：", prediction_positions)
    print("对应目标词元：", [VOCAB[i] for i in targets[targets != -100].tolist()])
    print("词表有 7 项，均匀预测的损失：", round(float(loss), 4))
    assert prediction_positions == [3, 4]
    assert torch.allclose(loss, torch.log(torch.tensor(float(len(VOCAB)))))
    text("两个目标分别是 red 与 EOS；对应预测来自 BOS 与 red 所在位置。答案第一个词元也必须计入损失。")
    text("### 交互观察：哪些输入位置作为学习目标？")
    interactive("images/lecture_04/vlm-lab/index.html", height=660, fragment="loss")


def runnable_mini_vlm():
    text("## 09 · 课堂训练：一个真的会按图回答的微型网络")
    image("images/lecture_04/mini-vlm.svg", width=760, style=FIGURE_STYLE)
    text("**数据**：8×16 RGB 图像，左右两块各选 red 或 blue，形成 4 幅图；每幅各问 left? 与 right?，共 8 条图像—问题—答案。")
    text("**模型**：卷积切块嵌入 → 线性连接器 → 与问题／答案前文拼接 → 两层因果 Transformer → 7 词元的输出头。")
    text("**训练**：随机初始化，全部小模型参数一起学习，执行 180 次 Adam 更新；不下载模型，不需要 GPU。它演示条件生成，不复现 LLaVA 的预训练或两阶段训练。")
    text("**解释边界**：8 条组合全部参与训练；后面的换问题与换图片属于训练范围内的受控检查，不是独立测试集或自然语言泛化。")
    text("### 模型实际使用的前向计算")
    text("```python\n" + inspect.getsource(MiniVLM.forward) + "```")
    text("PyTorch 的 `TransformerEncoder` 这里仅作为 Transformer 块的容器；传入因果掩码后，它执行解码器式的可见范围，名称本身不决定信息流。")
    text("### 实际的答案损失与参数更新")
    text("```python\n" + inspect.getsource(answer_loss) + "```")
    text("```python\noptimizer.zero_grad()\nloss = answer_loss(model(pixels, tokens), tokens)\nloss.backward()\noptimizer.step()\n```")
    model, report = train_demo()  # @stepover
    history = report["history"]  # @inspect history
    print("模型参数量：", report["parameters"])
    print("实际训练日志：", history)
    print("训练范围内完整答案匹配：", report["training_exact_match"])
    save_training_figure(report)  # @stepover
    image("images/lecture_04/minivlm-loss.svg", width=760, style=FIGURE_STYLE)
    text("损失曲线来自本次 Python 编译实际运行，浏览器播放时回放结果；网页没有重新启动训练。")
    text("### 同一张图，两个问题")
    pixels, tokens, names = make_data()  # @stepover
    case = pixels[2:3]
    answer_left = generate(model, case, IDS["left?"])  # @stepover
    answer_right = generate(model, case, IDS["right?"])  # @stepover
    print("图像：左 red，右 blue；left? →", answer_left)  # @inspect answer_left
    print("图像：左 red，右 blue；right? →", answer_right)  # @inspect answer_right
    swapped_answer = generate(model, case.flip(-1), IDS["left?"])  # @stepover
    print("把图像左右翻转，再问 left? →", swapped_answer)  # @inspect swapped_answer
    assert answer_left == ["red", "<eos>"]
    assert answer_right == ["blue", "<eos>"]
    assert swapped_answer == ["blue", "<eos>"]
    text("### 逐词元播放：把模型刚刚输出的词接回输入")
    prefix = torch.tensor([[IDS["left?"], IDS["<bos>"]]])  # @inspect prefix
    generated_words = []
    for step in range(3):
        with torch.no_grad():
            next_logits = model(case, prefix)[0, -1]  # @stepover
            probabilities = next_logits.softmax(-1).round(decimals=4)  # @inspect probabilities
        next_id = int(next_logits.argmax())  # @inspect next_id
        next_word = VOCAB[next_id]  # @inspect next_word
        generated_words.append(next_word)  # @inspect generated_words
        print("生成步", step + 1, "：", next_word)
        prefix = torch.cat([prefix, torch.tensor([[next_id]])], dim=1)  # @inspect prefix
        if next_id == IDS["<eos>"]:
            break
    text("**贪心解码（greedy decoding）**：每次选概率最高的词元；采样解码可按概率抽取，温度等设置影响输出的变化程度，不会自动增加事实正确性。")
    text("### 因果性检查：把未来答案换掉，前面的预测是否变化？")
    causal_delta = report["causal_future_token_delta"]  # @inspect causal_delta
    print("未来答案词元改变后，预测第一个答案位置的最大分数差：", causal_delta)
    text("结果为 0 表示本次检查没有从未来答案读取信息。损失下降、答案正确与无未来泄漏是三个不同的检查。")
    text("**可继续扩展的教学实验**：增加 green 或不同布局；重新划分训练／测试组合，比较见过与没见过的情况。当前微型模型的词表只包含本例七项，不处理任意中文问题。")


def details_and_video():
    text("## 10 · 从色块到真实图片：细节与成本")
    image("images/lecture_04/resolution-cost.svg", width=760, style=FIGURE_STYLE)
    text("**分辨率**影响小字、细线和远处物体能否被保留；把整张图片缩小后，语言模型不能凭空找回已经消失的像素细节。")
    text("**裁块（tiling／cropping）**：把高分辨率图像拆成多个局部输入；需要同时处理全局位置、块间对应与计算成本。")
    text("**视觉词元预算**：视觉位置越多，留给长问题和答案的上下文空间越少；查询压缩与块合并提供不同折中。")
    text("**多图**：比较两张图时，需要图像编号或其他位置线索，明确『第一张』『第二张』『相同位置』指向什么。")
    text("**视频**：通常先采样帧，再使用视觉表示；还需要时间顺序来处理动作与事件。抽帧过少可能遗漏关键瞬间，纯视觉视频模型也不自动具备音轨理解。")
    text("**多模态输入支持**与**实际任务成功**分别检查：能接收视频文件，不意味着能准确判断所有时间关系或听懂视频中的声音。")
    link(title="扩展阅读：LLaVA-OneVision（2024）· 单图、多图与视频任务", url="https://arxiv.org/abs/2408.03326")


def evaluation_and_language():
    text("## 11 · 看图回答的评价：流畅、真实与等值")
    image("images/lecture_04/grounding-evaluation.svg", width=760, style=FIGURE_STYLE)
    text("**视觉幻觉（visual hallucination）**：回答中的物体、属性或关系没有得到图像支持，例如把圆形说成汽车。语言表达可以很自然，事实却错误。")
    text("**物体幻觉**：图中没有车，却回答有车。**属性幻觉**：对象存在，颜色或数量却错误。**关系错误**：左右、包含、施事与受事被颠倒。")
    text("**POPE**是研究物体幻觉的一种代表性评测方法：围绕图中是否存在指定物体提出问题，观察回答；它不覆盖全部细粒度关系和长篇描述质量。")
    text("### 用不同评价维度拆开观察")
    text("| 任务 | 可计算的检查 | 需要额外核对的内容 |\n|---|---|---|\n| 短答案 VQA | 标准化后的答案匹配；有些基准支持多个人类答案 | 错误预设、数字格式、同义表达 |\n| 图像描述 | 与参考描述的文本指标；图文一致性分数 | 属性、关系、漏述与无依据增补 |\n| OCR／文档 | 文字、数字、字段的准确率 | 版面、表格、单位与证据位置 |\n| 双语图像说明 | 两语言分别检查视觉事实 | 指称、信息等值、语气与文化假设 |")
    text("**Exact Match（完全匹配）**：适合本课固定颜色答案；真实语言答案的同义改写可能被判为不匹配，因此不能直接把玩具评测当成完整 VQA 标准。")
    text("**CLIP 类相似度分数**能提供图文相关性线索，但单个高分不能证明每个数字、左右关系或因果解释都正确。第四课的生成任务与作业一的检索排名是不同任务。")
    text("### 语言学练习：对象没变，关系变了")
    text("**最小对照（minimal contrast）**：只改变一个关键因素，例如『红方块在蓝圆形左边』与『红方块在蓝圆形右边』。")
    text("『猫吃鱼』与『鱼吃猫』词语相同，施事和受事不同；描述图片时也必须核对『谁对谁做了什么』。不能只检查猫和鱼两个词有没有出现。")
    text("**指称检查**：『它在右边』中的它是否与前文同一对象？")
    text("**多语言检查**：中英回答是否都保留同一颜色、数量、左右关系与证据范围？")
    text("**语用检查**：要求只读展签时，应直接报告可见文字；无法确认的历史背景应与可见证据区分。")
    text("### 两人练习 · 给一条双语回答做标注（3 分钟）")
    text("使用本课场景。候选回答：『左边的蓝色圆形是一件 1280 年的藏品。The blue circle on the left was made in 1280.』")
    text("请分别标出：对象是否存在、左右是否正确、展签能否证明该句、两个语言版本是否表达同一事实。")
    text("参考：蓝圆形存在但在右侧；展签的年份未证明几何图形的制造年份；中文『藏品』与英文『was made』也不完全等值。")
    text("**实际应用**：可辅助双语展签、视觉词汇学习、资料图表阅读与无障碍描述；评价应依据具体使用场景，而不是只看模型回答长度。")
    link(title="POPE（EMNLP 2023）：Evaluating Object Hallucination in Large Vision-Language Models", url="https://arxiv.org/abs/2305.10355")


def recap_and_reading():
    text("## 12 · 本讲带走什么？")
    text("**表示**：视觉编码器把像素变成特征；视觉词元可以是连续向量，不必对应自然语言单词。")
    text("**连接**：投影拼接、查询压缩和交叉注意力，为视觉信息进入语言计算提供不同通路。")
    text("**生成**：图像＋问题＋答案前文 → 下一词元概率 → 逐步组成答案。")
    text("**学习**：训练数据给出所需行为，答案区损失给出更新信号；冻结参数与切断梯度是不同操作。")
    text("**评价**：视觉事实、关系与证据要与语言流畅性分别核对；多语言任务还需检查信息等值。")
    text("### 课后阅读路径")
    text("**先看通俗材料**：李宏毅『GPT-4 来了！』中关于模型接收图像的概念讨论，以及其生成式 AI 课程中的语言模型运作讲解。2023 年课件的图像接入示意含明确猜测标记，不能当成 GPT-4 已公开的真实架构。")
    link(title="李宏毅：GPT-4 来了！（2023）· 概念讲解与历史背景", url="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/GPT4%20%28v3%29.pdf")
    link(title="李宏毅：生成式人工智慧与机器学习导论（2025 秋）· 第 3 讲 解剖大型语言模型", url="https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall.php")
    text("**再读可执行讲义**：CS336 Lecture 17，关注『视觉编码器怎样连接语言模型』；本课例子和图示独立编写。")
    link(title="Stanford CS336 · Lecture 17", url="https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py")
    text("**最后读论文**：以 LLaVA 的结构图与两阶段训练为主线，再比较 BLIP-2 的查询压缩和 Flamingo 的交叉注意力。论文年份表示技术发展位置，不表示当前性能排名。")
    text("### 与后续课程连接")
    text("**作业一继续做 CLIP**：四语言图文对齐与检索；本课帮助理解这些视觉表示还能怎样用于生成。")
    text("**下一讲走向文本到图像**：本讲是图像 → 文字；下一讲介绍扩散模型怎样把文字条件用于生成图像。")


def save_training_figure(report):
    """Save a chart of observed losses and a reproducible, explicitly scoped report."""
    out = Path(__file__).resolve().parent / "images/lecture_04"
    history = report["history"]
    top = history[0]["loss"] * 1.1
    points = " ".join(f'{95 + r["step"] / report["updates"] * 820:.2f},'
                      f'{335 - r["loss"] / top * 225:.2f}' for r in history)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="455" '
           'viewBox="0 0 1000 455"><title>微型 VLM 的实际训练损失</title>'
           '<rect width="1000" height="455" rx="18" fill="#f5f9fc"/>'
           '<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif" fill="#17334a">'
           '<text x="30" y="43" font-size="28">实际运行结果：答案区交叉熵下降</text>'
           '<text x="30" y="84" font-size="20">全部 8 条训练组合 · 180 次更新 · CPU · 无预训练权重</text>'
           '<path d="M95 105V335H930" stroke="#8295a8" fill="none"/>'
           '<text x="24" y="123" font-size="18">Loss</text>'
           '<text x="65" y="339" font-size="18">0</text>'
           '<text x="91" y="365" font-size="18">0</text>'
           '<text x="875" y="365" font-size="18">180 步</text>'
           f'<polyline points="{points}" fill="none" stroke="#307f82" stroke-width="4"/>')
    for r in history:
        x, y = 95 + r["step"] / report["updates"] * 820, 335 - r["loss"] / top * 225
        svg += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#e57947"/>'
    svg += (f'<text x="200" y="142" font-size="21">起始 {history[0]["loss"]:.4f} → '
            f'末步 {history[-1]["loss"]:.6f}</text>'
            f'<text x="30" y="408" font-size="22">完整答案匹配：{report["training_exact_match"]}'
            '（训练范围内，非独立测试）</text></g></svg>')
    out.mkdir(parents=True, exist_ok=True)
    (out / "minivlm-loss.svg").write_text(svg, encoding="utf-8")
    result_path = Path(__file__).resolve().parent / "demos/lecture_04_minivlm_result.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
