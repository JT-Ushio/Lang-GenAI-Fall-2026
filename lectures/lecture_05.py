"""第五讲：从噪声到图像。45 分钟 × 2；备课说明见 notes/lecture_05.md。"""
import inspect
import numpy as np
import torch
from execute_util import image, interactive, link, text
from demos.lecture_05_diffusion import (
    NoisePredictor, add_noise, ddpm_step, load_sd_configs, make_schedule,
    make_templates, sample, sd_shape_walkthrough, train_demo, training_step,
)
from demos.lecture_05_assets import make_rgb_scene, prepare_assets, prepare_forward

FIGURE_STYLE = {"maxWidth": "min(100%, calc(100vw - 125px))", "height": "auto"}
LAB_STYLE = {"maxWidth": "min(100%, calc(100vw - 125px))"}


def main():
    text("# 第五讲 · 从噪声到图像")
    text("## 扩散模型与 Stable Diffusion：语言怎样指导画面生成")
    text("**课程**：《以语言为中心的生成式AI》｜**对象**：外语＋计算机双学位本科生｜**时长**：45 分钟 × 2 课时。")
    text("第三讲学习图文对应，第四讲学习看图回答；这一讲让文字成为条件，指导模型从随机噪声生成图像。")
    text("**目标一**：用代码区分加噪、学习预测噪声和从新噪声生成。")
    text("**目标二**：沿着『文字 → 条件特征；噪声 → 潜变量 → 像素』读懂 Stable Diffusion。")
    text("**目标三**：看懂模块形状，并用语言最小对照检查生成是否忠实表达要求。")
    text("**第一课时 · 45 分钟**：任务与讲图 7′ → 生成路线 5′ → 加噪代码 13′ → 去噪目标 8′ → 微型模型训练 12′。")
    text("**第二课时 · 45 分钟**：采样 7′ → 潜空间 6′ → SD 架构与维度 14′ → 文字条件与引导 10′ → 评价与回顾 8′。")
    task_and_video()
    generative_routes()
    clean, epsilon, alpha_bar = forward_noise()
    denoising_objective()
    model, report, data = runnable_training(clean, epsilon, alpha_bar)
    sampling_process(model, data)
    latent_diffusion()
    stable_diffusion_architecture()
    runnable_sd_dimensions()
    text_condition_and_attention()
    classifier_free_guidance()
    language_evaluation()
    recap_and_reading()


def task_and_video():
    text("## 01 · 从『这幅图说了什么』到『按这句话画图』")
    image("images/lecture_05/course-route.svg", width=760, style=FIGURE_STYLE)
    text("**文生图（text-to-image generation）**：输入文字条件，输出图像；一条描述通常允许多张合理的画面。")
    text("例：『桌上有一个红杯子』限定对象、颜色和位置，却没有完整规定杯子的形状、桌面材质、光照与相机角度。模型需要在这些可能性中生成一次具体实现。")
    text("**条件分布（conditional distribution）**：这里可理解为『在满足某些文字信息的前提下，哪些图像更可能出现』。同一句话并不唯一对应某张图。")
    text("CLIP 学到图文对应后，还需要生成网络、训练目标和采样过程，才能把随机变量转为图像。")
    text("### 三分钟讲图：匹配表示与图像生成之间还差什么")
    image("images/lecture_05/video-guide/02_clip-encoders.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · 中文版 **04:16**。先分别指出文字和图像编码器：二者输出可比较的向量，但图里还没有『从噪声生成图像』的网络与采样链。")
    text("**维度口径**：画面中的 512 是该视频所举 CLIP 检索表示；本课稍后 SD 1.5 使用的文本条件是 `[B,77,768]`，两者不是同一个接口。")
    text("已将参考视频整理为 **14 张关键帧＋4 段无声短动画**，按知识点讲解。完整视频留作课后材料，课堂无需连续观看。")
    link(title="离线讲解图册：逐图提示、GIF、可暂停 MP4", url="images/lecture_05/video-guide/index.html#02_clip-encoders")
    link(title="参考视频（课后）：B站中文版 · 技术Flow · 33:51", url="https://www.bilibili.com/video/BV1Z9ec69Ezz")
    link(title="英文原视频：But how do AI images and videos actually work?", url="https://www.youtube.com/watch?v=iv-5mZ_9CPY")
    link(title="作者动画与代码：WelchLabs/videos · _2025/sora", url="https://github.com/WelchLabs/videos/tree/master/_2025/sora")
    text("关键帧保留原画面与中文字幕；原作 Welch Labs／3Blue1Brown，中文版本发布者 技术Flow。截图为 852×480，来源和中文时间戳附在图册中。")


def generative_routes():
    text("## 02 · 生成模型的几条路线")
    image("images/lecture_05/generative-routes.svg", width=760, style=FIGURE_STYLE)
    text("**VAE（Variational Autoencoder，变分自编码器）**：用编码器把图像映射到潜变量分布，再用解码器重建；本课借它学习较紧凑的图像表示。")
    text("**GAN（Generative Adversarial Network，生成对抗网络）**：生成器与判别器共同训练，判别信号帮助生成器改进。")
    text("**扩散模型（diffusion model）**：构造不同噪声水平的数据，学习反向生成所需的预测，再从随机噪声逐步采样。它没有把随机噪声当成某张藏起来的原图。")
    text("**2015**：Sohl-Dickstein 等提出基于前向破坏结构、学习反向过程的扩散生成方法。")
    text("**2020**：DDPM 成为图像扩散的重要代表；DDIM 给出了与其训练目标兼容的另一类采样过程。")
    text("**2022**：潜空间扩散把主要计算转入压缩表示；Stable Diffusion 是这一方向的经典开放模型。")
    text("本课选择 **Stable Diffusion 1.5** 作为固定维度案例。SDXL、SD3 和其他新模型可能采用不同文本编码器、通道数或去噪骨干，不能直接照搬本课尺寸。")
    link(title="早期扩散生成：Sohl-Dickstein et al.（2015）", url="https://arxiv.org/abs/1503.03585")
    link(title="VAE：Auto-Encoding Variational Bayes", url="https://arxiv.org/abs/1312.6114")
    link(title="GAN：Generative Adversarial Networks", url="https://arxiv.org/abs/1406.2661")


def forward_noise():
    text("## 03 · 可执行加噪：一张图怎样逐渐失去结构")
    text("**高斯噪声（Gaussian noise）**：本例为每个通道、每个像素独立采样均值为 0、方差为 1 的随机数。噪声可以有负数，不是直接取 0–255 的随机颜色。")
    text("**噪声调度（noise schedule）**：规定每个时间索引加入多少噪声。β 是单步噪声方差，α=1−β，ᾱ 是累计保留系数的平方。")
    image("images/lecture_05/forward-formula.svg", width=760, style=FIGURE_STYLE)
    image("images/lecture_05/video-guide/04_forward-noise.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · **08:40**。沿左到右读图：结构逐渐被噪声遮盖。下面用实际 RGB 数组复现加噪，并查看每个通道的数值。")
    link(title="可选短动画：前向加噪（8 秒，默认静态，可手动播放）", url="images/lecture_05/video-guide/index.html#a02_forward-noising")
    text("### 代码一：真实 RGB 数组 → 指定噪声水平")
    clean = torch.from_numpy(make_rgb_scene())
    rng = torch.Generator().manual_seed(5)
    epsilon = torch.randn(clean.shape, generator=rng)
    beta, alpha, alpha_bar = make_schedule(1000, beta_end=0.02)
    image_shape = tuple(clean.shape)  # @inspect image_shape
    first_pixel = clean[:, 0, 0].tolist()  # @inspect first_pixel
    print("原图：CHW，浮点数 [-1,1]；shape =", image_shape)
    print("左上角三个通道：", first_pixel)
    for t in (49, 199, 499, 999):
        a = alpha_bar[t]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * epsilon
        signal_weight = round(float(a.sqrt()), 4)  # @inspect signal_weight
        noise_weight = round(float((1 - a).sqrt()), 4)  # @inspect noise_weight
        noisy_pixel = noisy[:, 0, 0].tolist()  # @inspect noisy_pixel
        print(f"t={t:3}：原图系数 {signal_weight:.4f}，噪声系数 {noise_weight:.4f}")
    prepare_forward(clean.numpy(), epsilon.numpy(), alpha_bar.numpy())  # @stepover
    image("images/lecture_05/forward-panels.svg", width=760, style=FIGURE_STYLE)
    text("图中为了控制变量，各时间点复用同一份噪声 ε。它比较不同噪声水平，不是按单步转移逐次抽噪的马尔可夫轨迹；两种构造在每个给定时间点的边缘分布一致。")
    text("**显示转换**：浮点计算可能超出 [-1,1]；绘图时才裁剪并映射到 0–255。模型输入仍使用未裁剪的浮点状态。")
    interactive("images/lecture_05/diffusion-lab/index.html", height=820, fragment="noise", style=LAB_STYLE)
    text("### 代码二：已知真实噪声时，验证公式")
    t = 199
    a = alpha_bar[t]
    noisy = a.sqrt() * clean + (1 - a).sqrt() * epsilon
    recovered = (noisy - (1 - a).sqrt() * epsilon) / a.sqrt()
    recovery_error = float((recovered - clean).abs().max())  # @inspect recovery_error
    print("已知真实噪声的还原最大误差：", recovery_error)
    assert recovery_error < 1e-5
    text("此处已经知道加入的 ε，所以能代数还原。实际生成时没有这份答案，需要让网络根据当前状态预测噪声。")
    text("**第一课时检查点**：区分 β 的单步规则、ᾱ 的累计作用，以及为了屏幕显示做的裁剪。")
    return clean, epsilon, alpha_bar


def denoising_objective():
    text("## 04 · 学习去噪：监督信号从哪里来")
    image("images/lecture_05/noise-objective.svg", width=760, style=FIGURE_STYLE)
    image("images/lecture_05/video-guide/06_noise-target.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · **11:02**。黄色 ε 是程序加入的噪声，紫色是网络预测；二者的差形成训练信号。可以直接跳到一个噪声水平构造训练样本，不必先跑完整条采样链。")
    text("**去噪网络（denoising network）**：读取噪声状态、时间索引和条件，输出与状态同形状的预测。本课采用 **ε 预测（噪声预测）**。")
    text("**均方误差（MSE，Mean Squared Error）**：对预测噪声与本次真实采样噪声的逐元素平方差求平均。误差通过反向传播更新网络。")
    text("**时间嵌入（timestep embedding）**：告诉网络当前处于怎样的噪声水平。同一张很模糊的图，下一步处理方式取决于噪声水平与采样规则。")
    text("**条件（conditioning）**：指定希望生成的内容。课堂小模型用『横条／竖条』类别；Stable Diffusion 使用文本编码器给出的连续特征。")
    text("训练通常随机抽一个时间索引，不需要为每张训练图先跑完整个反向采样链。不同时间点共用一套参数。")
    text("### 真正执行的训练更新函数")
    text("```python\n" + inspect.getsource(training_step) + "\n```")
    text("**无条件训练样本**：代码把部分条件替换为编号 2，让同一个网络也学习不指定横竖方向的预测；第二课时将用它介绍 CFG。")
    text("噪声标签可由程序构造，但高噪声下不能唯一识别原图。网络是在训练数据分布和条件下学习有用的预测，不能把它理解为每一步都精确删除真实噪声。")
    link(title="DDPM：Denoising Diffusion Probabilistic Models（2020）", url="https://arxiv.org/abs/2006.11239")


def runnable_training(clean, epsilon, alpha_bar):
    text("## 05 · 真实训练一个能运行的微型图像扩散模型")
    text("**数据**：6 张 8×8 灰度条纹，横竖各 3 个位置；背景 −1、前景 +1。所有模板都参与训练，不设置独立的干净图像测试集。")
    templates, labels = make_templates()  # @stepover
    template_shape = tuple(templates.shape)  # @inspect template_shape
    template_pixels = templates[0].reshape(8, 8).int()  # @inspect template_pixels
    label_values = labels.tolist()  # @inspect label_values
    print("模板 shape：", template_shape, "；条件编号：", label_values)
    print("第一张模板的实际像素：\n", template_pixels)
    text("**MLP（Multilayer Perceptron，多层感知机）**：把 64 个像素、32 维时间嵌入、16 维类别嵌入拼接，经过两个 128 维隐藏层，预测 64 个噪声值。这里暂不使用 U-Net。")
    text("```python\n" + inspect.getsource(NoisePredictor) + "\n```")
    model, report, data = train_demo()  # @stepover
    history = report["history"]  # @inspect history
    parameters = report["parameters"]  # @inspect parameters
    print("实际训练参数：", parameters, "；Adam 更新：", report["updates"])
    for checkpoint in history:
        print(f"更新 {checkpoint['updates']:4} 次，固定噪声评估 MSE = {checkpoint['noise_mse']:.4f}")
    shapes = sd_shape_walkthrough()  # @stepover
    prepare_assets(clean.numpy(), epsilon.numpy(), alpha_bar.numpy(), report, data, shapes)  # @stepover
    image("images/lecture_05/toy-training.svg", width=760, style=FIGURE_STYLE)
    image("images/lecture_05/toy-generated.svg", width=760, style=FIGURE_STYLE)
    accuracy = report["new_noise_orientation_accuracy"]  # @inspect accuracy
    print(f"64 张新噪声样本的方向匹配：{accuracy:.1%}")
    text("**怎样读这个结果**：评估用新的噪声和时间索引，但干净模板仍来自训练集合。生成图另从新噪声出发；方向指标只检查行／列均值的变化，不能代表自然图像质量或未见场景泛化。")
    text("**本例与真实文生图的距离**：6 张图、2 个类别、39,280 个参数，没有自然语言编码器；它用来验证完整训练与采样链条。")
    image("images/lecture_05/training-vs-sampling.svg", width=760, style=FIGURE_STYLE)
    text("### 第一课时带走")
    text("加噪提供可计算的训练目标；网络学习预测；训练后可以从新噪声生成。已知噪声的公式反算、模型训练和生成是三件事。")
    return model, report, data


def sampling_process(model, data):
    text("## 06 · 第二课时：从纯噪声到一张新图")
    image("images/lecture_05/sampling-loop.svg", width=760, style=FIGURE_STYLE)
    text("**采样（sampling）**：使用训练好的模型，从随机初始状态产生一次输出。每一步更新样本，通常不再更新模型参数。")
    text("**调度器／采样器（scheduler / sampler）**：组织时间索引、步长和状态更新。有些库统一称 scheduler，但不同算法的更新公式并不一样。")
    image("images/lecture_05/toy-sampling.svg", width=760, style=FIGURE_STYLE)
    text("### 代码三：训练后的网络真的从新噪声开始")
    generated, _ = sample(model, torch.tensor([0]), seed=42)  # @stepover
    generated_shape = tuple(generated.shape)  # @inspect generated_shape
    generated_pixels = generated[0].round().int()  # @inspect generated_pixels
    same_start = data["vertical"][0]["image"] == data["horizontal"][0]["image"]  # @inspect same_start
    print("生成 shape：", generated_shape, "；横竖条件是否共享初始噪声：", same_start)
    print("生成值四舍五入后：\n", generated_pixels)
    text("这条路径来自实际网络反复预测。图中的中间状态不要求每一步都肉眼更清楚；本例 DDPM 还会在最后一步之前加入受控随机噪声。")
    interactive("images/lecture_05/diffusion-lab/index.html", height=760, fragment="sampling", style=LAB_STYLE)
    text("### 可展开阅读：实际执行的单步 DDPM 更新")
    text("```python\n" + inspect.getsource(ddpm_step) + "\n```")
    text("`predicted_clean` 由预测噪声得到，并按本例数据范围裁剪；两个 coefficient 构成后验均值，variance 控制随机项。最后一步直接返回均值，不再加噪。")
    text("**DDIM（Denoising Diffusion Implicit Models，去噪扩散隐式模型）**：提供另一类采样路径。η=0 时，给定初始噪声与条件，路径可确定；η>0 可引入随机性。")
    image("images/lecture_05/video-guide/10_ddpm-ddim.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · **23:35**。观察两组路径的不同形态：去噪网络可以相同，采样规则可以改变。本帧说明采样机制，不是两种算法质量优劣的受控评测。")
    link(title="可选短动画：DDPM／DDIM 路径对照（8 秒）", url="images/lecture_05/video-guide/index.html#a03_ddpm-ddim")
    text("**训练时间网格 T 与推理次数 K**：可以不同，但减少次数必须配合对应的时间选择和更新公式。不能直接在本例循环中任意跳过若干步。")
    text("本例用 100 个训练时间索引和 100 次 DDPM 更新，β 从 0.0001 到 0.15；前面的 RGB 加噪图用 1,000 步、终点 0.02。这是两套教学设置，都不等于 SD 1.5 原始采样配置。")
    link(title="DDIM 原论文（2020）", url="https://arxiv.org/abs/2010.02502")
    link(title="实现参考：Diffusers 的 DDPMScheduler", url="https://huggingface.co/docs/diffusers/api/schedulers/ddpm")


def latent_diffusion():
    text("## 07 · 把扩散搬到潜空间")
    text("**潜空间（latent space）**：编码器学到的内部表示空间。它可以保留生成所需的结构，不要求每个通道都有可直接命名的意义。")
    text("**潜空间扩散（latent diffusion）**：先训练图像自编码器，再主要在压缩后的表示上训练和运行扩散网络，最后解码成像素。")
    image("images/lecture_05/vae-latent.svg", width=760, style=FIGURE_STYLE)
    text("**训练侧**：RGB → VAE 编码器 → 后验分布参数 → 采样潜变量 → 幅值缩放 → 加噪 → 学习噪声预测。")
    text("**纯文生图推理侧**：随机潜噪声 → 条件去噪与采样 → 反向幅值缩放 → VAE 解码器 → RGB；不必输入一张照片。")
    text("**四个潜通道**：不是 RGB 再加透明度。")
    text("**空间缩小 8 倍**：宽与高各缩小 8 倍，位置数变为 1/64。")
    text("**幅值乘 0.18215**：SD 1.x 的数值缩放，不改变 shape；解码前除回去。它和空间缩小 8 倍是不同操作。")
    scalar_ratio = (3 * 512 * 512) / (4 * 64 * 64)  # @inspect scalar_ratio
    print("原像素元素数 / 潜变量元素数 =", scalar_ratio)
    text("这个 48 比较的是标量元素数量，浮点精度和文件编码还会影响存储大小；不能称为 PNG/JPEG 的 48 倍文件压缩率。")
    link(title="Latent Diffusion：High-Resolution Image Synthesis with Latent Diffusion Models", url="https://arxiv.org/abs/2112.10752")
    link(title="原始 SD 1.x 推理配置：scale_factor 及网络设置", url="https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml")


def stable_diffusion_architecture():
    text("## 08 · Stable Diffusion 1.5 完整通路与模块维度")
    text("本节固定 **512×512 输出、B 张图、经典 SD 1.5 配置**。数值来自公开模型配置，快照与来源保存在本地 `demos/lecture_05_configs/`；不需要下载权重。")
    image("images/lecture_05/sd-overview.svg", width=760, style=FIGURE_STYLE)
    link(title="独立打开：Stable Diffusion 架构大图", url="images/lecture_05/sd-overview.svg")
    text("| 模块 | 输入 → 输出（先不合并 CFG） | 作用 |\n|---|---|---|\n| 分词器 | 提示词 → `[B,77]` | 词元编号与补齐 |\n| CLIP 文本编码器 | `[B,77]` → `[B,77,768]` | 每个位置的上下文化条件 |\n| 初始潜噪声 | 随机数 → `[B,4,64,64]` | 本次生成的初始状态 |\n| U-Net 去噪网络 | 潜状态＋时间＋文字 → `[B,4,64,64]` | 预测潜空间噪声 |\n| 采样器 | 当前潜状态＋噪声预测 → 同形状 | 更新状态并重复 |\n| VAE 解码器 | `[B,4,64,64]` → `[B,3,512,512]` | 潜变量转 RGB |")
    text("**77 个位置**：包含特殊标记与补齐，不等于 77 个单词。分词方式还会影响不同语言能容纳多少文字。")
    text("**本课沿用 CLIP 的部分**：文本 Transformer 的序列隐藏状态；不是把第三讲用于检索的整句向量复制 77 次。纯文生图不需要 CLIP 图像编码器参与这条推理链。")
    text("### U-Net：用多尺度特征处理空间结构")
    image("images/lecture_05/unet-dimensions.svg", width=760, style=FIGURE_STYLE)
    text("**U-Net（U 形网络）**：分辨率先下降、再上升；低分辨率层能汇聚更大范围的信息，高分辨率特征帮助定位细节。")
    text("**跳跃连接（skip connection）**：把下行的相同分辨率特征接到上行。拼接会临时增加通道，后续层再投影；示意图只画代表性层级。")
    text("**时间通路**：时间索引先转成 320 维嵌入，经时间 MLP 变成 1280 维特征，进入去噪网络的残差块。")
    text("SD 1.5 的 U-Net 把卷积、残差块与注意力组合起来。它的输入和输出都是潜变量空间的 4 通道状态，不直接输出 3 通道像素。")
    link(title="公开配置：SD 1.5 U-Net", url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/unet/config.json")
    link(title="公开配置：SD 1.5 CLIP 文本编码器", url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/text_encoder/config.json")
    link(title="公开配置：SD 1.5 VAE", url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/vae/config.json")


def runnable_sd_dimensions():
    text("### 代码四：读取配置，执行形状运算")
    config = load_sd_configs()  # @stepover
    channels = config["unet"]["block_out_channels"]  # @inspect channels
    text_width = config["text"]["hidden_size"]  # @inspect text_width
    print("U-Net 层级通道：", channels, "；文本隐藏宽度：", text_width)
    shapes = sd_shape_walkthrough(batch=1, cfg=True)  # @stepover
    token_shape = shapes["token_ids"]  # @inspect token_shape
    text_shape = shapes["text_embeddings"]  # @inspect text_shape
    latent_shape = shapes["initial_noise"]  # @inspect latent_shape
    cfg_input_shape = shapes["cfg_model_input"]  # @inspect cfg_input_shape
    guided_shape = shapes["guided_noise"]  # @inspect guided_shape
    for name, shape in shapes.items():
        print(f"{name:24} {shape}")
    text("**meta 张量**：只保存形状与类型，不分配大块像素／权重存储。示例实际执行 Embedding、卷积、池化、上采样和拼接的形状运算。")
    text("**演示边界**：Transformer、VAE 编解码使用输出接口占位；U-Net 只搭出缩减的层级通路，不是完整 SD 前向。能核对维度不代表已经验证预训练模型的图像生成质量。")
    interactive("images/lecture_05/diffusion-lab/index.html", height=850, fragment="architecture", style=LAB_STYLE)
    link(title="独立打开：扩散实验室（可切换加噪、采样与架构）", url="images/lecture_05/diffusion-lab/index.html#architecture")
    text("配置快照中的原始 scheduler 是 **PNDM**：训练时间网格为 1,000，β 使用 scaled_linear，端点 0.00085 与 0.012。本课微型 DDPM 是另一套设置；更换采样器需要检查与模型预测目标及调度的兼容性。")


def text_condition_and_attention():
    text("## 09 · 文字怎样影响正在去噪的空间位置")
    image("images/lecture_05/cross-attention.svg", width=760, style=FIGURE_STYLE)
    image("images/lecture_05/video-guide/11_text-conditioning.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · **26:28**。上方是文字条件，下方是重复调用的去噪模型；先读信息流，再对应回本课的交叉注意力。视频此帧用 Transformer 泛化示意，本课固定的 SD 1.5 则采用 U-Net，结构不能混用。")
    text("**交叉注意力（cross-attention）**：视觉特征发起查询 Q，文本特征提供键 K 与值 V。每个空间位置按匹配权重读取文字条件，再参与视觉特征更新。")
    text("语言学连接：『红色方块在蓝色圆的左边』同时给出对象、属性和关系。模型需要把 red 与 square 联系，把 blue 与 circle 联系，还要保留 left of 的方向。流畅的提示词并不保证这些绑定全部正确。")
    text("### 代码五：真实计算一张图像的词元对齐权重表")
    rng = torch.Generator().manual_seed(9)
    queries = torch.randn(1, 16, 8, generator=rng)
    keys = torch.randn(1, 5, 8, generator=rng)
    values = torch.randn(1, 5, 8, generator=rng)
    scores = queries @ keys.transpose(-1, -2) / (8 ** 0.5)
    attention = scores.softmax(-1)
    conditioned = attention @ values
    attention_shape = tuple(attention.shape)  # @inspect attention_shape
    first_query_weights = attention[0, 0].tolist()  # @inspect first_query_weights
    conditioned_shape = tuple(conditioned.shape)  # @inspect conditioned_shape
    print("16 个视觉位置 × 5 个文字位置：", attention_shape)
    print("首个视觉位置对 5 项的权重：", [round(x, 3) for x in first_query_weights])
    print("读取文字后的输出形状：", conditioned_shape)
    assert torch.allclose(attention.sum(-1), torch.ones(1, 16))
    text("此处特征是随机数，演示的是计算与形状，没有语言语义。真实 SD 还包括训练过的投影、多头、归一化和残差等结构。注意力图也不能单独作为『模型正确理解关系』的证据。")


def classifier_free_guidance():
    text("## 10 · CFG：怎样调节对文字条件的偏向")
    image("images/lecture_05/cfg.svg", width=760, style=FIGURE_STYLE)
    image("images/lecture_05/video-guide/13_cfg-vector.png", width=760, style=FIGURE_STYLE)
    text("视频关键帧 · **30:06**。灰色是无条件方向，黄色是条件方向，绿色表示差向量；再用尺度控制这种偏向。视频用向量场 f 表示，本课代码组合噪声预测 ε，阅读公式时先核对约定。")
    link(title="可选短动画：条件引导的更新路径（8 秒）", url="images/lecture_05/video-guide/index.html#a04_cfg-path")
    text("**CFG（Classifier-Free Guidance，无分类器引导）**：分别得到无条件／空提示与目标条件下的预测，再利用它们的差异调节采样；无需额外训练一个分类器。")
    text("它可以在一个批次中合并两条分支，也可以像课堂小模型那样顺序调用同一个网络两次。分支共享权重，输入条件不同。")
    text("### 代码六：引导尺度改变的是哪个方向")
    epsilon_unconditional = torch.tensor([0.2, -0.4])
    epsilon_conditional = torch.tensor([0.5, -0.2])
    for scale in (0., 1., 3.):
        guided = epsilon_unconditional + scale * (epsilon_conditional - epsilon_unconditional)  # @inspect guided
        print(f"s={scale:g}：引导后的预测 = {guided.tolist()}")
    assert torch.allclose(epsilon_unconditional + 1 * (epsilon_conditional - epsilon_unconditional), epsilon_conditional)
    text("本课采用常见 Diffusers 尺度约定：s=0 对应无条件，s=1 对应条件预测，s>1 沿两者差异方向外推。它并不是把图像清晰度乘以 s。")
    text("**空提示特征**：分词、特殊标记与文本编码器仍在工作，得到的特征不是全零。")
    text("**负面提示词（negative prompt）**：某些实现用它替换空提示这一基准条件；不能保证以逻辑规则彻底排除指定内容。")
    text("**尺度权衡**：更大可能增强条件偏向，也可能带来过饱和、伪影或多样性变化；需要实际比较，不能认定越大越好。")
    link(title="Classifier-Free Diffusion Guidance（2022）", url="https://arxiv.org/abs/2207.12598")


def language_evaluation():
    text("## 11 · 评价文生图：画得像，也要表达对")
    image("images/lecture_05/language-evaluation.svg", width=760, style=FIGURE_STYLE)
    text("**属性绑定**：红色属于方块还是圆形；改变一个形容词，检查对应对象是否改变。")
    text("**关系与角色**：『猫吃鱼』和『鱼吃猫』保留近似词汇，却交换施事与受事；对应画面应改变动作关系。")
    text("**多语言等值**：中英文提示保持对象、数量、位置与风格要求一致，再比较生成；先排除翻译本身改变要求的因素。")
    text("**随机种子（seed）**：控制随机数序列。同环境中固定模型、采样器、步数、引导尺度与种子，才能更清楚地比较一个提示因素；跨硬件或版本不保证逐像素一致。")
    text("**CLIP 相似度**：可以辅助衡量图文对应，但高分不能逐条认证数量、关系、文字拼写或文化表达正确。与人工细项标注配合。")
    text("### 三分钟课堂练习：先设计对照，再制定标准")
    text("提示 A：『一张展览海报：左边是红色方块，右边是蓝色圆形，不含文字。』")
    text("提示 B：只交换左右关系，其他要求不变。可再写出信息等值的英文版本。")
    text("标注表每行一个样本，分别记录：对象齐全、颜色绑定、左右关系、是否多出文字；另列整体观感。至少使用几组相同种子做成对比较。")
    text("上面是实验设计，尚未调用真实 SD 生成海报；不把标准答案或示意图当成模型的实测结果。")
    text("**发布前检查**：使用有授权的素材与模型，避免上传私人照片，标明合成内容；尤其留意冒名、虚假新闻或诈骗用途。生成细节自然，不代表内容真实。")


def recap_and_reading():
    text("## 12 · 回顾与阅读路径")
    text("**机制**：程序构造加噪样本与噪声目标 → 网络学习预测 → 采样器从新噪声逐步生成。")
    text("**结构**：CLIP 文本序列提供条件 → U-Net 在潜空间预测 → 调度器更新 → VAE 解码成像素。")
    text("**维度**：文字 `[B,77,768]`；潜状态 `[B,4,64,64]`；RGB `[B,3,512,512]`；CFG 的双分支可以把网络批量临时变成 2B。")
    text("**语言能力的作用**：写出对象、属性与关系，保持跨语言信息等值，设计可核查的最小对照，而非只堆积形容词。")
    text("**课后实践**：修改 8×8 模板位置或宽度，重新训练并查看采样；保留一个未参与训练的位置，才开始检验未见模板泛化。报告数据划分、随机种子、误差与生成图。")
    text("**基础阅读**：先用讲解图册回顾加噪、预测和引导，再复现代码；完整视频作为课后扩展，然后读 DDPM 的训练／采样算法框。")
    text("**结构阅读**：沿潜空间扩散论文核对『编码、去噪、解码』；用官方配置核对某个具体版本，避免混用不同模型的维度。")
    text("**进阶阅读**：比较 DDPM／DDIM 的采样规则与 CFG；本课不展开随机微分方程、流匹配及视频生成，它们可沿此基础继续学习。")
    link(title="扩散模型教程：Step-by-Step Diffusion（2024）", url="https://arxiv.org/abs/2406.08929")
    link(title="SD 1.5 模型卡：用途、训练与限制", url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5")
    link(title="原始 Latent Diffusion 实现", url="https://github.com/CompVis/latent-diffusion")


if __name__ == "__main__":
    main()
