# 第四讲备课说明：从图文对齐到看图说话

对应课纲第四周 Vision-Language Models（Captioning & VQA），面向外语＋计算机双学位本科生，按 45 分钟 × 2 课时设计。主文件 `lectures/lecture_04.py`，不改变作业一的 CLIP 检索任务。图示、代码与练习为本课程原创；论文和公开课程用于核对机制及组织教学。

## 课堂节奏

| 课时 | 时间 | 内容与操作 |
|---|---|---|
| 第一课时 | 0–5 分钟 | 第 01 节：复习 CLIP，对照匹配与生成 |
| 第一课时 | 5–10 分钟 | 第 02 节：描述、VQA、OCR、定位；图像中的指称与预设 |
| 第一课时 | 10–18 分钟 | 第 03 节：下一词元、教师强制、因果掩码；矩阵交互可择要演示 |
| 第一课时 | 18–31 分钟 | 第 04 节：视觉表示；横图／长文档预处理对照，RGB、标准化、换轴与视觉前缀 |
| 第一课时 | 31–35 分钟 | 第 05 节：三种连接方式；查询压缩代码只看输入／输出形状 |
| 第一课时 | 35–45 分钟 | 第 06 节：LLaVA-1.5 7B 完整架构、占位符替换、模块维度及 meta 形状检查 |
| 第二课时 | 0–11 分钟 | 第 07 节：训练数据、两阶段与冻结梯度演示 |
| 第二课时 | 11–19 分钟 | 第 08 节：答案区监督、-100、移位对齐与损失计算 |
| 第二课时 | 19–32 分钟 | 第 09 节：完整微型 VLM 的训练日志、曲线与逐词解码 |
| 第二课时 | 32–36 分钟 | 第 10 节：高分辨率、多图、视频与成本 |
| 第二课时 | 36–43 分钟 | 第 11 节：事实与语言评价，含 3 分钟双人标注练习 |
| 第二课时 | 43–45 分钟 | 第 12 节：回顾、阅读路径与扩散模型衔接 |

首次授课可以只读查询压缩代码中的输入／输出形状，不展开多头注意力内部细节。预处理重点讲横图／文档及数值转换；灰度、透明度与 EXIF 可作为补充。LLaVA 的具体维度用于示范读配置，不要求学生背诵；视觉和语言块内部细图及 meta 函数全源码可安排课后阅读。备课后可根据课堂速度压缩扩展阅读及视频段落。

## 本地编译与播放

常规播放素材已包含在仓库中；修改 Python 后需重新编译，再重新构建静态网站。

```bash
cd "/Users/taoji/Github/课程"
# 如果使用仓库虚拟环境，先执行 source .venv/bin/activate
cd lectures
python3 execute.py -m lecture_04
cd trace-viewer
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

打开 `http://127.0.0.1:4173/?trace=var/traces/lecture_04.json&step=1&animate=1`。同一网站改变 trace 文件即可切回 lecture_01、lecture_02、lecture_03 或 assignment_1。若 4173 已运行预览服务，不需要再启动一份；构建后刷新页面即可。

箭头逐步播放；Shift＋箭头跳过函数；Shift＋R 查看源码，Shift＋A 切换逐步展示。`@inspect` 的变量快照可在面板查看。本讲的重要变量包括 `blocked`、`visible`、`patch_features`、`visual_prefix`、`combined`、`weights`、`projector_grad`、`labels`、`prediction_positions`、`history`、`probabilities`、`prefix`。预处理新增 `gray_pixel`、`transparent_pixel`、`pixel_sample`、`normalized_sample`、`model_input_shape`；维度示例新增 `merged_shape`、`logits_shape`、`generation_shape`、`mask_tail`。大图片只展示样本像素与形状，避免变量面板展开全部像素。

浏览器回放编译时的结果。交互实验台使用本地 HTML 解释结构与掩码，不在浏览器中执行 PyTorch 或真实 VLM 推理。本讲的条件生成与损失公式直接绘制在 SVG 内，不依赖在线公式库；示意图、交互脚本和微型训练本身无外部资源依赖。框架其他课件的 LaTeX 公式仍沿用 MathJax CDN。

## 图片与交互素材

- `scripts/build_lecture_04_diagrams.py` 可重建 16 张概念 SVG，新增预处理数值流程、LLaVA 完整通路、视觉块内部维度、连接器／语言块内部维度和图像占位符替换。
- `demos/lecture_04_preprocess.py` 生成横图和长文档两个真实像素处理对照 SVG，以及 `images/lecture_04/preprocess/` 下的原始与处理结果 PNG。尺寸与数值结果来自实际 Pillow／NumPy 运算，不调用 OCR 或 VLM。
- `images/lecture_04/minivlm-loss.svg` 是实际微型模型的训练损失图，编译时更新；课件共 19 张图：16 张概念图、2 张处理案例图、1 张训练曲线。
- `images/lecture_04/vlm-lab/index.html` 提供视觉与语言连接、因果掩码和答案区监督三个交互视图，可单独打开。无 CDN 或第三方脚本。
- 教学图像是人造几何场景，答案由教师指定；不把示意答案写成某个模型的实测效果。图片最大 760 px，同时受窗口宽度限制，避免窄侧栏裁掉右侧；架构总览提供独立打开的大图入口。

## 六类可执行演示

1. **因果掩码**：真实生成一个 6×6 可见矩阵，检查 BOS 无法读取后面的答案。
2. **视觉前缀**：真实 RGB 数组 → 卷积切块 → 特征序列 → 线性投影 → 与文字嵌入拼接。随机权重只证明维度与信息通路，不证明语义。
3. **查询压缩**：真实 `nn.MultiheadAttention` 运算，4 个视觉位置压缩为 2 条查询输出；不是完整 Q-Former。
4. **冻结梯度**：阶段一冻结视觉和语言参数，连接器梯度仍非零；阶段二让语言分支也收到梯度。不可用 `torch.no_grad()` 包裹需要向前部连接器传梯度的语言模型。
5. **答案损失**：7 项词表的均匀 logits，答案红色及 EOS 两项参与交叉熵；检查移位位置与 `log(7)`。
6. **完整微型 VLM**：21,431 个参数的两层因果 Transformer，对 8 条色块问答做 180 次 Adam 更新，实际生成颜色＋EOS，并检查未来答案变化不会影响前面预测。

## 新增预处理与模块维度演示

预处理位于第 04 节，完整模块维度位于第 06 节。它们也可独立执行，不需要模型权重或 GPU：

```bash
python3 lectures/demos/lecture_04_preprocess.py
```

脚本验证 BCHW 格式、标准化与反变换一致性、中心裁剪丢失边缘红块、补边保留红块、灰度转 RGB、透明像素合成、EXIF 方向修正，以及完整形状通路；保存报告 `demos/lecture_04_preprocess_result.json`。

本地已与 Hugging Face 的 CLIP 图像处理器进行逐像素数值核对，两个 pad 案例最大绝对误差均为 0，版本与结果记录在 `demos/lecture_04_configs/processor_verification.json`。此核对是可选操作，常规编译不依赖 Transformers；如已安装该库，可复核：

```bash
python3 lectures/demos/lecture_04_preprocess.py --verify-hf
```

官方 LLaVA-1.5 7B、CLIP ViT-L/14-336 模型配置及图像处理器配置保存在 `demos/lecture_04_configs/`。`sources.json` 记录下载 URL、固定模型仓库 revision 与核对日期 2026-10-12；这些是公开配置 JSON，不是权重。

形状演示使用 `device='meta'`，实际执行 Conv2d、MLP、Embedding、拼接和输出头的形状运算；未运行的视觉／语言 Transformer 用 `empty_like` 表示输出接口。输入 21 项是人为构造的有效文字编号＋图像占位符，不是真实中文句子的分词结果。20 个普通词元加一个图像占位符，替换后为 `21−1+576=596`；两条文字长度不同的样本还演示 `[2,596]` 的批量补齐 mask。

完整微型网络在 `demos/lecture_04_minivlm.py`。可在仓库根目录独立运行：

```bash
python3 lectures/demos/lecture_04_minivlm.py
```

它不下载权重或数据，不需要 GPU。编译课件时保存的报告是 `demos/lecture_04_minivlm_result.json`。所有 8 条组合参与训练，报告中的完整匹配仅为训练集拟合；换问题与左右翻转也仍在训练组合范围内。不要称其为独立测试、零样本能力或完整 LLaVA 复现。真实 LLaVA 复用预训练视觉及语言模型，而本例从随机初始化训练全部参数。

## 技术口径

- VLM 是广义名称，CLIP 也属于 VLM；本讲聚焦能够生成文字的视觉语言模型。
- 连续视觉向量不必先转成词表编号；视觉词元数不等于物体数。
- 原始 LLaVA 使用线性投影；LLaVA-1.5 常用两层 MLP。本讲 336／14、576、1024→4096→4096 对应公开的 LLaVA-1.5 7B 配置，不泛指所有模型。
- 官方视觉分支取指定中间层、patch 模式去掉 CLS；并非拿 CLIP 检索用的最终整图向量直接复制 576 次。
- 本配置 image_aspect_ratio=pad，先以处理器均值对应的整数 RGB 颜色补方图，再交给 CLIP 处理器；不是无条件把所有原图中心裁掉。均值补边在标准化后接近 0，整数取整使它未必恰好为 0。
- 视觉网络含 24 个块、16 个注意力头，隐藏状态序列的 -2 项对应第 23 个块后；加入 CLS 是 577 个位置，取 patch 特征后是 576。视觉宽度 1024、前馈中间维度 4096。
- Vicuna 7B 为 32 层、32 头，隐藏维度 4096、门控前馈中间维度 11008、词表 32000；图示为该固定配置，不泛指所有 LLaVA。
- 原始 LLaVA 的图像占位符内部编号为 -200，先被拆出再由视觉向量替换；不能直接对负编号调用普通 Embedding。图像补边、序列补齐 mask、因果 mask、答案 labels 各有不同用途。
- Q-Former 的可学习查询不是自然语言问题；32 条查询来自 BLIP-2 原论文实验。
- Flamingo 使用重采样器与新加入的门控交叉注意力模块；不是把视觉和文字简单拼接后修改全部原有 LM 参数。
- `TransformerEncoder` 在微型代码中是块的容器，因果 mask 决定实际可见范围；不用这个 API 名称把模型错误称为双向编码器。
- -100 控制损失，注意力掩码控制可见信息；预测移位必须监督答案的第一个词和结束词。
- 视频输入不等于音轨输入；能接收多模态材料也不等于每个细节都能正确理解。
- 微型固定颜色答案的 Exact Match 不是所有真实 VQA 基准的评分方法；CLIP 类分数不能独自认证每个事实和关系。

## 参考材料与使用方式

初稿资料核对日期：2026-10-11；预处理与完整维度补充核对日期：2026-10-12。引用机制与历史阶段，避免把经典论文说成当前最新／最强模型；图片独立绘制，不复制原课件图。

| 材料 | 本课使用方式 |
|---|---|
| [Stanford CS336 Lecture 17](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py) | 参考从图像编码到接入语言模型的教学组织与可执行讲义形式；模型事实另查原论文及代码 |
| [李宏毅《GPT-4 来了！》2023](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/GPT4%20%28v3%29.pdf) | 概念讲解与历史背景；其图像接入页标明猜测，不当作已公开 GPT-4 架构 |
| [李宏毅 GenAI-ML 2025 秋](https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall.php) | 第 3 讲语言模型内部运作，供学生补习文字生成 |
| [Show and Tell](https://arxiv.org/abs/1411.4555) | CNN＋LSTM 图像描述的历史背景 |
| [LLaVA](https://arxiv.org/abs/2304.08485)、[官方项目](https://llava-vl.github.io/) | 原始连接器、训练阶段和合成指令数据 |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744)、[7B 配置](https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json) | 具体结构与维度；[视觉代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py)、[MLP 代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py) 核对实现 |
| [LLaVA 图像工具](https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py)、[图像处理器配置](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json) | pad 分支、背景颜色、尺寸及标准化常数；本地验证与官方处理器数值一致 |
| [CLIP 模型配置](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/config.json)、[LLaVA 序列组装](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py) | 视觉层数、头数、维度、占位符替换与序列补齐 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 查询压缩与冻结视觉／语言模型 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 重采样、层间交叉注意力与冻结 LM |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 自回归语言建模的背景阅读 |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | 单图、多图、视频扩展阅读 |
| [POPE](https://arxiv.org/abs/2305.10355) | 物体幻觉与评测范围 |

## 后续完善方向

可补充经授权的自然图像和真实 VLM 的推理对照，用问题变体检查左右、数量、小字、错误预设与双语等值。真实模型输出需记录模型版本、输入图片与提示；本稿没有把人造例子冒充真实大模型评测，也没有新增 GPU 训练任务。
