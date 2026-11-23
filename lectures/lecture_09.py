"""第九讲：视频与语言建模。节奏、来源与运行说明见 notes/lecture_09.md。"""
import inspect
import torch
from torch.nn import functional as F
from execute_util import text, image, video, interactive, link
from demos.lecture_09_video import (
    load_clip, uniform_indices, fps_indices, reorder_frames, pixel_motion,
    temporal_pooling_example, tubelet_example, stc_shape_example,
    qwen_shape_example, answer_loss_example, interval_iou, aligned_time_example,
)
STYLE = {'maxWidth': 'min(100%, calc(100vw - 120px))', 'height': 'auto'}
FENCE = chr(96) * 3


def sampling_code():
    frames, timestamps, metadata = load_clip()  # @stepover
    frames_shape = tuple(frames.shape)  # @inspect frames_shape
    indices = uniform_indices(len(frames), 8)  # @stepover
    selected_indices = indices.tolist()  # @inspect selected_indices
    selected_times = timestamps[indices].tolist()  # @inspect selected_times
    sampled = frames[indices]
    sampled_shape = tuple(sampled.shape)  # @inspect sampled_shape
    print('解码后的 T,C,H,W：', frames_shape, '均匀抽八帧：', selected_indices)
    print('原视频时间（秒）：', selected_times, '抽帧后形状：', sampled_shape)
    assert selected_indices == [6, 18, 30, 42, 54, 66, 78, 90]
    text(FENCE + 'python\n' + inspect.getsource(uniform_indices) + FENCE)


def short_event_code():
    frames, timestamps, _ = load_clip()  # @stepover
    two = fps_indices(timestamps, 2, 12)  # @stepover
    four = fps_indices(timestamps, 4, 12)  # @stepover
    hits_at_two = two[(two >= 10) & (two < 12)].tolist()  # @inspect hits_at_two
    hits_at_four = four[(four >= 10) & (four < 12)].tolist()  # @inspect hits_at_four
    print('2 帧/秒捕获红灯帧：', hits_at_two, '4 帧/秒：', hits_at_four)
    assert hits_at_two == [] and hits_at_four == [10]
    text(FENCE + 'python\n' + inspect.getsource(fps_indices) + FENCE)


def order_code():
    frames, _, _ = load_clip()  # @stepover
    shuffled, order, restored = reorder_frames(frames)  # @stepover
    first_source_indices = order[:8].tolist()  # @inspect first_source_indices
    restoration_error = int((restored.to(torch.int16) - frames).abs().max())  # @inspect restoration_error
    print('打乱后前八个位置来自原帧：', first_source_indices, '恢复误差：', restoration_error)
    text(FENCE + 'python\n' + inspect.getsource(reorder_frames) + FENCE)


def pooling_code():
    frames, _, _ = load_clip()  # @stepover
    means, deltas = temporal_pooling_example(frames)  # @stepover
    mean_difference = float((means[0] - means[1]).abs().max())  # @inspect mean_difference
    endpoint_change = float(deltas[0].abs().mean())  # @inspect endpoint_change
    opposite_error = float((deltas[0] + deltas[1]).abs().max())  # @inspect opposite_error
    print('正放与倒放的均值差：', mean_difference, '端点变化幅度：', endpoint_change)
    print('正反端点差相加：', opposite_error)
    text(FENCE + 'python\n' + inspect.getsource(temporal_pooling_example) + FENCE)


def motion_code():
    frames, _, _ = load_clip()  # @stepover
    changes = pixel_motion(frames)  # @stepover
    change_count = len(changes)  # @inspect change_count
    largest_change = float(changes.max())  # @inspect largest_change
    print('96 帧产生相邻差分：', change_count, '最大归一化差分：', largest_change)
    text(FENCE + 'python\n' + inspect.getsource(pixel_motion) + FENCE)


def tubelet_code():
    frames, _, _ = load_clip()  # @stepover
    resized, patches = tubelet_example(frames)  # @stepover
    input_shape = tuple(resized.shape)  # @inspect input_shape
    tubelet_shape = tuple(patches.shape)  # @inspect tubelet_shape
    print('八帧小图：', input_shape, '时空小块：', tubelet_shape)
    assert tubelet_shape == (64, 384)
    text(FENCE + 'python\n' + inspect.getsource(tubelet_example) + FENCE)


def onevision_code():
    grid = torch.arange(27 * 27 * 16, dtype=torch.float32).reshape(1, 16, 27, 27)
    reduced = F.interpolate(grid, size=(14, 14), mode='bilinear', align_corners=False)
    sequence = reduced.flatten(2).transpose(1, 2)
    token_shape = tuple(sequence.shape)  # @inspect token_shape
    print('教学隐藏维度 D=16；729 个空间位置重采样为 196 个：', token_shape)
    assert token_shape == (1, 196, 16)


def stc_code():
    features, reduced, tokens = stc_shape_example()  # @stepover
    input_shape = tuple(features.shape)  # @inspect input_shape
    reduced_shape = tuple(reduced.shape)  # @inspect reduced_shape
    token_shape = tuple(tokens.shape)  # @inspect token_shape
    print('B,C,T,H,W：', input_shape, '三维卷积后：', reduced_shape, '序列：', token_shape)
    text(FENCE + 'python\n' + inspect.getsource(stc_shape_example) + FENCE)


def qwen_code():
    shapes = qwen_shape_example()  # @stepover
    module_shapes = shapes  # @inspect module_shapes
    print('Qwen2.5-VL-7B 形状演算：', module_shapes)
    text(FENCE + 'python\n' + inspect.getsource(qwen_shape_example) + FENCE)


def loss_code():
    labels, loss, norms = answer_loss_example()  # @stepover
    target_ids = labels.tolist()  # @inspect target_ids
    position_gradient = norms.tolist()  # @inspect position_gradient
    measured_loss = loss  # @inspect measured_loss
    print('答案监督标签：', target_ids, '交叉熵：', measured_loss)
    print('每个位置的 logits 梯度范数：', position_gradient)
    text(FENCE + 'python\n' + inspect.getsource(answer_loss_example) + FENCE)


def grounding_code():
    prediction = (3.0, 6.0)
    reference = (3.6, 5.7)
    temporal_iou = interval_iou(prediction, reference)  # @stepover
    measured_tiou = temporal_iou  # @inspect measured_tiou
    print('预测区间：', prediction, '水流标注：', reference, '时间交并比：', measured_tiou)
    assert abs(temporal_iou - .7) < 1e-6
    text(FENCE + 'python\n' + inspect.getsource(interval_iou) + FENCE)


def alignment_code():
    video_times, audio_times, nearest = aligned_time_example()  # @stepover
    video_index = 2  # @inspect video_index
    audio_index = nearest  # @inspect audio_index
    matching_seconds = [float(video_times[2]), float(audio_times[nearest])]  # @inspect matching_seconds
    print('视频位置 2 与音频位置', audio_index, '都对应约 0.4 秒：', matching_seconds)
    text(FENCE + 'python\n' + inspect.getsource(aligned_time_example) + FENCE)


def opening():
    text('## 01 · 视频里发生了什么？')
    video('videos/lecture_09/operation.mp4', width=760, style=STYLE)
    text('一个人先取杯，再倒水，随后端杯饮水，最后放回。这里的描述来自人工编写的示意动画，不是模型预测。')
    text('一张图能呈现杯子、水壶与人物；视频还要表达动作、状态变化和先后关系。')
    text('| 任务 | 中文输入 | 输出形式 |\n|---|---|---|\n| 视频描述 | 描述人物的操作 | 一段文字 |\n| 视频问答 | 倒水之前做了什么？ | 取杯，并引用证据 |\n| 文本检索视频 | 找到“倒水后喝水”的片段 | 候选片段排序 |\n| 时间定位 | 哪段时间能看到水流？ | 起止时间 |')


def language_and_time():
    text('### 动词、论元和时间关系')
    text('**动作与论元**：谁把什么倒进哪里？“人—倒水—杯子”把动词连接到参与者与目标。')
    text('**状态变化**：杯中水量增加，是区别“拿着水壶”和“正在倒水”的证据之一。')
    text('**时序连接词**：“先取杯，再倒水，随后饮水”与“先饮水，再倒水”包含相似名词，却描述不同事件。')
    text('**证据边界**：先发生不等于导致；画面能支持动作判断，通常不能直接证明人物的意图或心理状态。')


def sampling():
    text('## 02 · 从连续播放到有限帧序列')
    text('原视频为 12 秒、每秒 8 帧，共 96 帧。解码得到 RGB 像素；本例缓存的是实际 MP4 解码结果，课堂运行不需要安装视频解码器。')
    text('**T** 表示帧数，**C** 表示颜色通道，**H/W** 表示高度与宽度。文件缓存用 T×H×W×C，PyTorch 示例转为 T×C×H×W。')
    sampling_code()
    image('images/lecture_09/storyboard.png', width=760, style=STYLE)
    text('均匀抽帧：把时间轴分成八段，各取中心位置。它覆盖整段视频，但不保证覆盖每个短事件。')
    video('videos/lecture_09/sparse-eight.mp4', width=760, style=STYLE)
    text('上方是八张抽取帧的保持播放，不是模型补出的中间帧。')
    text('### 源帧率与采样帧率')
    text('源帧率决定视频原有的时间分辨率；采样帧率决定送入模型的观测密度。提高采样率不能恢复源视频没有记录的细节。')
    text('**PTS（呈现时间戳）** 给出帧在播放时间轴上的位置。恒定帧率可用帧号÷帧率换算；可变帧率视频应读取实际时间戳。')
    text('### 一个会被漏掉的事件')
    image('images/lecture_09/brief-event.png', width=760, style=STYLE)
    text('指示灯只在第 10、11 帧变红，即 [1.25, 1.50) 秒。以下从 0 秒开始按固定时间间隔采样。')
    short_event_code()
    image('images/lecture_09/sampling.png', width=760, style=STYLE)
    text('2 帧/秒恰好漏掉红灯；4 帧/秒捕获到一次。这是本例采样相位下的结果，并不说明 4 帧/秒能覆盖所有短事件。')


def temporal_order():
    text('## 03 · 看见相同物体，不等于理解同一事件')
    text('### 倒放：帧集合相同，方向改变')
    video('videos/lecture_09/reversed.mp4', width=760, style=STYLE)
    text('倒放仅用于检验时序敏感性，不代表真实、合理的操作。以下倒放和打乱版本均去掉音轨，避免声音泄露原顺序。')
    text('### 打乱：保留全部帧，破坏相邻关系')
    video('videos/lecture_09/shuffled.mp4', width=760, style=STYLE)
    order_code()
    text('**排列实验的控制条件**：帧和它原有的时间戳应一起排列；若把原时间戳直接暴露给模型，模型可以据此恢复顺序。测试“只看呈现顺序”时，应重新标记呈现位置，并明确这一设置。')
    text('### 平均池化为什么会丢失方向？')
    pooling_code()
    text('对同一组帧取平均，与排列无关。端点差能保留部分方向信息，但像素差本身并不理解“倒水”或“饮水”。实际模型需要学习时空关系。')
    interactive('images/lecture_09/video-lab/index.html', height=920,
                style={'maxWidth': STYLE['maxWidth']})
    text('### 相邻帧差：先测变化，再谈语义')
    motion_code()
    image('images/lecture_09/motion.png', width=760, style=STYLE)
    text('差分峰值只表示像素变化较大。镜头晃动、灯光变化和物体运动都能产生峰值，不能直接当作动作标签。')


def representation():
    text('## 04 · 视频如何变成 token？')
    text('**Token（模型处理的序列单元）** 可以对应文字片段，也可以对应图像小块或时空小块。')
    image('images/lecture_09/tubelets.png', width=760, style=STYLE)
    text('**Tubelet（时空小块）**：同时覆盖连续若干帧与一个空间区域。本例将 8 帧缩为 32×32，每块覆盖 2 帧、8×8 像素、3 通道。')
    tubelet_code()
    text('块数：(8÷2)×(32÷8)×(32÷8)=64。每块原始维度：3×2×8×8=384；线性投影之后才得到模型的隐藏向量。')
    text('### 内容、空间位置与时间位置')
    text('内容向量回答“这里有什么”；空间位置区分左边与右边；时间位置区分之前与之后。只把帧拼起来而不保留时间信息，会削弱事件顺序表达。')
    text('同一个第 4 个采样位置，在 2 帧/秒下可对应 2 秒，在 4 帧/秒下可对应 1 秒。位置编号与物理时间不能混用。')
    text('### 时间覆盖与 token 预算')
    text('8 帧×每帧 576 个位置=4,608 个视觉 token；32 帧×每帧 196 个位置=6,272 个。帧数更多，不一定意味着总 token 更多。')
    text('对完整序列做稠密自注意力时，注意力分数矩阵随长度近似平方增长；实际速度还取决于压缩、窗口注意力、缓存与实现。长视频必须同时考虑覆盖范围和局部细节。')


def architectures():
    text('## 05 · 三条视频理解架构路线')
    image('images/lecture_09/video-language.png', width=760, style=STYLE)
    text('视觉编码器把像素变成特征；连接器把视觉特征映射到语言模型的表示空间；语言模型结合问题生成答案。连接器的输出是向量，不必先转成一段文字。')
    image('images/lecture_09/architecture-routes.png', width=760, style=STYLE)
    text('### LLaVA-OneVision：逐帧编码，压缩后交给语言模型')
    image('images/lecture_09/onevision-official.png', width=760, style=STYLE)
    text('SigLIP 视觉编码器 → 两层 MLP（多层感知机）连接器 → Qwen2 语言模型。视频先复用图像编码能力，再将多帧视觉表示交给语言模型处理。')
    text('报告配置中，384×384 的帧产生 27×27=729 个空间位置；视频分支双线性重采样到 14×14=196 个。32 帧对应 6,272 个视觉位置。27→14 是重采样，不是将 729 简单整除四。')
    onevision_code()
    text('代码仅演示空间压缩，隐藏维度 16 为教学设置，不是模型真实通道数。压缩会降低空间成本，也可能损失小字与小物体细节。')
    link(title='LLaVA-OneVision 技术报告与原图', url='https://arxiv.org/abs/2408.03326')
    text('### VideoLLaMA 2：在连接器里融合空间与时间')
    image('images/lecture_09/videollama-official.png', width=760, style=STYLE)
    text('**STC（时空卷积连接器）** 在视觉编码器与语言模型之间，引入空间处理与三维卷积下采样，把相邻帧的局部信息融合。')
    text('三维卷积沿时间、高度、宽度滑动。本例输入轴为 B×C×T×H×W；时间和空间均缩半，8×24×24 个位置变成 4×12×12=576 个。')
    stc_code()
    text('这里只执行随机初始化的小型 Conv3d，不是完整 STC，也不是预训练模型推理。实际连接器还包含空间模块与 MLP。模型另设音频分支；音频不是从 RGB 帧中自动得到的。')
    link(title='VideoLLaMA 2 技术报告与原图', url='https://arxiv.org/abs/2406.07476')
    text('### Qwen2.5-VL：时空小块、动态分辨率与真实时间')
    image('images/lecture_09/qwen-official.jpg', width=760, style=STYLE)
    text('视觉分支将连续两帧组成时空小块，采用以窗口注意力为主、穿插全局注意力的视觉 Transformer。相邻 2×2 空间位置合并，再映射到语言模型维度。')
    image('images/lecture_09/qwen-dimensions.png', width=760, style=STYLE)
    text('以 **Qwen2.5-VL-7B** 为例：8 帧、224×224 输入，时间块长 2、空间块边长 14。初始时空网格为 4×16×16；视觉隐藏维度 1,280；空间四块拼接后为 5,120，再映射为 3,584。最终为 256 个视觉 token。')
    qwen_code()
    text('meta 张量只验证形状，不分配模型权重，也不执行视觉识别。这里固定为 224×224 方便计算，真实处理器支持动态分辨率。')
    text('**MRoPE（多模态旋转位置编码）** 区分时间、高度、宽度。Qwen2.5-VL 将时间位置与绝对时间对齐，使不同采样帧率下的时间表示更一致；这与只按帧序号计数不同。')
    text('本节所讲 Qwen2.5-VL 的视频路径输入是画面，不把它当作原生音频理解模型。')
    link(title='Qwen2.5-VL 技术报告与原图', url='https://arxiv.org/abs/2502.13923')


def training():
    text('## 06 · 视频问答模型学什么？')
    text('| 数据形式 | 输入条件 | 监督目标 |\n|---|---|---|\n| 片段描述 | 视频 | 人工描述的文字 |\n| 视频问答 | 视频＋问题 | 回答文字 |\n| 时间定位 | 视频＋事件描述 | 起止时间或时间标记 |\n| 视频文本检索 | 视频、匹配文字 | 正确配对的相似度高于负例 |')
    text('CLIP 式对比学习适合配对与检索；生成式视频问答通常在给定视觉条件下预测答案 token。两者可以共存，但损失函数和输出接口不同。')
    text('### 只在答案位置计算语言损失')
    text('教学编号序列包含视频占位、问题与三个答案符号。“-100”表示该标签被交叉熵忽略；真实模型的中文分词结果不必与这里一致。')
    loss_code()
    text('被忽略的是对应输出位置的直接损失，不代表视觉编码器或问题表示永远没有梯度：它们仍可能通过答案的计算路径收到梯度，是否更新还取决于冻结设置。')
    text('预训练常先对齐连接器，再做指令微调；具体冻结哪些模块要查每个模型的训练阶段，不能套用统一规则。')
    text('### 文字答对，还要定位证据')
    text('“倒水操作阶段”可粗标为 [3,6) 秒；“可见水流”在动画里是 [3.6,5.7) 秒。标注粒度取决于任务定义，不应把两个区间混为一谈。')
    grounding_code()
    image('images/lecture_09/grounding.png', width=760, style=STYLE)
    text('**tIoU（时间交并比）**=两个时间段的交集时长÷并集时长。这里 2.1÷3=0.70。实际评测应结合定位误差与答案内容，单一分数不能解释所有错误。')


def evaluation():
    text('## 07 · 从“看起来合理”到可核验')
    text('**问答正确性**：区分“取杯”与“放杯”，核查动作主体与目标。')
    text('**时序敏感性**：原版、倒放、打乱版本分别作答；同时控制音轨、字幕和时间戳，避免模型从旁路恢复答案。')
    text('**检索性能**：用“倒水后喝水”检索候选视频，报告 Recall@K（前 K 个结果命中率）；含相同物体、动作顺序不同的片段应成为困难负例。')
    text('**定位性能**：报告 tIoU 或在指定 tIoU 阈值下的命中率，并明确边界标注规则。')
    text('**多语言一致性**：中文、英文、德文问题应保持相同事实约束，分别检查理解与输出；翻译流畅不保证视觉证据正确。')
    text('**幻觉与捷径**：问题写着“喝水”，模型不能仅凭常识回答“先倒水”；应引用画面时间证据。画面外的动作应标为无法确认。')
    text('### 长视频：先找到证据，再看细节')
    text('按镜头或时间段建立粗索引 → 找到候选事件 → 在候选区间提高采样密度 → 生成带时间戳的答案。检索失败与问答失败需要分别记录。')
    text('抽到 32 帧不等于观察了每一秒；回答应受已观测证据约束。可以保留片段层次摘要，但不要用摘要替代原视频的核验。')
    text('### 音画协同也需要时间对齐')
    video('videos/lecture_09/operation.mp4', width=760, style=STYLE)
    text('原版：水流声、提示音与画面同步。')
    video('videos/lecture_09/audio-delayed.mp4', width=760, style=STYLE)
    text('对照版：画面相同，音轨延后 0.5 秒。声音可以帮助理解，也可能来自后期配音或画外来源，不能自动证明画面中的因果关系。')


def mova():
    text('## 08 · MOVA：从理解视频到生成同步音视频')
    text('MOVA 来自 SII–OpenMOSS／复旦相关团队。它在这里作为通向下一讲的案例：研究核心是根据图像和文字生成同步的视频与音频，不是把它列成视频问答模型。')
    text('| 方向 | 条件 | 输出 |\n|---|---|---|\n| 视频理解 | 已有视频＋问题 | 答案、描述或定位 |\n| MOVA 音视频生成 | 参考图像＋文字 | 新的视频＋同步声音 |')
    image('images/lecture_09/mova-official.png', width=760, style=STYLE)
    text('**视频塔**：基于 Wan2.2 I2V A14B 的扩散 Transformer（DiT），在视频潜变量中生成运动画面。')
    text('**音频塔**：约 13 亿参数的音频 DiT，在音频潜变量中生成声音。')
    text('**双向 Bridge（桥接模块）**：约 26 亿参数，通过跨模态注意力让两塔交换信息，而不是先生成无声视频、最后任意贴一段背景音。')
    text('报告给出的整体规模约 320 亿总参数、180 亿激活参数；这是作者模型规格，不是课堂运行的模型规模。视频使用 Wan2.1 VAE，音频使用 DAC 风格 VAE，两个模态不共享同一套像素／波形网格。')
    text('### 相同的时刻，不同的序列下标')
    alignment_code()
    text('5 Hz 视频网格与 50 Hz 音频网格仅用于教学：下标 2 和 20 可以表示同一 0.4 秒。不能按“相同下标”直接对齐两个模态。')
    text('MOVA 的 **Aligned RoPE（时间对齐旋转位置编码）** 把跨模态位置映射到共同时间轴，让不同潜变量采样密度下的交互仍参照物理时间。上方代码不复现其完整注意力实现或 VAE 帧率。')
    text('### 如何构建同步生成能力？')
    text('数据需要联合描述画面、动作、语音、音乐和音效，并核对时间关系。只有场景名词而没有声音事件描述，不足以教会模型声画同步。')
    text('作者先做音频塔预训练，再联合训练音频塔、视频塔与 Bridge，采用不同学习率并逐步提升数据质量与分辨率；联合阶段并非只训练 Bridge。训练使用流匹配目标。')
    text('### MOVA 完整主展示')
    video('videos/lecture_09/mova-official-full.mp4', width=760, style=STYLE)
    text('完整保留官方主展示约 38 秒的画面和声音。换到说话片段时，观察口型、停顿和声音起点。')
    text('### 单人语音 · 完整样例')
    video('videos/lecture_09/mova-single-speaker.mp4', width=760, style=STYLE)
    text('### 多人语音 · 完整样例')
    video('videos/lecture_09/mova-multiple-speakers.mp4', width=760, style=STYLE)
    text('对照单人和多人样例，再检查声音来自哪位说话者，以及切换说话者时画面和声音是否同步。')
    link(title='MOVA 官方仓库与演示来源', url='https://github.com/OpenMOSS/MOVA#demo')
    link(title='MOVA 技术报告', url='https://arxiv.org/abs/2602.08794')


def generation_showcase():
    text('## 09 · Sora 与 Seedance／即梦')
    text('### Sora · 从时空潜变量到音视频生成')
    text('**2024 年 2 月 15 日**，OpenAI 展示 Sora。技术报告描述了如何将视频压缩到潜空间，切分为时空小块，再由带文字条件的扩散 Transformer 生成。这与前面用于理解的时空小块使用了相近的表示思路，训练目标分别对应生成视频和预测答案。')
    text('**东京街景**中，观察镜头移动时行人外观、背景位置和遮挡关系能否连贯。')
    video('https://cdn.openai.com/sora/videos/tokyo-walk.mp4', width=760, style=STYLE)
    link(title='OpenAI 官方 Sora 演示与提示词', url='https://openai.com/index/sora/')
    link(title='Sora 技术报告', url='https://openai.com/index/video-generation-models-as-world-simulators/')
    text('**2025 年 9 月 30 日**，Sora 2 发布，加入同步对白和音效。官方体操样例适合逐段观察角色动作，以及声音与动作的配合。')
    video('https://openaiassets.blob.core.windows.net/$web/nf2/blog-final2/dalmation_gymnastics.mp4', width=760, style=STYLE)
    text('Sora 网页与 App 已于 **2026 年 4 月 26 日**停止服务。本课使用公开发布页保留的历史样例。')
    link(title='Sora 2 官方发布与演示', url='https://openai.com/index/sora-2/')
    text('### Seedance 2.0 → 2.5 · 从多模态参考到按时间编辑')
    text('**2026 年 2 月 12 日**，字节跳动 Seed 团队发布 Seedance 2.0。它以文字、图像、音频和视频为条件，联合生成画面与声音。')
    link(title='Seedance 2.0 官方发布', url='https://seed.bytedance.com/zh/blog/official-launch-of-seedance-2-0')
    text('**2026 年 7 月 31 日**发布的 Seedance 2.5 将单次生成延长到约 30 秒，并支持按时间段指定音画编辑。Seedance 是模型系列，即梦 AI 是接入该模型的创作平台。')
    text('**参考视频编辑样例**按时间指定不同场景。观察动作是否延续、场景何时切换、画面变化和声音是否对应。')
    video('https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/user-upload/4xfa4ms8awtl3.mp4',
          width=760, style=STYLE)
    link(title='Seedance 2.5 官方发布与完整样例', url='https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5')
    link(title='即梦 AI 创作入口', url='https://jimeng.jianying.com/')


def generation_industry():
    text('## 10 · 视频生成有多少人在用？收入怎么读？')
    text('| 产品与统计时间 | 已披露数字 | 统计口径 |\n|---|---:|---|\n'
         '| Sora Android · 2025 年 11 月上线首日 | 超 100 万条生成视频 | OpenAI 披露的首 24 小时生成量 |\n'
         '| 即梦 AI App · 2026 年 3 月 | 1,493 万台 | 艾瑞监测的中国移动端月独立设备数，覆盖 App 内各类创作功能 |\n'
         '| 可灵 AI · 截至 2025 年 12 月 | 超 6,000 万创作者、超 6 亿条生成视频、超 3 万企业用户 | 快手披露的全球创作者及业务累计规模 |\n'
         '| 可灵 AI · 2025 年第四季度 | 3.4 亿元人民币 | 季度实际产品收入 |\n'
         '| 可灵 AI · 2026 年第一季度 | 超 6.5 亿元人民币 | 季度实际产品收入 |\n'
         '| 可灵 AI · 2026 年第二季度 | 超 8.5 亿元人民币 | 季度实际产品收入 |\n'
         '| 可灵 AI · 2025 年 12 月 | 单月超 2,000 万美元，折合约 2.4 亿美元年化运行率 | 将该月收入按 12 个月外推 |')
    text('同一个人可能使用多台设备，累计创作者还包括后来不再使用的人。统计对象、地区和时间窗口相同，用户规模才适合横向比较。')
    text('订阅、生成额度和企业 API 都可为创作产品带来收入。上表的人民币数据对应已经实现的单季收入，美元年化运行率展示的是一个月收入维持一年时的规模。')
    text('课堂案例把用量和收入分别定位到具体产品。即梦的设备数包含图像等功能，可灵的产品收入也不能用来推算整个视频生成行业规模。')
    link(title='OpenAI 披露 Sora Android 上线首日生成量', url='https://openai.com/index/shipping-sora-for-android-with-codex/')
    link(title='艾瑞咨询 2026 年一季度中国移动互联网流量报告，第 22 页', url='https://pdf.dfcfw.com/pdf/H3_AP202605191822433573_1.pdf')
    link(title='快手披露可灵创作者规模、生成量与年化运行率', url='https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-annualized-revenue-run-rate-hits-usd240-million/')
    link(title='快手 2025 年第四季度财报', url='https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-technology-announces-fourth-quarter-and-full-year-2025/')
    link(title='快手 2026 年第一季度财报', url='https://ir.kuaishou.com/static-files/51eda7e7-a9e8-43c1-b688-43f131104d98')
    link(title='快手 2026 年第二季度财报', url='https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-technology-announces-second-quarter-and-interim-2026')


def closing():
    text('## 11 · 把视频任务写成可检查的请求')
    text('“仅依据视频，列出取杯、可见水流、端杯饮水的起止时间；按发生顺序输出中文表格。每项附画面证据；看不清的边界标为不确定，不推断人物意图。”')
    text('| 架构 | 主要机制 | 要保留的区别 |\n|---|---|---|\n| LLaVA-OneVision | 逐帧视觉编码＋空间压缩＋LLM | 多帧输入与 token 预算 |\n| VideoLLaMA 2 | 时空卷积连接器＋LLM | 连接器内融合相邻时空信息 |\n| Qwen2.5-VL | 时空小块＋合并＋绝对时间位置 | 时间下标与物理秒数 |\n| MOVA | 视频／音频 DiT＋双向桥接 | 生成同步音视频，与理解任务分开 |')
    text('视频与语言建模需要同时处理：**观测到了什么、顺序怎样、发生在何时、文字回答是否有证据**。同步生成还要进一步让画面与声音在同一时间轴上成立。')


def main():
    text('# 第九讲 · 视频与语言建模')
    opening()
    language_and_time()
    sampling()
    temporal_order()
    representation()
    architectures()
    training()
    evaluation()
    mova()
    generation_showcase()
    generation_industry()
    closing()
