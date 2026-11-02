"""第七讲：语音与语言对齐。离线编译无需下载模型，备课说明见 notes/lecture_07.md。"""
import inspect
import torch
from execute_util import audio, image, link, text
from demos.lecture_07_speech import (
    read_wav, features, collapse, ctc_example, optimize_ctc,
    attention_demo, edit_distance, shape_demo,
)

STYLE = {'maxWidth': '100%', 'height': 'auto'}


def listening():
    text('### 正常语速')
    audio('videos/lecture_07/speech.wav')
    text('### 慢速 0.75 倍，保持音高')
    audio('videos/lecture_07/speech-slow.wav')


def wave_demo():
    rate, samples = read_wav('videos/lecture_07/speech.wav')  # @stepover
    sample_rate = rate  # @inspect sample_rate
    sample_shape = tuple(samples.shape)  # @inspect sample_shape
    first_samples = samples[:12].tolist()  # @inspect first_samples
    duration = len(samples) / rate  # @inspect duration
    print('采样率：', sample_rate, 'Hz；shape：', sample_shape)
    print('前 12 个采样值：', first_samples)
    print('时长：', round(duration, 3), '秒')
    assert rate == 16000


def feature_demo():
    rate, samples = read_wav('videos/lecture_07/speech.wav')  # @stepover
    power, mel = features(samples)  # @stepover
    power_shape = tuple(power.shape)  # @inspect power_shape
    mel_shape = tuple(mel.shape)  # @inspect mel_shape
    first_frame = mel[:8, 0].tolist()  # @inspect first_frame
    print('STFT 功率：', power_shape, '梅尔特征：', mel_shape)
    print('第一帧前 8 个频带：', first_frame)
    assert power.shape[0] == 201 and mel.shape[0] == 80
    text('### 实际执行的特征提取函数')
    text('```python\n' + inspect.getsource(features) + '\n```')


def collapse_demo():
    text('```python\n' + inspect.getsource(collapse) + '\n```')
    path = [1, 1, 0, 1, 1]  # 0=blank，1=哈
    collapsed = collapse(path)  # @inspect collapsed
    print('哈 哈 blank 哈 哈：', collapsed, '即“哈哈”')
    assert collapsed == (1, 1)
    text('若先删除 blank 再合并，会错误地把两个“哈”缩成一个。')


def probability_demo():
    probs, probability, valid, totals = ctc_example()  # @stepover
    target_probability = probability  # @inspect target_probability
    path_count = len(valid)  # @inspect path_count
    greedy = collapse(probs.argmax(-1).tolist())  # @inspect greedy
    loss = torch.nn.functional.ctc_loss(
        probs.log()[:, None, :], torch.tensor([1]),
        torch.tensor([3]), torch.tensor([1]), reduction='sum')
    actual_loss = float(loss)  # @inspect actual_loss
    print('输入：[时间 3，batch 1，类别 2]；目标“好”')
    for path, p in valid:
        print('路径', ['空白' if v == 0 else '好' for v in path], '概率', round(p, 3))
    print('有效路径数：', path_count, '总概率：', probability)
    print('PyTorch CTC 损失：', actual_loss, '贪心结果：', greedy)
    assert abs(float(torch.exp(-loss)) - probability) < 1e-12


def training_demo():
    text('```python\n' + inspect.getsource(optimize_ctc) + '\n```')
    probs, path, decoded, history = optimize_ctc()  # @stepover
    frame_probabilities = probs.tolist()  # @inspect frame_probabilities
    greedy_path = path  # @inspect greedy_path
    decoded_labels = decoded  # @inspect decoded_labels
    loss_history = history  # @inspect loss_history
    for row in history:
        print('优化步数', row['step'], '损失', round(row['loss'], 5))
    print('最终路径编号：', path, '折叠编号：', decoded)
    print('折叠文字：', ''.join({1: '看', 2: '一'}[v] for v in decoded))
    assert decoded == (1, 2, 1) and history[-1]['loss'] < history[0]['loss']


def dimensions_demo():
    shapes = shape_demo()  # @stepover
    mel_input, conv1_output, conv2_output, encoder_input = shapes  # @inspect encoder_input
    print('梅尔输入：', mel_input)
    print('卷积一：', conv1_output)
    print('卷积二：', conv2_output)
    print('编码器输入：', encoder_input)
    assert encoder_input == (1, 1500, 384)
    text('```python\n' + inspect.getsource(shape_demo) + '\n```')


def attention_code():
    weights, values, context = attention_demo()  # @stepover
    attention_weights = weights.tolist()  # @inspect attention_weights
    context_shape = tuple(context.shape)  # @inspect context_shape
    print('权重：', tuple(weights.shape), '值向量：', tuple(values.shape))
    print('各行权重之和：', weights.sum(-1).tolist())
    print('上下文 shape：', context_shape)
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    text('```python\n' + inspect.getsource(attention_demo) + '\n```')


def metric_demo():
    reference = ['我', '今天', '学习', '语音']
    hypothesis = ['我', '学习', '语音']
    errors, _ = edit_distance(reference, hypothesis)  # @stepover
    word_error_rate = errors / len(reference)  # @inspect word_error_rate
    print('参考：', reference, '预测：', hypothesis)
    print('删除“今天”：编辑距离', errors, 'WER', word_error_rate)
    assert word_error_rate == .25
    char_errors, _ = edit_distance(list('欢迎参观'), list('欢迎参光'))  # @stepover
    print('中文单字替换：CER =', char_errors / 4)
    text('```python\n' + inspect.getsource(edit_distance) + '\n```')



def section_01():
    text('## 语音和文字的对应关系')
    text('**语音识别（ASR，Automatic Speech Recognition）**：将声音中的语言内容转写为文字。')
    text('**语音翻译（speech translation）**：将源语言语音表达为目标语言文字。')
    text('**语音合成（TTS，Text-to-Speech）**：根据文字生成声音。同一句文字可以有不同说话人、语速和语调。')
    text('本讲重点是声音表示、识别与对齐，语音生成留到下一讲。')


def section_02():
    text('## 声音连续变化，文字按单位排列')
    text('声音在时间中展开，单个词的时长随说话人、语速和语境变化。')
    text('书写中的空格不保证对应声学上的静音，连续语音也不会自动标出词边界。')
    text('**对齐（alignment）**：建立声音片段与文字单位之间的对应关系，可以是软权重，也可以是显式时间边界。')


def section_03():
    text('## 音素、音位与书写单位')
    text('**音素（phone）** 是实际发出的声音，**音位（phoneme）** 是某种语言中具有区别意义作用的抽象单位。')
    text('普通话“八”（bā）与“趴”（pā）的声母主要区别是是否送气，拼音字母与实际语音需要区分。')
    text('字母、汉字、音节、音位和模型的子词 token，分别属于不同的表示层次。')
    text('同一个字母可以对应不同发音，同一种发音也可能有多种拼写。')


def section_04():
    text('## 连读与协同发音')
    text('**协同发音（coarticulation）**：一个音的实现受到相邻音的影响，声音边界常互相重叠。')
    text('“研究生命起源”通常切分为“研究／生命／起源”，声音中不会在每个词边界插入静音，分词需要结合上下文。')
    text('普通话“你好”中两个三声相连时，前一个通常发生变调；连续语音不是单字读音的简单拼接。')
    text('多语言识别需要同时处理发音系统、书写系统和语言使用习惯的差异。')


def section_05():
    text('## 试听：同一句语音的两种语速')
    text('素材为普通话合成语音：“你好，欢迎来到语音与语言对齐课堂。请听同一句话在不同语速下的变化。”')
    text('正常版与慢速版均为 16 kHz、单声道、16 位 PCM。慢速版使用保持音高的时间伸缩。')
    text('慢速使总时长增长，但预期文字没有增加，因此输入帧数与输出 token 数不能固定一一对应。')
    image('images/lecture_07/duration.png', width=760, style=STYLE)
    listening()


def section_06():
    text('## 波形：每个采样点记录振幅')
    text('**PCM（脉冲编码调制）**：把各采样时刻的振幅量化为数字。')
    text('16 kHz 表示每秒 16,000 个采样点，16 位表示每个采样值使用 16 个二进制位。')
    text('WAV 是容器，读取本例的 16 位 PCM 后得到一维浮点数组，归一化到约 [-1,1]。')
    image('images/lecture_07/waveform.png', width=760, style=STYLE)
    wave_demo()


def section_07():
    text('## 分帧：在局部时间段分析声音')
    text('**帧（frame）**：用于分析的一小段声音。采用 25 ms 窗口，每次前进 10 ms。')
    text('16 kHz 下，25 ms 对应 400 个采样点，10 ms 对应 160 个采样点。相邻窗口重叠 240 点。')
    text('**窗函数（window function）**：逐渐减弱片段两端的影响。这里使用 Hann 窗。')
    text('时间窗口中的“帧”与视频帧概念相关，但音频分析窗口允许重叠。')
    image('images/lecture_07/window.png', width=760, style=STYLE)


def section_08():
    text('## 频谱：一小段声音包含哪些频率')
    text('**傅里叶变换（Fourier transform）**：用不同频率的振荡成分描述信号。')
    text('**频率（frequency）** 以 Hz 为单位。频谱中的峰表示该窗口在相应频率附近的成分较强。')
    text('示例信号由 300 Hz 和 1,000 Hz 正弦波组成，改变它们的相对振幅会改变频谱。')
    text('语音比两个纯音复杂，元音的谐波、共振峰和擦音的宽带能量会形成不同结构。')
    image('images/lecture_07/fft.png', width=760, style=STYLE)


def section_09():
    text('## 短时傅里叶变换 STFT')
    text('**STFT（Short-Time Fourier Transform，短时傅里叶变换）**：对连续移动的窗口分别计算频谱。')
    text('把每帧功率谱并排放置，得到“频率 × 时间”的二维数组。')
    text('n_fft=400 时，实数音频的单边频率轴有 201 格，每格相隔 40 Hz。')
    text('这种二维图便于观察声音，但声音仍有时间、相位和听觉特征，不能直接等同于普通照片。')
    image('images/lecture_07/spectrogram.png', width=760, style=STYLE)


def section_10():
    text('## 梅尔频谱与对数压缩')
    text('**梅尔（Mel）尺度**：借助非线性频率刻度组织频带，低频较密，高频较疏。')
    text('**滤波器组（filter bank）**：把线性频率的功率加权汇总。本例将 201 个频率格转换为 80 个梅尔频带。')
    text('**对数压缩（log compression）**：减小强弱能量间的数值跨度。Whisper 还执行动态范围裁剪和缩放。')
    text('梅尔频谱是有损表示。本例仅用它作为识别特征，不从频谱反推原始声音。')
    image('images/lecture_07/mel.png', width=760, style=STYLE)


def section_11():
    text('## 代码：波形变成二维特征')
    text('采样点数 N 变为约 N/160 个时间帧，时间轴变短，每个位置包含多维频率信息。')
    text('Whisper 的实现使用居中补边，并移除最后一个 STFT 帧。不同库的补边和帧数规则可能不同。')
    text('以下代码实际读取音频并运行特征提取，滤波器使用仓库中保存的官方 80 频带矩阵。')
    feature_demo()


def section_12():
    text('## 对齐困难：同一句话有许多条路径')
    text('设声音编码后有 T 个时间位置，文字有 U 个单位，一般 T 大于 U。')
    text('“你好”可以先在多帧内持续预测“你”，再持续预测“好”，也可以在中间保留若干无输出帧。')
    text('训练只有整句文字时，通常没有逐帧标签。需要把多条可能路径共同纳入目标。')
    image('images/lecture_07/alignment.png', width=760, style=STYLE)


def section_13():
    text('## CTC：不需要逐帧标注的训练目标')
    text('**CTC（Connectionist Temporal Classification，连接时序分类）** 允许模型在每个时间位置预测文字标签或 blank。')
    text('**blank（空白标签）** 表示这个位置不输出文字，它不是空格，也不专指物理静音。')
    text('每条逐帧路径通过折叠规则变为文字序列，多条路径可以折叠成同一个答案。')
    text('常用编码器可以是 RNN、Transformer 等，CTC 本身是一种训练目标。')


def section_14():
    text('## 折叠路径：顺序很重要')
    text('第一步：合并相邻的相同标签。')
    text('第二步：删除 blank。')
    text('路径“你 你 blank 好 好”折叠为“你好”。')
    text('路径“哈 blank 哈”折叠为“哈哈”，blank 可以分隔两个相同的文字单位。')
    image('images/lecture_07/collapse.png', width=760, style=STYLE)
    collapse_demo()


def section_15():
    text('## CTC 概率：把正确路径相加')
    text('对同一条路径，各帧标签概率相乘。对同一个目标文字，将所有有效路径的概率相加。')
    text('P(y|x) = Σ[B(π)=y] ∏ₜ p(πₜ|x)')
    text('训练最小化负对数概率 −log P(y|x)。这通常通过动态规划高效计算，不需要枚举所有路径。')
    text('CTC 假设给定编码器表示后逐帧输出分布分解。编码器本身可以利用上下文。')
    image('images/lecture_07/ctc-prob.png', width=760, style=STYLE)


def section_16():
    text('## 代码：枚举结果与 PyTorch 相互核对')
    text('3 个时间位置，每帧空白标签概率 0.6，“好”的概率 0.4。')
    text('目标“好”对应 6 条路径，概率和为 0.688。目标空串概率 0.216，目标“好好”概率 0.096。')
    text('代码枚举全部 2³ 条路径，并将目标“好”的结果与 torch.nn.functional.ctc_loss 核对。')
    probability_demo()


def section_17():
    text('## 贪心路径不一定得到最可能文字')
    text('**贪心解码（greedy decoding）**：每帧选择概率最大的标签，再折叠。')
    text('本例贪心路径为三个空白标签，得到空串。但把路径概率相加后，“好”才是最可能的文字。')
    text('**束搜索（beam search）** 保留多个候选前缀，近似寻找概率更高的文字序列。')
    text('解码还可结合语言模型，但较强的语言先验也可能偏向常见表达而改错专名。')
    image('images/lecture_07/greedy.png', width=760, style=STYLE)


def section_18():
    text('## 代码：CTC 损失怎样改变逐帧预测')
    text('固定 8 个时间位置、3 个类别（空白、看、一），目标序列为“看一看”。')
    text('直接优化这 8×3 个 logits（未归一化分数），观察损失下降和折叠结果。')
    text('该示例没有音频输入，也没有声学编码器，只演示 CTC 梯度如何分配路径概率。')
    image('images/lecture_07/ctc-training.png', width=760, style=STYLE)
    training_demo()


def section_19():
    text('## CTC 的能力边界')
    text('CTC 偏向单调对齐：声音和目标文字基本按相同顺序前进。转写通常满足这个条件。')
    text('输入时间位置必须足够多。相邻重复目标还需要能分隔它们的额外位置。')
    text('跨语言翻译可能改变语序，不能简单用相同的逐帧单调输出方式处理所有翻译。')
    text('对齐路径的高概率位置也不自动等于精确的语音学边界。')


def section_20():
    text('## 强制对齐：文字已知，寻找对应时间')
    text('**强制对齐（forced alignment）**：已知音频和文字，寻找它们的时间对应。')
    text('语音识别需要同时预测内容，对齐则以给定文字为约束，两者输入条件不同。')
    text('字幕同步、语料标注和跟读反馈可以使用对齐结果。错误文字、漏词和重叠说话都会干扰对齐。')
    text('本讲手工构造的时间位置图用于解释机制，不作为实际语音边界标注。')


def section_21():
    text('## Whisper：编码器与解码器')
    text('Whisper 用音频编码器处理声音，用自回归文本解码器预测文字和任务相关 token。')
    text('原始论文使用约 68 万小时多语言、多任务弱监督数据。模型效果需要结合语言和测试条件评价。')
    text('这里固定采用多语言 tiny 的维度。与 large-v3 等版本的频带数、隐藏维度不能混用。')
    image('images/lecture_07/whisper-chinese.png', width=760, style=STYLE)


def section_22():
    text('## Whisper tiny 的音频通路')
    text('30 秒、16 kHz 单声道：每条音频 480,000 个采样点。')
    text('梅尔输入 [B,80,3000]，第一层 Conv1d 后 [B,384,3000]。')
    text('第二层卷积 stride=2 后 [B,384,1500]，转置为 [B,1500,384]。')
    text('4 层音频 Transformer，6 个注意力头，每头 64 维。编码器在本段音频内使用双向上下文。')


def section_23():
    text('## 代码：卷积压缩时间轴')
    text('Conv1d（一维卷积）沿时间轴提取局部模式。80 个梅尔频带在这里作为输入通道。')
    text('[B,通道,时间] 经卷积后转换为 [B,时间,特征]，供 Transformer 使用。')
    text('随机初始化的前端只能验证形状。要识别语音，需要训练好的卷积、编码器和解码器参数。')
    dimensions_demo()


def section_24():
    text('## Whisper tiny 的文本通路')
    text('已有 token 的嵌入为 [B,U,384]，解码器包含 4 层、6 个注意力头。')
    text('**因果自注意力（causal self-attention）** 只读取当前及此前位置。训练时输入正确文字的右移版本。')
    text('交叉注意力读取音频编码结果。输出投影为 [B,U,51865]，得到词表上的分数。')
    text('推理时重复预测下一 token，直到结束或达到长度限制。tiny 的文本位置上限为 448。')


def section_25():
    text('## 交叉注意力：文字位置读取声音')
    text('**Query（查询）** 来自当前文本位置，**Key（键）** 和 **Value（值）** 来自音频编码序列。')
    text('单头分数形状 [U,1500]，softmax 在声音时间轴上归一化。')
    text('每个文字位置用不同权重汇总声音表示，因此不要求文字数等于声音帧数。')
    text('注意力是模型内部的软读取权重，不能单凭一张热图宣称获得可靠的词级时间戳。')
    image('images/lecture_07/attention.png', width=760, style=STYLE)


def section_26():
    text('## 代码：用权重汇总声音表示')
    text('6 个声音位置，每个位置为 4 维向量。3 个文字位置分别读取不同的声音区域。')
    text('权重 [3,6] 乘声音值向量 [6,4]，得到 [3,4] 的上下文表示。')
    text('分数为手工设置，softmax 与矩阵乘法真实执行。它解释接口，不表示训练出的语音对齐。')
    attention_code()


def section_27():
    text('## 任务 token：同一声音，不同输出')
    text('多语言转写保留源语言文字。语音翻译则要求输出目标语言内容。')
    text('Whisper 的 translate 任务主要输出英语，不是任意目标语言翻译接口。')
    text('语言、任务和时间戳相关特殊 token 帮助规定解码任务。模型中的 token 边界不等于音位边界。')
    text('同一段中文录音，转写任务保留中文内容；Whisper 的翻译任务生成英语译文，两者的输出要求不同。')


def section_28():
    text('## 识别、翻译与合成的模块关系')
    text('转写：音频编码器处理声音，文本解码器输出源语言文字。')
    text('翻译：音频编码器处理声音，文本解码器输出目标语言文字，语序可能发生变化。')
    text('合成：文字条件控制声学表示和波形生成，涉及时长、音色与韵律。')
    text('Whisper 是语音识别与语音翻译模型，这一架构本身不承担波形生成。')


def section_29():
    text('## 词错误率 WER 与字错误率 CER')
    text('**WER（Word Error Rate，词错误率）** = (替换 S + 删除 D + 插入 I) / 参考词数 N。')
    text('**CER（Character Error Rate，字错误率）** 用字符作为计算单位，常用于中文等场景。')
    text('标点、大小写、数字格式和分词规则应在比较前固定。中英混合数据也需要明确单位。')
    text('错误率可能超过 100%，例如参考很短但预测插入很多内容。')
    metric_demo()


def section_30():
    text('## 错误类型与语言信息')
    text('“向左转”识别为“向右转”只替换一个方向词，却可能改变行动方向。')
    text('专名、否定词、数值和单位需要单独检查，整体 WER 不足以描述任务风险。')
    text('口音是语言变体。比较不同口音时，应控制录音条件、语料内容与样本规模。')
    text('转写会丢失部分语调、说话方式和情绪线索，需要根据任务保留音频证据。')


def section_31():
    text('## 时间戳、流式处理与等待时间')
    text('**时间戳（timestamp）** 标记文字或片段在音频中的位置，精度需要用标注数据验证。')
    text('**流式（streaming）** 边接收边处理音频，输出可能在后续声音到来后被修订。')
    text('固定 30 秒上下文的离线模型不自动具备低延迟流式接口。切块长度、缓存和上下文会影响效果。')
    text('语音交互中的全双工还需要打断、轮次管理和回声处理，不能只看识别速度。')


def section_32():
    text('## 离散语音单元与后续生成')
    text('频谱和编码器隐藏状态通常是连续数值，文字 token 是离散编号。')
    text('**音频编解码器（audio codec）** 可以把声音压缩为离散单元，再用解码器重建波形。')
    text('这些单元可能同时包含语音内容、音色与环境声，并非天然等价于音位或文字。')
    text('下一讲可以在这些表示上继续讨论文本到语音、韵律控制与音乐生成。')


def section_33():
    text('## 语音数据的隐私与可靠性')
    text('录音可能包含身份线索、住址和旁人的声音。采集、上传与保留需要明确范围。')
    text('识别器可能在噪声或静音片段输出不受声音支持的文字，不能把流畅结果直接当作原话。')
    text('带有时间戳的文字仍需回听核查，尤其是医疗、法律、考试与重要业务记录。')
    text('课堂示例使用合成语音，不需要收集学生声纹。')


def section_34():
    text('## 语音与语言对齐的核心关系')
    text('波形经过分帧和频谱变换，得到按时间组织的声学特征。')
    text('CTC 通过合并多条单调路径学习文字序列，Whisper 通过编码器与自回归解码器完成任务。')
    text('识别文字、确定时间边界、翻译语义和生成声音是相关但不同的任务。')
    text('评价同时关注错误率、关键语言信息、时间精度与具体使用条件。')


def section_35():
    text('## 阅读材料')
    text('Graves 等：Connectionist Temporal Classification，ICML 2006。')
    text('Radford 等：Robust Speech Recognition via Large-Scale Weak Supervision，ICML 2023。')
    text('OpenAI Whisper：音频处理、模型实现、tiny 配置与 MIT 许可。')
    text('本地演示：PCM 读取、STFT、梅尔频谱、CTC 折叠与求和、CTC 优化、交叉注意力、编辑距离。')


def main():
    text("# 第七讲 · 语音与语言对齐")
    text("声音表示、CTC 与 Whisper")
    section_01()
    section_02()
    section_03()
    section_04()
    section_05()
    section_06()
    section_07()
    section_08()
    section_09()
    section_10()
    section_11()
    section_12()
    section_13()
    section_14()
    section_15()
    section_16()
    section_17()
    section_18()
    section_19()
    section_20()
    section_21()
    section_22()
    section_23()
    section_24()
    section_25()
    section_26()
    section_27()
    section_28()
    section_29()
    section_30()
    section_31()
    section_32()
    section_33()
    section_34()
    section_35()
    link(title="CTC 原论文（2006）", url="https://www.cs.toronto.edu/~graves/icml_2006.pdf")
    link(title="Whisper 原论文（ICML 2023）", url="https://proceedings.mlr.press/v202/radford23a.html")
    link(title="Whisper 官方实现", url="https://github.com/openai/whisper")
