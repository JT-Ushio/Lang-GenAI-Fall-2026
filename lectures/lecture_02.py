"""第二讲：多种模态的数字表示与跨模态协同任务。素材已保存，编译无需联网。"""
from pathlib import Path
import wave

import numpy as np
from PIL import Image
from execute_util import audio, image, link, text, video


def main():
    text("# 第二讲 · 多模态数据与跨模态协同")
    text("## 第一部分 · 模态的数字表示")
    text("本讲先认识各种模态的**数组形状、数值含义和存储形式**，再介绍它们与语言协同完成的任务。")
    text("页面中的 print 输出来自编译时实际运行；按 R 可以切换源码。修改 Python 后需要重新编译，浏览器不是在线 Python 解释器。")
    grayscale_images()
    rgb_images()
    sampled_audio()
    audiovisual_video()
    point_clouds()
    summary()
    cross_modal_tasks()


def grayscale_images():
    text("## 01 · 灰度图：一张手写数字就是一个矩阵")
    image("images/lecture_02/grayscale.png", width=950, style={"maxWidth": "calc(100vw - 120px)"})
    text("这是 scikit-learn digits 数据集中的真实手写数字 **3**。它是 8×8，不是 MNIST 的 28×28；该数据的灰度取值为 0–16。")
    text("**数组（array）**：是按一定顺序组织的数值集合。")
    text("**shape（形状）**：给出各个维度的长度。")
    text("**dtype（数据类型）**：说明每个数如何存储。")
    digit = np.load("images/lecture_02/digit.npy")
    print("shape =", digit.shape)
    print("dtype =", digit.dtype)
    print(digit)
    text("shape=(8, 8) 表示 8 行、8 列。每个位置是一格灰度：这里 0 是黑，16 是白，中间值是不同深浅。")
    print("第 0 行 =", digit[0])
    print("第 0 行、第 2 列 =", digit[0, 2])
    print("范围 =", digit.min(), "到", digit.max())
    text("Python 从 0 开始计数。访问图像数组通常用 [行, 列]，也就是 [y, x]。显示时放大这些格子，不会增加原始数据的信息。")
    text("### 手写数字识别：输入和答案不是同一种数据")
    x = digit.reshape(64)
    y = 3
    print("输入展开后的 shape =", x.shape)
    print("标签 y =", y)
    text("识别任务是：让模型根据 64 个灰度值预测 0–9 的类别。标签 3 是数据集给出的答案，不是本例已经训练出了一个识别器。")
    text("reshape 改变数组的组织方式，不会凭空增减像素；后续模型还需要学习哪些空间排列对应哪些数字。")
    print("像素数量 =", digit.size, "；像素数组占用 =", digit.nbytes, "字节")
    text("这里把 0–16 的整数保存为 uint8（8 位无符号整数），每个数占 1 字节；1 字节（byte）等于 8 位（bit）。常见 8 位灰度图使用 0–255，但灰度范围与存储类型要看具体数据。")
    link(title="数据来源与取值说明：scikit-learn load_digits", url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html")


def rgb_images():
    text("## 02 · RGB 图像：红、绿、蓝三个颜色通道")
    image("images/lecture_02/rgb-channels.png", width=1100, style={"maxWidth": "calc(100vw - 120px)"})
    text("RGB 是 Red、Green、Blue（红、绿、蓝）的缩写。")
    text("**通道（channel）**：是每个像素中某一种颜色分量。")
    text("这里是一张程序绘制的小房子，右侧三张灰度图分别显示 R、G、B 通道的强度，亮不亮表示该通道的数值大小。")
    rgb = np.array(Image.open("images/lecture_02/rgb.png").convert("RGB"))
    print("shape =", rgb.shape)
    print("dtype =", rgb.dtype)
    print("天空像素 [0, 0] 的 RGB =", rgb[0, 0])
    print("屋顶像素 [20, 42] 的 RGB =", rgb[20, 42])
    print("太阳像素 [14, 74] 的 RGB =", rgb[14, 74])
    print("左上角 2×3 个像素 =\n", rgb[:2, :3])
    text("shape=(64, 96, 3)：高度 64、宽度 96、通道数 3。尺寸常写成 **96×64（宽×高）**，数组则常按 **高×宽×通道** 排列，注意顺序！")
    print("只取红色通道，shape =", rgb[:, :, 0].shape)
    print("像素数 =", rgb.shape[0] * rgb.shape[1])
    print("数值个数 =", rgb.size, "；解码后的数组 =", rgb.nbytes, "字节")
    text("在这个 uint8 RGB 示例中，像素取值为 0–255：红色可写 [255,0,0]，黑色 [0,0,0]，白色 [255,255,255]。颜色空间和透明通道以后再展开。")
    text("### 像素尺寸、像素密度与清晰度")
    image("images/lecture_02/resolution.png", width=1050, style={"maxWidth": "calc(100vw - 120px)"})
    text("三张图用相近的显示面积呈现，但原始像素数不同。**像素尺寸**描述宽×高；日常所说的图像『分辨率』常指它，但屏幕或印刷也会讨论每英寸的像素密度 PPI（Pixels Per Inch，每英寸像素数）。")
    text("相同像素尺寸，可以显示成不同物理大小。更多像素提供更大的采样容量，但失焦、噪声、压缩和简单插值都可能让图像仍不清楚；放大不等于恢复真实细节。")
    print("1920×1080 RGB uint8 的未压缩数组 =", 1920 * 1080 * 3, "字节")
    print("本例 PNG 文件大小 =", Path("images/lecture_02/rgb.png").stat().st_size, "字节")
    text("**文件大小 ≠ 数组大小。** PNG（便携式网络图形，一种图像文件格式）含格式信息和无损压缩数据；加载解码后才得到这里的像素数组。JPEG 是另一类常见图像编码标准，通常采用有损压缩，也不能直接把文件字节当像素。")


def read_pcm(path):
    with wave.open(path, "rb") as f:
        rate = f.getframerate()
        channels = f.getnchannels()
        assert f.getsampwidth() == 2, "这个读取函数仅适用于本课的 16 位 PCM WAV"
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").reshape(-1, channels)
    return rate, samples


def sampled_audio():
    text("## 03 · 音频：按时间记录振幅")
    text("**采样（sampling）**：是每隔一小段时间记录一次信号值。")
    text("**Hz（赫兹）**：在采样率中表示每秒采样次数，1 kHz（千赫兹）= 1,000 Hz。")
    text("**声道**是一条独立的声音信号。")
    text("**PCM（Pulse Code Modulation，脉冲编码调制）**：把声音在各个采样时刻的振幅量化并编码成数字。可以把它理解为：按固定时间间隔记下声音波形的高低，再用二进制数保存。")
    text("先听同一份素材的三个版本：同一句英语合成语音，停顿后是一段由低到高的扫频声。语音文本：She sees six shiny ships. Listen to the same sound at different sample rates.")
    text("### 48 kHz：每秒 48,000 个采样点（每声道）")
    audio("videos/lecture_02/speech_48000.wav")
    text("### 16 kHz：每秒 16,000 个采样点（每声道）")
    audio("videos/lecture_02/speech_16000.wav")
    text("### 8 kHz：每秒 8,000 个采样点（每声道）")
    audio("videos/lecture_02/speech_8000.wav")
    text("听一听 /s/、/ʃ/ 等声音的细节和扫频末段（扫频指频率随时间连续变化的声音）。使用相同音量比较；主观差异也受扬声器、听力与素材影响。")
    text("48 kHz 文件并不意味着语音源本身具有 24 kHz 的带宽；升采样不会补回源素材中没有的高频细节。")
    text("三个版本均为单声道、16 位 PCM；由同一母版得到；降采样时先做低通滤波（保留低频、衰减过高频率），再重采样，时长基本相同。降低采样率不应简单变成『放慢播放』。")
    rate48, pcm48 = read_pcm("videos/lecture_02/speech_48000.wav")  # @stepover
    rate16, pcm16 = read_pcm("videos/lecture_02/speech_16000.wav")  # @stepover
    rate8, pcm8 = read_pcm("videos/lecture_02/speech_8000.wav")  # @stepover
    print("48 kHz:", pcm48.shape, pcm48.dtype, "时长", round(len(pcm48) / rate48, 4), "秒")
    print("16 kHz:", pcm16.shape, pcm16.dtype, "时长", round(len(pcm16) / rate16, 4), "秒")
    print(" 8 kHz:", pcm8.shape, pcm8.dtype, "时长", round(len(pcm8) / rate8, 4), "秒")
    text("这里统一用 shape=(采样点数, 声道数)，单声道为 (N,1)。其他库也可能返回 (N,) 或把声道轴放在前面；立体声通常有两个声道。")
    peak = int(np.argmax(np.abs(pcm48[:, 0].astype(np.int32))))
    print("48 kHz 中一段非静音 PCM 数值 =", pcm48[peak:peak + 12, 0])
    print("16 位有符号整数范围 =", np.iinfo(np.int16).min, "到", np.iinfo(np.int16).max)
    print("48 kHz PCM 数据字节数 =", pcm48.nbytes)
    print(" 8 kHz PCM 数据字节数 =", pcm8.nbytes)
    image("images/lecture_02/audio-samples.png", width=950, style={"maxWidth": "calc(100vw - 120px)"})
    text("**采样率**决定时间轴上每秒取多少个点。")
    text("**位深**决定每个采样值的量化精度。这里都是 16 位，采样率变化没有改变位深。")
    text("理想带限信号的采样率须大于最高频率的两倍。8 kHz 采样的奈奎斯特频率为 4 kHz；降采样前要去除过高频率，避免混叠（过高的频率被错误地表现成低频），实际滤波还需要过渡带。")
    text("PCM 数据量 = 时长 × 采样率 × 声道数 × 每个采样值的字节数。")
    text("WAV（波形音频文件格式）是文件容器，本例装的是未压缩 PCM；文件头也占空间。")
    text("MP3、AAC（Advanced Audio Coding，高级音频编码）等压缩编码后的文件大小不能直接套这个公式。")
    text("语音是一维时间信号；双声道的第二个轴是声道，不是空间高度。声谱图则通过时频分析得到，不能只把这个数组随意 reshape 成一张图片。")

    multichannel_audio()


def multichannel_audio():
    text("### 单声道、立体声与多声道")
    text("**单声道（mono）**：一条音频信号，本课数组形状为 (N,1)。同一条信号可以由多个扬声器播放，扬声器数量不等于内容的声道数。")
    text("**立体声（stereo）**：通常有左、右两条信号，数组形状为 (N,2)。两个声道可以在振幅、到达时间和频谱上不同，形成方向感。")
    text("**多声道（multichannel）**：用多条信号组织声音，例如 5.1 环绕声。必须知道每个声道的含义与排列顺序，单看 shape 不够。")
    text("### 试听：左侧 → 中间 → 右侧 → 从左向右移动")
    text("建议戴耳机，以舒适音量试听。前面三声依次为左、双耳相同、右，停顿后连续向右移动；不要开启设备的单声道合并功能。")
    audio("videos/lecture_02/channels-stereo.wav")
    text("本例只通过改变左右声道的音量比例来移动声像（感知到的声音位置），不模拟前后、远近或高度。双耳相同的中间声可能听起来位于头部中央。")
    text("下面将同一示例的左右声道取平均，合成为单声道。横向变化会减弱或消失；混音后音量也会变化，并非严格的等响度听觉实验。")
    audio("videos/lecture_02/channels-mono.wav")
    rate, stereo = read_pcm("videos/lecture_02/channels-stereo.wav")  # @stepover
    mono_rate, mono = read_pcm("videos/lecture_02/channels-mono.wav")  # @stepover
    print("立体声：", stereo.shape, "采样率：", rate)
    print("单声道：", mono.shape, "采样率：", mono_rate)
    print("左侧声音的 5 个采样时刻 [L, R]：\n", stereo[12000:12005])
    print("右侧声音的 5 个采样时刻 [L, R]：\n", stereo[156000:156005])
    print("立体声 PCM 字节数：", stereo.nbytes)
    print("单声道 PCM 字节数：", mono.nbytes)
    text("每一行是同一时刻的左、右采样值。本例 WAV 的 PCM 数据按 L₀,R₀,L₁,R₁…交错保存；读取时整理成两列。")
    text("相同时长、采样率和位深下，两声道未压缩 PCM 的数据量是单声道的两倍；每个声道仍是每秒 48,000 次采样，播放不会加快。")
    text("### 5.1 环绕声：声道与扬声器布局")
    text("**5**：左前、右前、中置、左环绕、右环绕这五个主声道。")
    text("**.1**：LFE（Low-Frequency Effects，低频效果）声道；它的频率范围较窄，不表示只有十分之一个采样列。LFE 也不等于所有低频声音，低音管理还可把主声道低频交给低音炮。")
    channel_names = ["左前", "右前", "中置", "LFE", "左环绕", "右环绕"]
    surround = np.zeros((48000, 6), dtype=np.int16)
    print("1 秒 5.1 PCM 示意：", surround.shape)
    print("本例约定的列顺序：", channel_names)
    text("这个全零数组只演示六声道的存储形状，没有合成环绕音效。不同文件或接口的声道顺序可能不同；实际播放还需读取声道布局信息。")
    text("**5.1.2 扬声器布局**：五个平面主扬声器位置、一个低音炮、两个高度扬声器位置。这里最后的 2 表示高度位置，不是左右声道各增加两个。")
    text("### 空间音频：用方向和位置组织声音")
    text("**空间音频（spatial audio）**：关注声音相对于听者的位置、方向与环境，而不仅是声道数量。可通过多扬声器或耳机呈现，效果取决于内容、渲染方法和播放条件。")
    text("**基于声道**：把声音混入预定声道，再送到对应的播放位置，传统环绕声是常见例子。")
    text("**基于对象**：保存声音对象及位置等元数据，由渲染器按实际设备计算各输出声道。Dolby Atmos（杜比全景声）可结合声道床与声音对象；并非把声道数固定得更多。")
    text("**双耳音频（binaural audio）**：为左右耳准备带空间线索的两条信号。文件可以只有两声道，却包含比简单左右音量平衡更丰富的方向线索。")
    text("**HRTF（Head-Related Transfer Function，头部相关传递函数）**：描述声音从某个方向到达两耳时，受头部、耳廓等影响发生的变化，可用于合成双耳空间线索。")
    text("**头部追踪（head tracking）**：根据听者转头更新渲染，让声音可保持在场景中的固定位置；普通双声道录音本身不包含这种动态交互。")
    text("本课的左右声道试听是基础声像演示，没有使用 HRTF，也不是全景声或头部追踪演示。耳机双耳渲染的方位感还会受个体耳形差异影响。")
    text("应用示例：虚拟博物馆中，讲解声来自展品方向；多角色外语对话中，不同说话人位于不同位置。语音内容、说话人身份和空间位置是需要分别控制的信息。")
    link(title="扩展阅读：MDN 空间音频与 HRTF", url="https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Web_audio_spatialization_basics")
    link(title="扩展阅读：Dolby Atmos 的对象与空间位置", url="https://professionalsupport.dolby.com/s/article/Dolby-Atmos-Understanding-the-Concept")


def audiovisual_video():
    text("## 04 · 视频：帧、音轨与时间轴")
    video("videos/lecture_02/motion.mp4", width=800)
    text("本例是 2 秒钟的移动圆点，同时播放 440 Hz 合成音。它有图像和声音；视频文件也可以没有音轨，或包含多个音轨与字幕轨。")
    image("images/lecture_02/video-frames.png", width=1050, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 解码后：两种不同速度的数据流")
    text("**编码**把数据转换为适合保存或传输的形式。")
    text("**解码**把编码数据还原为可显示的图像或可播放的采样值。")
    text("**容器**像一个文件包，组织视频轨、音频轨和相关信息。")
    text("以下数组由素材脚本实际解码本例 MP4 后保存，方便课堂离线读取；不是假设 MP4 文件内部直接保存了一个 RGB 四维数组。")
    frames = np.load("images/lecture_02/video-decoded-frames.npy")
    fps = 24
    print("解码后的 RGB 帧 shape =", frames.shape)
    print("第 12 帧 shape =", frames[12].shape)
    print("第 12 帧的一个像素 =", frames[12, 48, 42])
    print("本例第 12 帧的显示时刻 =", 12 / fps, "秒")
    rate, sound = read_pcm("videos/lecture_02/video-decoded-audio.wav")  # @stepover
    print("解码后的音轨 shape =", sound.shape, "；采样率 =", rate)
    print("一帧显示期间对应的音频采样点数 =", rate // fps, "（每声道）")
    text("这个恒定帧率例子每秒有 24 帧，同时每声道每秒有 48,000 个音频采样点。两条轨道通过时间戳（标记各段数据应在何时播放的时间信息）同步；不是每张图像里都存一段声音。")
    text("AAC 编解码可能引入延迟与尾部填充，因此解码后的音频采样总数不一定恰好是 96,000；播放时还要结合容器时间信息。")
    text("### 磁盘上：MP4 容器中的压缩码流")
    print("解码 RGB 帧总字节数 =", frames.nbytes)
    print("MP4 文件总字节数 =", Path("videos/lecture_02/motion.mp4").stat().st_size)
    source = np.load("images/lecture_02/video-source-frames.npy")
    print("源帧与解码帧完全一致？", np.array_equal(source, frames))
    text("本例：**MP4（多媒体容器格式）→ H.264（一种视频压缩编码标准）视频轨 + AAC 音频轨 + 时间信息**。编码可以利用空间与帧间冗余；本例采用有损设置，解码值不必与源数组逐像素相同。")
    text("**帧率（fps，Frames Per Second，每秒帧数）**：描述每秒的帧数。")
    text("**像素尺寸**描述每帧多大。")
    text("**码率**描述每秒编码数据量。")
    text("它们相关，但不是同一个指标；一般视频还可能采用可变帧率。")
    link(title="进一步阅读：媒体容器与编码的区别（MDN）", url="https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Containers")


def point_clouds():
    text("## 05 · 3D 点云：一组在空间中定位的点")
    image("images/lecture_02/pointcloud-mesh.png", width=1000, style={"maxWidth": "calc(100vw - 120px)"})
    text("左边是在立方体表面采样的点，右边是同样外形的三角网格。两者都与 3D 建模有关，但保存的信息不同。")
    points = np.load("images/lecture_02/points.npy")
    print("点云 shape =", points.shape, "；dtype =", points.dtype)
    print("前 5 个点的 XYZ 坐标 =\n", points[:5])
    print("坐标数组大小 =", points.nbytes, "字节")
    text("shape=(N,3) 表示 N 个点，每点三个坐标。")
    text("float32 是 32 位浮点数类型，可表示带小数的坐标，每个数占 4 字节。")
    text("这里是教学用无量纲坐标，取值 -1 到 1；真实扫描必须约定单位、坐标系与朝向。")
    text("点云可来自激光雷达、深度相机或多视角重建，也可像本例一样合成。每点还可带颜色、法向量（指示局部表面朝向的向量）或类别标签；(N,6) 的后 3 列含义必须查数据说明。")
    text("### 点云与网格：两种三维表示")
    mesh = np.load("images/lecture_02/cube-mesh.npz")
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    print("网格顶点 shape =", vertices.shape)
    print("三角面 shape =", faces.shape)
    print("第一个三角面连接的顶点编号 =", faces[0])
    print("这些顶点的实际坐标 =\n", vertices[faces[0]])
    text("**点云**保存采样位置，通常没有显式的连边和面。")
    text("**网格**除了顶点，还保存哪些顶点组成表面。本例网格有 8 个顶点、12 个三角面。")
    text("建模可以从点云出发，经过清理、配准（把多次扫描对齐到同一坐标系）和表面重建得到网格；缺失、噪声与遮挡让这一步并不唯一。也能反过来从网格表面采样点云。")
    text("这里的点云和网格都由立方体几何直接构造，未执行自动表面重建。3D 还可用体素（三维空间中的小格子）、隐式场（用函数描述空间中的表面或属性）等形式表示，点云只是其中一种。")
    text("附带 points.ply 是可读的 ASCII（用字符编码记录内容的）文本点云文件：文件头说明顶点数和属性，后面每行是一个 XYZ 点。浮点数组大小与文本文件大小也不相同。")


def summary():
    text("## 这一部分带走：先读懂 shape，再解释每一个数")
    text("| 模态 | 本课解码后表示 | 一个数的含义 | 还必须知道什么 |\n|---|---|---|---|\n| 灰度图 | H×W | 灰度值 | 取值范围、行列顺序 |\n| RGB 图像 | H×W×3 | 一个颜色通道的值 | 通道顺序、颜色空间 |\n| 音频 | N×C | 一个时刻某声道的振幅 | 采样率、位深 |\n| 视频 | T×H×W×3，另有音轨 | 一帧中一个颜色通道的值 | 时间戳、帧率、编码 |\n| 点云 | N×3 | 一个空间坐标分量 | 单位、坐标系、附加属性 |")
    text("同样是三个数，RGB 是颜色，XYZ 是位置。**形状相同，不代表语义相同；磁盘文件，也不等于解码后的数组。**")
    text("只把音频采样率标签改为一半、却不重新采样，会使播放时长约翻倍、音高约降低一个八度；正确降采样会同时改变采样序列与采样率，保持原来的时长和音高。")
    text("接下来从数据表示转向任务：语言与图像、语音、视频和三维数据可以共同构成输入，也可以作为不同的输出。")


def cross_modal_tasks():
    text("## 第二部分 · 跨模态协同任务")
    text("**跨模态协同**：把不同模态中的相关信息联系起来，共同完成任务。例如，文字说明目标，图像提供证据，语音承担交互。")
    text("先区分四类任务：")
    text("**理解**是解释已有内容。")
    text("**检索**是从已有集合中找出匹配项。")
    text("**生成**是产生新内容。")
    text("**编辑**是在保留部分原内容的条件下修改它。")
    text("下面给出任务设计与预期输出的例子，不在本页调用在线模型。重点是看清输入、输出和评价标准。")
    image_text_tasks()
    speech_text_tasks()
    video_text_tasks()
    pointcloud_text_tasks()
    text("## 小结 · 同一条语言指令，可以连接不同模态")
    text("| 模态组合 | 典型协同任务 | 主要检查内容 |\n|---|---|---|\n| 图像＋文本 | 看图问答、图文检索、文生图、指令编辑 | 对象、属性、空间关系是否对应 |\n| 语音＋文本 | 转写、朗读、翻译、口语交互 | 内容、发音、语气与响应时机 |\n| 视频＋文本 | 视频问答、片段检索、字幕、生成与编辑 | 事件顺序、时间定位、跨帧一致性 |\n| 点云＋文本 | 三维描述、目标定位、检索、生成与编辑 | 空间位置、尺寸、结构与几何可用性 |")
    text("语言既可以描述内容，也可以指定目标与约束。评价跨模态系统时，要同时检查**语义对应、模态自身的质量，以及任务是否完成**。")


def image_text_tasks():
    text("## 06 · 图像与文本：理解、检索、生成、编辑")
    image("images/lecture_02/rgb.png", width=480, style={"imageRendering": "pixelated", "maxWidth": "calc(100vw - 120px)"})
    text("### 联合理解：依据图像回答文字问题")
    text("**输入：图像＋问题 → 输出：文本。** 以上图为例，问『屋顶是什么颜色？太阳位于房子的哪一侧？』，答案应是红色、画面右侧。")
    text("**VQA（Visual Question Answering，视觉问答）**：要求回答同时依据图像和问题。")
    text("**图像描述（image captioning）**：是给图像生成简短说明。")
    text("**OCR（Optical Character Recognition，光学字符识别）**：则是读出图中的文字。")
    text("外语示例：输入博物馆展签照片，先识别原文，再解释并翻译；识字、理解和翻译分别可能出错，应分别核对。")
    text("### 跨模态检索：从已有图片中找匹配项")
    text("**输入：『红屋顶、黄色太阳的小房子』＋图片库 → 输出：匹配图片的编号与排序。** 也可以反过来，用图片找对应文字说明。")
    text("检索选择已有内容，不会直接画出新图片。检查返回结果是否满足颜色、对象与位置条件，而不只是整体风格相近。")
    text("### 条件生成：根据文字产生新图像")
    text("**输入：文字描述 → 输出：新图像。** 例如『一栋红屋顶小房子，黄色太阳在画面右侧，草地上有两棵树』。文字在这里是控制生成的条件。")
    text("检查对象数量、颜色和左右关系。好看与遵循指令是两项不同的评价。")
    text("### 指令编辑：修改指定内容，保留其余部分")
    text("**输入：原图＋『把红屋顶改成蓝色，其他部分保持不变』 → 输出：编辑后的图像。** 有时还提供掩码（mask，即标出允许修改区域的图）。")
    text("编辑要同时满足『改对了什么』和『保留了什么』：屋顶变蓝，但房子形状、太阳位置与背景应保持。")


def speech_text_tasks():
    text("## 07 · 语音与文本：从内容转换到实时交互")
    text("### 三类基础任务")
    text("**ASR（Automatic Speech Recognition，自动语音识别）**：语音 → 文本，例如把英文访谈转成英文逐字稿。转写不自动等于翻译。")
    text("**TTS（Text-to-Speech，文本转语音／语音合成）**：文本 → 语音，例如把双语导览稿读出来；内容相同，也可以有不同语速与语气。")
    text("**语音翻译（speech translation）**：源语言语音 → 目标语言文本或语音，例如英语发言 → 中文字幕或中文配音。还要处理专名、数字、停顿和表达习惯。")
    text("### 语音助手的两种实现思路")
    text("**级联式**：语音 → ASR 转写 → 语言模型生成回答 → TTS 朗读。中间文字便于检查，但转写错误可能传递到后续步骤，语气等信息也可能丢失。")
    text("**端到端语音模型**：直接以语音为输入并产生语音响应，不必把完整转写文本作为各模块间唯一的接口；内部仍可能包含多个模块，也可能同时输出文字。")
    text("### 单工、半双工与全双工")
    image("images/lecture_02/duplex.svg", width=1000, style={"maxWidth": "calc(100vw - 120px)"})
    text("**单工（simplex）**：只朝一个方向传递，例如单向广播。")
    text("**半双工（half-duplex）**：双方都能传递，但同一时刻只有一个方向，例如对讲机。")
    text("**全双工（full-duplex）**：两个方向可以同时传递，例如电话通话。")
    text("在语音助手中，常见的**轮流交互**是『用户说完 → 系统回答』；全双工交互则要求系统在输出语音时仍持续接收并处理用户语音。网络能双向传输，不代表模型已经能边听边说。")
    text("**流式（streaming）**：指数据分成小段连续处理，不必等完整录音结束；半双工系统也可以流式输出。")
    text("**可打断（barge-in）**：指用户插话时系统能够停止或调整回答，仅有打断功能还不能证明具备完整的全双工对话能力。")


def video_text_tasks():
    text("## 08 · 视频与文本：语义需要对应时间")
    image("images/lecture_02/video-frames.png", width=1000, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 理解与描述")
    text("**输入：视频＋『描述圆点的运动方向』 → 输出：『圆点从左向右移动』。** 与单张图片相比，视频理解需要关联前后帧；对话内容、音乐等还需要使用音轨。")
    text("讲座示例：视频＋『概括讲者提出的三个观点』 → 带证据时间段的文字摘要。仅看字幕可能漏掉图表，仅看少数帧也可能漏掉动作。")
    text("### 检索与时间定位")
    text("**视频检索**：用『介绍语音识别的讲座』从视频库找出相关视频。")
    text("**时间定位（temporal grounding）**：在一个视频内，根据『讲者开始解释采样率』找出对应的起止时间。")
    text("预期输出示例：『02:10–02:45』。这是任务形式示例，不是前面两秒动画的实际片段；检查的是时间范围是否覆盖目标事件。")
    text("### 字幕、配音与生成编辑")
    text("**视频 → 带时间戳的双语字幕**：结合音轨转写、翻译并对齐时间。")
    text("**字幕／译稿 → 配音**：还需控制语速、时长和说话人对应关系。")
    text("**文本 → 视频**：『一个橙色圆点在两秒内从左向右匀速移动』。")
    text("**原视频＋指令 → 编辑后视频**：『把圆点改为蓝色，保持运动轨迹和音轨不变』。")
    text("编辑不能只让某一帧正确，还要保持**跨帧一致性**：同一个对象的身份、颜色、形状与运动应在时间上连贯；音画也要同步。")


def pointcloud_text_tasks():
    text("## 09 · 点云与文本：把语言对应到三维空间")
    image("images/lecture_02/pointcloud-mesh.png", width=950, style={"maxWidth": "calc(100vw - 120px)"})
    text("### 理解与目标定位")
    text("**输入：点云＋『描述这个物体的形状』 → 输出：文本描述**，例如『表面近似立方体』。真实场景还可能要求识别家具、估计尺寸或描述空间关系。")
    text("**三维指称定位（3D visual grounding）**：点云＋『桌子左边的椅子』 → 对应的点集合或三维包围框（围住目标的空间盒子）。『左边』要明确采用观察者还是场景坐标系。")
    text("这种任务把语言中的对象、属性与空间关系落实到具体几何位置。参照物不清楚、扫描缺失或遮挡，都可能造成定位错误。")
    link(title="研究实例：3DVG-Transformer，点云中的语言指称定位", url="https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_3DVG-Transformer_Relation_Modeling_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.pdf")
    text("### 检索、生成与编辑")
    text("**文本＋三维资产库 → 匹配模型**：例如检索『有靠背、没有扶手的椅子』。这是从已有资产中选择。")
    text("**文本 → 新三维表示**：例如生成『一个带四条腿的木凳』；输出可能是点云、网格或其他形式。文本生成一张立体感图片，并不等于生成了可旋转、可测量的三维几何。")
    text("**点云／网格＋编辑指令 → 修改后的几何**：例如『保持底面不变，把高度增加一倍』。生成结果还需检查尺寸、结构完整性与后续建模用途。")
    text("### 一个可执行的几何编辑示例")
    text("以下用明确的坐标变换演示编辑效果，不调用语言模型。原立方体 z 范围为 -1 到 1；固定底面 z=-1，将竖直方向高度拉伸为原来的两倍。")
    points = np.load("images/lecture_02/points.npy")
    edited = points.copy()
    edited[:, 2] = -1 + 2 * (points[:, 2] + 1)
    print("原始 z 范围：", points[:, 2].min(), points[:, 2].max())
    print("编辑后 z 范围：", edited[:, 2].min(), edited[:, 2].max())
    print("x、y 保持不变：", np.array_equal(points[:, :2], edited[:, :2]))
    text("语言指令给出目标，几何操作落实变化。真正的语言驱动编辑系统还需要识别目标对象、解析约束，再生成或执行合适的修改。")


if __name__ == "__main__":
    main()
