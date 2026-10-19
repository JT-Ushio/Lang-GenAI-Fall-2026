# 第五讲：视频关键帧与短动画

本素材包供教师逐图讲解，图册不是作者官方 PPT。已核对的作者公开视频目录有动画源码与实验笔记，未找到成套 PPT。

## 来源与规格

- 中文视频：[B站 技术Flow](https://www.bilibili.com/video/BV1Z9ec69Ezz)，BV1Z9ec69Ezz，CID 41943437272，33:51。
- 原作：[Welch Labs／3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY)。
- [作者动画源码](https://github.com/WelchLabs/videos/tree/master/_2025/sora)；只检索了公开材料，不声称所有渠道都没有 PPT。
- 14 张 PNG 直接提取自可访问的 852×480 视频流；保留完整画面与中文字幕，没有放大冒充高清。
- 4 段 GIF：640×360、10 fps，合计 36 秒；另有 852×480、30 fps 无声 MP4，便于暂停。
- 时间戳均对应中文版。未把英文版的章节时间直接套用。
- 保留原作者与转载来源；没有重新分发完整视频。编译课件不下载媒体、不调用 FFmpeg。

## 使用

直接双击 `index.html` 可离线讲解；也可通过课程网站打开。左右方向键换图，GIF 默认静态，点击播放或恢复静态。MP4 有暂停与进度控件。PNG／GIF 可插入自己的 PPT。

## 关键帧

| 图片 | 中文版时间 | 建议章节 | 内容 |
|---|---|---|---|
| [01_iterative-generation](01_iterative-generation.png) | [01:35](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=95) | 01／06 | 从随机状态开始，重复更新 |
| [02_clip-encoders](02_clip-encoders.png) | [04:16](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=256) | 01 | 复习 CLIP 的两个编码器 |
| [03_clip-pairing](03_clip-pairing.png) | [05:08](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=308) | 01／课前复习 | 批量图文配对与对比学习 |
| [04_forward-noise](04_forward-noise.png) | [08:40](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=520) | 03 | 加噪让图像结构逐渐消失 |
| [05_reverse-generation](05_reverse-generation.png) | [09:08](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=548) | 05／06 | 训练好后，从噪声朝图像更新 |
| [06_noise-target](06_noise-target.png) | [11:02](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=662) | 04 | 训练目标是这次人为加入的噪声 |
| [07_two-pixel-space](07_two-pixel-space.png) | [12:28](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=748) | 06／拓展 | 两个像素，也能组成一个数据空间 |
| [08_score-field](08_score-field.png) | [15:46](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=946) | 06／拓展 | 箭头场给出局部移动方向 |
| [09_ddpm-paths](09_ddpm-paths.png) | [18:23](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1103) | 06 | DDPM 的随机采样路径 |
| [10_ddpm-ddim](10_ddpm-ddim.png) | [23:35](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1415) | 06 | 同一训练目标，可以搭配不同采样路径 |
| [11_text-conditioning](11_text-conditioning.png) | [26:28](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1588) | 09 | 文字特征进入去噪模型 |
| [12_conditional-distribution](12_conditional-distribution.png) | [27:24](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1644) | 09／10 | 类别条件挑选分布中的一部分 |
| [13_cfg-vector](13_cfg-vector.png) | [30:06](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1806) | 10 | CFG：使用两种预测的差异 |
| [14_guidance-output](14_guidance-output.png) | [31:44](https://www.bilibili.com/video/BV1Z9ec69Ezz/?t=1904) | 10／11 | 引导尺度会改变生成外观 |

## 短动画

| 动画 | 区间 | 时长 |
|---|---|---|
| [从噪声逐渐出现结构](a01_noise-to-generation.gif) · [MP4](a01_noise-to-generation.mp4) | 01:53—02:05 | 12 秒 |
| [构造前向加噪链](a02_forward-noising.gif) · [MP4](a02_forward-noising.mp4) | 08:32—08:40 | 8 秒 |
| [DDPM／DDIM 采样路径对照](a03_ddpm-ddim.gif) · [MP4](a03_ddpm-ddim.mp4) | 23:28—23:36 | 8 秒 |
| [沿条件偏向更新状态](a04_cfg-path.gif) · [MP4](a04_cfg-path.mp4) | 30:15—30:23 | 8 秒 |

## 逐图讲解

### 01:35 · 从随机状态开始，重复更新

先指出输入是随机噪声；网络给出更新所需的预测，经过多次更新才得到结构。

口径：本帧来自视频对 Wan 的介绍，骨干标为 Transformer；本课经典 SD 1.5 使用 U-Net，不能混用架构。

### 04:16 · 复习 CLIP 的两个编码器

文字和图片各自编码，再比较向量。这让模型能匹配描述，还没有自动获得把噪声画成图的生成网络。

口径：图中的 512 是视频所举 CLIP 检索表示；本课 SD 1.5 条件采用 77×768 的文本序列隐藏状态。

### 05:08 · 批量图文配对与对比学习

按行看文本，按列看图像；配对的目标是让同一概念靠近，其他候选形成区分。

口径：这张表列的是配对的向量，不能直接把每格数字当成已经算好的相似度。

### 08:40 · 加噪让图像结构逐渐消失

沿左到右读：猫的细节越来越不明显，噪声越来越占主导；接着回到课堂的 RGB 加噪代码。

口径：这是视频的过程示意；具体信号和噪声系数以课件公式及代码为准。

### 09:08 · 训练好后，从噪声朝图像更新

沿箭头方向读生成链：当前状态送入网络，再执行采样更新。不同噪声水平重复使用同一套训练好的参数。

口径：多个网络图标表示多次调用，不表示每一步重新训练一个独立网络。

### 11:02 · 训练目标是这次人为加入的噪声

先找到黄色 ε，再找紫色网络预测：损失比较这两个量。干净图、时间和噪声可由程序构造训练样本。

口径：本课采用 ε 预测；预测原图或 velocity 的模型需要相应的目标与采样公式。

### 12:28 · 两个像素，也能组成一个数据空间

横轴记录第一个像素，纵轴记录第二个像素；一幅只有两个像素的图像变成平面上的一个点。

口径：二维平面是数据空间类比，不是原图的宽与高；真实图像每个标量都可能贡献一个维度。

### 15:46 · 箭头场给出局部移动方向

把螺旋上的黄点看成数据分布的样本，箭头表示在某个噪声水平下可参考的局部方向。

口径：score function 在这里指对数密度的梯度，不是 CLIP 的图文评分；不用本图替代完整采样公式。

### 18:23 · DDPM 的随机采样路径

每条彩色路径从随机状态出发；受控随机项会让轨迹弯折，最终样本落向学到的结构。

口径：单条轨迹不要求每一步肉眼更清晰，也不需要恢复某张预先指定的原图。

### 23:35 · 同一训练目标，可以搭配不同采样路径

比较左侧随机折线与右侧较平滑的路径，再说明网络预测和采样算法是两个层次。

口径：DDIM 的确定性需要 η=0 且初始噪声与条件给定；本帧不是两种算法优劣的受控质量评测。

### 26:28 · 文字特征进入去噪模型

上方文本编码器输出条件特征，下面的去噪模型在每次更新时读取条件。语言提供内容约束。

口径：视频本帧用 Transformer 作泛化示意；本课用 U-Net 内的交叉注意力讲 SD 1.5，不能照此图认定所有模型相同。

### 27:24 · 类别条件挑选分布中的一部分

先把整条螺旋看成数据集合，再想象不同颜色对应不同类别；给定条件改变的是希望生成哪部分样本。

口径：二维类别示意不等同于现实自然语言的全部语义，也不是实际像素坐标。

### 30:06 · CFG：使用两种预测的差异

先读灰色无条件方向，再读黄色条件方向；绿色差向量表示条件带来的偏向，随后按尺度组合。

口径：视频用向量场 f 表示，课堂代码组合的是噪声预测 ε；符号和尺度约定应先对齐，不能机械抄符号。

### 31:44 · 引导尺度会改变生成外观

指出提示词、引导尺度和输出三个位置，讨论条件遵循、颜色、细节和多样性应分别评价。

口径：这张截图本身不能证明所有控制变量相同；严谨比较还要固定模型、采样器、步数和随机种子。

## 重建

只在需要重新截取时运行，不属于常规课件编译：

```bash
python3 lectures/scripts/extract_lecture_05_video.py --source /path/to/source.mp4 --ffmpeg /path/to/ffmpeg
```

输入必须是相同剪辑、相同时间轴的中文版视频；换成英文版时需要重新选时刻。来源与文件校验和保存在 `manifest.json`。
