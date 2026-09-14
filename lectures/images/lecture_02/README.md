# 第二讲素材

- `digit.npy` / `grayscale.png`：scikit-learn `load_digits()` 第 3 项，真实标签 3，8×8，灰度 0–16。该数据源于 UCI Optical Recognition of Handwritten Digits；不是 MNIST。来源：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html 。本课转为 uint8，保留原始数值。
- RGB 房子、移动圆点、立方体点云与网格：本项目程序构造。
- 音频：macOS Samantha 合成英语语音，后接程序生成的 200–7000 Hz 扫频。并非真人录音。三个版本从同一母版以 `scipy.signal.resample_poly` 抗混叠重采样，PCM16、单声道；不单独归一化各版本。
- MP4：160×96、24 fps、48 帧、H.264 yuv420p + AAC，2 秒动画配合 440 Hz 合成音。`video-source-frames.npy` 为编码前 RGB；`video-decoded-frames.npy`、`video-decoded-audio.wav` 为 FFmpeg 实际解码结果。AAC 可能包含填充样本。
- 点云：立方体六个表面规则采样后去重，float32 XYZ，无量纲。网格直接定义 8 个顶点、12 个三角形；未从点云估计网格。PLY 是 ASCII 格式。
- 所有图表均从以上实际数据绘制，未使用 AI 生图。图表采用英文短标签，中文解释放在课件正文。

素材生成：`python lectures/scripts/build_lecture_02_assets.py`。额外依赖 `scipy matplotlib scikit-learn imageio-ffmpeg`；语音生成需要 macOS `say` / Samantha，或预先提供 `/private/tmp/course-speech.aiff`。常规课件编译不需要这些额外依赖、FFmpeg 或联网。

## 多声道扩展示例

`channels-stereo.wav` 与 `channels-mono.wav` 由 `python lectures/scripts/build_channel_demo.py` 生成，仅需 NumPy。7.5 秒、48 kHz、PCM16：440 Hz 音调左侧 1 秒、停顿 0.5 秒、居中 1 秒、停顿 0.5 秒、右侧 1 秒、停顿 0.5 秒、连续等功率向右声像移动 3 秒。单声道版本为左右均值。淡入淡出用于避免硬切换爆音。这里没有 HRTF、全景声、空间录音或头部追踪。
