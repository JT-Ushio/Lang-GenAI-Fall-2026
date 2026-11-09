# 第七讲备课说明

主题：语音与语言对齐。学生端不加入对象、目标、分钟表等组织性文字。配套 39 页演示文稿与 Python 课件内容对应。PPT 内容源为 `presentations/lecture_07/content.json`；Python 正文直接写在 `lecture_07.py`，方便延续前几讲的逐行展示与手工修改。修改其中一份正文后应同步另一份。

## 90 分钟建议

- 第一课时：语音与书写层次 8 分钟，试听与数字波形 7 分钟，分帧/STFT/梅尔频谱 15 分钟，CTC 折叠与路径 15 分钟。
- 第二课时：CTC 概率与优化 12 分钟，Whisper 架构和维度 15 分钟，交叉注意力和任务 8 分钟，评价、时间戳和局限 10 分钟。

PPT 第 1—19 页可用于第一课时，第 20—39 页用于第二课时。根据学生线性代数基础调整代码推导时间。

## 编译与打开

```bash
cd lectures
python3 execute.py -m lecture_07
cd trace-viewer
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

浏览 `http://127.0.0.1:4173/?trace=var%2Ftraces%2Flecture_07.json&step=1&animate=1`。

常规编译使用项目现有依赖，不下载权重，不调用云端 API，不需要 FFmpeg。`# @inspect` 可查看采样点、特征形状、CTC 路径、损失和注意力权重。浏览器展示的是编译时记录，Python 修改后重新编译。

## 演示的证据范围

- 普通话合成语音使用 macOS Tingting，语速设置 175，母版保存在 `videos/lecture_07/speech-source.wav`。原文：“你好，欢迎来到语音与语言对齐课堂。请听同一句话在不同语速下的变化。”慢速版由 FFmpeg `atempo=0.75` 预生成，保持音高不等于完美保持所有声学细节。
- 特征提取真实执行 `torch.stft` 与官方梅尔矩阵乘法，使用 16 kHz、400 点 Hann 窗、160 点 hop、80 频带和官方对数缩放。
- CTC 概率例子枚举全部 8 条路径。6 条折叠成“好”，概率和 0.688；空串 0.216，“好好”0.096。与 PyTorch 的 `reduction='sum'` 核对，避免默认按目标长度缩放混淆。
- CTC 优化仅学习一个固定“看一看”样本的 8×3 帧分数，没有声学编码器，不构成语音识别训练或泛化评测。
- Whisper tiny 卷积示例随机初始化，仅验证形状，没有下载或运行预训练识别器。不要把 shape 检查说成识别成功。
- 注意力图使用人工构造分数，矩阵计算是真实的。没有声称得到真实对齐或词级时间戳。
- WER/CER 只计算固定例子的编辑距离，不代表某个真实模型的准确率。

## 资料与版权

- CTC 原论文：https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Whisper 原论文：https://proceedings.mlr.press/v202/radford23a.html
- Whisper 官方仓库：https://github.com/openai/whisper
- 音频预处理：https://github.com/openai/whisper/blob/main/whisper/audio.py
- tiny 配置：https://huggingface.co/openai/whisper-tiny/blob/main/config.json
- 学生课件采用 `whisper-chinese.png`，依据官方 tiny 配置绘制、AI 辅助生成，转写样例为“欢迎参观”。图中的波形与频谱为示意，不对应真实标注。
- 官方架构图 `whisper-official.png` 留作备课参考，梅尔滤波器源自官方仓库，MIT 许可附于图片目录。
- 其他图均为本地数值或显式标注的示意路径。STFT 与梅尔图不是声学标注，不能据其色块直接给词或音素分段。

## 重新生成素材

仅素材构建需要 matplotlib、imageio-ffmpeg（不要为普通授课编译额外安装）。

```bash
python3 lectures/scripts/build_lecture_07_assets.py
```

图、WAV 与结果 JSON 已保存在仓库。图片源数据可由脚本重算。演示文稿脚本 `build_slides.mjs` 在 Artifact Session 中调用，共用上述 JSON。在线文稿可编辑，Python 课件中的音频可以直接试听。演示文稿用备注标明音频相对路径；授课试听使用浏览器课件。

中文 WER 示例明确给定分词：参考 `[我, 今天, 学习, 语音]`，假设 `[我, 学习, 语音]`，删除一个词，WER=1/4。CER 的“欢迎参观／欢迎参光”按字比较。两者不混用分词单位。


## 梅尔频谱对照素材

六段音频已转为 16 kHz、单声道、16 位 PCM，统一将重采样后的振幅乘以 0.95，避免峰值截断。课件里的 80 频带梅尔功率由同一段代码计算。每组使用相同的时间轴和相对于组内峰值的 dB 色标。录音增益会影响颜色亮度。Whisper 输入还会做本讲前面代码中的动态范围截断与缩放。

| 样本 | 来源及处理 |
|---|---|
| 鸟叫 | keweldog 的 [bird chirps.wav](https://freesound.org/people/keweldog/sounds/181132/)（CC0），取 [ESC-50 片段 3-181132-A-14.wav](https://github.com/karolpiczak/ESC-50/blob/master/audio/3-181132-A-14.wav)。ESC-50 整库采用 CC BY-NC 3.0。 |
| 轻雨 | Mijesty 的 [Sound of light rainfall.ogg](https://commons.wikimedia.org/wiki/File:Sound_of_light_rainfall.ogg)，取前 5 秒，音频改编和相应图示依 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供。 |
| 提米（Timmie） | 《原神》游戏配音，[Genshin Voice](https://huggingface.co/datasets/simon3000/genshin-voice) revision `68950dc52d799876d1e64c1ab6c47955502e79a1` 的各语言 `1-10/Timmie.zip`。语音 ID 均为 `vo_tips_hexenzirkel_010_3_timmie_01.wem`。版权归原权利人。 |

四个配音源文件按中文、日语、英语、韩语顺序分别为 `2dcd3bec7f9f2fce.wav`、`1ce97767fae5d1a4.wav`、`b4bc77e50aaac066.wav` 和 `0ef9596ac7e9e597.wav`。用图说明音节分布、停顿与能量差异时，可指出语言、译文和演员是同时变化的因素。

从已备好的 WAV 重画两组图需安装 matplotlib，再从仓库根目录运行 `python3 lectures/scripts/build_lecture_07_mel_examples.py`。网页课件编译使用已有图，无需 matplotlib。
