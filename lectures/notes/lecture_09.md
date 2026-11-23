# 第九讲备课说明：视频与语言建模

正文以中文操作台动画贯穿抽帧、事件顺序、视频表示、视频问答和时间定位，避免重复第二讲的通用媒体存储知识。MOVA 放在结尾作为通往下一讲同步音视频生成的桥梁，不能将它误列为视频问答模型。

## 两课时节奏（各 45 分钟）

| 第一课时 | 分钟 | 对应函数 |
|---|---:|---|
| 视频案例、任务与动词／论元／时序 | 8 | opening、language_and_time |
| 均匀抽帧、时间戳与短事件漏检 | 12 | sampling |
| 倒放、打乱、恢复、池化不变性与交互实验 | 13 | temporal_order |
| 时空小块、物理时间与 token 预算 | 12 | representation |
| 第二课时 | 分钟 | 对应函数 |
| 三种理解架构与模块维度 | 14 | architectures |
| 答案监督与时间定位 | 7 | training |
| 评价、长视频与音画错位 | 6 | evaluation |
| MOVA 双塔、时间对齐与完整演示 | 8 | mova |
| Sora 与 Seedance／即梦官方样例 | 6 | generation_showcase |
| 产品使用与收入的数据口径 | 3 | generation_industry |
| 任务指令与回顾 | 1 | closing |

操作台视频每段约 12 秒。MOVA 约 38 秒的主展示、单人和多人示例可离线播放。Sora 与 Seedance 播放官方在线素材，课前可用正文链接检查课堂网络。代码可逐行运行，变量监视只展示索引、小型结果与形状，不展开完整像素。

## 运行

在仓库根目录：

```bash
cd lectures
python3 execute.py -m lecture_09
cd trace-viewer
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

打开 <http://127.0.0.1:4173/?trace=var/traces/lecture_09.json&step=1&animate=1>。已有 4173 服务时复用即可，不重复启动。编译只依赖已有 Python/PyTorch 环境，无网络下载、预训练权重、GPU 或 FFmpeg 依赖。

### 实际解码本地视频

课件使用随库提供的 `teaching_clip.npz`：它由实际 MP4 解码得到，而不是临时构造的随机像素。可选工具对 MP4 实际解码后导出均匀抽帧图片与 `frames.json`：

```bash
python3 -m pip install imageio-ffmpeg
python3 demos/lecture_09_extract.py videos/lecture_09/operation.mp4 /tmp/lecture09-frames --count 8 --source-fps 8
```

工具的 `--source-fps` 要求已知恒定帧率；不要给任意可变帧率视频估填平均帧率。通用生产处理应读取解码器返回的逐帧 PTS，再按时间采样。本例展示时间选择算法，但不声称已实现通用视频解码服务。

原创动画、解析声音、示意图的可复现生成入口：

```bash
python3 -m pip install imageio-ffmpeg matplotlib
python3 scripts/build_lecture_09_assets.py
```

当前生成脚本的中文字体路径面向 macOS PingFang；其他平台需要设置可用中文字体。官方图已随库提供；MOVA 主展示重建还需原始展示片段位于脚本注明的临时路径，课堂播放不依赖该原始文件。所有已打包资源的 SHA256 及来源在 `demos/lecture_09_sources.json`，官方图 URL 在 `images/lecture_09/official_sources.json`。

## 实验口径与事实边界

- 原视频 12 秒、8 FPS、96 帧。图片缓存为 `[96,90,160,3]`，运行后转 `[96,3,90,160]`。
- 八帧采样使用等时段中心，原帧号 `[6,18,30,42,54,66,78,90]`。全帧打乱使用固定种子 9，可精确恢复。倒放／打乱移除音轨；交互图分别显示播放位置与源帧时间。
- 状态灯红色区间 `[1.25,1.50)` 秒。从零开始 2 FPS 采样漏掉，4 FPS 捕获第 10 帧；采样相位改变可能改变结论。
- 取杯／倒水／饮水／放杯是作者设计的粗阶段。水流可见区间单独定义为 `[3.6,5.7)` 秒，8 FPS 下首个实际可见帧在 3.625 秒附近。预测 `[3,6)` 的连续区间 tIoU 为 0.70；边界定义与离散帧观测精度需要区分。
- 均值池化正反不变、端点差相反是数值事实；它们不是动作识别模型。像素差分不是光流，更不是语义动作标签。
- Tubelet 用真实视频缩图，实算 `[64,384]`。OneVision 压缩示例的隐藏维度 16 为教学值；STC 示例的 8→16 通道为教学值，仅复现三维下采样原理，不复现完整连接器。
- Qwen2.5-VL-7B 形状按官方配置推导。meta 张量不执行视觉识别：8 帧、224²，2×14² tubelet，ViT 1280，2×2 空间合并 5120，投影到 LLM 3584，最终 `[256,3584]`。真实视频处理器动态调整尺寸与帧数，本课没有调用该处理器。
- 答案交叉熵代码使用小型随机 logits，确实反向传播并验证忽略位置的 logits 梯度为零；不是完整 VLM 训练。答案以外输入仍可能通过答案路径得到梯度，除非对应参数被冻结。
- 本课没有下载或运行预训练大模型，也没有生成新的 MOVA 视频。作者样例与本课原创操作台动画分别标明来源。5 Hz／50 Hz 对齐是教学网格，不是 MOVA 的真实 VAE 速率。

## 代表性架构与一手来源

选择明确版本用于讲机制，不按发布日期宣称“最新”或作未经实测的性能排名。

1. **LLaVA-OneVision**（2024-08）：SigLIP＋两层 MLP＋Qwen2；视频空间表示 27²→14²，上限 32 帧配置见报告。
   - <https://arxiv.org/abs/2408.03326>
   - <https://github.com/LLaVA-VL/LLaVA-NeXT>
2. **VideoLLaMA 2**（2024-06）：STC 时空连接器包含空间处理、三维下采样与 MLP；另有音频分支。
   - <https://arxiv.org/abs/2406.07476>
   - <https://github.com/DAMO-NLP-SG/VideoLLaMA2>
3. **Qwen2.5-VL**（2025-02）：动态分辨率／帧率，双帧时空块，2×2 合并器、窗口与全局注意力、绝对时间对齐 MRoPE。本课尺寸具体限定 7B 版本。
   - <https://arxiv.org/abs/2502.13923>
   - <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json>
4. **MOVA**（2026-02）：SII–OpenMOSS 团队；图像和文本到同步视频／音频。Wan2.2 I2V A14B 视频塔、约 1.3B 音频塔和约 2.6B 双向桥接；总参数约 32B、激活约 18B。视频 VAE 是 Wan2.1，不应将骨干模型版本与 VAE 版本混写。音频塔先预训练，联合阶段两塔与桥接同时更新。Aligned RoPE 以共同时间轴对齐不同模态。
   - <https://arxiv.org/abs/2602.08794>
   - <https://github.com/OpenMOSS/MOVA>
   - 样例源：官方 README 第一段内嵌展示视频 <https://gist.github.com/user-attachments/assets/cee573cc-56ce-4987-beef-0b374e1ed3b7>；保留完整画面与原音轨，缩放为 768 宽并重编码。官方仓库的单人和多人语音示例也完整保留。课堂引用需保留作者来源；仓库代码许可证不自动等同于全部演示素材的再分发许可。

官方架构图直接保留原图，不将英文模型模块擅自重写成不同结构；相邻中文正文与自制维度图辅助解释。模型来源核对日期：2026-11-23。

## 视频生成案例和行业数据

Sora 用于回顾 2024 年的时空潜变量路线和 2025 年 Sora 2 的同步声音。Seedance 用于观察多模态参考和按时间编辑，即梦是使用模型的创作平台。

- [Sora 官方样例](https://openai.com/index/sora/)与 [Sora 2 官方发布及停服公告](https://openai.com/index/sora-2/)
- [Seedance 2.0 发布](https://seed.bytedance.com/zh/blog/official-launch-of-seedance-2-0)与 [Seedance 2.5 发布及示例](https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5)
- [Sora Android 上线首日生成量](https://openai.com/index/shipping-sora-for-android-with-codex/)与[艾瑞监测的即梦 App 月独立设备数（报告第 22 页）](https://pdf.dfcfw.com/pdf/H3_AP202605191822433573_1.pdf)
- [快手公布的可灵累计创作者与生成量](https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-annualized-revenue-run-rate-hits-usd240-million/)及 [2026 年二季度产品收入](https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-technology-announces-second-quarter-and-interim-2026)
