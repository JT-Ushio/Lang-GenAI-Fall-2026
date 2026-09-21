# 第三讲备课说明：CLIP 与两条编码路线

面向外语＋计算机双学位本科生，按 45 分钟 × 2 课时设计。概念先行，公式和代码用于解释机制，不要求现场推导梯度。

| 课时 | 内容 | 时间 |
|---|---|---:|
| 第一课时 | 引入、表示与词袋 | 5 分钟 |
| 第一课时 | RNN、长距离依存、LSTM | 12 分钟 |
| 第一课时 | 注意力、Q/K/V 数值演示 | 13 分钟 |
| 第一课时 | Transformer 组件与上下文化 | 10 分钟 |
| 第一课时 | CNN 的局部性引入 | 5 分钟 |
| 第二课时 | CNN 卷积演示、层级与历史 | 7 分钟 |
| 第二课时 | ViT 切块与位置、CNN 对照 | 8 分钟 |
| 第二课时 | CLIP 双编码器、相似度与损失 | 18 分钟 |
| 第二课时 | 检索、零样本分类与语言学评价 | 9 分钟 |
| 第二课时 | 回顾与课后任务 | 3 分钟 |

若学生线性代数基础较弱，先读向量 shape 和矩阵行列含义，把注意力公式与损失公式作为扩展；保留 CLIP 的图文配对表与逐行／逐列比较。不要将大部分时间花在 RNN 门控推导上，CLIP 是本讲的落点。

## 教学例子与边界

- 词袋：猫吃鱼／鱼吃猫，说明词项相同但论元角色不同。
- RNN：两个维度的人为嵌入与固定递推，证明顺序敏感，不证明句法理解；循环代码逐步运行，浏览器行内通常显示该行最后一次运行输出。
- 长距离依存：The key to the cabinets near the windows is missing. 中心名词 key 控制主谓一致，不能以最近名词代替句法结构。
- 注意力：人为给定 Q/K/V，一条查询、三个候选，计算权重并加权求和；不是实际模型 attention map。注意力本身不能当成可靠的因果解释。
- CNN：3×3 手工核检测明暗左右变化；不是训练得到的卷积核。不要声称每层天然对应边缘／零件／物体。
- ViT：8×8×3 合成数组切成 16 个 2×2 块，每块 12 个数，投影成 6 维。标准示例 224×224、16×16 patch 得到 196 个块，加汇总位置为 197。
- CLIP：人为三对向量真实计算归一化、相似度、双向交叉熵；不下载权重，不伪造 CLIP 预测，不训练大模型。分数表图中的 8/2/1 是另一组纯概念示意，正文的向量计算是实际打印值。
- 原始 CLIP 采用因果文本注意力、结束标记表示，视觉可用改进 ResNet 或 ViT。不是标准翻译 Transformer 的整个 encoder-decoder，也不是图文 token 交叉注意力融合模型。
- 原始 LSTM（1997）与后续常用遗忘门版本区别已用“常见现代版本”注明。
- 自注意力去掉位置信息时，具有置换等变性质；结合对称汇总会丢失顺序区分。不要说“Transformer 只能看词袋”，因为位置表示／掩码等可提供结构信息。
- 零样本是迁移设置，不是没有预训练或保证不存在数据泄漏；候选 softmax 不是校准后的正确率。

## 课后任务参考

选择具有明确动作或空间关系的图片，制作属性、数量、位置、论元角色四种最小对照。保证图像确实能判定，而不是依赖画外知识；记录参照系。下一次可再加载真实 CLIP 检索分数，本讲不对未测例子预判模型成败。

## 编译与打开

在 `lectures` 执行 `python3 execute.py -m lecture_03`，随后在 `trace-viewer` 执行 `npm run build`。同一网站用 `?trace=var/traces/lecture_03.json&step=1&animate=1` 打开。修改课件需重新编译。

图示源脚本：`lectures/scripts/build_lecture_03_diagrams.py`，运行不需要第三方库。图示文件在 `images/lecture_03/`，均为本课程原创概念示意，不是论文图或模型测量结果。

## 主要原始资料

- Elman（1990），Finding Structure in Time：https://jeffelman.ucsd.edu/research/publications/
- Hochreiter & Schmidhuber（1997），Long Short-Term Memory：https://www.bioinf.jku.at/publications/older/2604.pdf
- Bahdanau et al.（2014/2015），Neural Machine Translation by Jointly Learning to Align and Translate：https://arxiv.org/abs/1409.0473
- Vaswani et al.（2017），Attention Is All You Need：https://arxiv.org/abs/1706.03762
- LeCun et al.（1998），Gradient-Based Learning Applied to Document Recognition：https://bottou.org/papers/lecun-98h
- Krizhevsky et al.（2012），AlexNet：https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- He et al.（2015/2016），ResNet：https://arxiv.org/abs/1512.03385
- Dosovitskiy et al.（2020/2021），ViT：https://arxiv.org/abs/2010.11929
- Radford et al.（2021），CLIP：https://proceedings.mlr.press/v139/radford21a.html
- CLIP 官方实现：https://github.com/openai/CLIP/blob/main/clip/model.py 与 https://github.com/openai/CLIP/blob/main/clip/clip.py

以上按原始研究核对历史及架构，正文使用教师设计的语言学例子。模型方法的时间线不是线性淘汰关系。

## 新增：实际网络与训练演示

五个 `runnable_*` 函数穿插在相关章节后，用现有 PyTorch 依赖执行，无需额外安装或下载权重。

- RNN：输入 `(1,3,2)`，每步状态 `(1,3,4)`，最终状态 `(1,1,4)`，比较顺序变化。
- Transformer：固定参数、关闭 dropout，只改末位置；无掩码首位置会变化，因果掩码下首位置保持相同，以断言验证。
- CNN：实际 Conv2d 扫描 6×6 边缘图，得到 4×4 响应。卷积权重是手工指定的教学滤波器。
- ViT：Conv2d 实现 patch embedding，拼入可学习 CLS 和位置向量，执行一个实际 Transformer 块，提取整图表示。随机初始化，没有预训练。
- MiniCLIP：四幅 RGB 色块＋四个颜色词编号，简化线性视觉分支和文本嵌入，执行 80 次 Adam 更新。可见训练代码直接从实际函数读取，运行日志来自编译时训练。检查损失下降、双向配对和候选打乱后的匹配。仅为训练集拟合，不是独立测试，更不是自然语言或零样本泛化。

本地验证记录（固定种子）：损失 2.3399 → 0.1545 → 0.0259（0/20/80 步），图→文训练集准确率 25% → 100% → 100%。不同库版本可能有微小数值差异，以实际输出为准。所有计算使用 CPU。

时间建议：不延长 90 分钟总时长时，用实际 RNN/CNN 演示替换前面的手算过程；优先完整讲解因果掩码实验和 MiniCLIP 训练，ViT 代码按 shape 快速浏览。五个新演示若全展开，额外约 15–20 分钟。

## 逐步查看变量

关键代码行使用 `# @inspect 变量名`；同一行查看多个值时，分别写 `@inspect`，例如 `# @inspect word, @inspect state1`。执行编译后，浏览器变量面板随播放步骤更新，可拖动到空白处。面板保留当前函数中已记录变量的最近值，切换函数后显示相应函数的变量。

- RNN：从全零状态开始，逐词查看 `word`、`state1`、`state2`，比较两个方向的最终状态。
- 注意力：查看查询、键、值、分数、归一化权重及加权结果。
- Transformer：查看因果掩码与第一个位置的变化量。
- CNN / ViT：查看卷积核、特征图、切块序列和整图向量。
- CLIP：查看相似度、双向概率与损失，以及训练前后匹配分数和训练记录。80 次训练更新仍整体跳过，展示实际采集的记录，避免逐步播放冗长训练循环。

这些值是 Python 执行时记录的快照，网页回放不会重新执行 Python；修改代码后需重新编译。
