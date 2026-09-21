# 第三讲图示

由 `lectures/scripts/build_lecture_03_diagrams.py` 生成的六张原创 SVG：两条路线总览、RNN 与注意力、CNN、ViT、CLIP 双编码器、对比学习分数表。

箭头表示教学概念中的信息流，不是模型可解释性实验；分数表为手工示意。无需网络或 AI 生图服务。

新增五张具象示意图由 `lectures/scripts/build_lecture_03_visual_examples.py` 生成：

- `rnn-memory.svg`：猫吃鱼／鱼吃猫的逐词状态，数值取自本课固定递推。
- `cnn-volume.svg`：RGB 通道叠层、3×3×3 卷积核与多张特征图。立体厚度表示通道，不是物体三维深度。
- `clip-pictorial.svg`：猫图像和英文描述通过双编码器形成可比较向量。
- `clip-learning.svg`：训练前后图文配对的概念位置，非真实嵌入投影。
- `clip-candidates.svg`：图片与多段描述比较，分数仅为手工示意。

所有图均为可编辑 SVG，显示宽度 760px，受内容区域宽度限制。
